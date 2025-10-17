import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

def topk_threshold(scores: np.ndarray, k: float) -> float:
    """Devuelve el umbral tal que el top-k de scores queda etiquetado como positivo."""
    k = float(k)
    n = len(scores)
    n_top = max(1, int(round(k * n)))
    # umbral = percentil de 1 - k (si scores más altos = más anómalo)
    thr = np.partition(scores, -n_top)[-n_top]
    return float(thr)

def metrics_at_k(y_true, scores, k):
    thr = topk_threshold(scores, k)
    y_pred = (scores >= thr).astype(int)
    return {
        "precision_at_k": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_k": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_k": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": thr
    }

def compute_metrics(df, k):
    y = df["label"].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    # Métricas globales
    out = {}
    # Cuidado: ROC requiere ambas clases
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
    else:
        out["roc_auc"] = None
    out["pr_auc"] = float(average_precision_score(y, s))
    out.update(metrics_at_k(y, s, k))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV con columnas mmsi,window_id,score")
    ap.add_argument("--labels", required=True, help="Parquet con columnas mmsi,window_id,label")
    ap.add_argument("--metrics", nargs="+", default=["pr_auc","roc_auc","precision_at_k","recall_at_k","f1_at_k"])
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--group-by", nargs="+", default=["global"], help="global y/o mmsi")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    lab = pd.read_parquet(args.labels)
    df = pred.merge(lab, on=["mmsi","window_id"], how="inner").dropna(subset=["score","label"])
    if df.empty:
        raise SystemExit("No hay intersección entre predicciones y etiquetas.")

    result = {}

    if "global" in args.group_by:
        result["global"] = compute_metrics(df, args.k)

    if "mmsi" in args.group_by:
        per_m = {}
        for mmsi, g in df.groupby("mmsi"):
            try:
                per_m[int(mmsi)] = compute_metrics(g, args.k)
            except Exception:
                # Si falla por clase única, lo marcamos como None
                per_m[int(mmsi)] = {"roc_auc": None, "pr_auc": None,
                                    "precision_at_k": None, "recall_at_k": None, "f1_at_k": None, "threshold": None}
        result["by_mmsi"] = per_m

    # Filtrar por lista solicitada (args.metrics) en la sección global
    # (dejamos by_mmsi completo para análisis posterior)
    if "global" in result:
        result["global"] = {k:v for k,v in result["global"].items() if k in set(args.metrics + ["threshold"])}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
