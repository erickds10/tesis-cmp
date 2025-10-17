#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera y guarda figuras de evaluación en runs/figs/ a partir de:
- JSONs de métricas en runs/if_eval/*_k*.json (salidas de eval_windows.py)
- (Opcional) PR/ROC/Hist si hay labels y preds y está duckdb disponible
- (Opcional) KS drift desde runs/diagnostics/ks_drift.csv

Requisitos:
- matplotlib, pandas, numpy, scikit-learn
- (opcional) duckdb para leer parquet sin pyarrow
"""

import os, json, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "runs/figs"
os.makedirs(OUTDIR, exist_ok=True)

# === Config de rutas (ajústalas si usas otras) ===
PRED_BASE = "runs/if_eval/preds_ctx_norm_dedup.csv"
PRED_ROB  = "runs/iforest_ctx_robust/preds_eval.csv"
PRED_ENS  = "runs/if_eval/preds_ens_robust_base.csv"
LABELS_PARQUET = "data/eval_labels_aligned.parquet"
KS_DRIFT = "runs/diagnostics/ks_drift.csv"

# === Utils ===
def save_plot(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def load_metrics_jsons():
    import re
    files = sorted(glob.glob("runs/if_eval/*_k*.json"))
    if not files:
        raise SystemExit("No se encontraron JSONs en runs/if_eval/*.json")
    rows=[]
    for fp in files:
        name = os.path.basename(fp)
        model = name.split("_k")[0]
        kpart = name.split("_k", 1)[1].replace(".json", "")

        # Extrae el primer bloque numérico (soporta '0015', '15', '0.015', '50_dedup', etc.)
        m = re.search(r'(\d+(?:\.\d+)?)', kpart)
        if not m:
            # si no hay número, saltamos
            continue
        token = m.group(1)
        # Normaliza a fracción (0.015, 0.05, 0.01, etc.)
        if "." in token:
            k = float(token)
        else:
            # sin punto: interpretamos según longitud (convención de tus archivos)
            # '0015' -> 0.015, '015' -> 0.015, '15' -> 0.015, '50' -> 0.05, '1' -> 0.1
            L = len(token)
            k = int(token) / (10 ** L)

        data = json.load(open(fp)).get("global", {})
        # por si algún JSON no tiene todos los campos
        rows.append({
            "model": model, "K": float(k),
            "precision": data.get("precision_at_k", float("nan")),
            "recall": data.get("recall_at_k", float("nan")),
            "f1": data.get("f1_at_k", float("nan")),
            "roc_auc": data.get("roc_auc", float("nan")),
            "pr_auc": data.get("pr_auc", float("nan")),
            "threshold": data.get("threshold", float("nan")),
        })
    df = pd.DataFrame(rows)
    name_map = {
        "preds_ctx_norm_dedup": "Base (ctx_norm)",
        "preds_eval": "Robust (subset)",
        "preds_ens_robust_base": "Ensamble (robust+base)"
    }
    df["model_label"] = df["model"].map(lambda m: name_map.get(m, m))
    return df


def plot_metric_vs_k(df, metric, fname, ylabel):
    fig = plt.figure()
    for m, g in df.groupby("model_label"):
        g = g.sort_values("K")
        plt.plot(g["K"], g[metric], marker="o", label=m)
    plt.xlabel("K (fracción investigada)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs K")
    plt.grid(True, linestyle=":")
    plt.legend()
    return save_plot(fig, fname)

def bar_compare_at_k(df, kvalue, tag):
    sel = df[df["K"].round(5) == kvalue].copy()
    if sel.empty:
        return []
    sel = sel.sort_values("model_label")
    out = []

    for metric, ylabel in [
        ("precision", "Precision@K"),
        ("recall",    "Recall@K"),
        ("f1",        "F1@K"),
        ("threshold", "Umbral (score)"),
    ]:
        fig = plt.figure()
        plt.bar(sel["model_label"], sel[metric])
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (K={kvalue:.3f})")
        plt.xticks(rotation=15, ha="right")
        out.append(save_plot(fig, f"bar_{metric}_k{tag}.png"))
    return out

def optional_pr_roc_hist():
    """Intenta generar PR/ROC/Hist leyendo labels desde parquet con duckdb (si está instalado)."""
    preds_exist = os.path.exists(PRED_BASE) and os.path.exists(PRED_ROB) and os.path.exists(PRED_ENS)
    if not preds_exist:
        return []

    try:
        import duckdb
    except Exception:
        print("duckdb no disponible en este entorno; omito PR/ROC/Hist.")
        return []

    # Carga labels
    con = duckdb.connect()
    labels = con.execute(
        f"SELECT mmsi::BIGINT AS mmsi, window_id::BIGINT AS window_id, label::BIGINT AS label "
        f"FROM read_parquet('{LABELS_PARQUET}')"
    ).df()

    def load_pred(path):
        df = pd.read_csv(path)
        need = {"mmsi","window_id","score"}
        assert need.issubset(df.columns), f"{path} debe tener columnas {need}"
        df["mmsi"] = df["mmsi"].astype("int64", errors="ignore")
        df["window_id"] = df["window_id"].astype("int64", errors="ignore")
        return df.merge(labels, on=["mmsi","window_id"], how="inner").dropna(subset=["score","label"])

    out = []
    try:
        base = load_pred(PRED_BASE)
        robust = load_pred(PRED_ROB) if os.path.exists(PRED_ROB) else None
        ens = load_pred(PRED_ENS) if os.path.exists(PRED_ENS) else None

        from sklearn.metrics import (
            precision_recall_curve, average_precision_score,
            roc_curve, roc_auc_score
        )

        def pr_roc(M, lab):
            y = M["label"].to_numpy(dtype=int)
            s = M["score"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            return (prec, rec, ap, fpr, tpr, auc)

        # PR
        fig = plt.figure()
        prec, rec, ap, fpr, tpr, auc = pr_roc(base, "Base")
        plt.plot(rec, prec, label=f"Base (AP={ap:.3f})")
        if robust is not None:
            prec, rec, ap, fprr, tprr, aucr = pr_roc(robust, "Robust")
            plt.plot(rec, prec, label=f"Robust (AP={ap:.3f})")
        if ens is not None:
            prec, rec, ap, fpre, tpre, auce = pr_roc(ens, "Ensamble")
            plt.plot(rec, prec, label=f"Ensamble (AP={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Curva Precision–Recall (eval)")
        plt.legend()
        out.append(save_plot(fig, "pr_curve.png"))

        # ROC
        fig = plt.figure()
        _, _, _, fpr, tpr, auc = pr_roc(base, "Base")
        plt.plot(fpr, tpr, label=f"Base (AUC={auc:.3f})")
        if 'fprr' in locals():
            plt.plot(fprr, tprr, label=f"Robust (AUC={aucr:.3f})")
        if 'fpre' in locals():
            plt.plot(fpre, tpre, label=f"Ensamble (AUC={auce:.3f})")
        plt.plot([0,1],[0,1], linestyle="--", label="azar")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("Curva ROC (eval)")
        plt.legend()
        out.append(save_plot(fig, "roc_curve.png"))

        # Histograma de scores (base)
        fig = plt.figure()
        sample_max = 2_000_000
        Mb = base.sample(n=min(len(base), sample_max), random_state=42)
        plt.hist(Mb.loc[Mb["label"]==0, "score"], bins=50, alpha=0.6, label="negativos")
        plt.hist(Mb.loc[Mb["label"]==1, "score"], bins=50, alpha=0.6, label="positivos")
        plt.xlabel("score"); plt.ylabel("frecuencia")
        plt.title("Distribución de scores (Base)")
        plt.legend()
        out.append(save_plot(fig, "score_hist_base.png"))

        # TP acumulados vs K (base)
        fig = plt.figure()
        m = len(base); pos = int((base["label"]==1).sum())
        Mb_sorted = base.sort_values("score", ascending=False).reset_index(drop=True)
        cum_tp = (Mb_sorted["label"]==1).cumsum().to_numpy()
        grid = np.linspace(0.005, 0.10, 20)
        xs, ys = [], []
        for k in grid:
            topk = int(math.ceil(k*m))
            xs.append(k); ys.append(cum_tp[topk-1]/pos if topk>0 else 0.0)
        plt.plot(xs, ys, marker="o")
        plt.xlabel("K (fracción investigada)"); plt.ylabel("Recall acumulado")
        plt.title("TP acumulados vs K (Base)")
        plt.grid(True, linestyle=":")
        out.append(save_plot(fig, "tp_acumulados_vs_k.png"))

    except Exception as e:
        print(f"[Aviso] Omitiendo PR/ROC/Hist por error: {e}")
    return out

def optional_ks_drift():
    if not os.path.exists(KS_DRIFT):
        return []
    try:
        ks = pd.read_csv(KS_DRIFT).sort_values("ks", ascending=False).head(15)
        fig = plt.figure(figsize=(8,5))
        plt.bar(ks["feature"], ks["ks"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("KS")
        plt.title("Drift por feature (Top 15 KS)")
        return [save_plot(fig, "ks_drift_top.png")]
    except Exception as e:
        print(f"[Aviso] Omitiendo KS drift por error: {e}")
        return []

def main():
    df = load_metrics_jsons()

    # Curvas vs K
    p1 = plot_metric_vs_k(df, "precision", "precision_vs_k.png", "Precision@K")
    p2 = plot_metric_vs_k(df, "recall",    "recall_vs_k.png",    "Recall@K")
    p3 = plot_metric_vs_k(df, "f1",        "f1_vs_k.png",        "F1@K")

    # Lift@K = recall@K / K
    df_lift = df.copy()
    df_lift["lift"] = df_lift["recall"] / df_lift["K"]
    fig = plt.figure()
    for m, g in df_lift.groupby("model_label"):
        g = g.sort_values("K")
        plt.plot(g["K"], g["lift"], marker="o", label=m)
    plt.xlabel("K (fracción investigada)")
    plt.ylabel("Lift@K (= recall/K)")
    plt.title("Lift@K vs K")
    plt.grid(True, linestyle=":")
    plt.legend()
    p4 = save_plot(fig, "lift_vs_k.png")

    # Comparativas en K=0.015 y K=0.05
    out_bars = []
    out_bars += bar_compare_at_k(df, 0.015, "015")
    out_bars += bar_compare_at_k(df, 0.050, "050")

    # PR-AUC y ROC-AUC por modelo (barras)
    auc_df = df.groupby("model_label", as_index=False).agg({"pr_auc":"max","roc_auc":"max"})
    fig = plt.figure()
    plt.bar(auc_df["model_label"], auc_df["pr_auc"])
    plt.ylabel("PR-AUC"); plt.title("PR-AUC por modelo")
    plt.xticks(rotation=15, ha="right")
    p5 = save_plot(fig, "pr_auc_by_model.png")

    fig = plt.figure()
    plt.bar(auc_df["model_label"], auc_df["roc_auc"])
    plt.ylabel("ROC-AUC"); plt.title("ROC-AUC por modelo")
    plt.xticks(rotation=15, ha="right")
    p6 = save_plot(fig, "roc_auc_by_model.png")

    # Opcionales (si hay insumos)
    extra = []
    extra += optional_pr_roc_hist()
    extra += optional_ks_drift()

    # Manifest
    generated = sorted(glob.glob(os.path.join(OUTDIR, "*.png")))
    print("\nFiguras generadas:")
    for g in generated:
        print("-", g)

if __name__ == "__main__":
    main()
