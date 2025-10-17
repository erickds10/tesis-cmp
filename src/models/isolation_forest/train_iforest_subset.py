# train_iforest_subset.py
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump

def feature_cols(df, exclude_idx=None):
    feats = sorted([c for c in df.columns if c.startswith("feat_")],
                   key=lambda x: int(x.split("_")[1]))
    if exclude_idx:
        excl = {int(i) for i in exclude_idx}
        feats = [c for c in feats if int(c.split("_")[1]) not in excl]
    return feats

def standardize_rank(scores):
    r = pd.Series(scores).rank(method="average") / len(scores)
    return r.to_numpy(dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)              # train/val/test (normalizado)
    ap.add_argument("--eval_windows", required=True)         # eval (normalizado)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_indices", default="")         # p.ej. "7,8,10,12,13"
    ap.add_argument("--labels_parquet", default="")          # opcional: fuente de labels si falta
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.windows)
    ev = pd.read_parquet(args.eval_windows)

    # Asegura mmsi/window_id como int64 para merges limpias
    for D in (df, ev):
        if "mmsi" in D: D["mmsi"] = D["mmsi"].astype("int64", errors="ignore")
        if "window_id" in D: D["window_id"] = D["window_id"].astype("int64", errors="ignore")

    # Recupera labels si faltan
    if "label" not in df.columns:
        if args.labels_parquet and os.path.exists(args.labels_parquet):
            lab = pd.read_parquet(args.labels_parquet)[["mmsi","window_id","label"]]
            lab["mmsi"] = lab["mmsi"].astype("int64", errors="ignore")
            lab["window_id"] = lab["window_id"].astype("int64", errors="ignore")
            before = len(df)
            df = df.merge(lab, on=["mmsi","window_id"], how="left")
            print(f"Labels añadidas por merge: {before}->{len(df)} | n_missing_labels = {df['label'].isna().sum()}")
            df["label"] = df["label"].fillna(0).astype("int64")
        else:
            print("[ADVERTENCIA] 'label' no existe y no se pasó --labels_parquet; se asume label=0.")
            df["label"] = 0

    excl = [int(x) for x in args.exclude_indices.split(",") if x.strip()!=""]
    feats = feature_cols(df, exclude_idx=excl)
    if not feats:
        raise RuntimeError("Sin columnas de features tras la exclusión.")
    print(f"Features usados: {len(feats)} | excluidos: {excl}")

    # Train solo con MMSI del split y label==0
    split = json.load(open(args.split_json))
    mmsi_train = set(map(int, split["train"]))
    df["mmsi_int"] = df["mmsi"].astype("int64")
    tr = df[(df["label"]==0) & (df["mmsi_int"].isin(mmsi_train))]

    X = tr[feats].to_numpy(dtype=float)
    print("Train windows:", X.shape)

    ifo = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        bootstrap=False,
        random_state=args.seed,
        n_jobs=-1
    ).fit(X)

    dump(ifo, os.path.join(args.out, "model.joblib"))
    print("Modelo guardado.")

    # Preds para trazabilidad en todo windows_in
    df_out = df[["mmsi","window_id","label"]].copy()
    df_out["score"] = standardize_rank(-ifo.score_samples(df[feats].to_numpy(dtype=float)))
    df_out.to_csv(os.path.join(args.out, "preds_trainvaltest.csv"), index=False)
    print("preds_trainvaltest.csv ->", df_out.shape)

    # Inferencia en eval
    use = [c for c in feats if c in ev.columns]
    s_ev = standardize_rank(-ifo.score_samples(ev[use].to_numpy(dtype=float)))
    ev_out = ev[["mmsi","window_id"]].copy()
    ev_out["score"] = s_ev
    ev_out.to_csv(os.path.join(args.out, "preds_eval.csv"), index=False)
    print("preds_eval.csv ->", ev_out.shape)

if __name__ == "__main__":
    main()
