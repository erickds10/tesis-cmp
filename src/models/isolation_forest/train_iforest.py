# train_iforest.py
import os
import argparse
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Reutilizamos tus utilidades
from utils import feature_columns, standardize_scores

def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_split(split_file, mmsis_present):
    """
    Carga split JSON con claves {train, val, test}. 
    Si no existe o no intersecta, crea uno 70/15/15 sobre mmsis_present.
    """
    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            sp = json.load(f)
        # Solo conservar MMSIs que existan en el dataset
        split = {k: [int(x) for x in v if int(x) in mmsis_present] for k, v in sp.items()}
        # Si todo quedó vacío, fall back a crear uno
        if sum(len(v) for v in split.values()) == 0:
            log("Split no coincide con el dataset. Creando uno 70/15/15.")
            split = None
    else:
        split = None

    if split is None:
        mmsis = sorted(list(mmsis_present))
        rng = np.random.default_rng(42)
        rng.shuffle(mmsis)
        n = len(mmsis)
        ntr = int(0.70 * n)
        nva = int(0.15 * n)
        split = {
            "train": mmsis[:ntr],
            "val": mmsis[ntr:ntr + nva],
            "test": mmsis[ntr + nva:]
        }
        os.makedirs("splits", exist_ok=True)
        with open("splits/split_autogen.json", "w") as f:
            json.dump(split, f)
        log("Split autogen guardado en splits/split_autogen.json")

    # Garantiza existencia de claves
    for k in ("train", "val", "test"):
        split.setdefault(k, [])
    return split

def make_iforest(n_estimators, max_samples, contamination, seed):
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
        verbose=0
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="Parquet con columnas mmsi, window_id, feat_* (y opcional label)")
    ap.add_argument("--features-set", required=True, help="Nombre de feature set definido en utils.feature_columns")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--split-file", default="", help="JSON con {train,val,test} por MMSI")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", dest="out_dir", required=True, help="Carpeta de salida (se crea si no existe)")
    # Inferencia opcional en set de evaluación
    ap.add_argument("--eval-windows", default="", help="Parquet de evaluación (mmsi, window_id, feat_*)")
    ap.add_argument("--eval-out", default="", help="Ruta CSV para preds de evaluación (mmsi,window_id,score)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Cargar ventanas y seleccionar features
    log(f"Cargando {args.windows} ...")
    df = pd.read_parquet(args.windows)
    if "mmsi" not in df.columns or "window_id" not in df.columns:
        raise ValueError("El parquet debe contener columnas 'mmsi' y 'window_id'.")

    feats = feature_columns(df, args.features_set)
    if not feats:
        # fallback: tomar todas feat_*
        feats = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feats:
        raise ValueError("No se encontraron columnas de features (feat_*).")

    # 2) Construir / validar split por MMSI
    mmsis_present = set(int(x) for x in df["mmsi"].unique())
    split = load_split(args.split_file, mmsis_present)

    parts = {}
    for k in ("train", "val", "test"):
        parts[k] = df[df["mmsi"].isin(split[k])].copy()

    for k in parts:
        log(f"{k}: {parts[k].shape} | MMSI únicos={parts[k]['mmsi'].nunique()}")

    # Sanity: si no hay nada en train, usar todo para train
    if len(parts["train"]) == 0:
        log("Advertencia: split 'train' vacío. Usando todo el dataset como train.")
        parts["train"] = df

    # 3) Entrenamiento Isolation Forest
    log(f"Entrenando IForest ({len(feats)} feats) ...")
    model = make_iforest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        seed=args.seed
    )
    Xtr = parts["train"][feats].to_numpy(dtype=float)
    model.fit(Xtr)
    joblib.dump(model, os.path.join(args.out_dir, "model.joblib"))
    log(f"Modelo guardado en {args.out_dir}/model.joblib")

    # 4) Scoring por split y guardado de preds
    rows = []
    for k in ("train", "val", "test"):
        if len(parts[k]) == 0:
            continue
        X = parts[k][feats].to_numpy(dtype=float)
        # score_samples: valores negativos = más anómalo → invertimos y estandarizamos
        scores = -model.score_samples(X)
        scores = standardize_scores(scores, method="rank")  # [0,1]
        part_out = parts[k][["mmsi", "window_id"]].copy()
        part_out["split"] = k
        part_out["score"] = scores
        rows.append(part_out)

    if rows:
        preds = pd.concat(rows, ignore_index=True)
        preds.to_csv(os.path.join(args.out_dir, "preds.csv"), index=False)
        log(f"preds.csv -> {preds.shape}")
    else:
        log("No hay filas en ningún split; no se generó preds.csv")

    # 5) Inferencia opcional sobre eval
    if args.eval_windows:
        if not args.eval_out:
            # si no especifica, escribir al lado del run
            base = os.path.basename(os.path.normpath(args.out_dir))
            eval_out = f"runs/if_eval/preds_{base}.csv"
        else:
            eval_out = args.eval_out

        os.makedirs(os.path.dirname(eval_out), exist_ok=True)
        log(f"Cargando eval: {args.eval_windows}")
        ev = pd.read_parquet(args.eval_windows)
        use = [c for c in feats if c in ev.columns]
        if len(use) == 0:
            raise ValueError("El eval no tiene columnas en común con los feats del modelo.")
        Sev = -model.score_samples(ev[use].to_numpy(dtype=float))
        Sev = standardize_scores(Sev, method="rank")
        ev_out = ev[["mmsi", "window_id"]].copy()
        ev_out["score"] = Sev
        ev_out.to_csv(eval_out, index=False)
        log(f"Inferencia eval -> {eval_out} {ev_out.shape}")

if __name__ == "__main__":
    main()
