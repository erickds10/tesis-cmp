# tools/save_cluster_anomaly.py
import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- utilidades -------------------------------------------------------------
def select_topk_mask(scores: pd.Series, kfrac: float):
    """
    Devuelve:
      mask -> booleano con EXACTAMENTE K verdaderos (top-K por score)
      thr  -> umbral numérico (= mínimo score dentro del top-K)
    Selección estable (respeta el orden relativo en empates).
    """
    s = scores.to_numpy()
    n = len(s)
    K = max(1, int(math.ceil(kfrac * n)))
    order = np.argsort(-s, kind="stable")        # mayor->menor
    keep_idx = order[:K]
    mask = np.zeros(n, dtype=bool)
    mask[keep_idx] = True
    thr = float(s[keep_idx].min())
    return mask, thr

def feat_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("feat_")]

# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV con columnas mmsi, window_id, score")
    ap.add_argument("--eval-parquet", required=True, help="Parquet con features de evaluación")
    ap.add_argument("--k", type=float, default=0.015, help="fracción top-K para marcar anomalías")
    ap.add_argument("--sample-normals", type=int, default=30000, help="muestra de normales para el scatter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/figs_extra")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Cargar predicciones y eval features
    P = pd.read_csv(args.pred)
    E = pd.read_parquet(args.eval_parquet)
    # tipos seguros
    for c in ("mmsi", "window_id"):
        if c in P.columns: P[c] = P[c].astype("int64", errors="ignore")
        if c in E.columns: E[c] = E[c].astype("int64", errors="ignore")

    # 2) Merge (sólo necesitamos features + score)
    M = P.merge(E, on=["mmsi", "window_id"], how="inner")
    if "score" not in M.columns:
        raise ValueError("El CSV de predicciones debe tener columna 'score'.")

    # 3) Selección top-K EXACTA
    mask_topk, thr = select_topk_mask(M["score"], args.k)
    M["is_topk"] = mask_topk

    n = len(M); k = int(mask_topk.sum())
    print(f"[DEBUG] N={n:,} | K={k} ({k/n:.3%}) | thr≈{thr:.6f} | "
          f"min={M['score'].min():.6f} max={M['score'].max():.6f}")

    # 4) Preparar muestra para el scatter (normales + top-K)
    normals = M.loc[~M["is_topk"]]
    if len(normals) > args.sample_normals:
        normals = normals.sample(args.sample_normals, random_state=args.seed)
    anoms = M.loc[M["is_topk"]]
    S = pd.concat([normals, anoms], ignore_index=True)

    # 5) PCA 2D sobre features
    fcols = feat_columns(S)
    if not fcols:
        raise ValueError("No se encontraron columnas feat_* en el parquet de evaluación.")
    X = S[fcols].to_numpy(dtype=float)
    # imputación segura SOLO para el plot
    X = np.nan_to_num(X, copy=False)

    # PCA fit sólo con normales si existen; si no, con todo
    if len(normals) > 0:
        X_norm = normals[fcols].to_numpy(dtype=float)
        X_norm = np.nan_to_num(X_norm, copy=False)
        pca = PCA(n_components=2, random_state=args.seed).fit(X_norm)
    else:
        pca = PCA(n_components=2, random_state=args.seed).fit(X)
    Z = pca.transform(X)

    # 6) Scatter: normales (azul) vs top-K (rojo con borde verde)
    is_top = S["is_topk"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 9))
    if (~is_top).any():
        ax.scatter(Z[~is_top, 0], Z[~is_top, 1], s=8, alpha=0.25, label="Normal", linewidths=0)
    if (is_top).any():
        ax.scatter(Z[is_top, 0], Z[is_top, 1], s=18, alpha=0.8,
                   facecolors="#d62728", edgecolors="#2ca02c", linewidths=0.8,
                   label=f"Anomalías (top {args.k*100:.1f}%)")
    ax.set_title("Ventanas de evaluación (PCA 2D) — Top-K resaltadas")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.legend()
    out_scatter = os.path.join(args.out_dir, f"anomaly_scatter_top{int(args.k*1000):03d}p.png")
    plt.tight_layout(); plt.savefig(out_scatter, dpi=160); plt.close()

    # 7) Curva de scores ordenados (tipo “elbow”)
    s_sorted = np.sort(M["score"].to_numpy())
    idx = np.arange(1, len(s_sorted) + 1)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(idx, s_sorted, linewidth=1.5)
    ax2.axvline(x=k, linestyle="--")
    ax2.set_title("Scores ordenados (línea punteada marca el top-K)")
    ax2.set_xlabel("Índices ordenados"); ax2.set_ylabel("Score")
    out_curve = os.path.join(args.out_dir, "scores_sorted_curve.png")
    plt.tight_layout(); plt.savefig(out_curve, dpi=160); plt.close()

    print("✅ Guardados:")
    print(f"- {out_scatter}")
    print(f"- {out_curve}")
    print(f"   Normales ploteadas: {len(normals):,} | Anomalías top-K: {len(anoms):,} | Total: {len(S):,}")

if __name__ == "__main__":
    main()
