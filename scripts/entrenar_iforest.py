import argparse, numpy as np, pathlib, yaml
from sklearn.ensemble import IsolationForest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/models/iforest.yaml")
    args = ap.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    X = data["X"]

    cfg = yaml.safe_load(open(args.config))
    model = IsolationForest(
        n_estimators=cfg.get("n_estimators", 400),
        max_samples=cfg.get("max_samples", 512),
        contamination=cfg.get("contamination", 0.05),
        random_state=cfg.get("seed", 42),
    )
    model.fit(X)
    scores = -model.score_samples(X)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir/"scores.npy", scores)
    print(f"Entrenamiento OK. Scores: {scores.shape}, salida={outdir}")

if __name__ == "__main__":
    main()
