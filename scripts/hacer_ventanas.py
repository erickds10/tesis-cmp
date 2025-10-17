import argparse, numpy as np, pandas as pd, pathlib
def make_tabular_windows(df, T=20, features=None):
    if features is None:
        features = [c for c in df.columns if df[c].dtype != 'O']
    X = []
    for i in range(0, len(df)-T+1):
        win = df[features].iloc[i:i+T].to_numpy().reshape(-1)
        X.append(win)
    return np.array(X), features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["tabular"], default="tabular")
    ap.add_argument("--seq-len", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    X, feats = make_tabular_windows(df, T=args.seq_len)
    np.savez(args.out, X=X, features=np.array(feats))
    print(f"Ventanas guardadas: {args.out}, shape={X.shape}")

if __name__ == "__main__":
    main()
