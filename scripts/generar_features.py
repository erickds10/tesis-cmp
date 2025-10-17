import argparse, pandas as pd, numpy as np, pathlib
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.input)
    # TODO: calcular distancia, velocidad, heading norm, etc.
    if "course" in df.columns:
        df["course_norm"] = df["course"] % 360
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Guardado: {args.out}, cols={list(df.columns)[:10]}...")

if __name__ == "__main__":
    main()
