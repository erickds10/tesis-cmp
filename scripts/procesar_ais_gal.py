import argparse, pandas as pd, pathlib
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # TODO: limpieza real; por ahora solo ejemplo
    df = df.dropna(subset=["lat","lon","timestamp"]).copy() if set(["lat","lon","timestamp"]).issubset(df.columns) else df
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Guardado: {args.out}, filas={len(df)}")

if __name__ == "__main__":
    main()
