def load_ais(path):
    import pandas as pd
    return pd.read_parquet(path)
