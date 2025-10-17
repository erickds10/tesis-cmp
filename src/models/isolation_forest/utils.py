# utils.py
# =============================================================================
# Utilidades comunes para entrenamiento/evaluación de anomalías por ventanas
# =============================================================================

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONFIG DE FEATURES
# -----------------------------------------------------------------------------
# Trabajamos con columnas tabulares aplanadas 'feat_0', 'feat_1', ...
# Si luego defines subconjuntos por prefijo (p.ej. 'dyn_', 'ctx_'), agrega más.
FEATURE_SETS: Dict[str, Sequence[str]] = {
    # Para nuestros datasets alineados: ambas usan el prefijo 'feat_' (19 comunes)
    "dynamics_only": ["feat_"],
    "dynamics_plus_context": ["feat_"],
}


def feature_columns(df: pd.DataFrame, set_name: str) -> List[str]:
    """
    Devuelve las columnas de features para el set seleccionado.
    Selección por prefijos definidos en FEATURE_SETS. Excluye identificadores/label.
    """
    if set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {set_name}. Available: {list(FEATURE_SETS)}")
    prefixes = FEATURE_SETS[set_name]
    exclude = {"mmsi", "window_id", "label", "split"}
    feats = [c for c in df.columns if c not in exclude and any(c.startswith(p) for p in prefixes)]
    if not feats:
        # Mensaje útil para depurar
        sample = list(df.columns)[:30]
        raise ValueError(
            f"No feature columns matched for set '{set_name}' with prefixes {prefixes}. "
            f"First columns: {sample}"
        )
    # Orden estable
    return sorted(feats, key=lambda x: (len(x), x))


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
def setup_logger(name: str = "ais") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path, columns=columns)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)


def save_json(obj, path: str | Path) -> None:
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# SPLITS (por MMSI)
# -----------------------------------------------------------------------------
def load_split(split_file: str | Path) -> Dict[str, List[int]]:
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"split file not found: {split_file}")
    with open(split_file, "r") as f:
        split = json.load(f)
    # Normalizamos a int
    for k in ("train", "val", "test"):
        if k in split:
            split[k] = [int(x) for x in split[k]]
    return split


def filter_by_split(df: pd.DataFrame, split: Dict[str, List[int]], part: str) -> pd.DataFrame:
    if part not in split:
        raise ValueError(f"split part '{part}' not found. Available: {list(split.keys())}")
    return df[df["mmsi"].isin(split[part])].copy()


# -----------------------------------------------------------------------------
# NORMALIZACIÓN DE SCORES
# -----------------------------------------------------------------------------
def standardize_scores(scores: np.ndarray, method: str = "rank") -> np.ndarray:
    """
    Normaliza scores a [0,1].
    - 'rank': mapea por ranking (percentil), robusto a outliers/escala.
    - 'minmax': min-max clásico.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        s = s.ravel()

    if method == "minmax":
        mn, mx = np.nanmin(s), np.nanmax(s)
        if mx - mn <= 1e-12:
            return np.zeros_like(s)
        z = (s - mn) / (mx - mn)
        z[np.isnan(z)] = 0.0
        return z

    # rank (default)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(s), endpoint=True)
    ranks[np.isnan(s)] = 0.0
    return ranks


# -----------------------------------------------------------------------------
# CHEQUEOS
# -----------------------------------------------------------------------------
def assert_no_feature_leak(df: pd.DataFrame) -> None:
    """Asegura que columnas de etiqueta no estén entre las features."""
    leaks = [c for c in df.columns if c.lower() in {"label", "is_suspicious"}]
    if leaks:
        # Está bien que existan, pero no deben entrar en X; lo recordamos aquí.
        pass


def describe_df(df: pd.DataFrame, name: str = "df") -> Dict[str, object]:
    info = {
        "name": name,
        "shape": tuple(df.shape),
        "cols": list(df.columns),
        "head": df.head(3).to_dict(orient="list"),
    }
    return info


# -----------------------------------------------------------------------------
# PEQUEÑAS UTILIDADES
# -----------------------------------------------------------------------------
def topk_by_score(df: pd.DataFrame, k: float | int, score_col: str = "score") -> pd.DataFrame:
    """
    Devuelve el top-k por score. k puede ser fracción (0<k<1) o entero.
    """
    n = len(df)
    if n == 0:
        return df
    if 0 < k < 1:
        k = int(np.ceil(k * n))
    k = max(1, int(k))
    return df.sort_values(score_col, ascending=False).head(k).copy()
