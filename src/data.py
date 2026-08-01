"""Dataset loading. Every loader returns (X, y, feature_names)."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_credit():
    """UCI 350: Default of Credit Card Clients (Taiwan, 2005)."""
    cache = DATA_DIR / "credit_default.csv"

    if cache.exists():
        df = pd.read_csv(cache)
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
        df = ds.data.features.copy()
        df["target"] = ds.data.targets.iloc[:, 0].values
        df.to_csv(cache, index=False)

    y = df["target"].to_numpy(dtype=int)
    X_df = df.drop(columns=["target"])
    if "ID" in X_df.columns:
        X_df = X_df.drop(columns=["ID"])

    return X_df.to_numpy(dtype=float), y, list(X_df.columns)


LOADERS = {"credit": _load_credit}


def load_dataset(name):
    """Load a dataset by name. Returns (X, y, feature_names)."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(LOADERS)}")

    X, y, names = LOADERS[name]()

    assert X.shape[0] == y.shape[0], "X and y row counts disagree"
    assert set(np.unique(y)) <= {0, 1}, "y must be binary 0/1"

    return X, y, names
