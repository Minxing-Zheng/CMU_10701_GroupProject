"""Utility loaders for LETOR-style data and MovieLens splits.

MovieLens helpers keep a light dependency stack (pandas + sklearn) and
optionally download ml-100k. Splits are saved as tab-separated txt files
with columns: user_id, item_id, rating, timestamp.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def parse_letor_line(line: str) -> Tuple[int, int, Dict[int, float]]:
    parts = line.strip().split()
    label = int(float(parts[0]))
    qid = None
    feats: Dict[int, float] = {}

    for tok in parts[1:]:
        if tok.startswith("qid:"):
            qid = int(tok.split(":")[1])
        else:
            k, v = tok.split(":")
            feats[int(k)] = float(v)

    if qid is None:
        raise ValueError(f"Missing qid in line: {line[:50]}...")
    return label, qid, feats


def load_letor_split(path: str | Path, feature_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a LETOR-style txt file into dense X, y, qid arrays."""
    groups_X, groups_y = {}, {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            label, qid, feats = parse_letor_line(line)
            groups_X.setdefault(qid, []).append(feats)
            groups_y.setdefault(qid, []).append(label)

    qids = sorted(groups_X.keys())
    X_rows, y_rows, qid_rows = [], [], []
    for qid in qids:
        rows = groups_X[qid]
        for d in rows:
            x = np.zeros(feature_dim, dtype=np.float32)
            for k, v in d.items():
                if 1 <= k <= feature_dim:
                    x[k - 1] = v
            X_rows.append(x)
        y_rows.extend(groups_y[qid])
        qid_rows.extend([qid] * len(groups_y[qid]))

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    qid_arr = np.array(qid_rows, dtype=np.int64)
    return X, y, qid_arr


def ensure_movielens(root: str | Path = "data/movielens", download: bool = False) -> Path:
    """Return path to extracted MovieLens directory, downloading ml-100k if requested."""
    root = Path(root)
    target = root / "ml-100k"
    if target.exists():
        return target
    if not download:
        raise FileNotFoundError(f"{target} not found. Set download=True or place the dataset manually.")

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-100k.zip"
    try:
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MovieLens download failed; please download manually.") from exc
    return target


def load_movielens_df(data_dir: str | Path) -> pd.DataFrame:
    """Load ratings from ml-100k (u.data) or ml-latest-small (ratings.csv)."""
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.csv"
    legacy_path = data_dir / "u.data"

    if ratings_path.exists():
        df = pd.read_csv(ratings_path)
        df = df.rename(columns={"userId": "user_id", "movieId": "item_id", "rating": "rating"})
        if "timestamp" not in df.columns:
            df["timestamp"] = 0
    elif legacy_path.exists():
        df = pd.read_csv(
            legacy_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    else:
        raise FileNotFoundError(f"Could not find ratings in {data_dir}")

    return df[["user_id", "item_id", "rating", "timestamp"]]


def split_movielens_to_txt(
    data_root: str | Path = "data/movie",
    download: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    train_size: Optional[float] = 0.6,
) -> Dict[str, Path]:
    """
    Split MovieLens into train/valid/test txt files (tab-separated).

    Defaults to 60% train, 20% valid, and 20% test. If the provided fractions
    sum to < 1, the remainder is left unused to honor the requested sizes.
    To recover the previous behavior (train = 1 - test - valid), pass
    train_size=None.
    """
    if download:
        # Download into a temporary directory and clean it up after writing splits.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "ml-100k.zip"
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            df = load_movielens_df(tmpdir / "ml-100k")
            splits = _write_movielens_splits(df, data_root, train_size, test_size, val_size, seed)
        return splits
    else:
        mv_dir = ensure_movielens(data_root, download=False)
        df = load_movielens_df(mv_dir)
        return _write_movielens_splits(df, data_root, train_size, test_size, val_size, seed)


def _write_movielens_splits(
    df: pd.DataFrame,
    data_root: str | Path,
    train_size: Optional[float],
    test_size: float,
    val_size: float,
    seed: int,
) -> Dict[str, Path]:
    """Helper to split and write MovieLens data without persisting source archives."""
    if train_size is None:
        train_size = 1.0 - test_size - val_size

    total = train_size + test_size + val_size
    if train_size <= 0 or test_size < 0 or val_size < 0:
        raise ValueError("train_size must be > 0 and test/val sizes must be non-negative.")
    if total - 1.0 > 1e-6:
        raise ValueError("train_size + test_size + val_size cannot exceed 1.0.")

    # First carve out the training set; then take validation and test as absolute fractions of the full set.
    train, remainder = train_test_split(df, train_size=train_size, random_state=seed)

    remainder_frac = 1.0 - train_size
    if val_size > remainder_frac + 1e-8:
        raise ValueError("val_size is larger than the non-train remainder.")
    val_ratio = val_size / remainder_frac if remainder_frac > 0 else 0.0
    if val_size > 0 and val_ratio >= 1 - 1e-8:
        val, remainder = remainder, df.iloc[0:0]
    elif val_size > 0:
        val, remainder = train_test_split(remainder, train_size=val_ratio, random_state=seed)
    else:
        val, remainder = df.iloc[0:0], remainder

    remainder_frac -= val_size
    if test_size > remainder_frac + 1e-8:
        raise ValueError("test_size is larger than the remaining pool after train/valid.")
    test_ratio = test_size / remainder_frac if remainder_frac > 0 else 0.0
    if test_size > 0 and test_ratio >= 1 - 1e-8:
        test, _ = remainder, df.iloc[0:0]
    elif test_size > 0:
        test, _ = train_test_split(remainder, train_size=test_ratio, random_state=seed)
    else:
        test, _ = df.iloc[0:0], remainder

    out_dir = Path(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.txt",
        "valid": out_dir / "valid.txt",
        "test": out_dir / "test.txt",
    }
    for name, part in zip(["train", "valid", "test"], [train, val, test]):
        part.to_csv(paths[name], sep="\t", index=False)
    return paths


def load_movielens_split(path: str | Path) -> pd.DataFrame:
    """Load a MovieLens split created by split_movielens_to_txt."""
    df = pd.read_csv(path, sep="\t")
    expected = {"user_id", "item_id", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    return df


def iter_movielens_splits(paths: Dict[str, Path]) -> Iterable[Tuple[str, pd.DataFrame]]:
    for name, p in paths.items():
        yield name, load_movielens_split(p)
