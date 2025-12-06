import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset"
CACHE_ROOT = DATASET_ROOT / "cache"
try:
    import pyarrow  

    PARQUET_ENGINE = "pyarrow"
except Exception:
    PARQUET_ENGINE = None

#### parameter setup of Linear, MLP and tree based models.
LINEAR_EPOCHS = 50
LINEAR_LR = 0.005
LINEAR_BATCH = 4096

PAIRWISE_EPOCHS = 40
PAIRWISE_MAX_PAIRS = 4096
NEURAL_EPOCHS = 50
NEURAL_BATCH = 256
NEURAL_LR = 1e-3
NEURAL_WEIGHT_DECAY = 1e-4
NEURAL_DROPOUT = 0.1
NEURAL_HIDDEN_DIMS = (512, 256)

LGBM_TREE_SHARED_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
    "early_stopping_rounds": 100,
    "label_gain": [0, 1, 3, 7, 15],
}
LGBM_POINTWISE_PARAMS = LGBM_TREE_SHARED_PARAMS
LGBM_LAMBDARANK_PARAMS = LGBM_TREE_SHARED_PARAMS

MODEL_CHOICES = [
    "pointwise_linear",
    "pointwise_mlp",
    "pointwise_tree",
    "lambdarank_linear",
    "lambdarank_mlp",
    "lambdarank_tree",
    "xgboost",
]
MODEL_ALIASES = {
    "pointwise": "pointwise_tree",
    "pointwise_nn": "pointwise_mlp",
    "lambdarank": "lambdarank_linear",
    "lambdarank_nn": "lambdarank_mlp",
    "lambdamart": "lambdarank_tree",
}


USAGE_GUIDE = """\
Available datasets
------------------
- mq2007, mq2008: --fold 1,2,3,4,5, default fold is 1.
- mslr: --fold 1,2,3,4,5
- istella_subset
- yahoo_set1, yahoo_set2

Model choices
-------------
- pointwise_linear
- pointwise_mlp     
- pointwise_tree    
- lambdarank_linear 
- lambdarank_mlp    
- lambdarank_tree   
- xgboost
- all: run all the models and produce a summary results
Example commands
----------------
python rank_pipeline.py --dataset mq2007 --fold 1 --model lambdamart --seed1
python rank_pipeline.py --dataset yahoo_set2 --model all 
"""

# as there are trailing comments for some datasets after the feature
# we use this to remove the comments 
def _strip_comment(line: str) -> str:
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()



def _parse_feature_token(token: str) -> Optional[Tuple[str, float]]:
    if ":" not in token:
        return None
    idx, val = token.split(":", 1)
    if not idx.isdigit():
        return None
    return f"f{int(idx)}", float(val)


def log(msg: str) -> None:
    print(f"[rank_pipeline] {msg}")


def sanitize_label(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


# Construct a dcit of cache file paths for input dataset 
# Returns paths for the cache directory and the train/valid/test Parquet files.

def cache_paths(dataset_label: str) -> Dict[str, Path]:
    sanitized = sanitize_label(dataset_label)
    base = CACHE_ROOT / sanitized
    return {
        "dir": base,
        "train": base / "train.parquet",
        "valid": base / "valid.parquet",
        "test": base / "test.parquet",
    }


def cache_is_fresh(cache_files: Dict[str, Path], source_files: Sequence[Path]) -> bool:
    if not cache_files["train"].exists() or not cache_files["valid"].exists() or not cache_files["test"].exists():
        return False
    newest_source = max(p.stat().st_mtime for p in source_files)
    oldest_cache = min(cache_files["train"].stat().st_mtime, cache_files["valid"].stat().st_mtime, cache_files["test"].stat().st_mtime)
    return oldest_cache >= newest_source


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    name: str

### function to parse the LETOR datasets into data frame, with columns label, q_id, doc_id, features
def read_letor_file(path: Path, max_docs: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    
    rows: List[Dict[str, float]] = []
    feature_names: set[str] = set()
    per_query_doc_idx: defaultdict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_comment(raw_line)
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            label = float(tokens[0])
            qid = tokens[1].split(":", 1)[1]
            doc_id = per_query_doc_idx[qid]
            per_query_doc_idx[qid] += 1
            row: Dict[str, float] = {"label": label, "qid": qid, "doc_id": doc_id}
            for token in tokens[2:]:
                parsed = _parse_feature_token(token)
                if parsed is None:
                    continue
                feature, value = parsed
                row[feature] = value
                feature_names.add(feature)
            rows.append(row)
            if max_docs is not None and len(rows) >= max_docs:
                break
    if not rows:
        raise ValueError(f"No training instances found in {path}")
    frame = pd.DataFrame(rows)
    for feat in feature_names:
        if feat not in frame.columns:
            frame[feat] = 0.0
    ordered_cols = ["qid", "doc_id", "label"] + sorted(feature_names)
    frame = frame[ordered_cols].copy()
    frame["qid"] = frame["qid"].astype(str)
    frame["doc_id"] = frame["doc_id"].astype(int)
    frame["label"] = frame["label"].astype(float)
    return frame, sorted(feature_names)


def align_feature_columns(
    frames: Sequence[pd.DataFrame],
    feature_universe: Sequence[str],
) -> None:
    for frame in frames:
        missing = [feat for feat in feature_universe if feat not in frame.columns]
        for feat in missing:
            frame[feat] = 0.0
        extra = [feat for feat in frame.columns if feat.startswith("f") and feat not in feature_universe]
        if extra:
            frame.drop(columns=extra, inplace=True)
        frame[feature_universe] = frame[feature_universe].fillna(0.0)


def fast_read_libsvm(
    path: Path,
    feature_dim: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fast parser for dense LibSVM-style LTR files (e.g., Yahoo set1/set2).
    Builds a dense matrix in one pass after determining feature dimensionality.
    """
    records: List[Tuple[str, int, float, Dict[int, float]]] = []
    per_query_doc_idx: defaultdict[str, int] = defaultdict(int)
    max_feat = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_comment(raw_line)
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = float(parts[0])
            qid_token = parts[1]
            if not qid_token.startswith("qid:"):
                continue
            qid = qid_token.split(":", 1)[1]
            doc_id = per_query_doc_idx[qid]
            per_query_doc_idx[qid] += 1
            feats: Dict[int, float] = {}
            for tok in parts[2:]:
                if ":" not in tok:
                    continue
                k_str, v_str = tok.split(":", 1)
                if not k_str.isdigit():
                    continue
                k = int(k_str)
                v = float(v_str)
                feats[k] = v
                if k > max_feat:
                    max_feat = k
            records.append((qid, doc_id, label, feats))
            if max_docs is not None and len(records) >= max_docs:
                break
    if not records:
        raise ValueError(f"No training instances found in {path}")
    final_dim = feature_dim or max_feat
    num_rows = len(records)
    data = np.zeros((num_rows, final_dim), dtype=np.float32)
    qids = np.empty(num_rows, dtype=object)
    doc_ids = np.empty(num_rows, dtype=np.int32)
    labels = np.empty(num_rows, dtype=np.float32)
    for idx, (qid, doc_id, label, feats) in enumerate(records):
        qids[idx] = qid
        doc_ids[idx] = doc_id
        labels[idx] = label
        for k, v in feats.items():
            if 1 <= k <= final_dim:
                data[idx, k - 1] = v
    feature_cols = [f"f{i}" for i in range(1, final_dim + 1)]
    frame = pd.DataFrame(data, columns=feature_cols)
    frame.insert(0, "label", labels)
    frame.insert(0, "doc_id", doc_ids)
    frame.insert(0, "qid", qids.astype(str))
    return frame, feature_cols

##### the min max feature normalization function
##### Scale features into [0, 1] using stats computed on the training split.
def min_max_normalize(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    eval_dfs: Optional[Sequence[pd.DataFrame]] = None,
) -> Dict[str, Tuple[float, float]]:
    
    mins = train_df[feature_cols].min()
    maxs = train_df[feature_cols].max()
    stats: Dict[str, Tuple[float, float]] = {}
    denom = maxs - mins
    denom.replace(0.0, 1.0, inplace=True)
    train_df[feature_cols] = (train_df[feature_cols] - mins) / denom
    if eval_dfs:
        for df in eval_dfs:
            df[feature_cols] = (df[feature_cols] - mins) / denom
    for feat in feature_cols:
        stats[feat] = (float(mins[feat]), float(maxs[feat]))
    return stats


##### the data loading and split function, load differernt real dataset
def load_dataset_splits(
    dataset: str,
    fold: Optional[int] = None,
    use_cache: bool = True,
    max_docs: Optional[int] = None,
) -> DatasetBundle:
    dataset_key = dataset.lower()
    dataset_alias: Optional[str] = None
    if dataset_key.startswith("mq2007_fold"):
        fold = int(dataset_key.split("fold", 1)[1])
        dataset_key = "mq2007"
    elif dataset_key.startswith("mq2008_fold"):
        fold = int(dataset_key.split("fold", 1)[1])
        dataset_key = "mq2008"
    elif dataset_key.startswith("mslr_fold"):
        fold = int(dataset_key.split("fold", 1)[1])
        dataset_key = "mslr"
    elif dataset_key in {"yahoo_set1", "yahoo1"}:
        dataset_alias = "set1"
        dataset_key = "yahoo"
    elif dataset_key in {"yahoo_set2", "yahoo2"}:
        dataset_alias = "set2"
        dataset_key = "yahoo"
    elif dataset_key == "istella_sample":
        dataset_key = "istella"
    if dataset_key == "mq2007":
        fold = fold or 1
        base = DATASET_ROOT / "LETOR" / "MQ2007" / f"Fold{fold}"
        name = f"MQ2007-Fold{fold}"
        train_path = base / "train.txt"
        valid_path = base / "vali.txt"
        test_path = base / "test.txt"
    elif dataset_key == "mq2008":
        fold = fold or 1
        base = DATASET_ROOT / "LETOR" / "MQ2008" / f"Fold{fold}"
        name = f"MQ2008-Fold{fold}"
        train_path = base / "train.txt"
        valid_path = base / "vali.txt"
        test_path = base / "test.txt"
    elif dataset_key == "mslr":
        fold = fold or 1
        base = DATASET_ROOT / "MSLR_Macro" / "MSLR-WEB10K" / f"Fold{fold}"
        name = f"MSLR-WEB10K-Fold{fold}"
        train_path = base / "train.txt"
        valid_path = base / "vali.txt"
        test_path = base / "test.txt"
    elif dataset_key == "istella_subset":
        base = DATASET_ROOT / "istella" / "istella_subset"
        name = "Istella-Subset"
        train_path = base / "train_subset.txt"
        valid_path = base / "vali_subset.txt"
        test_path = base / "test_subset.txt"
    elif dataset_key == "istella":
        base = DATASET_ROOT / "istella" / "istella-s-letor" / "sample"
        name = "Istella-S-Sample"
        train_path = base / "train.txt"
        valid_path = base / "vali.txt"
        test_path = base / "test.txt"
    elif dataset_key == "yahoo":
        dataset_alias = dataset_alias or "set1"
        base = DATASET_ROOT / "yahoo"
        name = f"Yahoo-{dataset_alias}"
        train_path = base / f"{dataset_alias}.train.txt"
        valid_path = base / f"{dataset_alias}.valid.txt"
        test_path = base / f"{dataset_alias}.test.txt"
    else:
        raise ValueError(f"Unsupported dataset identifier: {dataset}")
    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(p)
    cache_info = cache_paths(name)
    source_files = (train_path, valid_path, test_path)
    if use_cache and cache_is_fresh(cache_info, source_files):
        log(f"Loading cached parquet dataset '{name}' from {cache_info['dir']}")
        read_kwargs = {"engine": PARQUET_ENGINE} if PARQUET_ENGINE else {}
        train_df = pd.read_parquet(cache_info["train"], **read_kwargs)
        valid_df = pd.read_parquet(cache_info["valid"], **read_kwargs)
        test_df = pd.read_parquet(cache_info["test"], **read_kwargs)
        feature_cols = sorted([c for c in train_df.columns if c not in {"qid", "doc_id", "label"}])
    else:
        log(f"Loading dataset '{name}' from {base}")
        if dataset_key == "yahoo":
            expected_dim = 700 if dataset_alias in {"set1", "set2"} else None
            train_df, train_feats = fast_read_libsvm(train_path, feature_dim=expected_dim, max_docs=max_docs)
            valid_df, valid_feats = fast_read_libsvm(valid_path, feature_dim=expected_dim, max_docs=max_docs)
            test_df, test_feats = fast_read_libsvm(test_path, feature_dim=expected_dim, max_docs=max_docs)
            feature_cols = sorted(set(train_feats) | set(valid_feats) | set(test_feats))
        else:
            train_df, train_feats = read_letor_file(train_path, max_docs=max_docs)
            valid_df, valid_feats = read_letor_file(valid_path, max_docs=max_docs)
            test_df, test_feats = read_letor_file(test_path, max_docs=max_docs)
            feature_cols = sorted(set(train_feats) | set(valid_feats) | set(test_feats))
            align_feature_columns((train_df, valid_df, test_df), feature_cols)
        if use_cache:
            cache_info["dir"].mkdir(parents=True, exist_ok=True)
            to_kwargs = {"engine": PARQUET_ENGINE} if PARQUET_ENGINE else {}
            train_df.to_parquet(cache_info["train"], index=False, **to_kwargs)
            valid_df.to_parquet(cache_info["valid"], index=False, **to_kwargs)
            test_df.to_parquet(cache_info["test"], index=False, **to_kwargs)
            log(f"Wrote parquet cache for '{name}' to {cache_info['dir']}")
    log(
        "Loaded dataset summary:\n"
        f"  Train: {len(train_df)} docs across {train_df['qid'].nunique()} queries\n"
        f"  Valid: {len(valid_df)} docs across {valid_df['qid'].nunique()} queries\n"
        f"  Test : {len(test_df)} docs across {test_df['qid'].nunique()} queries\n"
        f"  Features: {len(feature_cols)}"
    )
    log(
        "Train labels stats: "
        f"min={train_df['label'].min():.2f}, max={train_df['label'].max():.2f}, "
        f"mean={train_df['label'].mean():.2f}"
    )
    return DatasetBundle(train=train_df, valid=valid_df, test=test_df, feature_cols=feature_cols, name=name)


def group_lengths(df: pd.DataFrame) -> List[int]:
    return df.groupby("qid", sort=False).size().tolist()


def to_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=np.float32)


##### linear scorer for pairwise loss 
class SimpleLambdaRanker:
    def __init__(
        self,
        feature_cols: Sequence[str],
        learning_rate: float = LINEAR_LR,
        epochs: int = PAIRWISE_EPOCHS,
        max_pairs_per_query: int = PAIRWISE_MAX_PAIRS,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_pairs_per_query = max_pairs_per_query
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.weights = np.zeros(len(self.feature_cols), dtype=np.float32)

    def fit(self, df: pd.DataFrame) -> None:
        log(
            f"Training SimpleLambdaRanker for {self.epochs} epochs "
            f"with lr={self.learning_rate}, max_pairs={self.max_pairs_per_query}."
        )
        rng = self.rng
        grouped = [
            (group[self.feature_cols].to_numpy(dtype=np.float32), group["label"].to_numpy(dtype=np.float32))
            for _, group in df.groupby("qid")
        ]
        for epoch in range(1, self.epochs + 1):
            rng.shuffle(grouped)
            total_loss = 0.0
            total_pairs = 0
            for features, labels in grouped:
                pairs = self._make_pairs(labels)
                if not pairs:
                    continue
                pair_indices = np.array(pairs, dtype=np.int32)
                if len(pair_indices) > self.max_pairs_per_query:
                    idx = rng.choice(len(pair_indices), size=self.max_pairs_per_query, replace=False)
                    pair_indices = pair_indices[idx]
                diffs = features[pair_indices[:, 0]] - features[pair_indices[:, 1]]
                score_diffs = diffs @ self.weights
                inv_terms = 1.0 / (1.0 + np.exp(score_diffs))
                gradient = -(inv_terms[:, None] * diffs).mean(axis=0)
                self.weights -= self.learning_rate * gradient
                total_loss += float(np.log1p(np.exp(-score_diffs)).sum())
                total_pairs += len(score_diffs)
            if total_pairs > 0:
                avg_loss = total_loss / total_pairs
            else:
                avg_loss = 0.0
            log(f"  Epoch {epoch}/{self.epochs}: pairs={total_pairs}, avg_pair_loss={avg_loss:.4f}")

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(data, dtype=np.float32)
        return features @ self.weights

    @staticmethod
    def _make_pairs(labels: np.ndarray) -> List[Tuple[int, int]]:
        order = np.argsort(-labels)
        pairs: List[Tuple[int, int]] = []
        for idx, anchor in enumerate(order[:-1]):
            anchor_label = labels[anchor]
            for target in order[idx + 1 :]:
                if anchor_label == labels[target]:
                    continue
                pairs.append((anchor, target))
        return pairs


##### linear scorer for pointwise loss 
class SimpleLinearRegressor:

    def __init__(
        self,
        feature_cols: Sequence[str],
        learning_rate: float = LINEAR_LR,
        epochs: int = LINEAR_EPOCHS,
        batch_size: int = LINEAR_BATCH,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.weights = np.zeros(len(self.feature_cols), dtype=np.float32)
        self.bias = 0.0

    def fit(self, df: pd.DataFrame) -> None:
        log(
            f"Training SimpleLinearRegressor for {self.epochs} epochs "
            f"with lr={self.learning_rate}, batch_size={self.batch_size}."
        )
        rng = self.rng
        features = df[self.feature_cols].to_numpy(dtype=np.float32)
        labels = df["label"].to_numpy(dtype=np.float32)
        n = len(features)
        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(n)
            total_loss = 0.0
            total_count = 0
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = features[batch_idx]
                yb = labels[batch_idx]
                preds = xb @ self.weights + self.bias
                error = preds - yb
                loss = float(np.mean(error**2))
                grad_w = (xb.T @ error) / len(xb)
                grad_b = error.mean()
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b
                total_loss += loss * len(xb)
                total_count += len(xb)
            avg_loss = total_loss / max(total_count, 1)
            log(f"  Epoch {epoch}/{self.epochs}: mse={avg_loss:.4f}")

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(data, dtype=np.float32)
        return features @ self.weights + self.bias


##### MLP scorer for pairwise loss 
class NeuralLambdaRanker:
    def __init__(
        self,
        feature_cols: Sequence[str],
        hidden_dims: Sequence[int] = NEURAL_HIDDEN_DIMS,
        learning_rate: float = NEURAL_LR,
        epochs: int = NEURAL_EPOCHS,
        batch_size: int = NEURAL_BATCH,
        max_pairs_per_query: int = PAIRWISE_MAX_PAIRS,
        dropout: float = NEURAL_DROPOUT,
        weight_decay: float = NEURAL_WEIGHT_DECAY,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.hidden_dims = list(hidden_dims)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_pairs_per_query = max_pairs_per_query
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self._init_weights()

    def _init_weights(self) -> None:
        rng = self.rng
        in_dim = len(self.feature_cols)
        h1, h2 = self.hidden_dims
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), size=(h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2 / h2), size=(h2, 1)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def _forward(self, X: np.ndarray, *, training: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        mask1 = None
        if training and self.dropout > 0:
            mask1 = (self.rng.random(a1.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            a1 *= mask1
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        mask2 = None
        if training and self.dropout > 0:
            mask2 = (self.rng.random(a2.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            a2 *= mask2
        scores = a2 @ self.W3 + self.b3
        cache = (X, z1, a1, mask1, z2, a2, mask2)
        return scores, cache

    def _backward(self, cache: Tuple[np.ndarray, ...], grad_output: np.ndarray) -> None:
        X, z1, a1, mask1, z2, a2, mask2 = cache
        batch = max(len(X), 1)
        grad_W3 = (a2.T @ grad_output) / batch + self.weight_decay * self.W3
        grad_b3 = grad_output.mean(axis=0)
        grad_a2 = grad_output @ self.W3.T
        if mask2 is not None:
            grad_a2 *= mask2
        grad_z2 = grad_a2 * self._relu_grad(z2)
        grad_W2 = (a1.T @ grad_z2) / batch + self.weight_decay * self.W2
        grad_b2 = grad_z2.mean(axis=0)
        grad_a1 = grad_z2 @ self.W2.T
        if mask1 is not None:
            grad_a1 *= mask1
        grad_z1 = grad_a1 * self._relu_grad(z1)
        grad_W1 = (X.T @ grad_z1) / batch + self.weight_decay * self.W1
        grad_b1 = grad_z1.mean(axis=0)

        self.W3 -= self.learning_rate * grad_W3
        self.b3 -= self.learning_rate * grad_b3
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1

    def fit(self, df: pd.DataFrame) -> None:
        log(
            "Training NeuralLambdaRanker "
            f"(epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate})."
        )
        rng = np.random.default_rng(self.random_state)
        grouped = [
            (group[self.feature_cols].to_numpy(dtype=np.float32), group["label"].to_numpy(dtype=np.float32))
            for _, group in df.groupby("qid")
        ]
        for epoch in range(1, self.epochs + 1):
            rng.shuffle(grouped)
            total_loss = 0.0
            total_pairs = 0
            for features, labels in grouped:
                pairs = SimpleLambdaRanker._make_pairs(labels)
                if not pairs:
                    continue
                pair_idx = np.array(pairs, dtype=np.int32)
                if len(pair_idx) > self.max_pairs_per_query:
                    choice = rng.choice(len(pair_idx), size=self.max_pairs_per_query, replace=False)
                    pair_idx = pair_idx[choice]
                pos = features[pair_idx[:, 0]]
                neg = features[pair_idx[:, 1]]
                for start in range(0, len(pos), self.batch_size):
                    pos_batch = pos[start : start + self.batch_size]
                    neg_batch = neg[start : start + self.batch_size]
                    if len(pos_batch) == 0:
                        continue
                    s_pos, cache_pos = self._forward(pos_batch, training=True)
                    s_neg, cache_neg = self._forward(neg_batch, training=True)
                    diff = s_pos - s_neg
                    sigma = 1.0 / (1.0 + np.exp(-diff))
                    batch_loss = np.logaddexp(0.0, -diff)
                    grad_pos = sigma - 1.0
                    grad_neg = 1.0 - sigma
                    self._backward(cache_pos, grad_pos)
                    self._backward(cache_neg, grad_neg)
                    total_loss += float(batch_loss.mean()) * len(pos_batch)
                    total_pairs += len(pos_batch)
            avg_loss = total_loss / max(total_pairs, 1)
            log(f"  Epoch {epoch}/{self.epochs}: pairs={total_pairs}, avg_pair_loss={avg_loss:.4f}")

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(data, dtype=np.float32)
        scores, _ = self._forward(features)
        return scores.ravel()

##### linear scorer for pointwise loss 
class NeuralPointwiseRegressor:
    def __init__(
        self,
        feature_cols: Sequence[str],
        hidden_dims: Sequence[int] = NEURAL_HIDDEN_DIMS,
        learning_rate: float = NEURAL_LR,
        epochs: int = NEURAL_EPOCHS,
        batch_size: int = NEURAL_BATCH,
        dropout: float = NEURAL_DROPOUT,
        weight_decay: float = NEURAL_WEIGHT_DECAY,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.hidden_dims = list(hidden_dims)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self._init_weights()

    def _init_weights(self) -> None:
        rng = self.rng
        in_dim = len(self.feature_cols)
        h1, h2 = self.hidden_dims
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), size=(h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2 / h2), size=(h2, 1)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def _forward(self, X: np.ndarray, *, training: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        mask1 = None
        if training and self.dropout > 0:
            mask1 = (self.rng.random(a1.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            a1 *= mask1
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        mask2 = None
        if training and self.dropout > 0:
            mask2 = (self.rng.random(a2.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            a2 *= mask2
        scores = a2 @ self.W3 + self.b3
        cache = (X, z1, a1, mask1, z2, a2, mask2)
        return scores, cache

    def _backward(self, cache: Tuple[np.ndarray, ...], grad_output: np.ndarray) -> None:
        X, z1, a1, mask1, z2, a2, mask2 = cache
        batch = max(len(X), 1)
        grad_W3 = (a2.T @ grad_output) / batch + self.weight_decay * self.W3
        grad_b3 = grad_output.mean(axis=0)
        grad_a2 = grad_output @ self.W3.T
        if mask2 is not None:
            grad_a2 *= mask2
        grad_z2 = grad_a2 * self._relu_grad(z2)
        grad_W2 = (a1.T @ grad_z2) / batch + self.weight_decay * self.W2
        grad_b2 = grad_z2.mean(axis=0)
        grad_a1 = grad_z2 @ self.W2.T
        if mask1 is not None:
            grad_a1 *= mask1
        grad_z1 = grad_a1 * self._relu_grad(z1)
        grad_W1 = (X.T @ grad_z1) / batch + self.weight_decay * self.W1
        grad_b1 = grad_z1.mean(axis=0)

        self.W3 -= self.learning_rate * grad_W3
        self.b3 -= self.learning_rate * grad_b3
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1

    def fit(self, df: pd.DataFrame) -> None:
        log(
            "Training NeuralPointwiseRegressor "
            f"(epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate})."
        )
        rng = np.random.default_rng(self.random_state)
        features = df[self.feature_cols].to_numpy(dtype=np.float32)
        labels = df["label"].to_numpy(dtype=np.float32)
        n = len(features)
        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(n)
            total_loss = 0.0
            total_count = 0
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = features[batch_idx]
                yb = labels[batch_idx][:, None]
                preds, cache = self._forward(xb, training=True)
                error = preds - yb
                loss = float(np.mean(error**2))
                grad = 2.0 * error
                self._backward(cache, grad)
                total_loss += loss * len(xb)
                total_count += len(xb)
            avg_loss = total_loss / max(total_count, 1)
            log(f"  Epoch {epoch}/{self.epochs}: mse={avg_loss:.4f}")

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            features = data[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(data, dtype=np.float32)
        scores, _ = self._forward(features)
        return scores.ravel()


# performance evaluation metric with NDCG, dcg/ideal dcg 
def ndcg_at_k(df: pd.DataFrame, k: Optional[int] = 10) -> float:
    scores = []
    for _, group in df.groupby("qid"):
        ranked = group.sort_values("pred_score", ascending=False)
        labels = ranked["label"].to_numpy(dtype=np.float32)
        cutoff = len(labels) if k is None else min(k, len(labels))
        if cutoff == 0:
            continue
        labels = labels[:cutoff]
        gains = (2.0 ** labels - 1.0)
        discounts = np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.sum(gains / discounts))
        ideal_labels = np.sort(group["label"].to_numpy(dtype=np.float32))[::-1][:cutoff]
        ideal_gains = (2.0 ** ideal_labels - 1.0)
        ideal_dcg = float(np.sum(ideal_gains / discounts))
        if ideal_dcg <= 0:
            continue
        scores.append(dcg / ideal_dcg)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    ndcg10 = ndcg_at_k(df, k=10)
    ndcg5 = ndcg_at_k(df, k=5)
    ndcg_all = ndcg_at_k(df, k=None)
    return {"ndcg@5": ndcg5, "ndcg@10": ndcg10, "ndcg@all": ndcg_all}


def sigmoid_transform(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def should_normalize_features(model_name: str) -> bool:
    """Trees do not need min-max scaling; keep neural/linear models normalized."""
    key = MODEL_ALIASES.get(model_name.lower(), model_name.lower())
    return key not in {"pointwise_tree", "lambdarank_tree", "xgboost"}


# training function for pointwise loss scorer
def train_pointwise_regressor(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    log(
        "Training LightGBM pointwise regressor "
        f"(trees={LGBM_POINTWISE_PARAMS['n_estimators']}, lr={LGBM_POINTWISE_PARAMS['learning_rate']}, "
        f"leaves={LGBM_POINTWISE_PARAMS['num_leaves']}, "
        f"early_stopping={LGBM_POINTWISE_PARAMS['early_stopping_rounds']})."
    )
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=LGBM_POINTWISE_PARAMS["n_estimators"],
        learning_rate=LGBM_POINTWISE_PARAMS["learning_rate"],
        num_leaves=LGBM_POINTWISE_PARAMS["num_leaves"],
        subsample=LGBM_POINTWISE_PARAMS["subsample"],
        colsample_bytree=LGBM_POINTWISE_PARAMS["colsample_bytree"],
        min_child_samples=LGBM_POINTWISE_PARAMS["min_child_samples"],
        random_state=seed,
    )
    model.fit(
        to_matrix(train_df, feature_cols),
        train_df["label"],
        eval_set=[(to_matrix(valid_df, feature_cols), valid_df["label"])],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(LGBM_POINTWISE_PARAMS["early_stopping_rounds"]),
            lgb.log_evaluation(period=25),
        ],
    )
    return model


# training function for pairwise loss scorer models
def train_lightgbm_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    num_leaves: int,
    n_estimators: int,
    learning_rate: float,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    early_stopping_rounds: int,
    label: str,
    seed: int = 42,
    label_gain: Optional[Sequence[float]] = None,
) -> lgb.LGBMRanker:
    log(
        f"Training LightGBM {label} ranker "
        f"(trees={n_estimators}, lr={learning_rate}, leaves={num_leaves}, "
        f"min_child_samples={min_child_samples})."
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        random_state=seed,
        importance_type="gain",
        label_gain=label_gain,
    )
    model.fit(
        to_matrix(train_df, feature_cols),
        train_df["label"],
        group=group_lengths(train_df),
        eval_set=[(to_matrix(valid_df, feature_cols), valid_df["label"])],
        eval_group=[group_lengths(valid_df)],
        eval_at=[1, 3, 5, 10],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(period=25)],
    )
    return model

# training function for xgboost
def train_xgboost_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    seed: int = 42,
) -> xgb.XGBRanker:
    log("Training XGBoost ranker...")
    def _sanitize_features(matrix: np.ndarray, split: str) -> np.ndarray:
        finite_mask = np.isfinite(matrix)
        if not finite_mask.all():
            bad = matrix.size - int(finite_mask.sum())
            log(f"Found {bad} non-finite feature values in {split}; replacing with NaN for XGBoost.")
            matrix = matrix.copy()
            matrix[~finite_mask] = np.nan
        return matrix

    X_train = _sanitize_features(to_matrix(train_df, feature_cols), "train")
    X_valid = _sanitize_features(to_matrix(valid_df, feature_cols), "valid")
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        missing=np.nan,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        reg_lambda=1.0,
    )
    model.fit(
        X_train,
        train_df["label"].to_numpy(),
        group=group_lengths(train_df),
        eval_set=[(X_valid, valid_df["label"].to_numpy())],
        eval_group=[group_lengths(valid_df)],
        verbose=True,
    )
    return model


def build_model(
    name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    linear_epochs: Optional[int] = None,
    mlp_epochs: Optional[int] = None,
    seed: int = 42,
) -> object:
    key = name.lower()
    key = MODEL_ALIASES.get(key, key)
    if key in {"pointwise_tree", "pointwise", "pointwise_regression"}:
        log("Selected model: Pointwise Regression (LightGBM tree).")
        return train_pointwise_regressor(train_df, valid_df, feature_cols, seed=seed)
    if key in {"pointwise_linear", "linear_pointwise"}:
        log("Selected model: Linear pointwise regressor.")
        model = SimpleLinearRegressor(
            feature_cols,
            epochs=linear_epochs or LINEAR_EPOCHS,
            random_state=seed,
        )
        model.fit(train_df)
        return model
    if key == "pointwise_mlp":
        log("Selected model: Neural pointwise MLP.")
        model = NeuralPointwiseRegressor(
            feature_cols,
            epochs=mlp_epochs or NEURAL_EPOCHS,
            random_state=seed,
        )
        model.fit(train_df)
        return model
    if key == "lambdarank_linear":
        log("Selected model: Linear LambdaRank.")
        ranker = SimpleLambdaRanker(
            feature_cols,
            epochs=linear_epochs or PAIRWISE_EPOCHS,
            random_state=seed,
        )
        ranker.fit(train_df)
        return ranker
    if key == "lambdarank_mlp":
        log("Selected model: Neural LambdaRank.")
        ranker = NeuralLambdaRanker(
            feature_cols,
            epochs=mlp_epochs or NEURAL_EPOCHS,
            random_state=seed,
        )
        ranker.fit(train_df)
        return ranker
    if key == "lambdarank_tree":
        log("Selected model: LambdaMART (LightGBM tree).")
        return train_lightgbm_ranker(
            train_df,
            valid_df,
            feature_cols,
            num_leaves=LGBM_LAMBDARANK_PARAMS["num_leaves"],
            n_estimators=LGBM_LAMBDARANK_PARAMS["n_estimators"],
            learning_rate=LGBM_LAMBDARANK_PARAMS["learning_rate"],
            min_child_samples=LGBM_LAMBDARANK_PARAMS["min_child_samples"],
            subsample=LGBM_LAMBDARANK_PARAMS["subsample"],
            colsample_bytree=LGBM_LAMBDARANK_PARAMS["colsample_bytree"],
            early_stopping_rounds=LGBM_LAMBDARANK_PARAMS["early_stopping_rounds"],
            label="LambdaMART",
            label_gain=LGBM_LAMBDARANK_PARAMS["label_gain"],
            seed=seed,
        )
    if key in {"xgboost", "xgbrank"}:
        log("Selected model: XGBoost ranker.")
        return train_xgboost_ranker(train_df, valid_df, feature_cols, seed=seed)
    raise ValueError(f"Unsupported model '{name}'")


def predict_scores(model: object, df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if hasattr(model, "predict"):
        log(f"Scoring {len(df)} samples.")
        if isinstance(model, (SimpleLambdaRanker, SimpleLinearRegressor, NeuralLambdaRanker, NeuralPointwiseRegressor)):
            return model.predict(df)
        return model.predict(to_matrix(df, feature_cols))
    raise ValueError("Model does not expose a predict method.")


def run_experiment(
    dataset: str,
    model_name: str,
    fold: Optional[int],
    output_dir: Path,
    *,
    linear_epochs: Optional[int] = None,
    mlp_epochs: Optional[int] = None,
    seed: int = 42,
    max_docs: Optional[int] = None,
) -> Tuple[Dict[str, float], Path, str]:
    bundle = load_dataset_splits(dataset, fold, max_docs=max_docs)
    return run_experiment_with_bundle(
        bundle,
        model_name,
        output_dir,
        linear_epochs=linear_epochs,
        mlp_epochs=mlp_epochs,
        seed=seed,
        max_docs=max_docs,
    )


def run_experiment_with_bundle(
    bundle: DatasetBundle,
    model_name: str,
    output_dir: Path,
    *,
    linear_epochs: Optional[int] = None,
    mlp_epochs: Optional[int] = None,
    seed: int = 42,
    max_docs: Optional[int] = None,
) -> Tuple[Dict[str, float], Path, str]:
    train_df = bundle.train.copy()
    valid_df = bundle.valid.copy()
    test_df = bundle.test.copy()
    feature_cols = list(bundle.feature_cols)
    dataset_label = bundle.name
    if should_normalize_features(model_name):
        log("Applying min-max normalization with statistics from the training split.")
        min_max_normalize(train_df, feature_cols, (valid_df, test_df))
    else:
        log("Skipping feature normalization for tree-based models.")
    log(f"Beginning training for dataset '{dataset_label}' using model '{model_name}'.")
    model = build_model(
        model_name,
        train_df,
        valid_df,
        feature_cols,
        linear_epochs=linear_epochs,
        mlp_epochs=mlp_epochs,
        seed=seed,
    )
    lin_tag = linear_epochs if linear_epochs is not None else LINEAR_EPOCHS
    mlp_tag = mlp_epochs if mlp_epochs is not None else NEURAL_EPOCHS
    result_df = test_df[["qid", "doc_id", "label"]].copy()
    result_df["pred_score"] = predict_scores(model, test_df, feature_cols)
    eval_df = result_df.sort_values(["qid", "pred_score"], ascending=[True, False])
    metrics = evaluate_predictions(eval_df)
    log(
        "Evaluation complete: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dataset = sanitize_label(dataset_label)
    sanitized_model = sanitize_label(model_name)
    output_path = output_dir / f"{sanitized_dataset}_{sanitized_model}_seed{seed}_lin{lin_tag}_mlp{mlp_tag}_predictions.csv"
    output_df = result_df.copy()
    output_df["pred_score"] = sigmoid_transform(output_df["pred_score"].to_numpy())
    output_df = output_df.sort_values(["qid", "doc_id"]).reset_index(drop=True)
    log(f"Writing predictions to {output_path} (sorted by qid, doc_id)")
    output_df.to_csv(output_path, index=False)
    return metrics, output_path, dataset_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train learning-to-rank models on provided datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_GUIDE,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier (e.g., mq2007, mq2007_fold2, mq2008, mslr, istella, yahoo_set1).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold override for datasets that support it.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Which model to train (pointwise, lambdarank, lambdamart, xgboost, or 'all').",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "outputs"),
        help="Directory used to store the prediction files.",
    )
    parser.add_argument(
        "--linear-epochs",
        type=int,
        default=None,
        help="Override epoch count for linear/linear LambdaRank models (defaults depend on model).",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        help="Override epoch count for MLP pointwise/LambdaRank models (default: NEURAL_EPOCHS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for model initialization and training.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on documents per split (useful for quick experiments on large datasets).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    model_spec = args.model.lower()
    linear_epochs = args.linear_epochs
    mlp_epochs = args.mlp_epochs
    seed = args.seed
    lin_tag = linear_epochs if linear_epochs is not None else LINEAR_EPOCHS
    mlp_tag = mlp_epochs if mlp_epochs is not None else NEURAL_EPOCHS
    sanitized_dataset_arg = sanitize_label(args.dataset)
    output_dir = base_output_dir / f"{sanitized_dataset_arg}_seed{seed}"
    log(f"Outputs will be written under {output_dir}")
    if model_spec == "all":
        models_to_run = MODEL_CHOICES
        log(f"Running all models: {', '.join(models_to_run)}")
    else:
        models_to_run = [args.model]
    summary_rows = []
    dataset_label_for_summary: Optional[str] = None
    shared_bundle: Optional[DatasetBundle] = None
    if len(models_to_run) > 1:
        shared_bundle = load_dataset_splits(args.dataset, args.fold, max_docs=args.max_docs)
    for model_name in models_to_run:
        if shared_bundle is not None:
            metrics, predictions_path, dataset_label = run_experiment_with_bundle(
                shared_bundle,
                model_name,
                output_dir,
                linear_epochs=linear_epochs,
                mlp_epochs=mlp_epochs,
                seed=seed,
            )
        else:
            metrics, predictions_path, dataset_label = run_experiment(
                dataset=args.dataset,
                model_name=model_name,
                fold=args.fold,
                output_dir=output_dir,
                linear_epochs=linear_epochs,
                mlp_epochs=mlp_epochs,
                seed=seed,
            )
        dataset_label_for_summary = dataset_label
        print(f"Model '{model_name}' saved predictions to {predictions_path}")
        for key, value in metrics.items():
            print(f"{model_name} -> {key}: {value:.4f}")
        summary_rows.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                "seed": seed,
                **metrics,
                "predictions_path": str(predictions_path),
            }
        )
    if len(summary_rows) > 1:
        summary_df = pd.DataFrame(summary_rows)
        sanitized_dataset = sanitize_label(dataset_label_for_summary or args.dataset)
        summary_path = output_dir / f"{sanitized_dataset}_seed{args.seed}_lin{lin_tag}_mlp{mlp_tag}_model_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
