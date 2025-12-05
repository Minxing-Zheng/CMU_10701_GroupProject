from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.data_loader import load_movielens_split


@dataclass
class MFConfig:
    embedding_dim: int = 64
    lr: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 2048
    patience: int = 3


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        dot = (u * v).sum(dim=-1)
        return dot + self.user_bias(user_idx).squeeze(-1) + self.item_bias(item_idx).squeeze(-1)


class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_map: Dict[int, int], item_map: Dict[int, int]):
        self.users = torch.as_tensor(df["user_id"].map(user_map).to_numpy(), dtype=torch.long)
        self.items = torch.as_tensor(df["item_id"].map(item_map).to_numpy(), dtype=torch.long)
        self.ratings = torch.as_tensor(df["rating"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]


def build_encoders(dfs: Iterable[pd.DataFrame]) -> Tuple[Dict[int, int], Dict[int, int]]:
    merged = pd.concat(dfs, ignore_index=True)
    user_ids = np.sort(merged["user_id"].unique())
    item_ids = np.sort(merged["item_id"].unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {m: i for i, m in enumerate(item_ids)}
    return user_map, item_map


def filter_unknown(df: pd.DataFrame, user_map: Dict[int, int], item_map: Dict[int, int]) -> pd.DataFrame:
    mask = df["user_id"].isin(user_map) & df["item_id"].isin(item_map)
    return df.loc[mask].copy()


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def train_one_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    cfg: MFConfig,
    device: torch.device,
) -> Tuple[MatrixFactorization, float]:
    model = MatrixFactorization(len(user_map), len(item_map), cfg.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = RatingsDataset(train_df, user_map, item_map)
    val_ds = RatingsDataset(val_df, user_map, item_map)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(cfg.epochs):
        model.train()
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            opt.zero_grad()
            preds = model(u, i)
            loss = loss_fn(preds, r)
            loss.backward()
            opt.step()

        val_pred, val_true = [], []
        model.eval()
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i = u.to(device), i.to(device)
                preds = model(u, i).cpu().numpy()
                val_pred.append(preds)
                val_true.append(r.numpy())
        cur_rmse = rmse(np.concatenate(val_pred), np.concatenate(val_true))
        if cur_rmse + 1e-4 < best_rmse:
            best_rmse = cur_rmse
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse


def tune_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    device: torch.device,
    search_space: Dict[str, List],
) -> Tuple[MatrixFactorization, MFConfig, float]:
    best_model = None
    best_cfg = None
    best_score = float("inf")

    dims = search_space.get("embedding_dim", [32, 64])
    lrs = search_space.get("lr", [1e-2, 5e-3])
    wds = search_space.get("weight_decay", [1e-5, 1e-4])
    epochs = search_space.get("epochs", [15])
    batch_sizes = search_space.get("batch_size", [1024])
    patience = search_space.get("patience", [3])

    for d in dims:
        for lr in lrs:
            for wd in wds:
                for ep in epochs:
                    for bs in batch_sizes:
                        for pat in patience:
                            cfg = MFConfig(
                                embedding_dim=d,
                                lr=lr,
                                weight_decay=wd,
                                epochs=ep,
                                batch_size=bs,
                                patience=pat,
                            )
                            model, score = train_one_model(train_df, val_df, user_map, item_map, cfg, device)
                            if score < best_score:
                                best_score = score
                                best_cfg = cfg
                                best_model = model
    if best_model is None or best_cfg is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a model.")
    return best_model, best_cfg, best_score


def predict_df(
    model: MatrixFactorization,
    df: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    device: torch.device,
    apply_sigmoid: bool = True,
) -> pd.DataFrame:
    df = filter_unknown(df, user_map, item_map)
    ds = RatingsDataset(df, user_map, item_map)
    loader = DataLoader(ds, batch_size=4096)
    preds = []
    model.eval()
    with torch.no_grad():
        for u, i, _ in loader:
            u, i = u.to(device), i.to(device)
            pred = model(u, i).cpu().numpy()
            preds.append(pred)
    pred_scores = np.concatenate(preds)
    if apply_sigmoid:
        pred_scores = 1 / (1 + np.exp(-pred_scores))
    df_out = df.copy()
    df_out["pred_score"] = pred_scores
    df_out = df_out.rename(columns={"user_id": "qid", "item_id": "doc_id", "rating": "label"})
    return df_out[["qid", "doc_id", "label", "pred_score"]]


def run_movielens_from_txt(
    train_path: str | Path,
    valid_path: str | Path,
    test_path: str | Path,
    search_space: Dict[str, List] | None = None,
    device: str | torch.device | None = None,
    output_csv: str | Path | None = None,
    metrics_path: str | Path | None = None,
    apply_sigmoid: bool = True,
) -> pd.DataFrame:
    train_df = load_movielens_split(train_path)
    val_df = load_movielens_split(valid_path)
    test_df = load_movielens_split(test_path)

    user_map, item_map = build_encoders([train_df, val_df, test_df])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if search_space is None:
        search_space = {"embedding_dim": [32, 64], "lr": [5e-3], "weight_decay": [1e-5], "epochs": [20]}

    model, cfg, val_rmse = tune_hyperparams(train_df, val_df, user_map, item_map, device, search_space)

    print(f"Best config: {cfg}  |  val RMSE={val_rmse:.4f}")
    df_summary = predict_df(model, test_df, user_map, item_map, device, apply_sigmoid=apply_sigmoid)

    # compute metrics on test
    test_rmse = rmse(df_summary["pred_score"].to_numpy(), df_summary["label"].to_numpy())
    test_mae = mae(df_summary["pred_score"].to_numpy(), df_summary["label"].to_numpy())

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_csv, index=False)
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "split": "val",
                    "rmse": val_rmse,
                    "mae": np.nan,
                },
                {
                    "split": "test",
                    "rmse": test_rmse,
                    "mae": test_mae,
                },
            ]
        ).to_csv(metrics_path, index=False)
    return df_summary
