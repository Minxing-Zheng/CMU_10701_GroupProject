
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.data_loader import load_letor_split
from models.mlp_scorer import MLPScorer


@dataclass
class LetorConfig:
    hidden: List[int] = None  
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 512
    patience: int = 3

    def __post_init__(self):
        if self.hidden is None:
            self.hidden = [256, 128]


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


class LetorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _train_one_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    cfg: LetorConfig,
    device: torch.device,
) -> Tuple[MLPScorer, float]:
    model = MLPScorer(train_X.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(LetorDataset(train_X, train_y), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(LetorDataset(val_X, val_y), batch_size=cfg.batch_size)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).squeeze(-1).cpu().numpy()
                preds.append(pred)
                targets.append(yb.numpy())
        cur_rmse = rmse(np.concatenate(preds), np.concatenate(targets))
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


def _predict_df(
    model: MLPScorer,
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    device: torch.device,
    apply_sigmoid: bool = True,
) -> pd.DataFrame:
    loader = DataLoader(LetorDataset(X, y), batch_size=2048)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(-1).cpu().numpy()
            preds.append(pred)
    pred_scores = np.concatenate(preds)
    if apply_sigmoid:
        pred_scores = 1 / (1 + np.exp(-pred_scores))
    doc_ids = np.arange(len(pred_scores))
    df_out = pd.DataFrame(
        {
            "qid": qids.astype(int),
            "doc_id": doc_ids.astype(int),
            "label": y.astype(int),
            "pred_score": pred_scores.astype(float),
        }
    )
    return df_out


def run_letor_from_txt(
    train_path: str | Path,
    valid_path: str | Path,
    test_path: str | Path,
    feature_dim: int,
    cfg: LetorConfig | None = None,
    device: str | torch.device | None = None,
    output_csv: str | Path | None = None,
    metrics_path: str | Path | None = None,
    scaler: StandardScaler | None = None,
    apply_sigmoid: bool = True,
) -> pd.DataFrame:
    cfg = cfg or LetorConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_X, train_y, _ = load_letor_split(train_path, feature_dim)
    val_X, val_y, _ = load_letor_split(valid_path, feature_dim)
    test_X, test_y, test_qids = load_letor_split(test_path, feature_dim)

    if scaler is None:
        scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    model, val_rmse = _train_one_model(train_X, train_y, val_X, val_y, cfg, device)
    df_summary = _predict_df(model, test_X, test_y, test_qids, device, apply_sigmoid=apply_sigmoid)

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
                {"split": "val", "rmse": val_rmse, "mae": np.nan},
                {"split": "test", "rmse": test_rmse, "mae": test_mae},
            ]
        ).to_csv(metrics_path, index=False)
    return df_summary
