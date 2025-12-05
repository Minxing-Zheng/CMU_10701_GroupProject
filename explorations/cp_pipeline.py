
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.fdp_eval import compute_fdp
from utils.split_utils import make_query_dict, split_query_dict


def _search_lambda(calib_dict: Dict, delta: float, alpha: float, step: float, label_threshold: int) -> Optional[float]:
    grid = np.arange(0.99, 0.0, -step)
    best = None
    for lam in grid:
        _, summary = compute_fdp(calib_dict, lam, label_threshold=label_threshold)
        n = len(summary["per_sample_fdp"])
        penalty = np.sqrt(np.log(1 / delta) / (2 * n))
        if np.mean(summary["per_sample_fdp"]) + penalty < alpha:
            best = lam
        else:
            break
    return best


def run_cp_once(
    df_summary: pd.DataFrame,
    alpha: float = 0.3,
    delta: float = 0.1,
    lambda_step: float = 0.001,
    label_threshold: int = 3,
    test_ratio: float = 0.5,
    random_state: int = 42,
) -> Dict:
    required = {"qid", "doc_id", "label", "pred_score"}
    if not required.issubset(df_summary.columns):
        raise ValueError(f"df_summary must contain {required}")

    query_dict = make_query_dict(df_summary)
    calib_dict, test_dict = split_query_dict(
        query_dict, test_ratio=test_ratio, random_state=random_state
    )

    lambda_hat = _search_lambda(calib_dict, delta=delta, alpha=alpha, step=lambda_step, label_threshold=label_threshold)
    if lambda_hat is None:
        raise ValueError(
            "No lambda_hat found "
        )
    fdp_by_qid, summary = compute_fdp(test_dict, lambda_hat, label_threshold=label_threshold)

    return {
        "lambda_hat": lambda_hat,
        "fdp_summary": summary,
        "fdp_per_qid": fdp_by_qid,
        "n_calib": len(calib_dict),
        "n_test": len(test_dict),
    }


def run_repeated_cp(
    df_summary: pd.DataFrame,
    n_runs: int = 20,
    alpha: float = 0.3,
    delta: float = 0.1,
    lambda_step: float = 0.001,
    label_threshold: int = 3,
    test_ratio: float = 0.5,
    seed: int = 123,
) -> Dict:
    results = []
    fdr_list, size_list, lambda_list = [], [], []
    for i in range(n_runs):
        res = run_cp_once(
            df_summary,
            alpha=alpha,
            delta=delta,
            lambda_step=lambda_step,
            label_threshold=label_threshold,
            test_ratio=test_ratio,
            random_state=seed + i,
        )
        results.append(res)
        fdr_list.append(res["fdp_summary"]["mean_fdp"])
        size_list.append(res["fdp_summary"]["mean_selected"])
        lambda_list.append(res["lambda_hat"])

        # Progress logging every ~20 runs (and the first run) to show empirical λ̂ and mean FDR
        if (i == 0) or ((i + 1) % 20 == 0) or (i + 1 == n_runs):
            print(
                f"[{i+1:03d}/{n_runs}]  λ̂={res['lambda_hat']:.3f}   "
                f"meanFDR={res['fdp_summary']['mean_fdp']:.3f}"
            )

    mean_fdr = float(np.mean(fdr_list))
    mean_size = float(np.mean(size_list))
    success = float(np.mean([f <= alpha for f in fdr_list]))
    return {
        "mean_fdr": mean_fdr,
        "mean_selected": mean_size,
        "p_success": success,
        "runs": results,
        "fdr_list": fdr_list,
        "size_list": size_list,
        "lambda_list": lambda_list,
    }


def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"qid", "doc_id", "label", "pred_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prediction file missing columns {required}")
    return df[["qid", "doc_id", "label", "pred_score"]]


def compare_datasets(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        summary = res["fdp_summary"]
        rows.append(
            {
                "dataset": name,
                "lambda_hat": res["lambda_hat"],
                "mean_fdp": summary["mean_fdp"],
                "mean_selected": summary["mean_selected"],
                "mean_true": summary["mean_true"],
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def plot_cp_hist(results: Dict, alpha: float = 0.3, bins: int = 20):
    """Plot FDR and selected-set size histograms from run_repeated_cp output."""
    if "fdr_list" not in results or "size_list" not in results:
        raise ValueError("results must be output of run_repeated_cp with fdr_list/size_list.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(results["fdr_list"], bins=bins, color="steelblue", alpha=0.75, edgecolor="black")
    axes[0].axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}")
    axes[0].set_xlabel("Mean FDR over queries")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Mean FDR Distribution ({len(results['fdr_list'])} runs)")
    axes[0].legend(frameon=False)

    axes[1].hist(results["size_list"], bins=bins, color="darkorange", alpha=0.75, edgecolor="black")
    axes[1].set_xlabel("Mean selected set size")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Selected Set Size Distribution")

    plt.tight_layout()
    return fig, axes


def print_cp_summary(results: Dict, alpha: float, delta: float):
    """Pretty print conformal summary using run_repeated_cp output."""
    if "fdr_list" not in results or "size_list" not in results:
        raise ValueError("results must come from run_repeated_cp (needs fdr_list/size_list).")
    fdr_list = np.array(results["fdr_list"])
    size_list = np.array(results["size_list"])
    p_hat = np.mean(fdr_list <= alpha)
    print("\n================== Conformal FDR Summary ==================")
    print(f"α (target FDR):{alpha}")
    print(f"δ (tolerance):{delta}")
    print(f"Target:P(FDR ≤ α) ≥ 1-δ = {1-delta:.3f}")
    print(f"Empirical:P̂ = {p_hat:.3f}")
    print(f"Mean FDR:{fdr_list.mean():.3f} ± {fdr_list.std():.3f}")
    print(f"Mean set size:{size_list.mean():.2f} ± {size_list.std():.2f}")
    print("============================================================")
