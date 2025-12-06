#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.mlp_scorer import MLPScorer
from utils.yahoo_loader import predict_and_summarize
from utils.split_utils import split_query_dict, make_query_dict
from utils.fdp_eval import compute_fdp, find_lambda_hat

from joblib import Parallel, delayed


def to_query_dict(df):
    return (
        df.groupby("qid")[["doc_id", "label", "pred_score"]]
          .apply(lambda g: list(zip(g["doc_id"], g["label"], g["pred_score"])))
          .to_dict()
    )

def run_experiment(query_dict, n_exp=100, alpha=0.3, delta=0.2, lambda_step=0.001, high_quality_threshold=3):
    mean_fdr_list = []
    mean_size_list = []
    mean_true_list = []
    lambda_list = []
    success = []
    for i in range(n_exp):
        cal, test = split_query_dict(query_dict, random_state=42 + i)
        lam = find_lambda_hat(cal, delta, alpha, lambda_step)
        lambda_list.append(lam)
        _, summary = compute_fdp(test, lam, high_quality_threshold)
        fdr = summary["mean_fdp"]
        mean_fdr_list.append(fdr)
        mean_size_list.append(summary["mean_selected"])
        mean_true_list.append(summary["mean_true"])
        success.append(fdr <= alpha)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i+1:03d}/{n_exp}]  λ̂={lam:.3f}   meanFDR={fdr:.3f}")
    return {
        "lambda_list": lambda_list,
        "mean_fdr_list": mean_fdr_list,
        "mean_size_list": mean_size_list,
        "mean_true_list": mean_true_list,
        "success": success,
    }

def single_run(i, query_dict, alpha, delta, lambda_step, high_quality_threshold):
    cal, test = split_query_dict(query_dict, random_state=42 + i)
    lam = find_lambda_hat(cal, delta, alpha, lambda_step)
    _, summary = compute_fdp(test, lam, high_quality_threshold)
    fdr = summary["mean_fdp"]
    size = summary["mean_selected"]
    true_size = summary["mean_true"]
    success = (fdr <= alpha)
    return lam, fdr, size, true_size, success

def run_experiment_parallel(
    query_dict,
    n_exp=100,
    alpha=0.3,
    delta=0.2,
    lambda_step=0.001,
    high_quality_threshold=3,
    n_jobs=-1,    # use all cores
):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(single_run)(
            i, query_dict, alpha, delta, lambda_step, high_quality_threshold
        )
        for i in range(n_exp)
    )

    # unzip results
    lambda_list, fdr_list, size_list, true_list, success_list = zip(*results)

    return {
        "lambda_list": list(lambda_list),
        "mean_fdr_list": list(fdr_list),
        "mean_size_list": list(size_list),
        "mean_true_list": list(true_list),
        "success": list(success_list),
    }
def run_experiments_on_files(files, n_exp=100, alpha=0.3, delta=0.2, lambda_step=0.001, high_quality_threshold=3):
    results = {}
    for label, path in files:
        df = pd.read_csv(path).dropna(subset=["pred_score"])
        qdict = to_query_dict(df)
        res = run_experiment(
            qdict,
            n_exp=n_exp,
            alpha=alpha,
            delta=delta,
            lambda_step=lambda_step,
            high_quality_threshold=high_quality_threshold,
        )
        results[label] = res
    return results

# Example usage:
files = [
    ("5",  "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp5_predictions.csv"),
    ("10", "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp10_predictions.csv"),
    ("20", "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp20_predictions.csv"),
    ("30", "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp30_predictions.csv"),
    ("40", "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp40_predictions.csv"),
]

#### set up parameters ####
alpha = 0.3
delta = 0.1
lambda_step = 0.001
high_quality_threshold = 3 


results = run_experiments_on_files(files)
mean_fdr_list   = np.array(results['10']['mean_fdr_list'])
mean_size_list  = np.array(results['10']['mean_size_list'])
mean_true_list  = np.array(results['10']['mean_true_list'])

# ---------- validity summary ----------
n_exp = len(mean_fdr_list)
fdr_mean = mean_fdr_list.mean()
fdr_std  = mean_fdr_list.std(ddof=1)

# empirical violation probability δ̂ = P(FDR > α)
delta_hat = np.mean(mean_fdr_list > alpha)

print(f"Mean FDR  = {fdr_mean:.3f} ± {fdr_std:.3f}")
print(f"Violation rate δ̂ = P(FDR > α) = {delta_hat:.3f}")

# ---------- figure ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)

# (a) Mean FDR distribution
ax = axes[0]
ax.hist(
    mean_fdr_list,
    bins=20,
    density=True,
    color="steelblue",
    alpha=0.55,
    edgecolor="0.3"

)
ax.axvline(alpha, color="red", linestyle="--", linewidth=1.5, label=r"target $\alpha$")
ax.set_xlabel("FDR")
ax.set_ylabel("Density")
ax.set_title(f"FDR control")
ax.legend(frameon=False,loc='upper left')

# (b) Selected set size distribution
ax = axes[1]
ax.hist(
    mean_size_list,
    bins=20,
    density=True,
    color="steelblue",
    alpha=0.55,
    edgecolor="0.3",
    label="Predicted"
)
ax.hist(
    mean_true_list,
    bins=20,
    density=True,
    color="darkorange",
    alpha=0.55,
    edgecolor="0.3",
    label="True"
)
ax.set_xlabel("Set size")
ax.set_ylabel("Density")
ax.set_title("Recommendation set size")
ax.legend(frameon=False)

plt.show()



path_10 = "outputs/yahoo_set2_seed1/Yahoo-set2_lambdarank_mlp_seed1_lin50_mlp10_predictions.csv" 
df10 = pd.read_csv(path_10) 
query_dict_10 = to_query_dict(df10)
query_dict_20 = to_query_dict(df20)
query_dict_5 = to_query_dict(df5)

alpha_grid = np.arange(0.1, 0.51, 0.05) # 0.10, 0.15, ..., 0.50 
delta = 0.20 
n_exp = 100


# In[160]:


fdr_dict = {}
records = []

for alpha in alpha_grid:
    print(f"\n==== alpha = {alpha:.2f} ====")

    res = run_experiment_parallel(
        query_dict_20,
        n_exp=n_exp,
        alpha=alpha,
        delta=delta,
        lambda_step=0.001,
        high_quality_threshold=3,
        n_jobs=-1,     # all CPU cores
    )

    fdr_vals = np.array(res["mean_fdr_list"])
    size_vals = np.array(res["mean_size_list"])

    fdr_mean = fdr_vals.mean()
    fdr_std  = fdr_vals.std(ddof=1)

    viol_rate = np.mean(fdr_vals > alpha)

    print(f"  mean FDR = {fdr_mean:.3f} ± {fdr_std:.3f}")
    print(f"  violation rate = {viol_rate:.3f}")

    key = round(alpha, 2)
    fdr_dict[key] = fdr_vals
    records.append((alpha, fdr_mean, fdr_std, viol_rate, size_vals.mean(), size_vals.std()))

records = np.array(records)
alphas     = records[:, 0]
fdr_means  = records[:, 1]
fdr_stds   = records[:, 2]
viol_rates = records[:, 3]
size_means = records[:, 4]
size_stds  = records[:, 5]


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "axes.linewidth": 1.1,
    "figure.dpi": 150,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, axes = plt.subplots(
    1, 2, figsize=(8.5, 3.3),
    constrained_layout=True
)

# ------------------------------------------------------------
# (A) Empirical FDR vs nominal α  (left panel)
# ------------------------------------------------------------
ax = axes[0]

fdr_means = [np.mean(fdr_dict[round(a,2)]) for a in alpha_grid]
fdr_stds  = [np.std (fdr_dict[round(a,2)], ddof=1) for a in alpha_grid]
fdr_means = np.array(fdr_means)
fdr_stds  = np.array(fdr_stds)

# shaded uncertainty
ax.fill_between(
    alpha_grid,
    fdr_means - fdr_stds,
    fdr_means + fdr_stds,
    color="#4c72b0",
    alpha=0.18,
    linewidth=0
)

# mean line + crisp markers
ax.plot(
    alpha_grid,
    fdr_means,
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgecolor="#4c72b0",
    color="#4c72b0",
    linewidth=2,
    label="Empirical FDR"
)

# diagonal reference y = α
ax.plot(
    alpha_grid,
    alpha_grid,
    "--",
    linewidth=1.4,
    color="red",
    alpha=0.9,
    label=r"$y=\alpha$"
)

ax.set_xlabel(r"Nominal level $\alpha$")
ax.set_ylabel(r"Empirical FDR")
ax.set_title("Empirical FDR vs nominal α")
ax.set_ylim(0, max(alpha_grid) + 0.08)

# subtle horizontal gridlines
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax.set_ylim(-0.1, 0.51)
ax.legend(frameon=False)


# ------------------------------------------------------------
# (B) Violation rate vs α  (right panel)
# ------------------------------------------------------------
ax = axes[1]

# mean line + crisp markers (δ-hat curve)
ax.plot(
    alpha_grid,
    viol_rates,
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgecolor="#e17c1f",
    color="#e17c1f",
    linewidth=2,
    label=r"Empirical $\hat{\delta}(\alpha)$"
)

# nominal δ reference
delta_nominal = delta
ax.axhline(
    delta_nominal,
    linestyle="--",
    linewidth=1.5,
    color="red",
    alpha=0.9,
    label=rf"Nominal $\delta={delta_nominal}$"
)

ax.set_xlabel(r"Nominal $\alpha$")
ax.set_ylabel(r"$\hat\delta$")
ax.set_title(r"Empirical violation rate $\hat{\delta}$ vs α")
ax.set_ylim(-0.1, delta_nominal + 0.03)

# subtle gridlines
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

ax.legend(frameon=False)

plt.show()

