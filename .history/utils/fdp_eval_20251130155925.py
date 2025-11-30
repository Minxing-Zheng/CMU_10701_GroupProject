import numpy as np

def compute_fdp(data_dict, lambda_value, label_threshold=3):
    fdp_dict = {}
    selected, n_true = [], []

    for qid, docs in data_dict.items():
        labels = np.array([d[1] for d in docs])
        scores = np.array([d[2] for d in docs])

        mask = scores >= lambda_value
        s = mask.sum()
        t = (mask & (labels >= label_threshold)).sum()
        fdp = (s - t) / s if s > 0 else 0.0

        fdp_dict[qid] = fdp
        selected.append(s)
        n_true.append(t)

    return fdp_dict, {
        "mean_fdp": float(np.mean(list(fdp_dict.values()))),
        "mean_selected": float(np.mean(selected)),
        "mean_true": float(np.mean(n_true)),
        "per_sample_fdp": list(fdp_dict.values())
    }


def aggregate_fdp_test(fdp_vec, delta, alpha):
    n = len(fdp_vec)
    mean = fdp_vec.mean()
    penalty = np.sqrt(np.log(1/delta) / (2*n))
    return mean + penalty < alpha


def find_lambda_hat(calib_dict, delta, alpha, step=0.001):
    grid = np.arange(0.99, 0.0, -step)
    best = None
    for lam in grid:
        _, summary = compute_fdp(calib_dict, lam)
        if aggregate_fdp_test(np.array(summary["per_sample_fdp"]), delta, alpha):
            best = lam
        else:
            break
    return best

