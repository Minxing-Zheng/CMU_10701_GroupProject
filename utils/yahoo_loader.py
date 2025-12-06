import os
import numpy as np
import pandas as pd
import torch

def parse_line(line):
    parts = line.strip().split()
    label = int(float(parts[0]))
    qid = None
    feats = {}

    for tok in parts[1:]:
        if tok.startswith("qid:"):
            qid = int(tok.split(":")[1])
        else:
            k, v = tok.split(":")
            feats[int(k)] = float(v)

    return label, qid, feats


def predict_and_summarize(model, data_dir, test_file_name, device, Fdim, scaler=None):
    path = os.path.join(data_dir, test_file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    groups_X, groups_y = {}, {}

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            label, qid, feats = parse_line(line)
            groups_X.setdefault(qid, []).append(feats)
            groups_y.setdefault(qid, []).append(label)

    qids = sorted(groups_X.keys())
    X_groups, y_groups = [], []

    for qid in qids:
        rows = groups_X[qid]
        X = np.zeros((len(rows), Fdim), dtype=np.float32)
        for i, d in enumerate(rows):
            for k, v in d.items():
                if 1 <= k <= Fdim:
                    X[i, k - 1] = v
        if scaler is not None:
            X = scaler.transform(X)

        X_groups.append(X)
        y_groups.append(np.array(groups_y[qid], dtype=np.int64))

    rows_out = []
    doc_id = 0
    with torch.no_grad():
        for qid, X, y in zip(qids, X_groups, y_groups):
            X_t = torch.from_numpy(X).float().to(device)
            scores = model(X_t).squeeze(-1).cpu().numpy()
            for i in range(len(scores)):
                rows_out.append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "label": int(y[i]),
                    "pred_score": float(scores[i])
                })
                doc_id += 1

    df = pd.DataFrame(rows_out)
    df["pred_score"] = 1 / (1 + np.exp(-df["pred_score"]))  # sigmoid

    return df
