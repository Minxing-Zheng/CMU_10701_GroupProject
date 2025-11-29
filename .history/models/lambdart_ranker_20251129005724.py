# models/lambdamart_ranker.py
# LambdaMART ranker wrapper using LightGBM
# ----------------------------------------
# This module wraps a LightGBM LGBMRanker model so that it can be used
# as a generic scoring model in your conformal prediction pipeline.
#
# It supports:
#   - fit(X, y, qid)
#   - predict(X)
#   - save(path) / load(path)
#
# For MSLR-like datasets, X is the feature matrix, y are relevance labels,
# and qid is the query group id for each row.

import json
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass, asdict


@dataclass
class LambdaMARTConfig:
    # Default hyperparameters as in your project text
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    # additional options
    random_state: int = 42
    objective: str = "lambdarank"
    metric: str = "ndcg"
    ndcg_at: tuple = (5, 10)


class LambdaMARTRanker:
    """
    A wrapper around LightGBM's LGBMRanker that exposes:
        - fit(X, y, qid)
        - predict(X)
        - save(path) / load(path)
    """

    def __init__(self, config: LambdaMARTConfig | None = None):
        if config is None:
            config = LambdaMARTConfig()
        self.config = config
        self.model = None

    def fit(self, X, y, qid):
        """
        Train a LambdaMART model.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,) - relevance labels
            qid: array-like of shape (n_samples,) - integer query group ids
        """

        X = np.asarray(X)
        y = np.asarray(y)
        qid = np.asarray(qid)

        # Compute group sizes for LightGBM from qid
        # group[k] = number of samples belonging to query k
        unique_qids, counts = np.unique(qid, return_counts=True)
        group = counts.tolist()

        lgb_train = lgb.Dataset(X, label=y, group=group)

        params = {
            "objective": self.config.objective,
            "metric": self.config.metric,
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "lambda_l2": self.config.reg_lambda,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "verbose": -1,
            "seed": self.config.random_state,
            "ndcg_eval_at": list(self.config.ndcg_at),
        }

        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.config.n_estimators,
        )

    def predict(self, X):
        """
        Predict scores for feature matrix X.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            scores: ndarray of shape (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        X = np.asarray(X)
        scores = self.model.predict(X)
        return scores

    def save(self, path_prefix: str):
        """
        Save model and config.
        Will create:
            - path_prefix + ".txt"   for the LightGBM model
            - path_prefix + ".json"  for the config
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Nothing to save.")

        model_path = path_prefix + ".txt"
        config_path = path_prefix + ".json"

        self.model.save_model(model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @staticmethod
    def load(path_prefix: str):
        """
        Load a LambdaMARTRanker from disk.

        Args:
            path_prefix: same prefix used in save()

        Returns:
            ranker: LambdaMARTRanker
        """
        model_path = path_prefix + ".txt"
        config_path = path_prefix + ".json"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = LambdaMARTConfig(**cfg_dict)

        ranker = LambdaMARTRanker(config)
        ranker.model = lgb.Booster(model_file=model_path)
        return ranker
