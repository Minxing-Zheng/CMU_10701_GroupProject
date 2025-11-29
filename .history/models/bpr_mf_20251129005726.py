# models/bpr_mf.py
# Bayesian Personalized Ranking with Matrix Factorization (BPR-MF)
# ---------------------------------------------------------------
# This module implements a simple BPR-MF model in PyTorch.
# It is designed for implicit-feedback recommendation:
#   - users: 0 ... num_users-1
#   - items: 0 ... num_items-1
#
# Core scoring function:
#   x_hat_{ui} = U_u^T V_i
#
# Core objective:
#   max_Theta sum_{(u, i, j)} ln sigma(x_hat_ui - x_hat_uj) - lambda * ||Theta||_2^2
#
# Example usage:
#   model = BPRMatrixFactorization(num_users, num_items, emb_dim=64)
#   loss = model.bpr_loss(u, i, j)
#   loss.backward(); optimizer.step()

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BPRConfig:
    num_users: int
    num_items: int
    emb_dim: int = 64
    l2_reg: float = 1e-4  # lambda in the paper
    device: str = "cpu"


class BPRMatrixFactorization(nn.Module):
    """
    Bayesian Personalized Ranking with Matrix Factorization (BPR-MF).

    - user embedding matrix U ∈ R^{num_users × emb_dim}
    - item embedding matrix V ∈ R^{num_items × emb_dim}

    Score:
        x_hat_ui = <U_u, V_i>  (dot product)
    """

    def __init__(self, config: BPRConfig):
        super().__init__()
        self.config = config

        self.user_embedding = nn.Embedding(
            num_embeddings=config.num_users,
            embedding_dim=config.emb_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=config.num_items,
            embedding_dim=config.emb_dim
        )

        # Optional: user/item bias terms (can be turned off if not needed)
        self.user_bias = nn.Embedding(config.num_users, 1)
        self.item_bias = nn.Embedding(config.num_items, 1)

        self._init_parameters()

    def _init_parameters(self):
        # Standard Gaussian init with small std
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        """
        Forward: return predicted affinity x_hat_{ui}.

        Args:
            user_ids: LongTensor of shape (batch_size,)
            item_ids: LongTensor of shape (batch_size,)

        Returns:
            scores: FloatTensor of shape (batch_size,)
        """
        u_emb = self.user_embedding(user_ids)         # (B, d)
        i_emb = self.item_embedding(item_ids)         # (B, d)
        u_bias = self.user_bias(user_ids).squeeze(-1) # (B,)
        i_bias = self.item_bias(item_ids).squeeze(-1) # (B,)

        dot = (u_emb * i_emb).sum(dim=-1)             # (B,)
        scores = dot + u_bias + i_bias
        return scores

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """
        Compute the BPR pairwise ranking loss for triples (u, i, j).

        Args:
            user_ids: LongTensor, shape (batch_size,)
            pos_item_ids: LongTensor, shape (batch_size,) - observed (positive) items
            neg_item_ids: LongTensor, shape (batch_size,) - unobserved (negative) items

        Returns:
            scalar loss (mean over batch)
        """
        # Scores for positive and negative items
        x_ui = self.forward(user_ids, pos_item_ids)   # (B,)
        x_uj = self.forward(user_ids, neg_item_ids)   # (B,)

        # Pairwise score difference
        x_uij = x_ui - x_uj                           # (B,)

        # BPR objective: - ln sigma(x_uij)
        log_sigmoid = F.logsigmoid(x_uij)             # ln σ(x_ui - x_uj)
        bpr_loss = -log_sigmoid.mean()

        # L2 regularization on embeddings
        l2_norm = (
            self.user_embedding(user_ids).pow(2).sum() +
            self.item_embedding(pos_item_ids).pow(2).sum() +
            self.item_embedding(neg_item_ids).pow(2).sum()
        )

        l2_loss = self.config.l2_reg * l2_norm / user_ids.shape[0]

        return bpr_loss + l2_loss

    @torch.no_grad()
    def predict_for_user(self, user_id, item_ids):
        """
        Predict scores x_hat_{ui} for a single user and a list of item ids.

        Args:
            user_id: int
            item_ids: LongTensor of shape (num_items_to_score,)

        Returns:
            scores: FloatTensor of shape (num_items_to_score,)
        """
        if not torch.is_tensor(item_ids):
            item_ids = torch.tensor(item_ids, dtype=torch.long)
        user_ids = torch.full_like(item_ids, fill_value=user_id, dtype=torch.long)

        self.eval()
        scores = self.forward(user_ids, item_ids)
        return scores

    def save_checkpoint(self, path):
        """
        Save model + config to a checkpoint dict.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path, map_location="cpu"):
        """
        Load model + config from a checkpoint dict.

        Returns:
            model: BPRMatrixFactorization
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = BPRConfig(**checkpoint["config"])
        model = BPRMatrixFactorization(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
