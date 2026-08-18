"""
Shared helpers used by every training/inference script:
  - reproducible seeding
  - safe WandB authentication (Kaggle Secrets, never hardcoded keys)
  - metrics: accuracy, macro-F1, and the competition's own MAP@3
"""
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb_auth() -> None:
    """
    Loads WANDB_API_KEY from Kaggle Secrets (Add-ons -> Secrets, label
    "WANDB_API_KEY"). Falls back to offline logging instead of crashing
    or ever hardcoding a key in source.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as e:
        print(
            f"Could not load WANDB_API_KEY from Kaggle Secrets ({e}). "
            f"Falling back to offline wandb logging so the rest of the script still runs."
        )
        os.environ["WANDB_MODE"] = "offline"


def map_at_3(y_true_idx, logits: torch.Tensor) -> float:
    """Mean Average Precision @ 3 -- the competition's actual leaderboard metric."""
    top3 = torch.topk(logits, 3, dim=1).indices
    scores = []
    for true_idx, pred_idx in zip(y_true_idx, top3):
        pred_list = pred_idx.tolist()
        scores.append(1.0 / (pred_list.index(true_idx) + 1) if true_idx in pred_list else 0.0)
    return sum(scores) / len(scores)


def compute_metrics(y_true, y_pred, logits: torch.Tensor) -> dict:
    """Common metrics reported identically by every model, so runs are comparable."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "map3": map_at_3(y_true, logits),
    }
