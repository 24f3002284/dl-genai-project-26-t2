"""
Central configuration for the Smart MCQ Solver project.

All paths default to the Kaggle competition mount point, but can be
overridden with environment variables so the same code runs locally too:

    export MCQ_DATA_DIR=/path/to/local/data
"""
import os

# ---- Data ----
DATA_DIR = os.environ.get(
    "MCQ_DATA_DIR", "/kaggle/input/competitions/smart-mcq-solver-challenge"
)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# ---- Labels ----
OPTIONS = ["A", "B", "C", "D", "E"]
LABEL_MAP = {opt: i for i, opt in enumerate(OPTIONS)}

# ---- Reproducibility ----
SEED = 42
VAL_SIZE = 0.2

# ---- WandB ----
WANDB_PROJECT = "24f3002284-t22026"

# ---- Output dirs (checkpoints, logits, submissions) ----
CKPT_DIR = os.environ.get("MCQ_CKPT_DIR", "./checkpoints")
LOGITS_DIR = os.environ.get("MCQ_LOGITS_DIR", "./logits")
SUBMISSION_DIR = os.environ.get("MCQ_SUBMISSION_DIR", "./submissions")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOGITS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
