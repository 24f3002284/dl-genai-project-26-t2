"""
Exploratory Data Analysis -- run this FIRST, before any train_*.py script.

Reproduces the numbers reported in Section 3 of the project report:
  - class (answer) distribution
  - average prompt / option length
  - majority-class baseline MAP@3 (the floor any model must clear)
  - TF-IDF cosine-similarity baseline MAP@3 (shows lexical overlap alone
    is not a useful signal on this dataset)
  - a class-distribution bar chart saved to ./outputs/

This is deliberately kept separate from preprocessing.py: preprocessing.py
holds only what every training script needs at runtime (split, leakage
check, vocab); this script is a one-off human-facing analysis, run once
to justify modelling choices in the report.

Usage:
    python src/eda.py
"""
import os

import matplotlib
matplotlib.use("Agg") # doesnot display on this machine, render to file only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OPTIONS, LABEL_MAP
from preprocessing import load_data, tokenize
from utils import map_at_3
import torch


def class_distribution(train: pd.DataFrame) -> None:
    counts = train["answer"].value_counts().reindex(OPTIONS) # reindex so bars appear in A,B,C,D,E order, not by frequency
    print("Answer distribution:\n", counts, "\n")

    os.makedirs("./outputs", exist_ok=True) # won't crash on a fresh clone where outputs/ doesn't exist yet
    ax = counts.plot(kind="bar", color="#2563eb")
    ax.set_ylabel("Count")
    ax.set_title("Answer distribution of train.csv")
    plt.tight_layout() # stops the title/labels from getting clipped
    plt.savefig("./outputs/eda_class_distribution.png", dpi=150)
    plt.close() # matplotlib keeps figures in memory otherwise - adds up over a long EDA run
    print("Saved ./outputs/eda_class_distribution.png\n")


def length_stats(train: pd.DataFrame) -> None:
    prompt_lens = train["prompt"].apply(lambda x: len(tokenize(x)))
    option_lens = pd.concat([train[c].apply(lambda x: len(tokenize(x))) for c in OPTIONS]) 
    print(f"Avg prompt length: {prompt_lens.mean():.1f} words "
          f"(max {prompt_lens.max()})")
    print(f"Avg option length: {option_lens.mean():.1f} words "
          f"(max {option_lens.max()})\n")


def majority_baseline(train: pd.DataFrame) -> float: #Always predict the 3 most frequent options, ranked by frequency
    top3_options = train["answer"].value_counts().index[:3].tolist()
    y_true = train["answer"].map(LABEL_MAP).tolist() # convert letter answers -> integer class ids

    # MAP@3 needs a score per option per row, so fake up logits: give the top-3 options descending scores (3, 2, 1) and everything else 0, same for every row
    fake_logits = torch.zeros(len(train), 5)
    for rank, opt in enumerate(top3_options):
        fake_logits[:, LABEL_MAP[opt]] = 3 - rank  # rank 0 -> 3, rank 1 -> 2, rank 2 -> 1
    score = map_at_3(y_true, fake_logits)
    print(f"Majority-class baseline MAP@3: {score:.4f} "
          f"(always guessing {top3_options})")
    return score


def tfidf_baseline(train: pd.DataFrame) -> float:
    """Rank each question's 5 options by TF-IDF cosine similarity to the prompt."""
    vectorizer = TfidfVectorizer(stop_words="english")
    all_text = pd.concat([train["prompt"]] + [train[c] for c in OPTIONS])
    vectorizer.fit(all_text)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    prompt_vecs = vectorizer.transform(train["prompt"])
    y_true = train["answer"].map(LABEL_MAP).tolist()

    all_logits = torch.zeros(len(train), 5)
    for i, opt in enumerate(OPTIONS):
        opt_vecs = vectorizer.transform(train[opt])
        sims = cosine_similarity(prompt_vecs, opt_vecs).diagonal()
        all_logits[:, i] = torch.tensor(sims, dtype=torch.float32)

    score = map_at_3(y_true, all_logits)
    print(f"TF-IDF cosine-similarity baseline MAP@3: {score:.4f}\n")
    return score


def main():
    train, _ = load_data() # second return value (val/test split) isn't needed for this EDA pass
    print(f"train.csv: {train.shape[0]} rows, {train.shape[1]} columns\n")
    class_distribution(train)
    length_stats(train)
    majority_baseline(train)
    tfidf_baseline(train)
    print("EDA done -- see Section 3 of the report for the write-up of these numbers.")


if __name__ == "__main__":
    main()
