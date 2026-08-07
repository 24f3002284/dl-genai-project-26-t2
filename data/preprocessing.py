from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from config import TRAIN_PATH, TEST_PATH, SEED, VAL_SIZE


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    A near-perfect val accuracy is a red flag, not a milestone -- usually it means val contains near-duplicates of training rows. This dataset has templated question variants (several differently-worded Heidegger questions 
    with near-identical options), so check before trusting any number that comes out of this split.
    """

    def _normalize(text):
        return str(text).lower().strip()  # so "Cat" and "cat " don't count as different rows

    tr_norm = set(train_df["prompt"].apply(_normalize))
    val_norm = set(val_df["prompt"].apply(_normalize))
    overlap = tr_norm & val_norm  # prompts appearing on both sides of the split
    print(f"Exact prompt overlap between train/val: {len(overlap)} rows")

    # catches the sneakier case: different prompt wording, but the exact same answer + option set underneath -- i.e. a templated duplicate question
    tr_sig = set(zip(train_df["answer"], train_df["A"], train_df["B"], train_df["C"], train_df["D"], train_df["E"]))
    val_sig = set(zip(val_df["answer"], val_df["A"], val_df["B"], val_df["C"], val_df["D"], val_df["E"]))
    sig_overlap = tr_sig & val_sig
    print(f"Rows with identical option sets in both train/val: {len(sig_overlap)}")

def stratified_split(train_df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED):
    """
    The one split every model has to use. Stratifying on `answer` keeps the class balance the same on both sides, and more fixes the seed means the same rows land in val every time, across all three notebooks. Without that, 
    "comparing" the three WandB runs would be comparing apples to oranges.
    """
    train_split, val_split = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df["answer"]
    )

    check_leakage(train_split, val_split)
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)
  
def tokenize(text: str):
    return str(text).lower().split()  

def build_vocab(train_df: pd.DataFrame, test_df: pd.DataFrame, min_count: int = 2):
    """Word-level vocabulary for the from-scratch embedding model only."""
    all_text = []
    for col in ["prompt", "A", "B", "C", "D", "E"]:
        # include test set too -- otherwise any word only seen at test time becomes <UNK>
        train_df[col].fillna("").apply(lambda x: all_text.extend(tokenize(x)))
        test_df[col].fillna("").apply(lambda x: all_text.extend(tokenize(x)))

    vocab = {"<PAD>": 0, "<UNK>": 1}  # reserve ids 0/1 before assigning real words
    for word, count in Counter(all_text).items():
        if count >= min_count:  # drop words seen only once -- probably typos/noise, not worth an embedding slot
            vocab[word] = len(vocab)
    print(f"Vocab size: {len(vocab)}")
    return vocab

def encode(text: str, vocab: dict, max_len: int = 64):
    tokens = tokenize(text)[:max_len]  # truncate long text so every sequence fits the model's fixed input size
    ids = [vocab.get(t, 1) for t in tokens]  # unknown words fall back to <UNK> (id 1)
    ids += [0] * (max_len - len(ids))  # pad the rest with <PAD> (id 0) so every sequence is the same length
    return ids
