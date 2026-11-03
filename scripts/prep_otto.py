"""Otto Group Product Classification Challenge (Kaggle, 2015), as notes.

`train.csv` is already one table: 61,878 products, an `id`, 93 obfuscated count
features and the product category. Nothing is changed. The notes are the
competition's data page, transcribed, and nothing more.
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "otto_group", "train.csv")
TARGET = "target"

ABOUT = """Products of the Otto Group, one row per product. Each row has 93 numerical
features, which represent counts of different events. All features have been
obfuscated and will not be defined any further. There are nine categories for all
products; each target category represents one of the most important product
categories (like fashion, electronics, etc.).

Scoring is the multi-class logarithmic loss of the predicted class probabilities."""

COLUMN_NOTES: dict[str, str] = {
    "id": "an anonymous id unique to a product.",
    TARGET: "the target: the class of a product (Class_1 to Class_9).",
}


def load() -> pd.DataFrame:
    return pd.read_csv(RAW)


if __name__ == "__main__":
    df = load()
    print(df.shape, df[TARGET].nunique(), "classes")
