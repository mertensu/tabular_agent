"""The UCI "default of credit card clients" table (Yeh & Lien 2009), as notes.

The Kaggle copy (`uciml/default-of-credit-card-clients-dataset`,
`UCI_Credit_Card.csv`) is already one table, so nothing is joined or reshaped.
Everything dataset-specific lives here (`CLAUDE.md`): the source's variable
description, transcribed, and three facts about the file it does not mention.

* The target is renamed from `default.payment.next.month` to `default_next_month`.
  Dots in a column name are a nuisance in every downstream tool and carry nothing.
* The repayment-status columns are `PAY_0, PAY_2, ..., PAY_6`: there is no
  `PAY_1`. `PAY_0` is September, the month `BILL_AMT1` and `PAY_AMT1` describe.
  The names are left as they are; the note says which month each is.
* `PAY_*`, `EDUCATION` and `MARRIAGE` hold values the description does not define
  (`-2` and `0`; `0, 5, 6`; `0`). They are left as they are and named in the notes.
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "credit_default", "UCI_Credit_Card.csv")
TARGET = "default_next_month"

ABOUT = """Credit card clients of a bank in Taiwan, one row per client, observed
April to September 2005. The target, default_next_month, is whether the client
defaulted on the payment due in the following month: 1 if they did, 0 if not.
About 22% of rows are defaults.

Amounts are in New Taiwan dollars. The six monthly columns of each family run
backwards in time: suffix 1 (or PAY_0) is September 2005, suffix 6 is April 2005.

Scoring is AUC: only the order the predictions put the rows in matters."""

_MONTHS = ["September", "August", "July", "June", "May", "April"]

COLUMN_NOTES: dict[str, str] = {
    "ID": "row identifier of the client.",
    "LIMIT_BAL": "amount of the given credit, including the client's individual "
                 "credit and their family (supplementary) credit.",
    "SEX": "1 = male, 2 = female.",
    "EDUCATION": "1 = graduate school, 2 = university, 3 = high school, 4 = others. "
                 "The file also has 0, 5 and 6, which the source does not define.",
    "MARRIAGE": "1 = married, 2 = single, 3 = others. The file also has 0, which "
                "the source does not define.",
    "AGE": "age in years.",
    TARGET: "the target: default payment next month, 1 = yes, 0 = no.",
}
for i, (pay, month) in enumerate(zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
                                     _MONTHS), start=1):
    COLUMN_NOTES[pay] = (
        f"repayment status in {month} 2005: -1 = paid duly, 1 = payment delay of one "
        "month, 2 = two months, ..., 9 = nine months and above. The file also has -2 "
        "and 0, which the source does not define."
    )
    COLUMN_NOTES[f"BILL_AMT{i}"] = f"amount of the bill statement in {month} 2005."
    COLUMN_NOTES[f"PAY_AMT{i}"] = f"amount paid in {month} 2005."


def load() -> pd.DataFrame:
    return pd.read_csv(RAW).rename(columns={"default.payment.next.month": TARGET})


if __name__ == "__main__":
    df = load()
    print(df.shape, f"{df[TARGET].mean():.4f} positive")
