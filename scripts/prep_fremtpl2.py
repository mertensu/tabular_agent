"""French motor third-party liability claim counts (freMTPL2freq), as one table and notes.

OpenML 41214, the `freMTPL2freq` table of the R package CASdatasets: 678,013 policies,
one row each, with the number of claims (`ClaimNb`) observed over the time the policy
was in force (`Exposure`, in years). Everything dataset-specific lives here
(`CLAUDE.md`), and as little as possible happens:

* `VehGas` loses its quotes. OpenML stores the values as `'Regular'` and `'Diesel'`,
  quotes included; that is an artefact of the export, not data.
* Nothing else is changed. The table has exposures above one year and policies with
  up to 16 claims, which the literature caps at 1 and 4 as recording errors; whether
  they are errors, and what to do about them, is the run's to work out. `Exposure`
  stays an ordinary column: how a model should use it is the question this dataset
  is here for.

The notes are the CASdatasets documentation, transcribed, and nothing more.
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "fremtpl2", "freMTPL2freq.parquet")
TARGET = "ClaimNb"
EXPOSURE = "Exposure"

ABOUT = """French motor third-party liability insurance, one row per policy, from one
insurer's portfolio. The target, ClaimNb, is the number of claims during the
exposure period of the policy.

Scoring is the mean Poisson deviance of the predicted claim counts."""

COLUMN_NOTES: dict[str, str] = {
    "IDpol": "the policy ID (used to link with the claims dataset).",
    TARGET: "the target: number of claims during the exposure period.",
    EXPOSURE: "the period of exposure for a policy, in years.",
    "Area": "the density value of the city community where the car driver lives in: "
            "from 'A' for rural area to 'F' for urban centre.",
    "VehPower": "the power of the car (ordered values).",
    "VehAge": "the vehicle age, in years.",
    "DrivAge": "the driver age, in years (in France, people can drive a car at 18).",
    "BonusMalus": "bonus/malus, between 50 and 350: <100 means bonus, >100 means "
                  "malus in France.",
    "VehBrand": "the car brand (unknown categories).",
    "VehGas": "the car gas, Diesel or regular.",
    "Density": "the density of inhabitants (number of inhabitants per square-kilometer) "
               "of the city where the car driver lives in.",
    "Region": "the policy region in France (based on the 1970-2015 classification).",
}


def load() -> pd.DataFrame:
    df = pd.read_parquet(RAW)
    df["VehGas"] = df["VehGas"].astype(str).str.strip("'")
    return df


if __name__ == "__main__":
    df = load()
    print(df.shape, f"{df[TARGET].sum() / df[EXPOSURE].sum():.4f} claims per policy-year")
