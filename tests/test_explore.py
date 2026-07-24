import numpy as np
import pandas as pd

from tabular_agent.explore import explore_dataframe


def _churn_df():
    rng = np.random.default_rng(0)
    n = 400
    city = rng.choice(["berlin", "munich", "cologne"], size=n, p=[0.5, 0.3, 0.2])
    base = {"berlin": 0.6, "munich": 0.8, "cologne": 0.2}
    churn = (rng.random(n) < np.array([base[c] for c in city])).astype(int)
    income = rng.lognormal(10, 0.4, n)  # right-skewed
    return pd.DataFrame({"city": city, "income": income, "churn": churn})


def test_skew_available_on_demand():
    # skew is not in the profile anymore; the agent computes it here instead.
    df = _churn_df()
    code = "def explore(df):\n    return float(df['income'].skew())"
    assert explore_dataframe(df, code) > 1.0


def test_groupby_churn_rate_hypothesis():
    df = _churn_df()
    code = (
        "def explore(df):\n"
        "    return df.groupby('city')['churn'].mean().to_dict()"
    )
    rates = explore_dataframe(df, code)
    assert set(rates) == {"berlin", "munich", "cologne"}
    assert rates["munich"] > rates["cologne"]


def test_full_length_column_cannot_be_smuggled_back():
    # Looking at a per-category target signal is fine; turning it into a
    # full-length feature vector is not — the cap makes that structurally
    # impossible, so a manual leaky feature can't be reconstructed here.
    df = _churn_df()
    code = "def explore(df):\n    return df.groupby('city')['churn'].transform('mean')"
    out = explore_dataframe(df, code, max_rows=25)
    assert out["__kind__"] == "series"
    assert out["n_rows"] == len(df)
    assert out["truncated"] is True
    assert len(out["data"]) == 25
