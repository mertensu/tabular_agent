"""Stateful, leakage-safe encoders. Fit on train rows only; serialize params."""

import numpy as np
import pandas as pd


class TargetEncoder:
    kind = "target"

    def __init__(self, columns: list[str], smoothing: float = 10.0):
        self.columns = list(columns)
        self.smoothing = float(smoothing)
        self.mapping_: dict[str, dict[str, float]] = {}
        self.global_mean_: float | None = None

    def fit(self, df: pd.DataFrame, y) -> "TargetEncoder":
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y))
        y_ser = pd.Series(y)
        for col in self.columns:
            keys = df[col].astype(str).to_numpy()
            grouped = y_ser.groupby(keys)
            stats = grouped.agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.mapping_[col] = {str(k): float(v) for k, v in smooth.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            mapped = df[col].astype(str).map(self.mapping_[col])
            out[col + "_te"] = mapped.fillna(self.global_mean_).astype(float)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "smoothing": self.smoothing,
            "mapping": self.mapping_,
            "global_mean": self.global_mean_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetEncoder":
        enc = cls(d["columns"], d["smoothing"])
        enc.mapping_ = {c: dict(m) for c, m in d["mapping"].items()}
        enc.global_mean_ = d["global_mean"]
        return enc


ENCODER_CLASSES: dict[str, type] = {"target": TargetEncoder}


def oof_target_encode(df, y, columns, folds, smoothing: float = 10.0):
    y = np.asarray(y, dtype=float)
    oof = pd.DataFrame(index=df.index)
    for col in columns:
        oof[col + "_te"] = np.nan
    for tr, va in folds:
        enc = TargetEncoder(columns, smoothing).fit(df.iloc[tr], y[tr])
        transformed = enc.transform(df.iloc[va])
        for col in columns:
            oof.iloc[va, oof.columns.get_loc(col + "_te")] = transformed[col + "_te"].to_numpy()
    final = TargetEncoder(columns, smoothing).fit(df, y)
    return oof, final
