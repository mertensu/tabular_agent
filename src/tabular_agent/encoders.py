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


class FrequencyEncoder:
    """Category -> how common it is (spec §5.2, v1 set A, the non-leaky encoder).

    Uses only ``X`` (never the target), so it needs no out-of-fold machinery: a
    category's frequency is not target-informed and cannot leak. It exists to make
    string columns usable by models that do not consume raw categoricals (e.g.
    XGBoost) without inventing a target-mean the way :class:`TargetEncoder` does.
    Unknown categories seen only at predict time map to ``0`` (unseen == frequency
    zero), which is a sane, learned-vocabulary fallback.
    """

    kind = "frequency"

    def __init__(self, columns: list[str], normalize: bool = True):
        self.columns = list(columns)
        self.normalize = bool(normalize)
        self.mapping_: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, y=None) -> "FrequencyEncoder":
        for col in self.columns:
            counts = df[col].astype(str).value_counts(normalize=self.normalize)
            self.mapping_[col] = {str(k): float(v) for k, v in counts.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            mapped = df[col].astype(str).map(self.mapping_[col])
            out[col + "_freq"] = mapped.fillna(0.0).astype(float)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "normalize": self.normalize,
            "mapping": self.mapping_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrequencyEncoder":
        enc = cls(d["columns"], d["normalize"])
        enc.mapping_ = {c: dict(m) for c, m in d["mapping"].items()}
        return enc


ENCODER_CLASSES: dict[str, type] = {
    "target": TargetEncoder,
    "frequency": FrequencyEncoder,
}


def fit_encoding(df, columns, method="target", *, y=None, folds=None,
                 smoothing: float = 10.0, normalize: bool = True):
    """Structured, leakage-safe encoding tool (spec §5.2).

    The typed entry point the MCP ``fit_encoding`` tool wraps. Unlike the
    free-code escape hatch (:func:`~tabular_agent.transform.transform_features`),
    an encoder that *learns from the target* is only reachable here, where the
    per-fold discipline that prevents leakage is enforced structurally.

    Returns ``(train_features, final_encoder)``:

    - ``train_features`` — the encoded column(s) to **train** on. For
      ``method="target"`` these are **out-of-fold** (honest); for
      ``method="frequency"`` they are the ordinary transform (no leakage to
      guard against).
    - ``final_encoder`` — fit on **all** rows, to add to the
      :class:`~tabular_agent.pipeline.Pipeline` and apply at predict.

    ``method="target"`` requires ``y`` and ``folds``. ``method="frequency"``
    requires neither.
    """
    if method == "target":
        if y is None or folds is None:
            raise ValueError("method='target' requires both y and folds (per-fold "
                             "out-of-fold fitting is what keeps it leakage-safe).")
        oof, final = oof_target_encode(df, y, columns, folds, smoothing=smoothing)
        return oof, final
    if method == "frequency":
        enc = FrequencyEncoder(columns, normalize=normalize).fit(df)
        new = [c + "_freq" for c in columns]
        return enc.transform(df)[new], enc
    raise ValueError(f"unknown encoding method: {method!r} (use 'target' or 'frequency')")


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
