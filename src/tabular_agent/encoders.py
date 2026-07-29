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


class OrdinalEncoder:
    """Ordered category -> its rank, from an order the caller states explicitly.

    For columns whose categories genuinely have an order (``small < medium <
    large``, education level, Likert scales) integer codes are the *right*
    encoding: one tree split at ``<= medium`` captures the concept, where
    :class:`FrequencyEncoder` would scramble the order. Like the frequency
    encoder it never touches ``y``, so it needs no out-of-fold machinery.

    The order is a required parameter, never inferred. ``pd.factorize`` orders by
    first appearance and ``.cat.codes`` on an unordered dtype orders
    lexicographically -- both would encode ``Large/Medium/Small`` as ``0/1/2``
    and look perfectly fine while doing it. Making the order an argument that is
    serialized with the encoder is the whole point of this class.

    Fit is strict, predict is lenient: a value present in the fit data but
    missing from the declared order is an error (a typo or a forgotten level,
    not something to paper over), while a category first seen at predict time
    maps to ``unknown`` -- ``-1`` by default, i.e. below every declared level, so
    a single split isolates it.
    """

    kind = "ordinal"

    def __init__(self, categories: dict[str, list], unknown: float = -1.0):
        self.categories = {str(c): [str(v) for v in levels] for c, levels in categories.items()}
        self.columns = list(self.categories)
        self.unknown = float(unknown)
        self.mapping_: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, y=None) -> "OrdinalEncoder":
        for col, levels in self.categories.items():
            if not levels:
                raise ValueError(
                    f"column {col!r}: no level order given. Pass every level in "
                    f"increasing order, e.g. ['small', 'medium', 'large']; the "
                    f"column's values are {sorted(df[col].astype(str).dropna().unique())[:10]}."
                )
            dupes = {v for v in levels if levels.count(v) > 1}
            if dupes:
                raise ValueError(
                    f"column {col!r}: duplicate levels in the declared order: {sorted(dupes)}"
                )
            self.mapping_[col] = {v: float(i) for i, v in enumerate(levels)}
            seen = set(df[col].astype(str).dropna().unique())
            undeclared = sorted(seen - set(levels))
            if undeclared:
                raise ValueError(
                    f"column {col!r}: values present in the data but missing from the "
                    f"declared order: {undeclared}. Declared was {levels}. Add them in "
                    f"the right position (an ordinal encoding is only meaningful if the "
                    f"order is complete)."
                )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            mapped = df[col].astype(str).map(self.mapping_[col])
            out[col + "_ord"] = mapped.fillna(self.unknown).astype(float)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "categories": self.categories,
            "unknown": self.unknown,
            "mapping": self.mapping_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrdinalEncoder":
        enc = cls(d["categories"], d["unknown"])
        enc.mapping_ = {c: dict(m) for c, m in d["mapping"].items()}
        return enc


ENCODER_CLASSES: dict[str, type] = {
    "target": TargetEncoder,
    "frequency": FrequencyEncoder,
    "ordinal": OrdinalEncoder,
}


def fit_encoding(df, columns, method="target", *, y=None, folds=None,
                 smoothing: float = 10.0, normalize: bool = True,
                 categories: dict[str, list] | None = None, unknown: float = -1.0):
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
    requires neither. ``method="ordinal"`` requires ``categories`` -- the level
    order per column -- and no target either; use it when the column's
    categories are genuinely ordered, frequency/target when they are nominal.
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
    if method == "ordinal":
        if categories is None:
            raise ValueError("method='ordinal' requires categories={column: [level, ...]} "
                             "(the order is never inferred, see OrdinalEncoder).")
        missing = [c for c in columns if c not in categories]
        if missing:
            raise ValueError(f"method='ordinal' is missing a level order for: {missing}")
        enc = OrdinalEncoder({c: categories[c] for c in columns}, unknown=unknown).fit(df)
        new = [c + "_ord" for c in columns]
        return enc.transform(df)[new], enc
    raise ValueError(
        f"unknown encoding method: {method!r} (use 'target', 'frequency' or 'ordinal')"
    )


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
