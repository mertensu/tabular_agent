"""Stateful, leakage-safe encoders. Fit on train rows only; serialize params."""

import numpy as np
import pandas as pd


#: The statistics of the target a group may be summarised by.
#:
#: ``mean`` is the classical target encoding and the only one that is smoothed.
#: The other three describe the *rest* of the group's target distribution, which
#: the mean alone throws away: how spread out it is, where its middle is when the
#: mean is dragged by a tail, and how many observations stand behind the estimate
#: at all.
#:
#: Deliberately not on the menu: ``min``/``max``, because an extreme of the
#: target within a group is essentially one row's target and survives
#: out-of-fold fitting only by luck of which fold that row landed in; and
#: ``sum``, which is ``count * mean`` and adds no information to a model that can
#: already see both.
TARGET_STATS = ("mean", "std", "median", "count")


class TargetEncoder:
    """Per-group statistics of the **target**, fitted once and looked up later.

    The classical case is ``stats=("mean",)`` — a category replaced by the
    average target of the rows sharing it — which is why the default emits
    exactly that and names the column ``<col>_te``.

    The mean alone is a lossy summary of a group, and the loss matters. Two
    categories with the same average can have completely different target
    distributions, and an estimate standing on three rows should not be read like
    one standing on three hundred. ``std`` and ``median`` describe the shape;
    ``count`` is the trust signal, and it was already being computed here as the
    smoothing denominator and then discarded.

    **Only the mean is smoothed**, and the asymmetry is deliberate rather than an
    omission. Smoothing shrinks a group's estimate toward the population value in
    proportion to how little evidence it has, and the weighted-average formula
    that does this is a statement about means. Shrinking a *median* toward the
    global median by a count weight, or a standard deviation toward the global
    standard deviation, would be inventing an estimator here and giving it the
    same name as a standard one. Instead the group's count is emitted as its own
    column and the model is left to learn how far to trust a thin group — which
    is the same information, expressed rather than baked in.

    Groups whose statistic is undefined are not stored, so the lookup falls
    through to the population value: a one-row group has no sample standard
    deviation, and "no usable estimate for this group" should mean the same thing
    whether the group was thin or was never seen at all. ``count`` is the
    exception, and unseen means ``0`` there — the same reasoning as
    :class:`FrequencyEncoder`'s unseen-means-zero.

    Every statistic here reads ``y``, so every one of them needs the out-of-fold
    discipline in :func:`oof_target_encode`. That is as true of ``count`` as of
    ``mean``: a per-fold count is not target-informed in itself, but it is fitted
    on the same rows and there is no benefit to giving it a second, looser path.
    """

    kind = "target"

    def __init__(self, columns: list[str], smoothing: float = 10.0, stats=("mean",)):
        self.columns = list(columns)
        self.smoothing = float(smoothing)
        self.stats = tuple(dict.fromkeys(str(s) for s in stats))
        if not self.stats:
            raise ValueError("stats must name at least one statistic")
        unknown = [s for s in self.stats if s not in TARGET_STATS]
        if unknown:
            raise ValueError(
                f"unknown target statistic(s) {unknown}; use one or more of "
                f"{', '.join(TARGET_STATS)}"
            )
        # {column: {stat: {category: value}}}. Nested by stat rather than one
        # mapping per emitted column so that a round trip through JSON keeps the
        # encoder's shape self-describing.
        self.mapping_: dict[str, dict[str, dict[str, float]]] = {}
        self.defaults_: dict[str, float] = {}
        self.global_mean_: float | None = None

    def column_name(self, col: str, stat: str) -> str:
        """``<col>_te`` for the mean, ``<col>_te_<stat>`` for the rest.

        The mean keeps the bare name it has always had, so a table, a lineage
        sidecar or a journal row written before the other statistics existed
        still names its column the same thing.
        """
        return f"{col}_te" if stat == "mean" else f"{col}_te_{stat}"

    @property
    def output_columns(self) -> list[str]:
        return [self.column_name(c, s) for c in self.columns for s in self.stats]

    def fit(self, df: pd.DataFrame, y) -> "TargetEncoder":
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y))
        y_ser = pd.Series(y)

        self.defaults_ = {}
        for stat in self.stats:
            if stat == "count":
                self.defaults_[stat] = 0.0
            else:
                population = y_ser.agg(stat)
                self.defaults_[stat] = (
                    self.global_mean_ if pd.isna(population) else float(population)
                )

        # "count" is always aggregated because the smoothing weight needs it,
        # even when the caller did not ask for it as a feature.
        wanted = list(dict.fromkeys([*self.stats, "count"]))
        for col in self.columns:
            keys = df[col].astype(str).to_numpy()
            agg = pd.Series(y).groupby(keys).agg(wanted)
            per_stat: dict[str, dict[str, float]] = {}
            for stat in self.stats:
                if stat == "mean":
                    values = (agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing) / (
                        agg["count"] + self.smoothing
                    )
                else:
                    values = agg[stat]
                per_stat[stat] = {
                    str(k): float(v) for k, v in values.items() if pd.notna(v)
                }
            self.mapping_[col] = per_stat
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            keys = df[col].astype(str)
            for stat in self.stats:
                mapped = keys.map(self.mapping_[col][stat])
                out[self.column_name(col, stat)] = (
                    pd.to_numeric(mapped, errors="coerce")
                    .fillna(self.defaults_[stat])
                    .astype(float)
                )
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "smoothing": self.smoothing,
            "stats": list(self.stats),
            "mapping": self.mapping_,
            "defaults": self.defaults_,
            "global_mean": self.global_mean_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetEncoder":
        # Encoders serialized before the other statistics existed have no
        # "stats" key and a flat {column: {category: value}} mapping. They are
        # mean-only by definition, and a bundle on disk has to keep predicting
        # exactly what it predicted when it was written.
        stats = d.get("stats", ["mean"])
        enc = cls(d["columns"], d["smoothing"], stats)
        enc.global_mean_ = d["global_mean"]
        if "stats" in d:
            enc.mapping_ = {
                c: {s: dict(m) for s, m in per_stat.items()}
                for c, per_stat in d["mapping"].items()
            }
            enc.defaults_ = dict(d["defaults"])
        else:
            enc.mapping_ = {c: {"mean": dict(m)} for c, m in d["mapping"].items()}
            enc.defaults_ = {"mean": enc.global_mean_}
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


#: Separator joining a multi-column group key into the single string a JSON
#: object can use as a name. A control character rather than ``"_"`` or ``"|"``
#: because it cannot appear in a category label, so ``("a_b", "c")`` and
#: ``("a", "b_c")`` stay distinct keys instead of colliding into one group.
_KEY_SEP = "\x1f"

#: The statistics a group may be summarised by. A fixed menu rather than an
#: arbitrary callable: the encoder is serialized and replayed at predict time, so
#: whatever computes the number has to survive a round trip through JSON, and a
#: name does while a lambda does not.
GROUP_STATS = ("mean", "median", "std", "min", "max", "sum", "count", "nunique")

#: Statistics that scale with the size of the group. For a group the fit never
#: saw, zero is the honest answer to "how much of it was there" — the same
#: reasoning as :class:`FrequencyEncoder`'s unseen-means-zero. The remaining
#: (intensive) statistics have no such answer, so they fall back to the
#: population value instead.
_EXTENSIVE_STATS = frozenset({"sum", "count", "nunique"})


class GroupStatsEncoder:
    """Per-group summary of *another column*, learned once and looked up later.

    The missing third case. :class:`TargetEncoder` learns a per-group statistic
    of the **target**; :func:`~tabular_agent.transform.transform_features` does
    row-local arithmetic on the row's **own** values. Neither expresses "what is
    typical for the group this row belongs to" — the average basket size for this
    customer, the median duration for this route, how often this device is
    active — which is a feature family every panel-shaped dataset has and which
    the free-code path cannot produce.

    It cannot, because a ``groupby(...).transform("mean")`` written in
    ``transform_features`` recomputes the mean from whatever rows are in the
    frame. That is batch-dependent by construction: one row alone gets its own
    value back, and 0011's stability check rejects it — correctly. What makes it
    legal here is that the group's number is **fitted once and stored**, so a
    single row maps to the same value a million rows would.

    Two consequences worth stating, because they are why this is useful rather
    than merely allowed:

    * The value column does **not** have to exist at predict time. Only the
      ``by`` columns are read at lookup, so a quantity you can measure
      historically but not on the day you predict becomes a legitimate feature —
      as its group's history, never as itself.
    * A group absent from the fit gets ``default_``: zero for the extensive
      statistics (see :data:`_EXTENSIVE_STATS`), the population statistic
      otherwise.

    Why the value column is never assumed target-independent, and so why
    :func:`fit_group_stats` always takes the out-of-fold path: a column
    correlated with the target makes its per-group mean *nearly* a per-group mean
    of the target, and computing it including the row's own contribution is
    target encoding with the guard removed. Large groups hide it — a row moves a
    thousand-row group's mean by a thousandth — but group size is a property of
    the data, not of this class, and nothing here can inspect the correlation to
    decide.
    """

    kind = "group_stats"

    def __init__(self, by: list[str], value: str, stat: str = "mean"):
        self.by = [str(c) for c in by]
        self.value = str(value)
        self.stat = str(stat)
        if not self.by:
            raise ValueError("by must name at least one column to group on")
        if self.value in self.by:
            raise ValueError(
                f"value column {self.value!r} is also in by={self.by}: grouping a "
                "column by itself gives one distinct value per group and no signal"
            )
        if self.stat not in GROUP_STATS:
            raise ValueError(
                f"unknown stat {self.stat!r}; use one of {', '.join(GROUP_STATS)}"
            )
        self.mapping_: dict[str, float] = {}
        self.default_: float | None = None

    @property
    def output_column(self) -> str:
        """The one column this adds, named after everything that defines it.

        ``by`` + value + stat is exactly the encoder's identity, so two different
        aggregations can never land on the same column name and silently
        overwrite one another when both are added to the same table.
        """
        return "_".join([*self.by, self.value, self.stat])

    def _keys(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.by if c not in df.columns]
        if missing:
            raise ValueError(
                f"cannot look up group statistics: the grouping columns {missing} "
                f"are not in this frame. These are read at predict time as well as "
                f"at fit time, so they have to exist in both."
            )
        keys = df[self.by[0]].astype(str)
        for col in self.by[1:]:
            keys = keys + _KEY_SEP + df[col].astype(str)
        return keys

    def fit(self, df: pd.DataFrame, y=None) -> "GroupStatsEncoder":
        if self.value not in df.columns:
            raise ValueError(
                f"value column {self.value!r} is not in this frame; columns are "
                f"{list(df.columns)}"
            )
        stats = df.groupby(self._keys(df).to_numpy(), sort=False)[self.value].agg(self.stat)
        # NaN groups are dropped rather than stored: a one-row group has no
        # standard deviation, and leaving it out means the lookup falls through
        # to `default_` — the same treatment an unseen group gets, which is what
        # "no usable estimate for this group" should mean in both cases.
        self.mapping_ = {
            str(k): float(v) for k, v in stats.items() if pd.notna(v)
        }
        if self.stat in _EXTENSIVE_STATS:
            self.default_ = 0.0
        else:
            population = df[self.value].agg(self.stat)
            self.default_ = None if pd.isna(population) else float(population)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mapped = self._keys(df).map(self.mapping_)
        out[self.output_column] = pd.to_numeric(mapped, errors="coerce").astype(float)
        if self.default_ is not None:
            out[self.output_column] = out[self.output_column].fillna(self.default_)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "by": self.by,
            "value": self.value,
            "stat": self.stat,
            "mapping": self.mapping_,
            "default": self.default_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroupStatsEncoder":
        enc = cls(d["by"], d["value"], d["stat"])
        enc.mapping_ = dict(d["mapping"])
        enc.default_ = d["default"]
        return enc


ENCODER_CLASSES: dict[str, type] = {
    "target": TargetEncoder,
    "frequency": FrequencyEncoder,
    "ordinal": OrdinalEncoder,
    "group_stats": GroupStatsEncoder,
}


def fit_encoding(df, columns, method="target", *, y=None, folds=None,
                 smoothing: float = 10.0, normalize: bool = True,
                 categories: dict[str, list] | None = None, unknown: float = -1.0,
                 stats=("mean",)):
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

    ``stats`` applies to ``method="target"`` only and defaults to the mean, i.e.
    target encoding as it has always been. Ask for more of
    :data:`TARGET_STATS` to get the group's spread, middle and size as
    additional columns; see :class:`TargetEncoder` for why only the mean is
    smoothed.
    """
    if method == "target":
        if y is None or folds is None:
            raise ValueError("method='target' requires both y and folds (per-fold "
                             "out-of-fold fitting is what keeps it leakage-safe).")
        oof, final = oof_target_encode(df, y, columns, folds, smoothing=smoothing,
                                       stats=stats)
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


def fit_group_stats(df, by, value, stat="mean", *, folds):
    """Per-group statistics of a column, fitted out-of-fold (the tool entry point).

    Returns ``(train_features, final_encoder)`` on the same contract as
    :func:`fit_encoding`: the frame is what the model may **train** on, the
    encoder is what runs at **predict**.

    ``folds`` is required and there is no way to opt out of it. The alternative
    — deciding per call whether the value column is target-correlated enough to
    need holding out — is a judgement the caller would have to get right silently
    on every column, and getting it wrong reintroduces target leakage under a
    name that does not mention the target. One pass per fold is cheap; a silent
    leak is not. See :class:`GroupStatsEncoder` for why the question arises at
    all.

    Under forward-chaining folds the first block is validated by nothing, so
    those rows keep ``NaN`` — the same gap out-of-fold target encoding leaves,
    and for the same reason.
    """
    enc = GroupStatsEncoder(by, value, stat)
    column = enc.output_column
    oof = pd.DataFrame(index=df.index)
    oof[column] = np.nan
    position = oof.columns.get_loc(column)
    for tr, va in folds:
        fold_enc = GroupStatsEncoder(by, value, stat).fit(df.iloc[tr])
        oof.iloc[va, position] = fold_enc.transform(df.iloc[va])[column].to_numpy()
    return oof, enc.fit(df)


def oof_target_encode(df, y, columns, folds, smoothing: float = 10.0, stats=("mean",)):
    y = np.asarray(y, dtype=float)
    template = TargetEncoder(columns, smoothing, stats)
    names = template.output_columns
    oof = pd.DataFrame(index=df.index)
    for name in names:
        oof[name] = np.nan
    positions = {name: oof.columns.get_loc(name) for name in names}
    for tr, va in folds:
        enc = TargetEncoder(columns, smoothing, stats).fit(df.iloc[tr], y[tr])
        transformed = enc.transform(df.iloc[va])
        for name in names:
            oof.iloc[va, positions[name]] = transformed[name].to_numpy()
    final = TargetEncoder(columns, smoothing, stats).fit(df, y)
    return oof, final
