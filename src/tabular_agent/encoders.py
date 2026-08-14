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


#: Most observations any one group may keep. A cap rather than a guideline: the
#: whole point of storing a tail is that the encoder's size depends on the number
#: of groups and this constant, never on how long the table is.
MAX_KEEP = 1000


def to_periods(values) -> np.ndarray:
    """The ordering column as floats that can be subtracted from one another.

    Every number this module produces is a *difference* of ordering values — how
    far a row sits from the observation it reads — so the ordering column has to
    support subtraction. Dates become days, numbers stay themselves, and anything
    else is refused here rather than producing a difference in unnamed units.

    Refusing is the point. A run ordered on a string month (``"2015-07"``) sorts
    correctly and subtracts to nonsense, and the nonsense would arrive as a
    plausible-looking float in an ``_age`` column.
    """
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        # Divided by a timedelta rather than scaled from the integer
        # representation, because pandas may hand back datetime64 in seconds,
        # milliseconds, microseconds or nanoseconds depending on how the column
        # was built. Assuming one of them puts every age out by a factor of a
        # thousand, which is exactly the kind of plausible-looking wrong number
        # the `_age` columns exist to avoid. `NaT` falls through as `NaN`.
        return (s.to_numpy() - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.to_numpy().astype(float)
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        # A date column read back from CSV is a string, which is the normal case
        # for every table this harness loads — refusing it outright would refuse
        # the datasets history exists for. **ISO 8601 only**, and deliberately:
        # `pd.to_datetime` left to guess reads "01/02/2015" as either the 1st of
        # February or the 2nd of January depending on the rest of the column, and
        # a silently transposed calendar is precisely the plausible-looking wrong
        # number this function exists to refuse.
        try:
            parsed = pd.to_datetime(s, format="ISO8601")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"cannot measure age along this column: its values are text that is "
                f"not ISO 8601 dates ({exc}). History needs an ordering column it can "
                "subtract. An ambiguous format like '01/02/2015' is not guessed at "
                "here, because guessing wrong silently transposes days and months — "
                "convert it to a real date or a number first."
            ) from exc
        return (parsed.to_numpy() - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    raise ValueError(
        f"cannot measure age along a column of dtype {s.dtype}: history needs an "
        "ordering column it can subtract, so it must be a date/datetime or a "
        "number. A sortable value that cannot be subtracted orders the rows "
        "correctly and still gives no meaning to the distance between two of them."
    )


class HistoryEncoder:
    """A group's own past, frozen at fit time and replayed as last-known + age.

    The fourth case, and the one the others cannot reach. :class:`TargetEncoder`
    and :class:`GroupStatsEncoder` summarise a group across *all* of its rows at
    once — the average, the spread — which throws away the order the rows came
    in. Nothing in the vocabulary could say "what this store sold most recently",
    because that is a statement about *when*, not about *how much*.

    What it stores, per group, is the tail of the series: the last ``keep``
    observations of ``value`` in ``order``. Bounded by construction — the
    encoder's size is (number of groups × ``keep``), never a function of how many
    rows the table has, which is what makes it storable in a lineage at all.

    What it emits, per row, is a pair of columns and never a bare value:

    * ``..._last`` — the most recent observation for this row's group, as of the
      freeze point.
    * ``..._last_age`` — how far that observation is from *this row*, in the
      ordering column's units.

    **The pair is the design, not a convenience.** Serving the last known value
    alone is the same number with the honesty removed: it drifts further from the
    truth the further past the freeze point you predict, and says nothing about
    it. This project's characteristic failure is a number that looks fine and
    describes something else, and a bare carry-forward is that failure in feature
    form. The age column is what lets a model discount a value it should not
    trust.

    ``offsets`` add true lags on the same machinery — ``..._lag7`` is the most
    recent observation at or before ``row - 7``. They degrade into ``_last``
    rather than into a lie: ask for a 7-day lag on a row 60 days past the freeze
    point and you get the newest thing available with an ``_age`` of 60, which is
    the truth about what was answerable. Whether an offset is *legal* is a
    separate question, and one this class does not answer — see
    :func:`fit_history`.

    Two properties worth stating, because they are why this is allowed where the
    obvious ``groupby().shift()`` is not:

    * **Batch independence.** The value comes from stored state, not from the
      frame in front of it. One row alone and a batch of a million get identical
      answers, because neither can see the other's rows. A ``shift()`` inside
      ``transform_features`` is batch-dependent by construction and 0011's
      stability check rejects it, correctly.
    * **The value column need not exist at predict time** — and for the target it
      never does. Only ``by`` and ``order`` are read at lookup. That is what makes
      a lag *of the target* expressible at all: it can only ever come from stored
      history, so there is no served frame it could be smuggled out of.
    """

    kind = "history"

    def __init__(self, by: list[str], value: str, order: str, offsets=(), keep: int | None = None):
        self.by = [str(c) for c in by]
        self.value = str(value)
        self.order = str(order)
        self.offsets = [float(o) for o in offsets]
        if not self.by:
            raise ValueError("by must name at least one column to group on")
        if self.value in self.by or self.order in self.by:
            raise ValueError(
                f"by={self.by} may not contain the value column {self.value!r} or the "
                "ordering column: grouping a series by its own timestamp gives one "
                "observation per group and no history at all"
            )
        if self.value == self.order:
            raise ValueError(
                f"value and order are both {self.value!r}; the history of a column "
                "against itself is the column"
            )
        if any(o <= 0 for o in self.offsets):
            raise ValueError(
                f"offsets must be positive, got {self.offsets}. An offset of zero is "
                "the row's own value and a negative one is the future."
            )
        # Enough tail to answer the furthest offset, assuming roughly one
        # observation per unit of the ordering column. Irregular series have to
        # say so by passing `keep` — an assumption that is wrong is better than a
        # default that silently truncates, and the `_age` columns show it when it
        # happens.
        self.keep = int(keep) if keep is not None else int(max(self.offsets, default=0)) + 1
        if not 1 <= self.keep <= MAX_KEEP:
            raise ValueError(f"keep must be between 1 and {MAX_KEEP}, got {self.keep}")
        self.history_: dict[str, list[list[float]]] = {}
        self.frozen_at_: float | None = None

    @property
    def base_name(self) -> str:
        return "_".join([*self.by, self.value])

    @property
    def output_columns(self) -> list[str]:
        names = [f"{self.base_name}_last", f"{self.base_name}_last_age"]
        for o in self.offsets:
            names += [f"{self.base_name}_lag{o:g}", f"{self.base_name}_lag{o:g}_age"]
        return names

    def _keys(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in [*self.by, self.order] if c not in df.columns]
        if missing:
            raise ValueError(
                f"cannot look up history: {missing} are not in this frame. The "
                "grouping columns and the ordering column are read at predict time "
                "as well as at fit time, so they have to exist in both."
            )
        keys = df[self.by[0]].astype(str)
        for col in self.by[1:]:
            keys = keys + _KEY_SEP + df[col].astype(str)
        return keys.to_numpy()

    def fit(self, df: pd.DataFrame, y=None) -> "HistoryEncoder":
        if self.value not in df.columns:
            raise ValueError(
                f"value column {self.value!r} is not in this frame; columns are "
                f"{list(df.columns)}"
            )
        values = pd.to_numeric(df[self.value], errors="coerce").to_numpy().astype(float)
        frame = pd.DataFrame({
            "key": self._keys(df),
            "t": to_periods(df[self.order]),
            "v": values,
        }).dropna(subset=["t", "v"])
        # Stable, so observations sharing an ordering value keep their original
        # relative order and the stored tail is reproducible rather than dependent
        # on sort internals — the same reason `make_folds` sorts stably.
        frame = frame.sort_values("t", kind="stable")
        tail = frame.groupby("key", sort=False).tail(self.keep)
        self.history_ = {
            str(key): [g["t"].tolist(), g["v"].tolist()]
            for key, g in tail.groupby("key", sort=False)
        }
        self.frozen_at_ = float(frame["t"].max()) if len(frame) else None
        return self

    def _asof(self, keys: np.ndarray, t: np.ndarray, offset: float | None):
        """Most recent stored observation at or before ``t - offset``, and its age.

        ``offset=None`` means "the newest thing there is", which is the ``_last``
        pair. Age is always measured from **the row**, never from the anchor, so a
        lag that could not be answered exactly reports the distance to what it
        actually read rather than the distance it was asked for.
        """
        n = len(t)
        vals = np.full(n, np.nan)
        ages = np.full(n, np.nan)
        anchor = np.full(n, np.inf) if offset is None else t - offset
        for key in pd.unique(keys):
            hist = self.history_.get(str(key))
            if hist is None:
                continue
            ts = np.asarray(hist[0], dtype=float)
            vs = np.asarray(hist[1], dtype=float)
            rows = keys == key
            idx = np.searchsorted(ts, anchor[rows], side="right") - 1
            found = idx >= 0
            picked = idx[found]
            v = np.full(int(rows.sum()), np.nan)
            a = np.full(int(rows.sum()), np.nan)
            v[found] = vs[picked]
            a[found] = t[rows][found] - ts[picked]
            vals[rows] = v
            ages[rows] = a
        # A row with no ordering value has no position to measure from, so it gets
        # nothing. Without this it would fall through `searchsorted`'s NaN-sorts-last
        # behaviour and quietly receive the newest observation in its group.
        vals[np.isnan(t)] = np.nan
        ages[np.isnan(t)] = np.nan
        return vals, ages

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        keys = self._keys(df)
        t = to_periods(df[self.order])
        last, last_age = self._asof(keys, t, None)
        out[f"{self.base_name}_last"] = last
        out[f"{self.base_name}_last_age"] = last_age
        for o in self.offsets:
            v, a = self._asof(keys, t, o)
            out[f"{self.base_name}_lag{o:g}"] = v
            out[f"{self.base_name}_lag{o:g}_age"] = a
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "by": self.by,
            "value": self.value,
            "order": self.order,
            "offsets": self.offsets,
            "keep": self.keep,
            "history": self.history_,
            "frozen_at": self.frozen_at_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryEncoder":
        enc = cls(d["by"], d["value"], d["order"], d.get("offsets", ()), d["keep"])
        enc.history_ = {str(k): [list(v[0]), list(v[1])] for k, v in d["history"].items()}
        enc.frozen_at_ = d.get("frozen_at")
        return enc


ENCODER_CLASSES: dict[str, type] = {
    "target": TargetEncoder,
    "frequency": FrequencyEncoder,
    "ordinal": OrdinalEncoder,
    "group_stats": GroupStatsEncoder,
    "history": HistoryEncoder,
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


def _require_ordered_folds(periods: np.ndarray, folds) -> None:
    """Refuse folds whose training side is not wholly earlier than its validation side.

    This is checked as a *property* of the index pairs rather than by trusting a
    flag, because the property is the thing that matters: freezing history at the
    end of the training fold only means anything if the training fold is a prefix
    in time.

    Under random k-fold it is not. A validation row's most recent stored
    observation is then whichever of its neighbours happened to land in the other
    fold, so the age it reports is an artifact of the seed — change the seed and
    both the value and its age change, and neither corresponds to any serving
    situation. At serving time history ends at one point for every row and ages
    grow smoothly across the horizon; random folds punch holes in history that no
    served frame will ever have. That is a different distribution rather than a
    noisier one, which is why this refuses instead of warning.
    """
    for i, (tr, va) in enumerate(folds):
        if len(tr) == 0 or len(va) == 0:
            continue
        latest_train = np.nanmax(periods[tr])
        earliest_val = np.nanmin(periods[va])
        if not latest_train < earliest_val:
            raise ValueError(
                f"history needs forward-chaining folds and fold {i + 1} is not one: it "
                f"trains on rows up to {latest_train:g} and validates from "
                f"{earliest_val:g}, so the training side is not wholly in the past. "
                "Stored history frozen at a training fold that is scattered through "
                "time reports an age that depends on the fold seed rather than on the "
                "data, and matches no serving contract. Build folds with "
                "make_folds(..., order=<the ordering column>), which is what a run "
                "split on that column already does."
            )


def fit_history(df, by, value, order, *, folds, offsets=(), keep=None,
                forecast_horizon: float | None = None):
    """A group's recent past, frozen per fold (the tool entry point).

    Returns ``(train_features, final_encoder)`` on the same contract as
    :func:`fit_encoding` and :func:`fit_group_stats`: the frame is what the model
    may **train** on, the encoder is what runs at **predict**.

    **The fold loop here looks like** :func:`fit_group_stats`'s **and is not the
    same thing, and the difference is load-bearing.** That one iterates folds to
    exclude *the row's own contribution* from a statistic it would otherwise be
    inside — which a strictly backward lag cannot suffer from at any group size,
    since the row at ``t`` only ever reads ``t - k``. This one iterates folds to
    reproduce *the serving horizon*: each fold's history is frozen at the end of
    its training prefix, so a validation row reads exactly what a served row at
    the same distance would read, and the score measures a model that can
    actually be deployed.

    The consequence for anyone extending this: fitting on "all rows outside the
    validation fold" would satisfy the older rule and be a straight future leak
    here, because *last known* is defined relative to the fitting set. Put a later
    block in that set and the newest observation comes from the future.

    Expect the feature to look **weaker** than a naive backward lag. It is
    supposed to. The gap between the two numbers is the size of the lie, not the
    size of the loss.

    ``offsets`` are refused unless the task declares ``forecast_horizon`` and
    every offset is at least that large. Below the declared gap an offset cannot
    be answered from stored history for every served row, so ``_lag1`` on a block
    starting seven periods out would carry a seven-period-old number under a name
    that promises one. Undeclared, the last-known pair is what you get, and it is
    honest without needing the declaration.

    Under forward chaining the first block is validated by nothing, so those rows
    keep ``NaN`` — the same gap out-of-fold target encoding leaves, for the same
    reason.
    """
    offsets = [float(o) for o in offsets]
    if offsets:
        if forecast_horizon is None:
            raise ValueError(
                f"offsets={offsets} need the task to declare how far ahead it "
                "forecasts. Set TaskSpec.forecast_horizon to the gap between the end "
                "of history and the earliest row you are asked to predict, in the "
                "ordering column's units. Without it, the last-known value and its "
                "age are what history can answer honestly."
            )
        too_short = [o for o in offsets if o < forecast_horizon]
        if too_short:
            raise ValueError(
                f"offsets {too_short} are shorter than the declared forecast horizon "
                f"of {forecast_horizon:g}. A lag that short is not answerable from "
                "stored history for every served row, so the column would hold a "
                "value older than its own name claims. Use offsets of at least "
                f"{forecast_horizon:g}, or drop them and take the last-known pair."
            )

    periods = to_periods(df[order])
    _require_ordered_folds(periods, folds)

    template = HistoryEncoder(by, value, order, offsets, keep)
    names = template.output_columns
    oof = pd.DataFrame(index=df.index)
    for name in names:
        oof[name] = np.nan
    positions = {name: oof.columns.get_loc(name) for name in names}
    for tr, va in folds:
        # `df.iloc[tr]` is the training *prefix* under forward chaining, which
        # `_require_ordered_folds` has already established. That is the freeze
        # point, and it is the whole mechanism.
        fold_enc = HistoryEncoder(by, value, order, offsets, keep).fit(df.iloc[tr])
        transformed = fold_enc.transform(df.iloc[va])
        for name in names:
            oof.iloc[va, positions[name]] = transformed[name].to_numpy()
    return oof, template.fit(df)
