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

#: What may be asked of a trailing window, and the menu is short on purpose.
#:
#: ``mean`` is the one the whole slice exists for. ``std`` says how settled the
#: group has been lately, which no other encoder can express. ``median`` is the
#: statistic that survives a spike, and 0050 is a record of a spike destroying a
#: model — the last open day before a holiday sold 53% above normal, and a mean
#: over a short window inherits that where a median shrugs it off.
#:
#: **``min``/``max`` are off the menu, for 0046's reason at a shorter range.** An
#: extreme over a trailing window is close to a single observation, and "a single
#: observation is not a summary" is exactly what 0049 was written about. **``sum``
#: is off it too**: it is ``count * mean`` and the count is emitted beside every
#: window already, so it is a column a model can form and does not need given.
WINDOW_STATS = ("mean", "std", "median")


def _clean_windows(windows) -> list[tuple[float, str]]:
    """``[(90, "mean"), ...]`` validated, de-duplicated, order preserved.

    Accepts what JSON hands back, so a tool call carrying ``[[90, "mean"]]``
    arrives here as nested lists and is treated the same as tuples.
    """
    cleaned: list[tuple[float, str]] = []
    for item in windows or ():
        try:
            length, stat = item
        except (TypeError, ValueError):
            raise ValueError(
                f"windows takes (length, statistic) pairs, got {item!r}. A length "
                "on its own does not say what to do with the observations in it — "
                f"write [({item!r}, 'mean')]."
            ) from None
        length = float(length)
        stat = str(stat)
        if length <= 0:
            raise ValueError(
                f"window lengths must be positive, got {length:g}. A window of zero "
                "spans nothing and a negative one runs into the future."
            )
        if stat not in WINDOW_STATS:
            raise ValueError(
                f"unknown window statistic {stat!r}; pick from {list(WINDOW_STATS)}. "
                "min/max and sum are deliberately absent — see WINDOW_STATS."
            )
        if (length, stat) not in cleaned:
            cleaned.append((length, stat))
    return cleaned


def to_periods(values) -> np.ndarray:
    """The ordering column as floats that can be subtracted from one another.

    Every number this module produces is a *difference* of ordering values — how
    far a row sits from the observation it reads — so the ordering column has to
    support subtraction. Dates become days, numbers stay themselves, and anything
    else is refused here rather than producing a difference in unnamed units.

    Refusing is the point. A run ordered on a string month (``"2015-07"``) sorts
    correctly and subtracts to nonsense, and the nonsense would arrive as a
    plausible-looking float in an ``_ago`` column.

    **Known limitation: dates always become days, whatever their granularity.**
    That is right for a daily panel and awkward for a second-resolution event log,
    where an age reads ``0.00347`` and an hour has to be asked for as an offset of
    ``0.0417``. Nothing here is *wrong* — the arithmetic holds and the ``_ago``
    columns stay honest — but the numbers are unusable at that scale. The fix is
    a declared unit rather than an inferred one (inferring it from the data would
    make column values depend on the spacing that happened to be present), and it
    is not built. A numeric ordering column sidesteps it entirely: it keeps its
    own units, so a log indexed in epoch seconds already works in seconds.
    """
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        # Divided by a timedelta rather than scaled from the integer
        # representation, because pandas may hand back datetime64 in seconds,
        # milliseconds, microseconds or nanoseconds depending on how the column
        # was built. Assuming one of them puts every age out by a factor of a
        # thousand, which is exactly the kind of plausible-looking wrong number
        # the `_ago` columns exist to avoid. `NaT` falls through as `NaN`.
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
    """A group's own recent past, frozen at fit time and replayed as trailing windows.

    The fourth case, and the one the others cannot reach. :class:`TargetEncoder`
    and :class:`GroupStatsEncoder` summarise a group across *all* of its rows at
    once — the average, the spread — which throws away the order the rows came
    in. Nothing else in the vocabulary can say what level a group has been running
    at *lately*, because that is a statement about *when* as well as *how much*.

    What it stores, per group, is the tail of the series: the last ``keep``
    observations of ``value`` in ``order``. Bounded by construction — the
    encoder's size is (number of groups × ``keep``), never a function of how many
    rows the table has, which is what makes it storable in a lineage at all.

    **What it emits is always an average over a stretch, never the value of one
    row** (0072). That is the rule the whole class is organised around, and it has
    a price and a reason.

    A feature that is one row's value carries the noise of one row. Worse, in this
    encoder's fold geometry it does not stay one row's value: a *training* row
    reads its own fold's freeze, so across the folds the column takes thousands of
    values at ages spanning the whole table, while every *served* row reads the
    **final** freeze, so the column is one draw per group, constant for the whole
    horizon, with only its age moving. The model is fitted on a varying
    recent-value feature and served a frozen constant wearing the same name. Where
    that one draw lands on an unrepresentative period it is not idle but
    confidently wrong — measured at 0.014 rmspe on one dataset (the draw fell on a
    public holiday, and 98% of groups were shut) and 0.002 rmsle on another, each
    time against the table that would otherwise have shipped.

    A window has the same freeze and degrades gracefully instead: its left edge
    moves with the row, so each row averages its own stretch, and its ``_count``
    says how much it actually found.

    ``windows`` aggregate the stored tail rather than picking one point out of it.
    ``windows=[(90, "mean")]`` emits ``..._90_mean`` — the mean of every stored
    observation in ``[row - 90, row)`` — beside ``..._90_count``, how many
    observations that actually was.

    **The window is anchored at the row, never at the freeze point**, which is the
    single property that makes it worth building. A value anchored to the freeze is
    one draw shared by every served row, and 0050 is the record of that costing a
    working model two thirds of its score. ``[row - w, row)`` moves with the row,
    so each row averages its own stretch of the past.

    **The count is emitted with every window and is not optional.** A mean over
    two observations and a mean over ninety are the same number wearing the same
    name, and only one of them is a summary. A window that quietly averages two
    points on the rows that matter has collapsed back into the single draw this
    class exists to avoid, and the count is what makes that visible. It is emitted
    once per distinct window length, however many statistics that length was asked
    for.

    That is also the honest limit of the rule above: a window is only an average
    while it holds several observations. At exactly the forecast horizon it holds
    one — the freeze-point observation — which is the old ``_latest`` with extra
    arithmetic. The count says so; :func:`fit_history` refuses lengths below the
    horizon outright.

    **The right endpoint is strict.** The window ends *before* the row's own
    ordering value, so the row can never be inside its own summary. A self-included
    point in a mean over ninety is diluted to invisibility rather than showing up
    as a perfect correlation, so this cannot be left to the per-fold freeze the way
    a point lookup could (0048). A guarantee that only holds when the numbers are
    checked is not a guarantee.

    **Two families this class still knows how to compute and will not fit any
    more**: the last-known pair (``_latest`` / ``_latest_ago``) and true lags
    (``offsets`` → ``..._7_ago`` and its ``_exact`` flag). Both are one row's
    value, both are refused by :func:`fit_history` since 0072, and the machinery
    survives here for exactly one reason: **bundles written before that change must
    still load and predict.** An encoder restored by :meth:`from_dict` emits what
    it emitted when it was fitted, or the artifact silently produces a different
    column set than the model it serves was trained on (0055). Nothing new
    constructs them.

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

    def __init__(self, by: list[str], value: str, order: str, offsets=(),
                 keep: int | None = None, windows=(), latest: bool | None = None):
        self.by = [str(c) for c in by]
        self.value = str(value)
        self.order = str(order)
        self.offsets = [float(o) for o in offsets]
        self.windows = _clean_windows(windows)
        #: Whether to emit the last-known pair. Defaults to "only when there is
        #: nothing else to emit", so a construction that predates 0072 — offsets
        #: alone, or nothing at all — still produces what it used to, while a
        #: windowed one does not acquire it. :meth:`from_dict` passes it
        #: explicitly, because a restored bundle's column set is not negotiable.
        self.latest = (
            bool(latest) if latest is not None else not (self.offsets or self.windows)
        )
        if not self.latest and not self.offsets and not self.windows:
            raise ValueError(
                "nothing to emit: no windows, no offsets, and latest=False. A "
                "history encoder needs at least one trailing window — see "
                "fit_history for why a single stored observation is no longer a "
                "feature this builds."
            )
        #: Distinct lengths in first-asked order — one count column each, not one
        #: per (length, statistic) pair, since the count is a property of the
        #: window and not of what was asked of it.
        self.window_lengths = list(dict.fromkeys(w for w, _ in self.windows))
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
        # Enough tail to answer the furthest offset and fill the longest window,
        # assuming roughly one observation per unit of the ordering column.
        # Irregular series have to say so by passing `keep` — an assumption that is
        # wrong is better than a default that silently truncates, and the `_ago`
        # and `_count` columns show it when it happens.
        reach = max([*self.offsets, *self.window_lengths], default=0.0)
        self.keep = int(keep) if keep is not None else int(reach) + 1
        if not 1 <= self.keep <= MAX_KEEP:
            # A window longer than the cap is refused rather than quietly served
            # from a shortened tail. A window that spans a year and averages the
            # most recent 1000 observations of a series that has thousands is a
            # wrong number under a confident name, which is the failure this whole
            # area keeps producing.
            if keep is None and reach > MAX_KEEP:
                raise ValueError(
                    f"a reach of {reach:g} in {self.order!r} needs {int(reach) + 1} "
                    f"stored observations per group and the cap is {MAX_KEEP}. If the "
                    "series really is that dense, this is more history than an encoder "
                    "may carry; if it is sparser than one observation per unit — weekly "
                    "readings over a year-long window, say — pass keep explicitly to say "
                    "so, and the _count columns will show whether it was enough."
                )
            raise ValueError(f"keep must be between 1 and {MAX_KEEP}, got {self.keep}")
        self.history_: dict[str, list[list[float]]] = {}
        self.frozen_at_: float | None = None

    @property
    def base_name(self) -> str:
        return "_".join([*self.by, self.value])

    @property
    def output_columns(self) -> list[str]:
        names = []
        if self.latest:
            names += [f"{self.base_name}_latest", f"{self.base_name}_latest_ago"]
        for o in self.offsets:
            names += [f"{self.base_name}_{o:g}_ago", f"{self.base_name}_{o:g}_ago_exact"]
        for w, stat in self.windows:
            names.append(f"{self.base_name}_{w:g}_{stat}")
        for w in self.window_lengths:
            names.append(f"{self.base_name}_{w:g}_count")
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

    def _window_stats(self, keys: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        """Aggregates of the stored tail over ``[row - w, row)``, per requested pair.

        Cumulative sums do the work: with the tail already sorted, the two ends of
        a window are one ``searchsorted`` each, and mean, count and standard
        deviation follow by subtraction. That is what keeps a window the same order
        of cost as a lag rather than a scan per row. ``median`` cannot be had that
        way and slices per row, which is why it is the expensive member of
        :data:`WINDOW_STATS`.
        """
        n = len(t)
        wanted: dict[float, list[str]] = {w: [] for w in self.window_lengths}
        for w, stat in self.windows:
            wanted[w].append(stat)
        names = [f"{self.base_name}_{w:g}_{s}" for w, s in self.windows]
        names += [f"{self.base_name}_{w:g}_count" for w in self.window_lengths]
        out = {name: np.full(n, np.nan) for name in names}

        for key in pd.unique(keys):
            hist = self.history_.get(str(key))
            if hist is None:
                continue
            ts = np.asarray(hist[0], dtype=float)
            vs = np.asarray(hist[1], dtype=float)
            csum = np.concatenate([[0.0], np.cumsum(vs)])
            csq = np.concatenate([[0.0], np.cumsum(vs * vs)])
            rows = keys == key
            rt = t[rows]
            # `side="left"` on the right end is the strict endpoint: an observation
            # sharing the row's ordering value is not in the row's past, and letting
            # it in would put the row inside its own summary.
            hi = np.searchsorted(ts, rt, side="left")
            for w, stats in wanted.items():
                lo = np.searchsorted(ts, rt - w, side="left")
                count = (hi - lo).astype(float)
                filled = count > 0
                total = csum[hi] - csum[lo]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(filled, total / np.where(filled, count, 1.0), np.nan)
                out[f"{self.base_name}_{w:g}_count"][rows] = count
                if "mean" in stats:
                    out[f"{self.base_name}_{w:g}_mean"][rows] = mean
                if "std" in stats:
                    # One observation has no spread, so it gets NaN rather than the
                    # zero a population formula would hand back — a confident zero
                    # for "this group never moves" off a single point is the same
                    # class of lie as a mean over one.
                    pair = count > 1
                    sq = csq[hi] - csq[lo]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        var = (sq - count * mean * mean) / np.where(pair, count - 1.0, 1.0)
                    # Float error can drive a genuinely-zero variance slightly
                    # negative, and a NaN out of a square root would read as "no
                    # data" when the truth is "no movement".
                    out[f"{self.base_name}_{w:g}_std"][rows] = np.where(
                        pair, np.sqrt(np.maximum(var, 0.0)), np.nan
                    )
                if "median" in stats:
                    med = np.full(len(rt), np.nan)
                    for i in np.flatnonzero(filled):
                        med[i] = np.median(vs[lo[i]:hi[i]])
                    out[f"{self.base_name}_{w:g}_median"][rows] = med

        # Same reason as `_asof`: a row with no ordering value has no window to
        # take, and `searchsorted` would otherwise hand it an empty one at the end
        # of the tail — a count of 0 reading as "this group had nothing to say"
        # when the truth is that the row never asked.
        for name in out:
            out[name][np.isnan(t)] = np.nan
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        keys = self._keys(df)
        t = to_periods(df[self.order])
        if self.latest:
            last, last_age = self._asof(keys, t, None)
            out[f"{self.base_name}_latest"] = last
            out[f"{self.base_name}_latest_ago"] = last_age
        for o in self.offsets:
            v, a = self._asof(keys, t, o)
            out[f"{self.base_name}_{o:g}_ago"] = v
            # A flag rather than the age, because for an offset the age is
            # near-constant: it equals `o` whenever the lookup succeeded, which on
            # Rossmann v8 was 80% of served rows. All of its information sits in
            # the failures, so 1/0 carries the same signal in a column whose
            # meaning is legible — and a lag that fell back is exactly the case
            # worth flagging, since it has silently become the last-known value
            # (0050). `_latest` keeps a real age, which genuinely varies per row.
            #
            # `isclose` rather than `==`: ages are differences of floats, and an
            # exact match that fails on the last bit would report a working lag as
            # broken.
            exact = np.where(np.isnan(a), np.nan, np.isclose(a, o).astype(float))
            out[f"{self.base_name}_{o:g}_ago_exact"] = exact
        for name, values in self._window_stats(keys, t).items():
            out[name] = values
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "by": self.by,
            "value": self.value,
            "order": self.order,
            "offsets": self.offsets,
            "keep": self.keep,
            # Pairs go out as lists so the round trip through JSON is the identity;
            # `_clean_windows` takes them back in either shape.
            "windows": [[w, s] for w, s in self.windows],
            "latest": self.latest,
            "history": self.history_,
            "frozen_at": self.frozen_at_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryEncoder":
        # `windows` absent is a bundle written before this existed, and it loads as
        # what it was: last-known and offsets, no windows.
        #
        # `latest` absent is the same situation and the default must be `True`
        # rather than the constructor's `None`: every bundle written before 0072
        # emitted the pair unconditionally, so resolving it from the offsets and
        # windows here would hand back an encoder that produces *fewer columns
        # than the model it is loaded for was trained on*. That is a KeyError at
        # predict time on an artifact that used to work — 0055's failure, and the
        # reason this line is not `d.get("latest")`.
        enc = cls(d["by"], d["value"], d["order"], d.get("offsets", ()), d["keep"],
                  d.get("windows", ()), latest=d.get("latest", True))
        enc.history_ = {str(k): [list(v[0]), list(v[1])] for k, v in d["history"].items()}
        enc.frozen_at_ = d.get("frozen_at")
        return enc


#: What may be asked of the records a stream holds for one row, and the menu is
#: short for :data:`WINDOW_STATS`' reasons at a different grain.
#:
#: ``sum`` is the one this exists for: a stream of bookings, orders or claims is
#: about a *total*, and unlike a trailing window — where ``sum`` was left off
#: because it is ``count * mean`` and the count is already there — here the total
#: is the quantity and the mean is the derived one. Both are offered anyway,
#: because "forty people booked" and "the average booking was for two" are
#: different features. ``std`` says how uniform the records were.
#:
#: **``min``/``max``/``median`` are off the menu.** A bucket in a stream typically
#: holds a handful of records, and an extreme or a middle over three of them is
#: close to a single record — 0049's "a sample of size one" at the other end of
#: the harness.
STREAM_STATS = ("sum", "mean", "std")


class StreamAggregateEncoder:
    """An aggregate of a second input, restricted to what was knowable in time.

    A **stream** is a related file at a finer grain than the table being
    modelled, whose records carry *two* times: the time they are **about**
    (``event_time``) and the time they became **knowable** (``observed_at``). The
    canonical shapes are reservations against a visit, orders against a delivery
    date, claims against an accident date — the pattern is old enough to have
    names in two fields at once (bitemporality's *valid* vs *transaction* time in
    data warehousing, "incurred but not reported" in insurance).

    Such a file cannot be a column: several records collapse into one number per
    row of the table. Collapsing them is the obvious thing to do and it throws
    away ``observed_at``, which is the only thing that says which of those records
    a *forecast* could have seen. The collapsed column is then systematically
    larger in training than at serving, by an amount that **grows with the
    horizon** — and no guard in this harness sees it: no fold boundary is crossed,
    no encoder is misfitted, the column exists at predict time, and validation
    *confirms* it, because validation is drawn from the same uncut half.

    What this emits instead, per row and per requested ``lead``, is the aggregate
    over the stream records that

    * carry this row's key (the ``by`` columns, which the stream must also have),
    * are **about** this row's position in ``order`` — ``event_time`` equal to the
      row's ordering value, and
    * were **observed at or before** ``order - lead``.

    So ``lead=7`` answers "what would a forecast issued a week ahead have been
    able to see", whatever the file happens to contain now.

    **Emitted as a pair, never a bare value.** ``..._lead7_sum`` always comes with
    ``..._lead7_count``, for the reason every window in :class:`HistoryEncoder`
    carries a count: a sum over one record and a sum over forty are the same
    number wearing the same name, and only the count says which one a given row
    got. It is also where this design's known limit shows up — see below.

    **Several leads are separate columns, not competing settings.** A short lead
    and a long one are different features: one is what the near edge of a forecast
    window can see, the other what the far edge can. Asking for both and letting
    the model use both is the point.

    Three properties worth stating outright, because they are what makes this
    legal where the obvious ``merge`` + ``groupby`` in ``transform_features`` is
    not:

    * **No stored aggregate.** The encoder persists ``by``, ``value``,
      ``event_time``, ``observed_at``, ``order``, ``leads`` and ``stats``, and
      nothing else. The stream is handed in again at transform time and the
      aggregate recomputed — 0010's "store the code, not the columns" applied to
      an aggregate. Precomputing ``{(key, event_time) -> value}`` at fit time and
      storing it would need no plumbing at all, and it only works while the stream
      file already contains the future: true of a downloaded competition file,
      false of every real serving situation, where next month's bookings arrive
      after the fit. 0050 is the record of a frozen anchor costing a working model
      two thirds of its score.
    * **Batch independence** (0011). What a row reads is a function of the row's
      own key and time and of the stream, never of which other rows are in the
      frame. One row alone and a batch of a million get identical answers, which
      an aggregate computed inside ``transform_features`` cannot promise.
    * **No fold loop, and not by oversight.** The lead restriction is already
      strictly backward *per row*: a training row at ``t`` reads only records
      observed by ``t - lead``, so it cannot see anything recorded after its own
      position, whichever fold it is in. That is the property :func:`fit_history`
      needs its per-fold freeze to reproduce. Nothing here reads the target.

    **The known limit, stated rather than hidden.** ``forecast_horizon`` is the
    *floor* of a scored block (0047), so a lead set at the floor is honest for the
    near edge of that block and under-restrictive for the far edge — a row 39
    periods out is served an aggregate a row 1 period out could have had. The
    ``_count`` column is what makes that visible per row. Declaring the far edge
    too would close it by construction and is not built: it is one more field the
    data's owner has to get right on every ordered run, and the count already
    reports the damage where it lands.

    **The grain is measured, stored and replayed — it is not left to the caller.**
    A stream is almost always finer than the table it feeds: a store-day table
    against bookings stamped ``2017-03-13 18:00``. Something has to say which
    row a record at 18:00 belongs to, and there are three ways to decide it,
    two of them bad:

    * *Require the two to line up already*, and let a mismatch produce a
      well-formed table of zeros. That pushes a load-bearing rule into a prep
      script where nothing records it and nothing enforces it — an unwritten
      precondition is not a design.
    * *Infer a bucket at transform time* from the rows in front of it. That is
      batch-dependent, and 0011 refuses it for good reason.
    * **Measure the table's own spacing once, at fit time, and store it.** That is
      what happens. :meth:`fit` reads the distinct ordering values, takes the
      commonest gap between consecutive ones as ``grain_``, and keeps it — with
      ``origin_`` — in the encoder, so it survives into the lineage and is
      replayed byte-identically rather than re-derived from whatever frame turns
      up at predict time.

    Both sides are then floored onto that grid: a record at ``2017-03-13 18:00``
    and a row at ``2017-03-13`` land in the same bucket, whichever way round the
    resolutions happen to be. The *lead* is unaffected — it is still measured from
    the row's true ordering value against the record's true ``observed_at``, both
    at full resolution.

    This is a measurement, not 0051's inference. 0051 is about a column whose
    *unit* is unknown; the spacing of the rows is a fact about the table, in the
    ordering column's own units, and one this class can check. So it does: if the
    distinct ordering values do not sit on the grid the commonest gap implies,
    the measurement is not trustworthy and :meth:`fit` **refuses**, asking for
    ``event_grain`` to be declared instead of guessing. A declared grain is used
    as given — the data's owner outranks the measurement.
    """

    kind = "stream_aggregate"

    def __init__(self, stream: str, by: list[str], value: str, event_time: str,
                 observed_at: str, order: str, leads=(), stats=("sum",),
                 event_grain: float | None = None):
        self.stream = str(stream)
        self.by = [str(c) for c in by]
        self.value = str(value)
        self.event_time = str(event_time)
        self.observed_at = str(observed_at)
        self.order = str(order)
        self.leads = [float(x) for x in leads]
        # De-duplicated with the asked-for order preserved, so the column order a
        # caller gets back is the one they wrote down.
        self.stats = list(dict.fromkeys(str(s) for s in stats))
        #: Declared bucket width, or ``None`` to measure it from the table at fit
        #: time. Declared always wins: the data's owner knows things the spacing
        #: of the rows cannot show.
        self.event_grain = None if event_grain is None else float(event_grain)
        if self.event_grain is not None and self.event_grain <= 0:
            raise ValueError(
                f"event_grain must be positive, got {self.event_grain:g}. It is the "
                "width of the period one row covers, in the ordering column's units."
            )
        if not self.by:
            raise ValueError("by must name at least one column to group on")
        if not self.leads:
            raise ValueError(
                "leads must name at least one lead time. A stream aggregate with no "
                "lead is the collapsed column this encoder exists to replace — it "
                "counts records the forecast could not have seen."
            )
        if not self.stats:
            raise ValueError("stats must name at least one statistic")
        if self.event_time == self.observed_at:
            raise ValueError(
                f"event_time and observed_at are both {self.event_time!r}. A stream "
                "whose two times are one column is not bitemporal: every record "
                "would count as knowable exactly when it happened, and there is "
                "nothing for a lead to restrict."
            )
        if self.value in (self.event_time, self.observed_at) or self.value in self.by:
            raise ValueError(
                f"value column {self.value!r} is also a key or a time column. The "
                "value is what gets aggregated; a timestamp aggregated by its own "
                "grouping is not a feature."
            )
        for name in (self.event_time, self.observed_at):
            if name in self.by:
                raise ValueError(
                    f"by={self.by} may not contain the stream's time column {name!r}: "
                    "grouping records by the moment they happened gives one record "
                    "per group and nothing to aggregate"
                )
        if self.order in self.by:
            raise ValueError(
                f"by={self.by} may not contain the ordering column {self.order!r}"
            )
        bad = [s for s in self.stats if s not in STREAM_STATS]
        if bad:
            raise ValueError(
                f"unknown stream statistic(s) {bad}; pick from {list(STREAM_STATS)}. "
                "min/max and median are deliberately absent — see STREAM_STATS."
            )
        negative = [x for x in self.leads if x < 0]
        if negative:
            raise ValueError(
                f"leads must not be negative, got {negative}. A negative lead admits "
                "records observed *after* the row it describes, which is the leak "
                "this encoder exists to close."
            )
        #: The grid both sides are floored onto. Fitted, persisted, and replayed
        #: rather than re-derived — a bucket width that changed between fit and
        #: predict would silently change what every column means.
        self.grain_: float | None = self.event_grain
        self.origin_: float | None = None
        #: Diagnostics from the last :meth:`fit`, for the tool to report. Not
        #: persisted: they describe the stream that was fitted against, and the
        #: stream handed to :meth:`transform` is a different one by design.
        self.n_records_: int | None = None
        self.n_buckets_: int | None = None
        self.matched_rows_: float | None = None

    @property
    def base_name(self) -> str:
        """Everything that identifies the aggregate, including the stream's name.

        The stream name is in there because two streams can carry the same key
        and value column names — a table joined to both an internal and a partner
        booking feed is the ordinary case — and two aggregates landing on one
        column name would silently overwrite each other.
        """
        return "_".join([self.stream, *self.by, self.value])

    def count_column(self, lead: float) -> str:
        return f"{self.base_name}_lead{lead:g}_count"

    @property
    def output_columns(self) -> list[str]:
        names = []
        for lead in self.leads:
            names += [f"{self.base_name}_lead{lead:g}_{s}" for s in self.stats]
            names.append(self.count_column(lead))
        return names

    def _keys(self, frame: pd.DataFrame, what: str) -> np.ndarray:
        missing = [c for c in self.by if c not in frame.columns]
        if missing:
            raise ValueError(
                f"cannot aggregate stream {self.stream!r}: the grouping columns "
                f"{missing} are not in {what}. The key columns are read on both "
                "sides of the join, so they have to carry the same names in the "
                "table and in the stream — renaming them is the prep script's job."
            )
        keys = frame[self.by[0]].astype(str)
        for col in self.by[1:]:
            keys = keys + _KEY_SEP + frame[col].astype(str)
        return keys.to_numpy()

    def _clean_stream(self, stream: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in (self.value, self.event_time, self.observed_at)
                   if c not in stream.columns]
        if missing:
            raise ValueError(
                f"stream {self.stream!r} is missing {missing}; its columns are "
                f"{list(stream.columns)}"
            )
        s = pd.DataFrame({
            "key": self._keys(stream, f"stream {self.stream!r}"),
            "et": to_periods(stream[self.event_time]),
            "ot": to_periods(stream[self.observed_at]),
            "v": pd.to_numeric(stream[self.value], errors="coerce").to_numpy(dtype=float),
        })
        # A record missing either time cannot be placed or dated, and one with no
        # value would turn a whole bucket's sum into NaN. Dropping them costs the
        # count a record it could arguably have kept; a NaN sum beside a count of
        # forty would be worse, since it reads as "nothing was knowable" when the
        # truth is "one record was malformed".
        return s.dropna(subset=["key", "et", "ot", "v"]).reset_index(drop=True)

    def _measure_grid(self, t: np.ndarray) -> tuple[float, float]:
        """The bucket width and phase, taken from the table's own spacing.

        The commonest gap between consecutive distinct ordering values is the
        grain — and it is only *believed* if every distinct value sits on the grid
        that gap implies. A daily panel with a few table-wide holidays missing
        passes (the gaps are 1, 1, 2, 1 and every value is a whole number of days
        from the first). An irregular event log fails, and gets asked for a
        declaration rather than served a bucket width invented on its behalf.

        A ragged spine is fine and is not what this measures: the gaps are taken
        over the ordering values present *anywhere in the table*, so a store that
        reports on four days a week does not make the grain four days.
        """
        finite = t[np.isfinite(t)]
        if not len(finite):
            raise ValueError(
                f"every value of the ordering column {self.order!r} is missing, so "
                "there is no grain to measure and nothing to align the stream to."
            )
        origin = float(finite.min())
        if self.event_grain is not None:
            return self.event_grain, origin
        u = np.unique(finite)
        if len(u) < 2:
            raise ValueError(
                f"the table has one distinct value of {self.order!r}, which shows no "
                "spacing to measure. Pass event_grain=<width of the period one row "
                "covers, in that column's units> to say how wide a row's bucket is."
            )
        gaps = np.round(np.diff(u), 9)
        widths, counts = np.unique(gaps, return_counts=True)
        grain = float(widths[int(counts.argmax())])
        residual = (u - origin) / grain
        off_grid = np.abs(residual - np.round(residual)) > 1e-6
        if off_grid.any():
            example = float(u[np.flatnonzero(off_grid)[0]])
            raise ValueError(
                f"cannot measure the grain of {self.order!r}: its commonest gap is "
                f"{grain:g}, but {int(off_grid.sum())} of {len(u)} distinct values do "
                f"not sit on a grid of that width (e.g. {example:g}, which is "
                f"{(example - origin) / grain:.3f} grains from the first). Rows spaced "
                "irregularly have no bucket width to read off them, and guessing one "
                "would decide which records belong to which row by accident. Pass "
                "event_grain=<width of the period one row covers> to declare it."
            )
        return grain, origin

    def _bucket(self, values: np.ndarray) -> np.ndarray:
        """Floor onto the fitted grid, so both sides meet at the same resolution."""
        # A relative tolerance, not an absolute one: ordering values are often days
        # since the epoch, where float64 resolves to tens of nanoseconds and a value
        # sitting exactly on a boundary can land a hair below it.
        steps = np.floor((values - self.origin_) / self.grain_ + 1e-6)
        return self.origin_ + steps * self.grain_

    def fit(self, df: pd.DataFrame, y=None, *, stream: pd.DataFrame | None = None
            ) -> "StreamAggregateEncoder":
        """Measure the grid, check the stream against the table, record what matched.

        No aggregate is learned — that is the design (see the class docstring).
        What is learned is the *grain*: how wide a period one row of the table
        covers, so that a record stamped finer than the table can be placed on the
        row it belongs to. That plus an early refusal is what this is for; the
        failure being refused is a stream that matches nothing and therefore
        produces a perfectly well-formed table of zeros, which is the class of
        quietly-wrong number this whole area keeps producing.
        """
        if stream is None:
            raise ValueError(
                f"fitting a stream aggregate needs the stream itself: pass "
                f"stream=<the {self.stream!r} frame>."
            )
        s = self._clean_stream(stream)
        if self.order not in df.columns:
            raise ValueError(
                f"ordering column {self.order!r} is not in this frame; columns are "
                f"{list(df.columns)}"
            )
        keys = self._keys(df, "the table")
        t = to_periods(df[self.order])
        self.grain_, self.origin_ = self._measure_grid(t)
        self.n_records_ = int(len(s))
        if self.n_records_ == 0:
            raise ValueError(
                f"stream {self.stream!r} has no usable records: every row is missing "
                f"one of {self.value!r}, {self.event_time!r} or {self.observed_at!r}."
            )
        buckets = pd.DataFrame({
            "key": s["key"].to_numpy(), "bucket": self._bucket(s["et"].to_numpy()),
        }).drop_duplicates()
        # A hash join rather than an index lookup: both sides carry a text key, and
        # a `MultiIndex.get_indexer` over object levels is minutes where this is
        # milliseconds on a table of a quarter of a million rows.
        buckets["_hit"] = True
        rows = pd.DataFrame({"key": keys, "bucket": self._bucket(t)})
        matched = rows.merge(buckets, on=["key", "bucket"], how="left")["_hit"].notna()
        self.n_buckets_ = int(len(buckets))
        self.matched_rows_ = float(matched.mean()) if len(matched) else 0.0
        if self.matched_rows_ == 0.0:
            raise ValueError(
                f"no row of the table matches any record in stream {self.stream!r}, "
                f"bucketing both onto a grid of {self.grain_:g} in {self.order!r}. The "
                f"keys {self.by} are the likely culprit — they are compared as text, so "
                "a store id stored as 'A01' on one side and 'a01' on the other never "
                "meets. Left alone this would have produced a column of zeros reading "
                "'nothing was ever recorded'."
            )
        return self

    def _aggregate(self, df: pd.DataFrame, stream: pd.DataFrame) -> dict[str, np.ndarray]:
        """One pass per lead over the stream's per-bucket running totals.

        The work is done by :func:`pandas.merge_asof`: with the records of each
        ``(key, event_time)`` bucket sorted by ``observed_at`` and carrying their
        running count and running sums, the last record at or before
        ``row - lead`` *is* the prefix aggregate. So a lead costs one merge rather
        than a scan per row, and adding leads is cheap enough that asking for
        several is the normal thing to do.
        """
        if self.order not in df.columns:
            raise ValueError(
                f"ordering column {self.order!r} is not in this frame; columns are "
                f"{list(df.columns)}"
            )
        if self.grain_ is None or self.origin_ is None:
            raise ValueError(
                "this encoder has no grid yet, so it cannot say which row a record "
                "belongs to. Call fit(df, stream=...) — or load one that was fitted, "
                "which carries the grid with it."
            )
        s = self._clean_stream(stream)
        keys = self._keys(df, "the table")
        t = to_periods(df[self.order])
        n = len(df)

        # Both sides floored onto the *fitted* grid. Serving rows sit past the end
        # of the table this was fitted on, which is exactly why the grid is stored
        # arithmetic (width + phase) and not the list of dates that were seen.
        #
        # The bucket stays a (key, floored time) *pair* rather than being resolved
        # to an integer id: `merge_asof` groups on several `by` columns natively
        # and factorizes them internally, where doing it here by hand meant a
        # `MultiIndex.get_indexer` over a text level — minutes on a quarter of a
        # million rows against milliseconds for the merge that follows.
        s = s.assign(bucket=self._bucket(s["et"].to_numpy()), v2=s["v"] ** 2)
        # Sorted within bucket so the cumulative columns are prefix aggregates of
        # the records a lead can reach; `stable` so records sharing an
        # `observed_at` keep the file's order and the result is reproducible.
        s = s.sort_values(["key", "bucket", "ot"], kind="stable").reset_index(drop=True)
        grouped = s.groupby(["key", "bucket"], sort=False)
        s["csum"] = grouped["v"].cumsum()
        s["csq"] = grouped["v2"].cumsum()
        s["cnt"] = grouped.cumcount() + 1.0
        # `merge_asof` wants its right side monotonic in the on-key across the
        # whole frame, not just within `by` groups; the cumulative columns travel
        # with their rows, so re-sorting here does not disturb them.
        right = s.sort_values("ot", kind="stable")[
            ["ot", "key", "bucket", "cnt", "csum", "csq"]
        ]

        row_bucket = self._bucket(t)
        # "This key never appears in the stream at all" and "this key had nothing
        # to report on this day" are different facts, and only the first one is a
        # missing value. A store absent from a booking feed gets NaN; a store in
        # the feed with no bookings that day gets a count of 0 and a sum of 0,
        # which is the true total. Same split `_EXTENSIVE_STATS` makes.
        known = pd.Series(keys).isin(set(s["key"])).to_numpy() & np.isfinite(t)

        out: dict[str, np.ndarray] = {}
        for lead in self.leads:
            anchor = t - lead
            cnt = np.zeros(n)
            csum = np.zeros(n)
            csq = np.zeros(n)
            usable = np.flatnonzero(np.isfinite(anchor))
            if len(usable) and len(right):
                left = pd.DataFrame({
                    "anchor": anchor[usable], "key": keys[usable],
                    "bucket": row_bucket[usable], "pos": usable,
                }).sort_values("anchor", kind="stable")
                merged = pd.merge_asof(
                    left, right, left_on="anchor", right_on="ot",
                    by=["key", "bucket"],
                    direction="backward", allow_exact_matches=True,
                )
                pos = merged["pos"].to_numpy()
                # No match means no record of this bucket was observed in time,
                # which is a real zero rather than an absence: the bucket exists,
                # so this key was in the stream.
                cnt[pos] = merged["cnt"].fillna(0.0).to_numpy()
                csum[pos] = merged["csum"].fillna(0.0).to_numpy()
                csq[pos] = merged["csq"].fillna(0.0).to_numpy()

            filled = cnt > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(filled, csum / np.where(filled, cnt, 1.0), np.nan)
            out[self.count_column(lead)] = np.where(known, cnt, np.nan)
            for stat in self.stats:
                name = f"{self.base_name}_lead{lead:g}_{stat}"
                if stat == "sum":
                    out[name] = np.where(known, csum, np.nan)
                elif stat == "mean":
                    out[name] = np.where(known, mean, np.nan)
                else:
                    # One record has no spread, so it gets NaN rather than the zero
                    # a population formula hands back — a confident zero off a
                    # single booking is the same lie as a mean over one.
                    pair = cnt > 1
                    with np.errstate(invalid="ignore", divide="ignore"):
                        var = (csq - cnt * mean * mean) / np.where(pair, cnt - 1.0, 1.0)
                    # Float error can drive a genuinely-zero variance slightly
                    # negative, and a NaN out of a square root would read as "no
                    # records" when the truth is "no variation".
                    std = np.where(pair, np.sqrt(np.maximum(var, 0.0)), np.nan)
                    out[name] = np.where(known, std, np.nan)
        return out

    def transform(self, df: pd.DataFrame, stream: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
        if stream is None:
            raise ValueError(
                f"this table's features come from stream {self.stream!r}, which has "
                "to be handed in again at predict time — the encoder stores the "
                "recipe, not the numbers, precisely so that a stream which has moved "
                "on since the fit is the one that gets used."
            )
        out = df.copy()
        for name, values in self._aggregate(df, stream).items():
            out[name] = values
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "stream": self.stream,
            "by": self.by,
            "value": self.value,
            "event_time": self.event_time,
            "observed_at": self.observed_at,
            "order": self.order,
            "leads": self.leads,
            "stats": self.stats,
            "event_grain": self.event_grain,
            # The grid goes out with the recipe. It is the one thing here that was
            # *measured*, so replaying it is the difference between a column that
            # means the same at predict time and one that quietly re-derives its
            # bucket width from whatever frame turned up.
            "grain": self.grain_,
            "origin": self.origin_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StreamAggregateEncoder":
        enc = cls(d["stream"], d["by"], d["value"], d["event_time"], d["observed_at"],
                  d["order"], d["leads"], d["stats"], d.get("event_grain"))
        enc.grain_ = d.get("grain")
        enc.origin_ = d.get("origin")
        return enc


ENCODER_CLASSES: dict[str, type] = {
    "target": TargetEncoder,
    "frequency": FrequencyEncoder,
    "ordinal": OrdinalEncoder,
    "group_stats": GroupStatsEncoder,
    "history": HistoryEncoder,
    "stream_aggregate": StreamAggregateEncoder,
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


def fit_history(df, by, value, order, *, folds, keep=None,
                forecast_horizon: float | None = None, windows=()):
    """A group's recent past as trailing windows, frozen per fold (the tool entry point).

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

    **``windows`` is required, and single-row features are not built here** — no
    lags, no last-known pair (0072). Both are the value of one row, and one row's
    value carries one row's noise; a trailing average over the same stretch does
    not. The failure is not hypothetical arithmetic: this encoder's own freeze
    turns a single stored observation into a per-group **constant** across the
    whole served horizon, while the same column varies freely in training, and two
    datasets priced that at 0.014 and 0.002 against the table that would otherwise
    have shipped.

    What a lag was reached for is still expressible, and usually better: put the
    structure in the **key** and average over the stretch. "Sales on the same
    weekday a year ago" is ``by=[store, weekday]`` with a 364-length window, which
    is what the Rossmann winner's own feature set does, rather than an offset of
    364 that reads one day.

    ``windows`` are gated on ``forecast_horizon``. A row ``d`` ahead of the freeze
    has stored history only up to ``row - d``, so a window of ``w`` covers
    ``[row - w, row - d]`` and is **empty for every served row** when ``w < d``. At
    exactly the horizon it holds the freeze-point observation alone, which is the
    single draw this function exists to refuse, and the ``_count`` column says so.

    Under forward chaining the first block is validated by nothing, so those rows
    keep ``NaN`` — the same gap out-of-fold target encoding leaves, for the same
    reason.
    """
    windows = _clean_windows(windows)
    if not windows:
        raise ValueError(
            "fit_history needs at least one window, e.g. windows=[(90, 'mean')]. "
            "The last-known value and true lags are no longer fitted: both are one "
            "row's value, and this encoder's freeze makes a single stored "
            "observation a per-group constant for every row you predict while it "
            "varies across the folds you train on. Ask for a trailing average over "
            "the stretch instead — and if you wanted a seasonal lag, put the season "
            "in `by` and take a window over it."
        )
    if windows:
        if forecast_horizon is None:
            raise ValueError(
                f"windows={[(f'{w:g}', s) for w, s in windows]} need the task to "
                "declare how far ahead it forecasts. Set TaskSpec.forecast_horizon "
                "to the gap between the end of history and the earliest row you are "
                "asked to predict, in the ordering column's units. Without it there "
                "is no way to tell a window that spans real history from one that "
                "lands entirely in the gap and is empty for every row."
            )
        short = sorted({w for w, _ in windows if w < forecast_horizon})
        if short:
            raise ValueError(
                f"window lengths {[f'{w:g}' for w in short]} are shorter than the "
                f"declared forecast horizon of {forecast_horizon:g}. Stored history "
                "stops one horizon before the earliest row you predict, so a window "
                "that short covers only the gap and would be empty for every served "
                f"row. Use lengths longer than {forecast_horizon:g} — comfortably "
                "longer, or the window holds the freeze-point observation alone."
            )

    periods = to_periods(df[order])
    _require_ordered_folds(periods, folds)

    template = HistoryEncoder(by, value, order, (), keep, windows, latest=False)
    names = template.output_columns
    oof = pd.DataFrame(index=df.index)
    for name in names:
        oof[name] = np.nan
    positions = {name: oof.columns.get_loc(name) for name in names}
    for tr, va in folds:
        # `df.iloc[tr]` is the training *prefix* under forward chaining, which
        # `_require_ordered_folds` has already established. That is the freeze
        # point, and it is the whole mechanism.
        fold_enc = HistoryEncoder(by, value, order, (), keep, windows,
                                   latest=False).fit(df.iloc[tr])
        transformed = fold_enc.transform(df.iloc[va])
        for name in names:
            oof.iloc[va, positions[name]] = transformed[name].to_numpy()
    return oof, template.fit(df)
