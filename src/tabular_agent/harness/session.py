"""RunSession: the harness-side handle on one run.

Everything durable lives in ``run_dir/`` (spec §4) — this object holds only the
pointers needed to find it again, so a fresh process (or a fresh agent after a
context reset) can be handed the same ``run_dir`` and continue.

The input table is split once, before any tool runs, into a **work** half the
agent builds features on and a **validation** half it is scored against. Both
live in ``run_dir/data/``, so a run directory stays movable (0003).
"""

import os

import numpy as np

from ..cv import boundary_cut
from ..io import load_table, save_parquet
from ..runctx import RunContext
from ..taskspec import ResolvedSpec, TaskSpec, resolve_taskspec
from .stop import StopPolicy

#: Hard floor, and deliberately about *computability* rather than informativeness.
#: Below this a half cannot produce a metric worth the name at all. How many rows
#: make a validation score trustworthy is a judgement about the data, and that
#: judgement is the user's — a 90-row dataset is small, not invalid, and refusing
#: to run on it would be the library overruling the person who has the data.
MIN_SPLIT_ROWS = 5

#: Above the floor but still thin enough that the leaderboard is ranking mostly
#: noise. Not refused — flagged, in the split record and again in the report.
SMALL_VAL_ROWS = 30


class RunSession:
    """One run: a raw table, a task spec, and a run directory to write into."""

    def __init__(
        self,
        run_dir: str,
        data_path: str,
        spec: TaskSpec,
        stop_policy: StopPolicy | None = None,
        val_fraction: float = 0.2,
        split_on: str | None = None,
        serving_columns: list[str] | None = None,
        fit_time_budget_s: float | None = None,
    ):
        self.ctx = RunContext(run_dir)
        # The file the user handed us. Never trained on directly — `data_path`
        # below is the work half — but kept so the split can be re-derived and
        # so task detection does not depend on which rows landed where.
        self.source_path = os.path.abspath(data_path)
        self.spec = spec
        # How long this run is allowed to keep going without getting better
        # (§19.3). On the session rather than as a module constant because it is
        # a property of *this* run's budget, not of the library.
        self.stop_policy = stop_policy or StopPolicy()
        self.val_fraction = val_fraction
        # Column to order by when splitting. Left `None` on purpose rather than
        # auto-detected: guessing which column is time and being wrong produces a
        # split that looks principled and is not, and the failure is silent.
        self.split_on = split_on
        # Which columns will exist when this model is asked for a prediction.
        # `None` means "assume everything here will be there", which is the right
        # default for a table that is its own serving format and wrong for every
        # dataset where an outcome was recorded alongside the target.
        #
        # This is the only fact in the harness that cannot be derived from the
        # training data by any means. Rossmann's `Customers` correlates 0.895 with
        # the target, sits in the training file, is absent from the test file, and
        # nothing in a run can tell — the validation half is drawn from the same
        # file and has the column too, so it *confirms* the leak instead of
        # catching it.
        self.serving_columns = list(serving_columns) if serving_columns else None
        # Wall-clock ceiling on one cross-validated fit. `None` means no ceiling,
        # which is right for a library that cannot know how big the data is or what
        # machine it is on — and why this sits beside `stop_policy` rather than
        # being a module constant: it is a property of *this* run's budget.
        #
        # It is a safety net, not a tuning policy. Rossmann v6 lost 113 of its 132
        # minutes to a single fit whose parameters were identical to one that took
        # 3.8 — so the blowup was environmental, and no instruction to the agent
        # and no cap on parameters would have caught it. Only wall clock does.
        self.fit_time_budget_s = fit_time_budget_s
        self._resolved: ResolvedSpec | None = None
        self._split: tuple[str, str] | None = None

    @property
    def run_dir(self) -> str:
        return self.ctx.run_dir

    # ------------------------------------------------------------------ split

    def _require_target(self, df) -> None:
        if self.spec.target not in df.columns:
            raise ValueError(
                f"target column {self.spec.target!r} is not in {self.source_path}; "
                f"columns are {list(df.columns)}"
            )

    def _ensure_split(self) -> tuple[str, str]:
        """Materialise ``data/work.parquet`` and ``data/val.parquet`` once.

        Idempotent, and re-reads what is on disk when both files already exist —
        a resumed run must get the *same* split, or its leaderboard compares
        scores measured against different rows.
        """
        if self._split is not None:
            return self._split

        work_path = os.path.join(self.run_dir, "data", "work.parquet")
        val_path = os.path.join(self.run_dir, "data", "val.parquet")
        if os.path.exists(work_path) and os.path.exists(val_path):
            self._split = (work_path, val_path)
            return self._split

        df = load_table(self.source_path)
        # Before the size check, so a run whose target column is misspelled hears
        # about the target rather than about row counts — the split is downstream
        # of the spec being right, and the confusing error is the one that
        # answers a question nobody asked.
        self._require_target(df)
        n = len(df)
        n_val = int(round(n * self.val_fraction))
        if n_val < MIN_SPLIT_ROWS or n - n_val < MIN_SPLIT_ROWS:
            raise ValueError(
                f"cannot split {n} rows at val_fraction={self.val_fraction}: both halves "
                f"must have at least {MIN_SPLIT_ROWS} rows (got work={n - n_val}, val={n_val})"
            )

        if self.split_on is not None:
            if self.split_on not in df.columns:
                raise ValueError(
                    f"split_on column {self.split_on!r} is not in {self.source_path}; "
                    f"columns are {list(df.columns)}"
                )
            # Last rows by `split_on` — so validation is the *future* of work when
            # the column is a date. A random split would let the model see past its
            # own prediction horizon, and would be optimistic in exactly the way
            # cross-validation is already optimistic.
            #
            # The cut lands on a *value* boundary, not on the exact row count. At
            # 1115 stores per day, cutting at 20% of rows put 603 of one day's
            # stores in the work half and 512 of the same day in validation — and
            # rows sharing a day share the promo calendar, the holiday, the
            # weather. The model learns that day's level from one group and applies
            # it to the other, which is the leak the ordered split exists to
            # prevent. The realised fraction moves to the nearest honest boundary.
            values = df[self.split_on].to_numpy()
            order = np.argsort(values, kind="stable")
            cut = boundary_cut(values[order], n - n_val)
            val_pos = order[cut:]
        else:
            val_pos = np.random.default_rng(self.spec.seed).permutation(n)[:n_val]

        mask = np.zeros(n, dtype=bool)
        mask[val_pos] = True
        save_parquet(df.iloc[np.flatnonzero(~mask)].reset_index(drop=True), work_path)
        save_parquet(df.iloc[np.flatnonzero(mask)].reset_index(drop=True), val_path)
        self._split = (work_path, val_path)
        return self._split

    @property
    def data_path(self) -> str:
        """The work half — what ``table=None`` means to every tool."""
        return self._ensure_split()[0]

    @property
    def val_path(self) -> str:
        """The validation half. Scored by the harness; never handed to a tool."""
        return self._ensure_split()[1]

    def split_record(self) -> dict:
        """What the split was, for the journal — so two runs can be compared."""
        work, val = self._ensure_split()
        n_val = len(load_table(val))
        return {
            "source_path": self.source_path,
            "mode": "ordered" if self.split_on else "random",
            "split_on": self.split_on,
            "seed": self.spec.seed,
            "val_fraction": self.val_fraction,
            "n_work": len(load_table(work)),
            "n_val": n_val,
            "val_is_thin": n_val < SMALL_VAL_ROWS,
        }

    # ------------------------------------------------------------------ specs

    def resolved(self) -> ResolvedSpec:
        """Task type + metric, auto-detected from the target column on first use.

        Cached: detection reads the target column, and the answer must not drift
        between tool calls or the leaderboard would compare incomparable scores.
        Detected on the *whole* input rather than the work half so the answer does
        not depend on the split.
        """
        if self._resolved is None:
            df = load_table(self.source_path)
            self._require_target(df)
            self._resolved = resolve_taskspec(self.spec, df[self.spec.target])
        return self._resolved

    def resolve_table(self, table: str | None) -> str:
        """Turn a tool's ``table`` argument into an absolute path.

        Tools speak in paths (§4). ``None`` means the work table; otherwise the
        only thing that resolves is a derived table under ``tables/``.

        This is an **allowlist, not a convention**. It used to accept any path
        that existed, including absolute ones, which made `available_tables()` a
        display detail rather than a boundary — and with a validation half now on
        disk, "the agent has no reason to read it" is not the same guarantee as
        "the agent cannot". Anything else raises with the menu, because the caller
        is usually a model that mistyped or invented a table.

        The work table resolves under its own path as well as under ``None``:
        tool results name it (``source_table`` on every leaderboard row), and a
        path the agent is shown has to be a path the agent can pass back. Its
        sibling ``data/val.parquet`` is the whole point of the allowlist and is
        reachable by neither.
        """
        if table is None:
            return self.data_path

        candidate = os.path.normpath(os.path.join(self.run_dir, table))
        tables_dir = os.path.abspath(os.path.join(self.run_dir, "tables"))
        allowed = (
            not os.path.isabs(table)
            and os.path.exists(candidate)
            and (
                os.path.abspath(candidate).startswith(tables_dir + os.sep)
                or os.path.abspath(candidate) == os.path.abspath(self.data_path)
            )
        )
        if not allowed:
            raise ValueError(
                f"no such table: {table!r}. Available: None (the work table)"
                + "".join(f", {t!r}" for t in self.available_tables())
            )
        return candidate

    def available_tables(self) -> list[str]:
        """Derived feature tables that exist in the run dir, as the agent sees them."""
        tables_dir = os.path.join(self.run_dir, "tables")
        if not os.path.isdir(tables_dir):
            return []
        return sorted(
            f"tables/{name}" for name in os.listdir(tables_dir) if name.endswith(".parquet")
        )

    def rel(self, path: str) -> str:
        """Path as the agent should see it: relative to run_dir when inside it."""
        abspath = os.path.abspath(path)
        run_dir = os.path.abspath(self.run_dir)
        if abspath.startswith(run_dir + os.sep):
            return os.path.relpath(abspath, run_dir)
        return abspath
