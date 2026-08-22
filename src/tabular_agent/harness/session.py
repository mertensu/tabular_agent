"""RunSession: the harness-side handle on one run.

Everything durable lives in ``run_dir/`` (spec §4) — this object holds only the
pointers needed to find it again, so a fresh process (or a fresh agent after a
context reset) can be handed the same ``run_dir`` and continue.

The input table is split once, before any tool runs, into a **work** half the
agent builds features on and a **validation** half it is scored against —
optionally with a third, **sealed** slice no tool can reach, scored once at
`finalize_run` (0023). All of them live in ``run_dir/data/``, so a run directory
stays movable (0003).
"""

import json
import os
import re

import numpy as np

from ..cv import boundary_cut
from ..io import load_table, save_parquet
from ..runctx import RunContext
from ..taskspec import ResolvedSpec, TaskSpec, resolve_taskspec
from .stop import StopPolicy

#: A stream's name ends up inside feature column names and inside a filename, so
#: it is held to what both can carry. Refused at declaration rather than mangled
#: silently: a name that survives one and not the other produces a column nobody
#: asked for or a file nobody can find.
_STREAM_NAME = re.compile(r"[A-Za-z0-9_]+")

#: What a stream declaration must say, and it is exactly the part that cannot be
#: recovered by looking at the file. Which column is *when a record is about* and
#: which is *when it became knowable* is invisible to any profiler — both are
#: timestamps, and the wrong way round is a feature that reads the future.
STREAM_REQUIRED = ("path", "event_time", "observed_at")

#: Optional, and both are documentation rather than mechanism: ``notes`` is the
#: owner's prose about the file, and every other key is refused so a typo in a
#: declaration is an error instead of a setting that silently does nothing.
STREAM_OPTIONAL = ("notes",)

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
        holdout_fraction: float | None = None,
        split_on: str | None = None,
        serving_columns: list[str] | None = None,
        fit_time_budget_s: float | None = None,
        streams: dict[str, dict] | None = None,
        profile: str = "full",
    ):
        if profile not in {"full", "minimal"}:
            raise ValueError(f"unknown run profile {profile!r}; use 'full' or 'minimal'")
        if profile == "minimal" and holdout_fraction is not None:
            raise ValueError(
                "the minimal profile has one train/evaluation split; an internal "
                "holdout belongs to an offline audit, not the product run"
            )
        if profile == "minimal" and streams:
            raise ValueError(
                "the minimal profile accepts one table; declared streams require "
                "the full research profile"
            )
        self.ctx = RunContext(run_dir)
        journal_path = os.path.join(self.ctx.run_dir, "run_journal.json")
        if os.path.exists(journal_path):
            with open(journal_path) as handle:
                existing = json.load(handle)
            recorded = ((existing.get("split") or {}).get("profile") or "full")
            if recorded != profile:
                raise ValueError(
                    f"this run was created with profile {recorded!r}; refusing to "
                    f"resume it as {profile!r} because the tool and evaluation "
                    "contracts would change"
                )
        # The file the user handed us. Never trained on directly — `data_path`
        # below is the work half — but kept so the split can be re-derived and
        # so task detection does not depend on which rows landed where.
        self.source_path = os.path.abspath(data_path)
        self.spec = spec
        self.profile = profile
        # How long this run is allowed to keep going without getting better
        # (§19.3). On the session rather than as a module constant because it is
        # a property of *this* run's budget, not of the library.
        self.stop_policy = stop_policy or (
            StopPolicy(min_gain_ratio=0.0) if profile == "minimal" else StopPolicy()
        )
        self.val_fraction = val_fraction
        # The sealed third slice (0023). `None` — off — is the default, and the
        # default is deliberate rather than lazy: turning it on takes rows out of
        # the work half, so the first run with a holdout is not comparable to the
        # runs before it. That is a decision belonging to whoever is running the
        # experiment, and it should be visible in their driver script rather than
        # arriving as a library default that silently re-cuts everyone's data.
        #
        # A fraction of the *whole* input, like `val_fraction`, not a slice taken
        # out of the validation half. Under a random split that keeps the
        # validation rows themselves unchanged (see `_ensure_split`), so `val_se`
        # and every val-denominated comparison mean the same thing with the
        # holdout on as off — only the work half shrinks.
        self.holdout_fraction = holdout_fraction
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
        self.serving_columns = (
            None if serving_columns is None else list(serving_columns)
        )
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
        # Related files at a finer grain than the table, whose records carry both
        # the time they are *about* and the time they became *knowable* (0062).
        #
        # Declared, for `serving_columns`' reason: nothing recovers it by
        # inspection. A profiler sees two datetime columns and cannot say which is
        # which, and getting them the wrong way round builds a feature out of the
        # future. The owner of the data knows; nobody else can.
        #
        # Not split into work and validation, and that is not an oversight. Every
        # aggregate drawn from a stream is restricted per row to records observed
        # a stated span before that row, so a work row and a validation row are
        # bound by the same rule and neither can read past its own position. A
        # stream cut at the work/val boundary would instead give validation rows
        # *less* than a served row gets, which is a different lie.
        self.streams = self._validate_streams(streams)
        self._resolved: ResolvedSpec | None = None
        self._split: tuple[str, str, str | None] | None = None

    @property
    def run_dir(self) -> str:
        return self.ctx.run_dir

    # ----------------------------------------------------------------- streams

    @staticmethod
    def _validate_streams(streams: dict[str, dict] | None) -> dict[str, dict]:
        """Check every declaration at construction, before a run has started.

        Loudly and early, because the alternative is a run that gets ten tables in
        before discovering that a key was misspelled — and because a declaration
        with a silently-ignored key is worse than one that fails: it looks like it
        said something.
        """
        if not streams:
            return {}
        checked: dict[str, dict] = {}
        for name, spec in streams.items():
            if not isinstance(name, str) or not _STREAM_NAME.fullmatch(name):
                raise ValueError(
                    f"stream name {name!r} must be letters, digits and underscores. "
                    "It becomes part of the feature column names this stream "
                    "produces and part of the file it is stored under, so it has to "
                    "survive both."
                )
            if not isinstance(spec, dict):
                raise ValueError(
                    f"stream {name!r} must be declared as a dict with "
                    f"{list(STREAM_REQUIRED)}, got {type(spec).__name__}"
                )
            missing = [k for k in STREAM_REQUIRED if not spec.get(k)]
            if missing:
                raise ValueError(
                    f"stream {name!r} is missing {missing}. 'event_time' is when a "
                    "record is *about* — the day of the visit, the delivery date — "
                    "and 'observed_at' is when it became knowable. Nothing can work "
                    "out which is which by looking at the file, and the wrong way "
                    "round builds a feature out of the future."
                )
            unknown = sorted(set(spec) - set(STREAM_REQUIRED) - set(STREAM_OPTIONAL))
            if unknown:
                raise ValueError(
                    f"stream {name!r} declares {unknown}, which this harness does not "
                    f"read. Allowed: {list(STREAM_REQUIRED + STREAM_OPTIONAL)}. Which "
                    "columns to group and aggregate by is the agent's call at tool "
                    "time, not the declaration's."
                )
            if spec["event_time"] == spec["observed_at"]:
                raise ValueError(
                    f"stream {name!r} names {spec['event_time']!r} as both its event "
                    "time and its observation time. A file whose two times are one "
                    "column is not a stream in this sense — every record would count "
                    "as knowable exactly when it happened, and there is nothing for a "
                    "lead to restrict."
                )
            checked[name] = {
                "path": os.path.abspath(str(spec["path"])),
                "event_time": str(spec["event_time"]),
                "observed_at": str(spec["observed_at"]),
                **({"notes": str(spec["notes"])} if spec.get("notes") else {}),
            }
        return checked

    def stream_names(self) -> list[str]:
        return sorted(self.streams)

    def stream_spec(self, name: str) -> dict:
        if name not in self.streams:
            raise ValueError(
                f"no such stream: {name!r}. Declared: "
                + (", ".join(repr(n) for n in self.stream_names()) or "none")
                + ". A stream has to be declared on the run — the harness cannot "
                "tell a stream from an ordinary file by looking at it."
            )
        return self.streams[name]

    def stream_path(self, name: str) -> str:
        """The stream, copied into the run directory on first use.

        Copied for 0003's reason — a run directory has to stay movable, and a run
        whose second input lives at an absolute path somewhere else is not. It
        also fixes the file for the length of the run, which is what makes two
        experiments in the same run comparable.

        That is the opposite of what happens at *predict* time, where the stream
        is handed in fresh precisely so a file that has moved on since the fit is
        the one that gets used (0062). Both are right: a run is a fixed experiment
        against fixed data; serving is not.
        """
        spec = self.stream_spec(name)
        dest = os.path.join(self.run_dir, "data", "streams", f"{name}.parquet")
        if os.path.exists(dest):
            return dest
        df = load_table(spec["path"])
        missing = [
            spec[k] for k in ("event_time", "observed_at") if spec[k] not in df.columns
        ]
        if missing:
            raise ValueError(
                f"stream {name!r} declares {missing} as time columns, but "
                f"{spec['path']} has {list(df.columns)}"
            )
        save_parquet(df, dest)
        return dest

    def stream_record(self) -> dict:
        """What was declared, for the journal — the run's second inputs, named."""
        return {
            name: {
                "path": spec["path"],
                "event_time": spec["event_time"],
                "observed_at": spec["observed_at"],
                **({"notes": spec["notes"]} if spec.get("notes") else {}),
            }
            for name, spec in sorted(self.streams.items())
        }

    # ------------------------------------------------------------------ split

    def _require_target(self, df) -> None:
        if self.spec.target not in df.columns:
            raise ValueError(
                f"target column {self.spec.target!r} is not in {self.source_path}; "
                f"columns are {list(df.columns)}"
            )

    def _ensure_split(self) -> tuple[str, str, str | None]:
        """Materialise ``data/work.parquet``, ``data/val.parquet`` and — when
        ``holdout_fraction`` is set — ``data/holdout.parquet``, once.

        Idempotent, and re-reads what is on disk when the files already exist —
        a resumed run must get the *same* split, or its leaderboard compares
        scores measured against different rows.
        """
        if self._split is not None:
            return self._split

        work_path = os.path.join(self.run_dir, "data", "work.parquet")
        val_path = os.path.join(self.run_dir, "data", "val.parquet")
        holdout_path = os.path.join(self.run_dir, "data", "holdout.parquet")
        if os.path.exists(work_path) and os.path.exists(val_path):
            # What is on disk wins over what this object was constructed with.
            # Re-splitting to honour a `holdout_fraction` the original run did not
            # have would move the validation rows under a leaderboard already
            # written against them, and refusing to resume would strand a run dir
            # over a constructor argument. A run that was cut without a sealed
            # slice simply does not have one to score.
            on_disk = holdout_path if os.path.exists(holdout_path) else None
            self._split = (work_path, val_path, on_disk)
            return self._split

        df = load_table(self.source_path)
        # Before the size check, so a run whose target column is misspelled hears
        # about the target rather than about row counts — the split is downstream
        # of the spec being right, and the confusing error is the one that
        # answers a question nobody asked.
        self._require_target(df)
        n = len(df)
        n_val = int(round(n * self.val_fraction))
        n_holdout = int(round(n * (self.holdout_fraction or 0.0)))
        n_work = n - n_val - n_holdout
        parts = {"work": n_work, "val": n_val}
        if self.holdout_fraction:
            parts["holdout"] = n_holdout
        if min(parts.values()) < MIN_SPLIT_ROWS:
            fractions = f"val_fraction={self.val_fraction}"
            if self.holdout_fraction:
                fractions += f", holdout_fraction={self.holdout_fraction}"
            raise ValueError(
                f"cannot split {n} rows at {fractions}: every part must have at least "
                f"{MIN_SPLIT_ROWS} rows (got "
                + ", ".join(f"{k}={v}" for k, v in parts.items())
                + ")"
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
            # work | val | holdout, in time order: the sealed slice is the *latest*
            # period, because it is the one asked to stand in for unseen data and
            # the competition's test set is always the future. The price is that
            # turning the holdout on moves the validation rows earlier, so an
            # ordered run with a holdout is not comparable row-for-row to one
            # without — unlike the random case below, where val is unchanged.
            #
            # It also means val and holdout are *different periods*, so the gap
            # between them carries drift as well as selection optimism. That
            # confound is real, it is named in the report, and it is the price of
            # keeping the sealed slice the most test-like rows in the file.
            cut_work = boundary_cut(values[order], n_work)
            if self.holdout_fraction:
                # A second boundary, and it must not land on the first: two cuts
                # in the same place is an empty slice, which would look like a
                # split that worked and produce a sealed score over no rows.
                cut_val = boundary_cut(
                    values[order], n_work + n_val, exclude=(cut_work,)
                )
                if cut_val <= cut_work:
                    raise ValueError(
                        f"cannot cut {n} rows ordered by {self.split_on!r} into "
                        f"work/val/holdout: the nearest honest boundary to the "
                        f"validation end lands at or before the work end "
                        f"({cut_val} <= {cut_work}), so a slice would be empty. "
                        "Order on a finer column, or drop holdout_fraction."
                    )
            else:
                cut_val = n
            val_pos = order[cut_work:cut_val]
            holdout_pos = order[cut_val:]
        else:
            # One permutation, validation taken off the front of it, so which rows
            # are validation depends only on `n` and the seed — not on whether a
            # holdout was asked for. Adding one takes rows out of *work* and
            # leaves every val-denominated number comparable to a run without it.
            perm = np.random.default_rng(self.spec.seed).permutation(n)
            val_pos = perm[:n_val]
            holdout_pos = perm[n_val : n_val + n_holdout]

        held = np.zeros(n, dtype=bool)
        held[val_pos] = True
        held[holdout_pos] = True
        save_parquet(df.iloc[np.flatnonzero(~held)].reset_index(drop=True), work_path)
        save_parquet(df.iloc[np.sort(val_pos)].reset_index(drop=True), val_path)
        if self.holdout_fraction:
            save_parquet(df.iloc[np.sort(holdout_pos)].reset_index(drop=True), holdout_path)
        self._split = (work_path, val_path, holdout_path if self.holdout_fraction else None)
        return self._split

    @property
    def data_path(self) -> str:
        """The work half — what ``table=None`` means to every tool."""
        return self._ensure_split()[0]

    @property
    def val_path(self) -> str:
        """The validation half. Scored by the harness; never handed to a tool."""
        return self._ensure_split()[1]

    @property
    def holdout_path(self) -> str | None:
        """The sealed slice, or ``None`` when this run has none (0023).

        Read by exactly one caller, `finalize_run`, exactly once. Every other
        consumer of held-out rows — `probe_table`, `train_tabular_model`,
        `explain_table` — goes through `val_path`, and that asymmetry is the
        whole mechanism: a number consulted repeatedly is a number the search
        starts optimising against, which is why the validation half stopped being
        an independent check in the first place.

        It is not defended by an allowlist entry of its own. `resolve_table`
        admits nothing outside ``tables/``, so ``data/holdout.parquet`` is
        unreachable to a tool for the same reason ``data/val.parquet`` is.
        """
        return self._ensure_split()[2]

    def split_record(self) -> dict:
        """What the split was, for the journal — so two runs can be compared."""
        work, val, holdout = self._ensure_split()
        n_val = len(load_table(val))
        n_holdout = len(load_table(holdout)) if holdout else 0
        return {
            "source_path": self.source_path,
            "profile": self.profile,
            "mode": "ordered" if self.split_on else "random",
            "split_on": self.split_on,
            "seed": self.spec.seed,
            "val_fraction": self.val_fraction,
            "holdout_fraction": self.holdout_fraction,
            "n_work": len(load_table(work)),
            "n_val": n_val,
            "n_holdout": n_holdout,
            "val_is_thin": n_val < SMALL_VAL_ROWS,
            # Flagged separately from `val_is_thin` because it is a different
            # failure: a thin validation half makes the *ranking* noise, a thin
            # holdout makes the honesty number itself noise. Both can be true,
            # and a run can have a trustworthy leaderboard and an untrustworthy
            # gap, or the reverse.
            "holdout_is_thin": bool(holdout) and n_holdout < SMALL_VAL_ROWS,
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
            if (
                self.profile == "minimal"
                and self._resolved.task_type == "binary"
                and self._resolved.positive_class is not None
            ):
                labels = sorted(df[self.spec.target].dropna().unique())
                if labels and self._resolved.positive_class != labels[-1]:
                    raise ValueError(
                        f"positive_class={self._resolved.positive_class!r} is not "
                        f"supported by the minimal profile; use {labels[-1]!r}, the "
                        "model's positive probability class, or omit positive_class"
                    )
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
