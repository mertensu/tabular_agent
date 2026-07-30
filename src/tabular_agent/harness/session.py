"""RunSession: the harness-side handle on one run.

Everything durable lives in ``run_dir/`` (spec §4) — this object holds only the
pointers needed to find it again, so a fresh process (or a fresh agent after a
context reset) can be handed the same ``run_dir`` and continue.
"""

import os

from ..io import load_table
from ..runctx import RunContext
from ..taskspec import ResolvedSpec, TaskSpec, resolve_taskspec
from .stop import StopPolicy


class RunSession:
    """One run: a raw table, a task spec, and a run directory to write into."""

    def __init__(
        self,
        run_dir: str,
        data_path: str,
        spec: TaskSpec,
        stop_policy: StopPolicy | None = None,
    ):
        self.ctx = RunContext(run_dir)
        self.data_path = os.path.abspath(data_path)
        self.spec = spec
        # How long this run is allowed to keep going without getting better
        # (§19.3). On the session rather than as a module constant because it is
        # a property of *this* run's budget, not of the library.
        self.stop_policy = stop_policy or StopPolicy()
        self._resolved: ResolvedSpec | None = None

    @property
    def run_dir(self) -> str:
        return self.ctx.run_dir

    def resolved(self) -> ResolvedSpec:
        """Task type + metric, auto-detected from the target column on first use.

        Cached: detection reads the target column, and the answer must not drift
        between tool calls or the leaderboard would compare incomparable scores.
        """
        if self._resolved is None:
            df = load_table(self.data_path)
            if self.spec.target not in df.columns:
                raise ValueError(
                    f"target column {self.spec.target!r} is not in {self.data_path}; "
                    f"columns are {list(df.columns)}"
                )
            self._resolved = resolve_taskspec(self.spec, df[self.spec.target])
        return self._resolved

    def resolve_table(self, table: str | None) -> str:
        """Turn a tool's ``table`` argument into an absolute path.

        Tools speak in paths (§4). ``None`` means the raw input table; a relative
        path is interpreted inside ``run_dir`` so the agent can pass back paths
        the tools handed it.

        A path that does not exist raises ``ValueError`` listing the ones that do,
        rather than failing later with a bare ``FileNotFoundError``: the caller is
        usually a model that mistyped or invented a table, and the fix is to show
        it the menu.
        """
        if table is None:
            return self.data_path
        path = table if os.path.isabs(table) else os.path.join(self.run_dir, table)
        if not os.path.exists(path):
            raise ValueError(
                f"no such table: {table!r}. Available: None (the raw input table)"
                + "".join(f", {t!r}" for t in self.available_tables())
            )
        return path

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
