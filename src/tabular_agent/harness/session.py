"""RunSession: the harness-side handle on one run.

Everything durable lives in ``run_dir/`` (spec §4) — this object holds only the
pointers needed to find it again, so a fresh process (or a fresh agent after a
context reset) can be handed the same ``run_dir`` and continue.
"""

import os

from ..io import load_table
from ..runctx import RunContext
from ..taskspec import ResolvedSpec, TaskSpec, resolve_taskspec


class RunSession:
    """One run: a raw table, a task spec, and a run directory to write into."""

    def __init__(self, run_dir: str, data_path: str, spec: TaskSpec):
        self.ctx = RunContext(run_dir)
        self.data_path = os.path.abspath(data_path)
        self.spec = spec
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
        """
        if table is None:
            return self.data_path
        if os.path.isabs(table):
            return table
        return os.path.join(self.run_dir, table)

    def rel(self, path: str) -> str:
        """Path as the agent should see it: relative to run_dir when inside it."""
        abspath = os.path.abspath(path)
        run_dir = os.path.abspath(self.run_dir)
        if abspath.startswith(run_dir + os.sep):
            return os.path.relpath(abspath, run_dir)
        return abspath
