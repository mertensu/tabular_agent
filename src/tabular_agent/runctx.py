"""RunContext: the run directory is the state. Every artifact lives here."""

import os


class RunContext:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        full = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(os.path.abspath(full)), exist_ok=True)
        return full
