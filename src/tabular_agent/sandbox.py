"""Subprocess sandbox for agent-supplied Python (default tier from spec §6).

A fresh Python subprocess runs the user's code with:

- an **import whitelist** — ``pandas``/``numpy``/``scipy`` plus the standard
  library are allowed; the model libraries (``lightgbm``/``xgboost``/
  ``catboost``/``sklearn``) are blocked with a *teaching* error, so the code is
  structurally unable to train a model or learn state;
- **resource limits** (CPU seconds, and address space off macOS) and a hard
  wall-clock ``timeout``;
- its **own temp working directory** as ``cwd``;
- a validated result contract.

This is an *isolation* boundary (survives runaway/buggy code), not a hard
*security* boundary — the child runs with the same OS rights. That matches the
threat model: your machine, your data.

Only the read-only path (:func:`run_readonly`) is wired up in this phase; the
``df -> df`` ``transform_features`` path reuses the same child and is a small
follow-up.
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile

# Extra third-party roots allowed on top of the standard library. pandas pulls
# some of these in lazily at call time (dateutil/pytz for datetimes, the
# optional accelerators), so they must be importable or ordinary ops break.
_EXTRA_ALLOWED = [
    "pandas", "numpy", "scipy",
    "dateutil", "pytz", "six", "numexpr", "bottleneck", "pyarrow",
]

# Model libraries: blocked with a message that tells the agent what to use
# instead, so it does not loop trying variations.
_TRAINING_LIBS = {
    "lightgbm": "train_tabular_model",
    "xgboost": "train_tabular_model",
    "catboost": "train_tabular_model",
    "sklearn": "train_tabular_model (models) or fit_encoding (encoders)",
}


class SandboxError(RuntimeError):
    """Raised when sandboxed code fails, is blocked, or times out.

    The message is meant to be shown verbatim to the agent — for blocked
    imports it carries the "use X instead" hint.
    """


# Runner executed as a standalone file inside the temp cwd. Kept hermetic (no
# import of tabular_agent) and config-driven via config.json so nothing needs
# to be interpolated into this source.
_RUNNER = r'''
import builtins, json, pickle, sys

try:
    import resource
except ImportError:  # non-unix
    resource = None

cfg = json.load(open("config.json"))

# Resource caps, best-effort. RLIMIT_AS is unreliable on macOS (numpy/pandas
# mmap large regions and trip it spuriously), so skip address-space caps there
# and lean on the wall-clock timeout from the parent instead.
if resource is not None:
    try:
        if cfg["cpu_s"]:
            resource.setrlimit(resource.RLIMIT_CPU, (cfg["cpu_s"], cfg["cpu_s"]))
    except Exception:
        pass
    try:
        if cfg["mem_bytes"] and sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (cfg["mem_bytes"], cfg["mem_bytes"]))
    except Exception:
        pass

# Our own imports happen BEFORE the guard is installed, so pandas/numpy finish
# loading their C extensions and land in sys.modules unguarded.
import pandas as pd
import numpy as np

with open("df.pkl", "rb") as f:
    df = pickle.load(f)

ALLOWED = set(sys.stdlib_module_names) | set(cfg["extra_allowed"])
TRAINING = cfg["training_libs"]
_real_import = builtins.__import__

def _guarded_import(name, *a, **k):
    root = name.split(".")[0]
    if root in TRAINING:
        raise ImportError(
            "'%s' is not available in the sandbox: exploration code cannot "
            "train models or learn state. -> use %s instead." % (root, TRAINING[root])
        )
    if root not in ALLOWED:
        raise ImportError(
            "'%s' is not in the sandbox import whitelist. Allowed: pandas, "
            "numpy, scipy, and the Python standard library." % root
        )
    return _real_import(name, *a, **k)

builtins.__import__ = _guarded_import


def _cap_records(frame, max_rows):
    n = int(len(frame))
    body = frame.head(max_rows)
    return n, (n > max_rows), body


def _to_jsonable(x, max_rows):
    if isinstance(x, pd.DataFrame):
        n, trunc, body = _cap_records(x, max_rows)
        return {"__kind__": "dataframe", "n_rows": n, "truncated": trunc,
                "columns": [str(c) for c in x.columns],
                "rows": [_to_jsonable(r, max_rows) for r in body.to_dict("records")]}
    if isinstance(x, pd.Series):
        n, trunc, body = _cap_records(x, max_rows)
        return {"__kind__": "series", "n_rows": n, "truncated": trunc,
                "name": None if x.name is None else str(x.name),
                "data": {str(k): _to_jsonable(v, max_rows) for k, v in body.items()}}
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v, max_rows) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v, max_rows) for v in x[:max_rows]]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v, max_rows) for v in x.tolist()[:max_rows]]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)

try:
    ns = {}
    exec(cfg["code"], ns)
    fn = ns.get(cfg["func_name"])
    if not callable(fn):
        raise ValueError(
            "define a function `%s(df)` that returns a small summary" % cfg["func_name"]
        )
    result = fn(df)
    out = {"ok": True, "result": _to_jsonable(result, cfg["max_rows"])}
except BaseException as e:  # includes the CPU-limit SIGXCPU-turned-exception
    out = {"ok": False, "etype": type(e).__name__, "error": str(e)}

builtins.__import__ = _real_import
with open("result.json", "w") as f:
    json.dump(out, f, default=str)
'''


def run_readonly(df, code, *, func_name="explore", timeout=10.0,
                 mem_mb=2048, max_rows=200):
    """Run read-only exploration ``code`` in the sandbox; return a summary.

    ``code`` must define ``func_name(df)`` (default ``explore``) returning a
    JSON-serializable summary — a scalar, dict, or a (capped) Series/DataFrame.
    The result is returned to the caller but never persisted and never wired
    into the model.

    Raises :class:`SandboxError` on blocked imports, timeouts, or exceptions in
    the user code; the message carries any teaching hint.
    """
    with tempfile.TemporaryDirectory(prefix="tabular_sandbox_") as work:
        with open(os.path.join(work, "df.pkl"), "wb") as f:
            pickle.dump(df, f)
        with open(os.path.join(work, "config.json"), "w") as f:
            json.dump({
                "code": code,
                "func_name": func_name,
                "max_rows": int(max_rows),
                "cpu_s": max(1, int(timeout) + 1),
                "mem_bytes": int(mem_mb) * 1024 * 1024,
                "extra_allowed": _EXTRA_ALLOWED,
                "training_libs": _TRAINING_LIBS,
            }, f)
        runner = os.path.join(work, "_runner.py")
        with open(runner, "w") as f:
            f.write(_RUNNER)

        try:
            proc = subprocess.run(
                [sys.executable, "_runner.py"],
                cwd=work, timeout=timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"sandboxed code timed out after {timeout:g}s — it may loop or "
                f"be too heavy for an exploration call."
            ) from None

        result_path = os.path.join(work, "result.json")
        if not os.path.exists(result_path):
            # Child died before writing a result (e.g. hard resource kill).
            detail = (proc.stderr or "").strip().splitlines()
            tail = detail[-1] if detail else f"exit code {proc.returncode}"
            raise SandboxError(f"sandboxed code crashed: {tail}")

        with open(result_path) as f:
            out = json.load(f)

    if not out.get("ok"):
        raise SandboxError(out.get("error", "sandboxed code failed"))
    return out["result"]
