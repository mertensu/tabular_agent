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

Two entry points share the same child and guard, differing only in what leaves
the sandbox:

- :func:`run_readonly` — the code returns a small **summary** that is JSON-capped
  on the way out (the read-only ``explore`` tool, §17.1). Nothing full-length can
  escape, so a hand-built target-leaky column cannot be smuggled back.
- :func:`run_transform` — the code returns a **DataFrame** that leaves whole (via
  pickle) to become a persisted, replayed-at-predict pipeline step
  (``transform_features``, §5.1). The contract requires ``df -> df`` that
  preserves the row index, so it stays a row-local, stateless transform.
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

def _run_transform(fn, df):
    result = fn(df)
    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            "transform_features must return a pandas DataFrame (df -> df); "
            "got %s. Row-local math only: add/replace columns, keep the rows."
            % type(result).__name__
        )
    if not result.index.equals(df.index):
        raise ValueError(
            "transform_features must preserve the row index — do not reorder, "
            "drop, or add rows (it is a stateless, row-local transform re-run at "
            "predict time). Add columns instead of reindexing."
        )
    with open("out.pkl", "wb") as f:
        pickle.dump(result, f)
    new_cols = [str(c) for c in result.columns if c not in set(df.columns)]
    return {"ok": True, "n_rows": int(len(result)),
            "columns": [str(c) for c in result.columns], "new_columns": new_cols}


try:
    ns = {}
    exec(cfg["code"], ns)
    fn = ns.get(cfg["func_name"])
    if not callable(fn):
        raise ValueError(
            "define a function `%s(df)`" % cfg["func_name"]
        )
    if cfg["mode"] == "transform":
        out = _run_transform(fn, df)
    else:
        out = {"ok": True, "result": _to_jsonable(fn(df), cfg["max_rows"])}
except BaseException as e:  # includes the CPU-limit SIGXCPU-turned-exception
    out = {"ok": False, "etype": type(e).__name__, "error": str(e)}

builtins.__import__ = _real_import
with open("result.json", "w") as f:
    json.dump(out, f, default=str)
'''


def _run(df, code, *, mode, func_name, timeout, mem_mb, max_rows, load_out):
    """Shared subprocess driver. ``load_out(work, out)`` reads whatever the
    child wrote for this mode and produces the caller's return value."""
    with tempfile.TemporaryDirectory(prefix="tabular_sandbox_") as work:
        with open(os.path.join(work, "df.pkl"), "wb") as f:
            pickle.dump(df, f)
        with open(os.path.join(work, "config.json"), "w") as f:
            json.dump({
                "mode": mode,
                "code": code,
                "func_name": func_name,
                "max_rows": int(max_rows),
                "cpu_s": max(1, int(timeout) + 1),
                "mem_bytes": int(mem_mb) * 1024 * 1024,
                "extra_allowed": _EXTRA_ALLOWED,
                "training_libs": _TRAINING_LIBS,
            }, f)
        with open(os.path.join(work, "_runner.py"), "w") as f:
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
                f"be too heavy for a single call."
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
        return load_out(work, out)


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
    return _run(df, code, mode="readonly", func_name=func_name, timeout=timeout,
                mem_mb=mem_mb, max_rows=max_rows,
                load_out=lambda work, out: out["result"])


def run_transform(df, code, *, func_name="transform_features", timeout=30.0,
                  mem_mb=2048):
    """Run a stateless ``df -> df`` transform in the sandbox (spec §5.1).

    ``code`` must define ``func_name(df)`` (default ``transform_features``)
    returning a pandas DataFrame with the **same row index** as the input —
    row-local math that adds or replaces columns. The full transformed DataFrame
    leaves the sandbox (via pickle), unlike :func:`run_readonly`; this is the
    output that gets persisted as a pipeline step and re-run at predict.

    Returns ``(transformed_df, summary)`` where ``summary`` has ``n_rows``,
    ``columns`` and ``new_columns``. Raises :class:`SandboxError` on blocked
    imports, timeouts, contract violations, or user errors.
    """
    def load_out(work, out):
        with open(os.path.join(work, "out.pkl"), "rb") as f:
            transformed = pickle.load(f)
        return transformed, {k: out[k] for k in ("n_rows", "columns", "new_columns")}

    return _run(df, code, mode="transform", func_name=func_name, timeout=timeout,
                mem_mb=mem_mb, max_rows=0, load_out=load_out)
