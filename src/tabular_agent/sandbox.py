"""Subprocess sandbox for agent-supplied Python (default tier from spec §6).

A fresh Python subprocess runs the user's code with:

- an **import whitelist** — ``pandas``/``numpy``/``scipy`` plus the standard
  library are allowed; the model libraries (``lightgbm``/``xgboost``/
  ``catboost``/``sklearn``) are blocked with a *teaching* error, so the code is
  structurally unable to train a model or learn state;
- **resource limits** (CPU seconds, and address space off macOS) and a hard
  wall-clock ``timeout``;
- its **own temp working directory** as ``cwd``, ``HOME`` and ``TMPDIR``;
- an **empty environment** apart from a few non-secret variables Python needs
  (:func:`_child_env`), so API keys and tokens in the caller's environment are
  not readable from agent code;
- a validated result contract.

This is an *isolation* boundary (survives runaway/buggy code), not a hard
*security* boundary — the child runs with the same OS rights, so the standard
library, including file and network access, is still reachable. A prompt-injected
model could read files the user can read or open a connection; it cannot read
secrets from the environment. An OS-level sandbox that would close the rest is
written up, and deliberately deferred, in ``docs/sandbox-threat-model.md``.

Three entry points share the same child and guard, differing only in what leaves
the sandbox:

- :func:`run_readonly` — the code returns a small **summary** that is JSON-capped
  on the way out (the read-only ``explore`` tool, §17.1). Nothing full-length can
  escape, so a hand-built target-leaky column cannot be smuggled back.
- :func:`run_transform` — the code returns a **DataFrame** that leaves whole (via
  pickle) to become a persisted, replayed-at-predict pipeline step
  (``transform_features``, §5.1). The contract requires ``df -> df`` that
  preserves the row index, so it stays a row-local, stateless transform.
- :func:`run_filter` — the code returns a **boolean mask** saying which rows to
  keep (``filter_rows``). Unlike the other two this feeds a step that is applied
  once, at training time, and never replayed at predict.

Both take a **namespace**: a dict of named frames pickled into the child, of
which the agent's function receives only the ones its signature asks for. That
is what lets one sandbox serve a tool handing over a single ``df`` and a tool
handing over several run-record frames, without the agent having to unpack a
container. It is *per call* — nothing survives between calls, because a live
kernel would be run state that is not in ``run_dir`` and so could not be resumed
(``docs/decisions/0020``).
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile

#: Characters of captured stdout returned to the caller. Prints are a debugging
#: channel, not a second result path, so this is well under the row cap: enough
#: for a `describe()` or the trail leading to an exception, not enough to walk a
#: full-length column out of the sandbox a chunk at a time.
MAX_STDOUT_CHARS = 2000

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
import builtins, contextlib, inspect, io, json, pickle, sys

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

with open("frames.pkl", "rb") as f:
    frames = pickle.load(f)

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

def _bind(fn, frames):
    """Call `fn` with the frames its signature names, and only those.

    The alternative — passing every frame positionally — would force the agent
    to declare parameters it does not use and to get their *order* right, which
    is a rule it cannot check and we would have to state in a docstring. A name
    it asks for and we do not have is a typo or an invented frame, so the error
    lists what exists rather than failing on a KeyError deeper in.
    """
    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    unknown = [p.name for p in params if p.name not in frames]
    if unknown:
        raise ValueError(
            "%s() asked for %s, which is not available here. The frames in this "
            "call are: %s." % (fn.__name__, ", ".join(repr(u) for u in unknown),
                               ", ".join(sorted(frames)))
        )
    args = [frames[p.name] for p in params if p.kind is not p.KEYWORD_ONLY]
    kwargs = {p.name: frames[p.name] for p in params if p.kind is p.KEYWORD_ONLY}
    return fn(*args, **kwargs)


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
            "predict time). Add columns instead of reindexing. To train on fewer "
            "rows, use the filter_rows tool: dropping rows is a separate step "
            "because it must not be replayed at predict time."
        )
    with open("out.pkl", "wb") as f:
        pickle.dump(result, f)
    new_cols = [str(c) for c in result.columns if c not in set(df.columns)]
    return {"ok": True, "n_rows": int(len(result)),
            "columns": [str(c) for c in result.columns], "new_columns": new_cols}


def _run_filter(fn, df):
    """Contract for a row filter: df -> boolean mask over the same index.

    A mask rather than a filtered frame, deliberately. Returning rows would let
    the same call reorder them, rename columns, or hand back rows that were never
    in the input, and every one of those would have to be checked for. A boolean
    Series indexed like the input can only ever *select*.
    """
    result = fn(df)
    if isinstance(result, np.ndarray) and result.dtype == bool and len(result) == len(df):
        result = pd.Series(result, index=df.index)
    if not isinstance(result, pd.Series):
        raise ValueError(
            "keep_rows must return a boolean pandas Series, one value per row, "
            "indexed like the frame it was given; got %s. Write the condition "
            "directly, e.g. `return df[\"amount\"] > 0`." % type(result).__name__
        )
    if result.dtype != bool:
        raise ValueError(
            "keep_rows must return a *boolean* Series (True = keep the row); got "
            "dtype %s. If your condition can be null, decide what a null means: "
            "`(df[\"x\"] > 0).fillna(False)`." % result.dtype
        )
    if not result.index.equals(df.index):
        raise ValueError(
            "keep_rows must return a mask indexed like the frame it was given — "
            "do not reset, sort or subset the index. Return the condition itself "
            "rather than a mask built from already-filtered rows."
        )
    with open("out.pkl", "wb") as f:
        pickle.dump(result, f)
    return {"ok": True, "n_rows": int(len(result)), "n_kept": int(result.sum())}


# Everything the code prints is captured rather than going to the pipe and
# being dropped. A model writes `print(...)` in exploration code by reflex, and
# silently discarding it is the same trap as a profile that omits the categories
# an ordinal encoding needs: the agent behaves reasonably and nothing tells it
# why it saw nothing back.
printed = io.StringIO()
try:
    with contextlib.redirect_stdout(printed):
        ns = {}
        exec(cfg["code"], ns)
        fn = ns.get(cfg["func_name"])
        if not callable(fn):
            raise ValueError(
                "define a function `%s(df)`" % cfg["func_name"]
            )
        if cfg["mode"] == "transform":
            out = _run_transform(fn, frames["df"])
        elif cfg["mode"] == "filter":
            out = _run_filter(fn, frames["df"])
        else:
            out = {"ok": True,
                   "result": _to_jsonable(_bind(fn, frames), cfg["max_rows"])}
except BaseException as e:  # includes the CPU-limit SIGXCPU-turned-exception
    out = {"ok": False, "etype": type(e).__name__, "error": str(e)}

# Attached in both outcomes on purpose. On the failure path it is the only
# record of what the code computed before it raised, which is most of what a
# model needs to fix code written against a frame it has never seen.
text = printed.getvalue()
cap = cfg["max_stdout_chars"]
if len(text) > cap:
    text = text[:cap] + "\n... [stdout truncated]"
out["stdout"] = text

builtins.__import__ = _real_import
with open("result.json", "w") as f:
    json.dump(out, f, default=str)
'''


#: Environment variables the child may inherit. None of them carries a secret;
#: they are what the interpreter and its libraries need to start and to decode
#: text (``SYSTEMROOT`` is required by Python itself on Windows).
_INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH", "SYSTEMROOT")


def _child_env(work):
    """The child's whole environment: an allowlist, never the caller's copy.

    Inheriting ``os.environ`` would hand agent-written code every API key the
    caller exported — the provider key the run itself uses included — and a
    denylist of secret-looking names would miss whatever a user named theirs.
    """
    env = {k: os.environ[k] for k in _INHERITED_ENV if k in os.environ}
    env.update(HOME=work, TMPDIR=work)
    return env


def _run(frames, code, *, mode, func_name, timeout, mem_mb, max_rows, load_out):
    """Shared subprocess driver. ``load_out(work, out)`` reads whatever the
    child wrote for this mode and produces the caller's return value."""
    with tempfile.TemporaryDirectory(prefix="tabular_sandbox_") as work:
        with open(os.path.join(work, "frames.pkl"), "wb") as f:
            pickle.dump(frames, f)
        with open(os.path.join(work, "config.json"), "w") as f:
            json.dump({
                "mode": mode,
                "code": code,
                "func_name": func_name,
                "max_rows": int(max_rows),
                "max_stdout_chars": int(MAX_STDOUT_CHARS),
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
                cwd=work, env=_child_env(work), timeout=timeout,
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
            message = out.get("error", "sandboxed code failed")
            printed = (out.get("stdout") or "").strip()
            if printed:
                # The prints are part of the error report, not a separate
                # channel: code that fell over on line 8 is fixed with what
                # lines 1-7 printed, and the exception alone does not have it.
                message = f"{message}\n--- printed before the error ---\n{printed}"
            raise SandboxError(message)
        return load_out(work, out)


def run_readonly(frames, code, *, func_name="explore", timeout=10.0,
                 mem_mb=2048, max_rows=200):
    """Run read-only exploration ``code`` over ``frames``; return result + stdout.

    ``frames`` is a ``{name: object}`` namespace. ``code`` must define
    ``func_name`` (default ``explore``) taking any subset of those names as
    parameters and returning a JSON-serializable summary — a scalar, dict, or a
    (capped) Series/DataFrame.

    Returns ``{"result": ..., "stdout": str}``. Both are capped; neither is
    persisted or wired into the model.

    Raises :class:`SandboxError` on blocked imports, timeouts, an unknown frame
    name, or exceptions in the user code. The message carries any teaching hint
    and whatever the code printed before it failed.
    """
    return _run(frames, code, mode="readonly", func_name=func_name, timeout=timeout,
                mem_mb=mem_mb, max_rows=max_rows,
                load_out=lambda work, out: {"result": out["result"],
                                            "stdout": out.get("stdout", "")})


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

    return _run({"df": df}, code, mode="transform", func_name=func_name,
                timeout=timeout, mem_mb=mem_mb, max_rows=0, load_out=load_out)


def run_filter(df, code, *, func_name="keep_rows", timeout=30.0, mem_mb=2048):
    """Run a row-selection predicate in the sandbox; return ``(mask, summary)``.

    ``code`` must define ``func_name(df)`` (default ``keep_rows``) returning a
    boolean pandas Series indexed like the input — ``True`` keeps the row.

    The third sandbox mode, and the one that is *not* replayed at predict time
    (see :func:`~tabular_agent.rows.select_rows`). That difference is why it gets
    its own contract rather than reusing :func:`run_transform`: a transform must
    preserve every row because its output has to line up with predictions, and a
    filter exists precisely to break that.

    ``summary`` has ``n_rows`` and ``n_kept``. Raises :class:`SandboxError` on
    blocked imports, timeouts, contract violations, or user errors.
    """
    def load_out(work, out):
        with open(os.path.join(work, "out.pkl"), "rb") as f:
            mask = pickle.load(f)
        return mask, {k: out[k] for k in ("n_rows", "n_kept")}

    return _run({"df": df}, code, mode="filter", func_name=func_name,
                timeout=timeout, mem_mb=mem_mb, max_rows=0, load_out=load_out)
