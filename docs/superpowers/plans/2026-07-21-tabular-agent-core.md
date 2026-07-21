# tabular-core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe tabular-ML core library that powers the `tabular-agent` MCP server — everything from metrics to a serializable predictor, usable as a plain Python library with no MCP or subprocess dependency.

**Architecture:** A `src/`-layout Python package `tabular_agent` of small, single-responsibility modules. Pure functions and small classes; state lives in files and serializable dicts, never in long-lived objects. Correctness-by-construction for the two footguns the spec targets: (1) stateful feature encodings are fit **per CV fold** so validation is honest, and (2) the fitted parameters are serialized so `predict` replays them instead of recomputing on new data. Heavy lifting is delegated to scikit-learn / LightGBM to keep custom code minimal.

**Tech Stack:** Python ≥3.11, numpy, pandas, scipy, scikit-learn, pyarrow, lightgbm, xgboost, catboost, joblib. Tests: pytest. Env/build: uv + hatchling.

## Global Constraints

- **Python floor:** `requires-python = ">=3.11"`. Use `X | None` union syntax, `list[...]` generics.
- **Determinism:** every stochastic operation takes a seed; default `seed = 42`, default `n_splits = 5`. Identical inputs must yield identical folds, OOF, and metrics.
- **Leakage rule:** any transform that learns statistics from the target (target-encoding, scaling, imputation, category vocab) MUST be fit only on the training rows of each fold. Never fit such a transform on the full dataset and then use it inside cross-validation.
- **No recompute-at-serve rule:** `predict` must apply *stored* fitted parameters. It must never recompute a learned statistic on new data and must never require the target column to be present.
- **DRY:** reuse scikit-learn (`KFold`, `StratifiedKFold`, metric functions, `StandardScaler`, `SimpleImputer`) and LightGBM/XGBoost/CatBoost rather than hand-rolling equivalents.
- **Package import root:** `tabular_agent` under `src/`. All modules are `tabular_agent.<module>`.

---

### Task 1: Project scaffold + metrics registry

**Files:**
- Create: `pyproject.toml`
- Create: `src/tabular_agent/__init__.py`
- Create: `src/tabular_agent/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tabular_agent.metrics.compute_metric(name: str, y_true, y_pred) -> float`
  - `tabular_agent.metrics.greater_is_better(name: str) -> bool`
  - `tabular_agent.metrics.SUPPORTED_METRICS: set[str]` = `{"rmse","rmsle","mae","logloss","auc","accuracy"}`
  - Prediction-format contract used everywhere downstream: regression → 1-D array of values; binary → 1-D array of P(positive class); multiclass → 2-D array `(n, k)` of class probabilities.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "tabular-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "pyarrow>=15",
    "lightgbm>=4.3",
    "xgboost>=2.0",
    "catboost>=1.2",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tabular_agent"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package init and set up the env**

Create `src/tabular_agent/__init__.py`:

```python
"""tabular_agent: leakage-safe tabular-ML core."""

__version__ = "0.1.0"
```

Run:

```bash
uv venv
uv pip install -e ".[dev]"
```

Expected: install completes; `uv run python -c "import tabular_agent; print(tabular_agent.__version__)"` prints `0.1.0`.

- [ ] **Step 3: Write the failing test**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pytest

from tabular_agent.metrics import compute_metric, greater_is_better, SUPPORTED_METRICS


def test_rmse_known_value():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])  # errors: 0,0,2 -> mse=4/3 -> rmse=1.1547
    assert compute_metric("rmse", y, p) == pytest.approx(1.1547005, rel=1e-5)


def test_rmsle_rejects_negative():
    with pytest.raises(ValueError):
        compute_metric("rmsle", np.array([-1.0, 2.0]), np.array([1.0, 2.0]))


def test_mae_known_value():
    assert compute_metric("mae", np.array([1.0, 3.0]), np.array([2.0, 3.0])) == pytest.approx(0.5)


def test_binary_logloss_and_auc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_metric("logloss", y, p) < 0.3
    assert compute_metric("auc", y, p) == pytest.approx(1.0)


def test_accuracy_from_binary_probs():
    y = np.array([0, 1, 1])
    p = np.array([0.2, 0.9, 0.4])  # -> preds 0,1,0 -> 2/3 correct
    assert compute_metric("accuracy", y, p) == pytest.approx(2 / 3)


def test_multiclass_accuracy_from_proba_matrix():
    y = np.array([0, 1, 2])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    assert compute_metric("accuracy", y, p) == pytest.approx(1.0)


def test_greater_is_better_directions():
    assert greater_is_better("auc") is True
    assert greater_is_better("accuracy") is True
    assert greater_is_better("rmse") is False
    assert greater_is_better("logloss") is False


def test_supported_metrics_set():
    assert SUPPORTED_METRICS == {"rmse", "rmsle", "mae", "logloss", "auc", "accuracy"}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.metrics'`.

- [ ] **Step 5: Write minimal implementation**

Create `src/tabular_agent/metrics.py`:

```python
"""Metric registry. See prediction-format contract in the module test."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

SUPPORTED_METRICS = {"rmse", "rmsle", "mae", "logloss", "auc", "accuracy"}

_GREATER_IS_BETTER = {
    "rmse": False,
    "rmsle": False,
    "mae": False,
    "logloss": False,
    "auc": True,
    "accuracy": True,
}


def greater_is_better(name: str) -> bool:
    return _GREATER_IS_BETTER[name]


def compute_metric(name: str, y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "rmsle":
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("rmsle requires non-negative targets and predictions")
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "logloss":
        return float(log_loss(y_true, y_pred))
    if name == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if name == "accuracy":
        labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else (y_pred >= 0.5).astype(int)
        return float(accuracy_score(y_true, labels))
    raise ValueError(f"unknown metric: {name}")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (8 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/tabular_agent/__init__.py src/tabular_agent/metrics.py tests/test_metrics.py
git commit -m "feat: project scaffold + metrics registry"
```

---

### Task 2: TaskSpec + auto-detect with override

**Files:**
- Create: `src/tabular_agent/taskspec.py`
- Test: `tests/test_taskspec.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TaskSpec` dataclass: `target: str`, `task_type: str | None = None`, `metric: str | None = None`, `positive_class=None`, `notes: str | None = None`, `n_splits: int = 5`, `seed: int = 42`.
  - `ResolvedSpec` dataclass: same fields but `task_type: str` and `metric: str` always populated.
  - `detect_task_type(y: pd.Series) -> str` → `"regression" | "binary" | "multiclass"`.
  - `default_metric(task_type: str, y: pd.Series) -> str`.
  - `resolve_taskspec(spec: TaskSpec, y: pd.Series) -> ResolvedSpec`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_taskspec.py`:

```python
import pandas as pd

from tabular_agent.taskspec import (
    TaskSpec,
    detect_task_type,
    default_metric,
    resolve_taskspec,
)


def test_detect_binary():
    assert detect_task_type(pd.Series([0, 1, 1, 0])) == "binary"


def test_detect_multiclass_integer_low_cardinality():
    assert detect_task_type(pd.Series([0, 1, 2, 3, 2, 1])) == "multiclass"


def test_detect_regression_float():
    assert detect_task_type(pd.Series([0.5, 1.7, 3.2, 9.1])) == "regression"


def test_detect_regression_high_cardinality_integer():
    assert detect_task_type(pd.Series(range(100))) == "regression"


def test_default_metric_regression_nonneg_is_rmsle():
    assert default_metric("regression", pd.Series([0.0, 1.0, 2.0])) == "rmsle"


def test_default_metric_regression_negative_falls_back_to_rmse():
    assert default_metric("regression", pd.Series([-1.0, 2.0])) == "rmse"


def test_default_metric_classification_is_logloss():
    assert default_metric("binary", pd.Series([0, 1])) == "logloss"
    assert default_metric("multiclass", pd.Series([0, 1, 2])) == "logloss"


def test_resolve_fills_gaps_but_respects_override():
    y = pd.Series([0.0, 1.0, 2.0])
    resolved = resolve_taskspec(TaskSpec(target="t"), y)
    assert resolved.task_type == "regression"
    assert resolved.metric == "rmsle"

    overridden = resolve_taskspec(TaskSpec(target="t", task_type="regression", metric="mae"), y)
    assert overridden.metric == "mae"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taskspec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.taskspec'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/taskspec.py`:

```python
"""Task specification: the single source of truth; auto-detect fills gaps."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TaskSpec:
    target: str
    task_type: str | None = None
    metric: str | None = None
    positive_class: object | None = None
    notes: str | None = None
    n_splits: int = 5
    seed: int = 42


@dataclass
class ResolvedSpec:
    target: str
    task_type: str
    metric: str
    positive_class: object | None
    notes: str | None
    n_splits: int
    seed: int


def detect_task_type(y: pd.Series) -> str:
    n_unique = int(y.nunique(dropna=True))
    if n_unique <= 2:
        return "binary"
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_integer_dtype(y):
        return "multiclass" if n_unique <= 20 else "regression"
    return "multiclass"


def default_metric(task_type: str, y: pd.Series) -> str:
    if task_type == "regression":
        if pd.api.types.is_numeric_dtype(y) and bool((y.dropna() >= 0).all()):
            return "rmsle"
        return "rmse"
    return "logloss"


def resolve_taskspec(spec: TaskSpec, y: pd.Series) -> ResolvedSpec:
    task_type = spec.task_type or detect_task_type(y)
    metric = spec.metric or default_metric(task_type, y)
    return ResolvedSpec(
        target=spec.target,
        task_type=task_type,
        metric=metric,
        positive_class=spec.positive_class,
        notes=spec.notes,
        n_splits=spec.n_splits,
        seed=spec.seed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taskspec.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/taskspec.py tests/test_taskspec.py
git commit -m "feat: TaskSpec with task-type/metric auto-detect and override"
```

---

### Task 3: IO helpers + RunContext

**Files:**
- Create: `src/tabular_agent/io.py`
- Create: `src/tabular_agent/runctx.py`
- Test: `tests/test_io_runctx.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `io.load_table(path: str) -> pd.DataFrame` (`.csv` → `read_csv`, `.parquet` → `read_parquet`).
  - `io.save_parquet(df: pd.DataFrame, path: str) -> str` (returns the path; creates parent dirs).
  - `runctx.RunContext(run_dir: str)` with `.path(*parts: str) -> str` (joins under run_dir, creates parent dirs) and `.run_dir: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_runctx.py`:

```python
import os

import pandas as pd

from tabular_agent.io import load_table, save_parquet
from tabular_agent.runctx import RunContext


def test_save_and_load_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = save_parquet(df, str(tmp_path / "sub" / "data.parquet"))
    assert os.path.exists(path)
    loaded = load_table(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = load_table(str(p))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_runcontext_path_creates_dirs(tmp_path):
    ctx = RunContext(str(tmp_path / "run1"))
    p = ctx.path("model", "final.pkl")
    assert p.endswith(os.path.join("run1", "model", "final.pkl"))
    assert os.path.isdir(os.path.dirname(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io_runctx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.io'`.

- [ ] **Step 3: Write minimal implementations**

Create `src/tabular_agent/io.py`:

```python
"""Table IO: CSV and Parquet by file extension."""

import os

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported extension: {path}")


def save_parquet(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    return path
```

Create `src/tabular_agent/runctx.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io_runctx.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/io.py src/tabular_agent/runctx.py tests/test_io_runctx.py
git commit -m "feat: table IO helpers and RunContext"
```

---

### Task 4: Data profiling (`inspect`)

**Files:**
- Create: `src/tabular_agent/inspect.py`
- Test: `tests/test_inspect.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `inspect.profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict` with keys `n_rows`, `n_cols`, `columns` (per-column `dtype`/`n_missing`/`n_unique`), and optional `target` block.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inspect.py`:

```python
import numpy as np
import pandas as pd

from tabular_agent.inspect import profile_dataframe


def test_profile_basic_shape_and_columns():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "x", "y"]})
    prof = profile_dataframe(df)
    assert prof["n_rows"] == 3
    assert prof["n_cols"] == 2
    assert prof["columns"]["a"]["n_missing"] == 1
    assert prof["columns"]["b"]["n_unique"] == 2


def test_profile_includes_target_block():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    prof = profile_dataframe(df, target="y")
    assert prof["target"]["name"] == "y"
    assert prof["target"]["n_unique"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.inspect'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/inspect.py`:

```python
"""Dataset profiling for the inspect_data tool."""

import pandas as pd


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    prof: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": {}}
    for col in df.columns:
        s = df[col]
        prof["columns"][col] = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
    if target is not None and target in df.columns:
        s = df[target]
        prof["target"] = {
            "name": target,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
        }
    return prof
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspect.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/inspect.py tests/test_inspect.py
git commit -m "feat: dataframe profiling for inspect_data"
```

---

### Task 5: Deterministic CV folds

**Files:**
- Create: `src/tabular_agent/cv.py`
- Test: `tests/test_cv.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cv.make_folds(y, task_type: str, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]` returning `(train_idx, valid_idx)` integer-position arrays. Stratified for `binary`/`multiclass`, plain KFold for `regression`. Deterministic given identical args (both `fit_encoding` and `train_tabular_model` call this to agree on folds).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cv.py`:

```python
import numpy as np

from tabular_agent.cv import make_folds


def test_folds_cover_all_rows_exactly_once_in_validation():
    y = np.arange(50)
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    assert len(folds) == 5
    seen = np.concatenate([va for _, va in folds])
    assert sorted(seen.tolist()) == list(range(50))


def test_folds_are_deterministic():
    y = np.arange(50)
    a = make_folds(y, "regression", n_splits=5, seed=42)
    b = make_folds(y, "regression", n_splits=5, seed=42)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


def test_stratified_preserves_class_ratio():
    y = np.array([0] * 40 + [1] * 10)
    folds = make_folds(y, "binary", n_splits=5, seed=42)
    for _, va in folds:
        ratio = y[va].mean()
        assert 0.1 <= ratio <= 0.3  # ~0.2 in every fold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.cv'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/cv.py`:

```python
"""Deterministic fold generation shared by encoding and training."""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(y, task_type: str, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y)
    if task_type in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)), y)]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cv.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/cv.py tests/test_cv.py
git commit -m "feat: deterministic CV fold generation"
```

---

### Task 6: Leakage-safe encoders

This task carries the design's core correctness claim. It includes a **leakage regression test** proving that fitting on the full dataset leaks the target while out-of-fold fitting does not.

**Files:**
- Create: `src/tabular_agent/encoders.py`
- Test: `tests/test_encoders.py`

**Interfaces:**
- Consumes: `cv.make_folds` (in the test only).
- Produces:
  - `TargetEncoder(columns: list[str], smoothing: float = 10.0)` with `.fit(df, y) -> self`, `.transform(df) -> pd.DataFrame` (adds `<col>_te` columns), `.to_dict() -> dict`, classmethod `.from_dict(d) -> TargetEncoder`. `kind = "target"`.
  - `ENCODER_CLASSES: dict[str, type]` mapping `"target"` → `TargetEncoder`.
  - `oof_target_encode(df, y, columns, folds, smoothing=10.0) -> tuple[pd.DataFrame, TargetEncoder]` returning `(oof_encoded_columns_df, final_encoder_fit_on_all)`. The OOF frame has one column per `<col>_te`, indexed like `df`, where each row's value was produced by a fold in which that row was in validation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_encoders.py`:

```python
import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.encoders import TargetEncoder, ENCODER_CLASSES, oof_target_encode


def test_target_encoder_maps_stored_means_not_recomputed():
    df = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = np.array([10.0, 20.0, 100.0, 200.0])
    enc = TargetEncoder(["c"], smoothing=0.0).fit(df, y)
    # New data has NO target column; transform must still work from stored means.
    out = enc.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out["c_te"].tolist() == [15.0, 150.0]


def test_target_encoder_unseen_category_falls_back_to_global_mean():
    df = pd.DataFrame({"c": ["a", "b"]})
    y = np.array([10.0, 30.0])
    enc = TargetEncoder(["c"], smoothing=0.0).fit(df, y)
    out = enc.transform(pd.DataFrame({"c": ["zzz"]}))
    assert out["c_te"].iloc[0] == 20.0  # global mean


def test_target_encoder_serialization_roundtrip():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    y = np.array([1.0, 3.0, 9.0])
    enc = TargetEncoder(["c"], smoothing=1.0).fit(df, y)
    restored = ENCODER_CLASSES["target"].from_dict(enc.to_dict())
    a = enc.transform(df)["c_te"].to_numpy()
    b = restored.transform(df)["c_te"].to_numpy()
    assert np.allclose(a, b)


def test_oof_target_encoding_is_less_leaky_than_naive():
    # Each category appears a few times; target is pure noise w.r.t. category.
    rng = np.random.default_rng(0)
    n = 300
    cats = rng.integers(0, 60, size=n)  # ~5 rows per category
    df = pd.DataFrame({"c": [f"k{c}" for c in cats]})
    y = rng.normal(size=n)  # no real signal

    # Naive: fit on ALL rows, then read back -> each row sees its own target.
    naive = TargetEncoder(["c"], smoothing=0.3).fit(df, y).transform(df)["c_te"].to_numpy()

    # Out-of-fold: each row encoded from folds where it was NOT present.
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    oof_df, _ = oof_target_encode(df, y, ["c"], folds, smoothing=0.3)
    oof = oof_df["c_te"].to_numpy()

    corr_naive = abs(np.corrcoef(naive, y)[0, 1])
    corr_oof = abs(np.corrcoef(oof, y)[0, 1])
    # Signal is noise: the naive encoding fabricates correlation; OOF stays near zero.
    assert corr_naive > 0.25
    assert corr_oof < 0.15
    assert corr_naive > corr_oof + 0.1


def test_oof_returns_final_encoder_fit_on_all():
    df = pd.DataFrame({"c": ["a", "a", "b", "b", "b"]})
    y = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    _, final = oof_target_encode(df, y, ["c"], folds, smoothing=0.0)
    out = final.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out["c_te"].tolist() == [2.0, 20.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encoders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.encoders'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/encoders.py`:

```python
"""Stateful, leakage-safe encoders. Fit on train rows only; serialize params."""

import numpy as np
import pandas as pd


class TargetEncoder:
    kind = "target"

    def __init__(self, columns: list[str], smoothing: float = 10.0):
        self.columns = list(columns)
        self.smoothing = float(smoothing)
        self.mapping_: dict[str, dict[str, float]] = {}
        self.global_mean_: float | None = None

    def fit(self, df: pd.DataFrame, y) -> "TargetEncoder":
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y))
        y_ser = pd.Series(y)
        for col in self.columns:
            keys = df[col].astype(str).to_numpy()
            grouped = y_ser.groupby(keys)
            stats = grouped.agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.mapping_[col] = {str(k): float(v) for k, v in smooth.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            mapped = df[col].astype(str).map(self.mapping_[col])
            out[col + "_te"] = mapped.fillna(self.global_mean_).astype(float)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "smoothing": self.smoothing,
            "mapping": self.mapping_,
            "global_mean": self.global_mean_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetEncoder":
        enc = cls(d["columns"], d["smoothing"])
        enc.mapping_ = {c: dict(m) for c, m in d["mapping"].items()}
        enc.global_mean_ = d["global_mean"]
        return enc


ENCODER_CLASSES: dict[str, type] = {"target": TargetEncoder}


def oof_target_encode(df, y, columns, folds, smoothing: float = 10.0):
    y = np.asarray(y, dtype=float)
    oof = pd.DataFrame(index=df.index)
    for col in columns:
        oof[col + "_te"] = np.nan
    for tr, va in folds:
        enc = TargetEncoder(columns, smoothing).fit(df.iloc[tr], y[tr])
        transformed = enc.transform(df.iloc[va])
        for col in columns:
            oof.iloc[va, oof.columns.get_loc(col + "_te")] = transformed[col + "_te"].to_numpy()
    final = TargetEncoder(columns, smoothing).fit(df, y)
    return oof, final
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encoders.py -v`
Expected: PASS (5 passed). The `test_oof_target_encoding_is_less_leaky_than_naive` test is the design's headline guarantee.

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/encoders.py tests/test_encoders.py
git commit -m "feat: leakage-safe target encoder with out-of-fold fitting"
```

---

### Task 7: Model wrappers + cross-validated training

**Files:**
- Create: `src/tabular_agent/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `metrics.compute_metric`.
- Produces:
  - `predict_model(model, X: pd.DataFrame, task_type: str) -> np.ndarray` following the prediction-format contract (regression → 1-D; binary → 1-D P(class 1); multiclass → 2-D `(n,k)`).
  - `TrainResult` dataclass: `oof: np.ndarray`, `fold_metrics: list[float]`, `models: list`, `final_model`, `feature_names: list[str]`, `model_type: str`.
  - `train_cv(X, y, folds, task_type, metric, model_type="lgbm", params=None, seed=42) -> TrainResult`. Supported `model_type`: `"lgbm"`, `"xgb"`, `"catboost"`, `"baseline"`. Assumes multiclass `y` is label-encoded `0..k-1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.models import train_cv, predict_model, TrainResult


def _regression_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=n)
    return pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y)


def test_baseline_regression_oof_is_constant_mean():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    res = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    assert isinstance(res, TrainResult)
    assert res.oof.shape == (len(y),)
    # Baseline predicts a (near-)constant; std across predictions is tiny.
    assert res.oof.std() < 0.2


def test_lgbm_regression_beats_baseline():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    base = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    lgbm = train_cv(X, y, folds, "regression", "rmse", model_type="lgbm")
    assert np.mean(lgbm.fold_metrics) < np.mean(base.fold_metrics)
    assert lgbm.feature_names == ["a", "b", "c"]
    assert len(lgbm.models) == 5
    assert lgbm.final_model is not None


def test_binary_oof_is_probability_vector():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] + rng.normal(scale=0.3, size=200) > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    assert res.oof.shape == (200,)
    assert res.oof.min() >= 0.0 and res.oof.max() <= 1.0
    assert np.mean(res.fold_metrics) > 0.8  # AUC


def test_predict_model_binary_returns_positive_class_prob():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    preds = predict_model(res.final_model, X, "binary")
    assert preds.shape == (100,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.models'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/models.py`:

```python
"""Model wrappers over LightGBM/XGBoost/CatBoost plus a trivial baseline,
and cross-validated training that emits OOF predictions and fold metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metric


class _Baseline:
    """Predicts the training mean (regression) or class priors (classification)."""

    def __init__(self, task_type: str):
        self.task_type = task_type

    def fit(self, X, y):
        y = np.asarray(y)
        if self.task_type == "regression":
            self.value_ = float(np.mean(y))
        else:
            self.classes_ = np.unique(y)
            self.proba_ = np.array([float((y == c).mean()) for c in self.classes_])
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def predict_proba(self, X):
        return np.tile(self.proba_, (len(X), 1))


def _make_model(model_type: str, task_type: str, params: dict | None, seed: int):
    p = dict(params or {})
    if model_type == "baseline":
        return _Baseline(task_type)
    if model_type == "lgbm":
        import lightgbm as lgb

        if task_type == "regression":
            return lgb.LGBMRegressor(random_state=seed, n_estimators=200, verbose=-1, **p)
        return lgb.LGBMClassifier(random_state=seed, n_estimators=200, verbose=-1, **p)
    if model_type == "xgb":
        import xgboost as xgb

        if task_type == "regression":
            return xgb.XGBRegressor(random_state=seed, n_estimators=200, verbosity=0, **p)
        return xgb.XGBClassifier(random_state=seed, n_estimators=200, verbosity=0, **p)
    if model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if task_type == "regression":
            return CatBoostRegressor(random_state=seed, n_estimators=200, verbose=False, **p)
        return CatBoostClassifier(random_state=seed, n_estimators=200, verbose=False, **p)
    raise ValueError(f"unknown model_type: {model_type}")


def predict_model(model, X: pd.DataFrame, task_type: str) -> np.ndarray:
    if task_type == "regression":
        return np.asarray(model.predict(X), dtype=float)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if task_type == "binary":
        return proba[:, 1]
    return proba


@dataclass
class TrainResult:
    oof: np.ndarray
    fold_metrics: list[float]
    models: list
    final_model: object
    feature_names: list[str]
    model_type: str


def train_cv(X, y, folds, task_type, metric, model_type="lgbm", params=None, seed=42) -> TrainResult:
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    n = len(X)
    if task_type == "multiclass":
        k = int(y.nunique())
        oof = np.zeros((n, k))
    else:
        oof = np.zeros(n)

    models = []
    fold_metrics = []
    for tr, va in folds:
        model = _make_model(model_type, task_type, params, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = predict_model(model, X.iloc[va], task_type)
        oof[va] = preds
        fold_metrics.append(compute_metric(metric, y.iloc[va].to_numpy(), preds))
        models.append(model)

    final = _make_model(model_type, task_type, params, seed)
    final.fit(X, y)
    return TrainResult(oof, fold_metrics, models, final, list(X.columns), model_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (4 passed). (First run compiles LightGBM; allow a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/models.py tests/test_models.py
git commit -m "feat: model wrappers and cross-validated training with OOF"
```

---

### Task 8: Ensemble blending

**Files:**
- Create: `src/tabular_agent/blend.py`
- Test: `tests/test_blend.py`

**Interfaces:**
- Consumes: `metrics.compute_metric`, `metrics.greater_is_better`.
- Produces:
  - `optimize_blend(oofs: list[np.ndarray], y, metric: str, task_type: str) -> np.ndarray` — non-negative weights summing to 1, chosen to optimize `metric` on the blended OOF predictions.
  - `apply_blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_blend.py`:

```python
import numpy as np

from tabular_agent.blend import optimize_blend, apply_blend
from tabular_agent.metrics import compute_metric


def test_weights_are_nonnegative_and_sum_to_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    oof_a = np.array([1.1, 2.1, 2.9, 3.8])
    oof_b = np.array([3.0, 3.0, 3.0, 3.0])
    w = optimize_blend([oof_a, oof_b], y, "rmse", "regression")
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6


def test_blend_prefers_the_better_model():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    good = y + np.array([0.05, -0.05, 0.05, -0.05, 0.0])  # near perfect
    bad = np.full(5, 3.0)  # constant
    w = optimize_blend([good, bad], y, "rmse", "regression")
    assert w[0] > w[1]


def test_blend_beats_or_matches_best_single_model():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = y + np.array([0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    b = y + np.array([-0.3, 0.3, -0.3, 0.3, -0.3, 0.3])
    w = optimize_blend([a, b], y, "rmse", "regression")
    blended = apply_blend([a, b], w)
    best_single = min(compute_metric("rmse", y, a), compute_metric("rmse", y, b))
    assert compute_metric("rmse", y, blended) <= best_single + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_blend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.blend'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/blend.py`:

```python
"""Non-negative blend weights over out-of-fold predictions."""

import numpy as np
from scipy.optimize import minimize

from .metrics import compute_metric, greater_is_better


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.ones(len(w)) / len(w)


def optimize_blend(oofs: list[np.ndarray], y, metric: str, task_type: str) -> np.ndarray:
    y = np.asarray(y)
    m = len(oofs)
    sign = -1.0 if greater_is_better(metric) else 1.0

    def objective(raw: np.ndarray) -> float:
        w = _normalize(raw)
        pred = sum(wi * o for wi, o in zip(w, oofs))
        return sign * compute_metric(metric, y, pred)

    result = minimize(
        objective,
        np.ones(m) / m,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6},
    )
    return _normalize(result.x)


def apply_blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(wi * p for wi, p in zip(weights, preds))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_blend.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/blend.py tests/test_blend.py
git commit -m "feat: non-negative OOF blend weight optimization"
```

---

### Task 9: Pipeline serialization + predict (train/serve consistency)

**Files:**
- Create: `src/tabular_agent/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `encoders.ENCODER_CLASSES`, `models.predict_model`, `blend.apply_blend`.
- Produces:
  - `Pipeline(task_type: str)` with `.add_encoder(encoder)`, `.transform(df) -> pd.DataFrame`, `.to_dict()/from_dict()`. The `steps` list holds `{"type": "encode", "encoder": <dict>}` entries applied in order. (Stateless transform-code steps are added by Plan 2; the `type` field is the extension point.)
  - `save_bundle(run_ctx, pipeline, models: list, weights, feature_names, task_type) -> str` — writes `pipeline.json` + joblib model files under the run dir, returns the bundle directory.
  - `load_bundle(bundle_dir: str) -> dict` with keys `pipeline`, `models`, `weights`, `feature_names`, `task_type`.
  - `predict_bundle(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray` — applies the pipeline, selects `feature_names`, runs each model, blends. Must not require the target column.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.encoders import oof_target_encode
from tabular_agent.models import train_cv
from tabular_agent.pipeline import (
    Pipeline,
    save_bundle,
    load_bundle,
    predict_bundle,
)
from tabular_agent.runctx import RunContext


def test_pipeline_transform_roundtrip_dict():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    y = np.array([1.0, 3.0, 9.0])
    folds = make_folds(y, "regression", seed=42)
    _, final_enc = oof_target_encode(df, y, ["c"], folds)
    p = Pipeline("regression")
    p.add_encoder(final_enc)
    restored = Pipeline.from_dict(p.to_dict())
    pd.testing.assert_frame_equal(p.transform(df), restored.transform(df))


def test_predict_bundle_no_target_needed_and_consistent(tmp_path):
    # Build a tiny regression problem with one categorical driver.
    rng = np.random.default_rng(0)
    cats = rng.integers(0, 4, size=120)
    base = np.array([0.0, 10.0, 20.0, 30.0])[cats]
    df = pd.DataFrame({"c": [f"k{c}" for c in cats]})
    y = pd.Series(base + rng.normal(scale=0.5, size=120))

    folds = make_folds(y, "regression", seed=42)
    oof_enc, final_enc = oof_target_encode(df, y, ["c"], folds)

    # Train on the encoded feature.
    X = oof_enc  # single column c_te (OOF-encoded for honest training)
    res = train_cv(X, y, folds, "regression", "rmse", model_type="lgbm")

    pipeline = Pipeline("regression")
    pipeline.add_encoder(final_enc)
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(
        ctx,
        pipeline,
        models=[res.final_model],
        weights=np.array([1.0]),
        feature_names=["c_te"],
        task_type="regression",
    )

    bundle = load_bundle(bundle_dir)
    # New raw data: only the raw category column, NO target.
    new_raw = pd.DataFrame({"c": ["k0", "k1", "k2", "k3"]})
    preds = predict_bundle(bundle, new_raw)
    assert preds.shape == (4,)
    # Predictions must track the category means monotonically.
    assert preds[0] < preds[1] < preds[2] < preds[3]


def test_predict_is_deterministic(tmp_path):
    df = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = pd.Series([1.0, 2.0, 8.0, 9.0])
    folds = make_folds(y, "regression", seed=42)
    oof_enc, final_enc = oof_target_encode(df, y, ["c"], folds)
    res = train_cv(oof_enc, y, folds, "regression", "rmse", model_type="baseline")
    pipeline = Pipeline("regression")
    pipeline.add_encoder(final_enc)
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(ctx, pipeline, [res.final_model], np.array([1.0]), ["c_te"], "regression")
    bundle = load_bundle(bundle_dir)
    raw = pd.DataFrame({"c": ["a", "b"]})
    p1 = predict_bundle(bundle, raw)
    p2 = predict_bundle(bundle, raw)
    assert np.array_equal(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/pipeline.py`:

```python
"""Serializable feature pipeline + predictor bundle. predict replays stored
parameters and never needs the target column."""

import json
import os

import joblib
import numpy as np
import pandas as pd

from .blend import apply_blend
from .encoders import ENCODER_CLASSES
from .models import predict_model


class Pipeline:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.steps: list[dict] = []

    def add_encoder(self, encoder) -> None:
        self.steps.append({"type": "encode", "encoder": encoder.to_dict()})

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        for step in self.steps:
            if step["type"] == "encode":
                enc = ENCODER_CLASSES[step["encoder"]["kind"]].from_dict(step["encoder"])
                out = enc.transform(out)
            else:
                raise ValueError(f"unknown step type: {step['type']}")
        return out

    def to_dict(self) -> dict:
        return {"task_type": self.task_type, "steps": self.steps}

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        p = cls(d["task_type"])
        p.steps = d["steps"]
        return p


def save_bundle(run_ctx, pipeline: Pipeline, models: list, weights, feature_names, task_type) -> str:
    bundle_dir = run_ctx.path("bundle", "_")
    bundle_dir = os.path.dirname(bundle_dir)
    model_paths = []
    for i, model in enumerate(models):
        mp = os.path.join(bundle_dir, f"model_{i}.joblib")
        joblib.dump(model, mp)
        model_paths.append(f"model_{i}.joblib")
    manifest = {
        "pipeline": pipeline.to_dict(),
        "model_paths": model_paths,
        "weights": np.asarray(weights, dtype=float).tolist(),
        "feature_names": list(feature_names),
        "task_type": task_type,
    }
    with open(os.path.join(bundle_dir, "pipeline.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return bundle_dir


def load_bundle(bundle_dir: str) -> dict:
    with open(os.path.join(bundle_dir, "pipeline.json")) as f:
        manifest = json.load(f)
    models = [joblib.load(os.path.join(bundle_dir, mp)) for mp in manifest["model_paths"]]
    return {
        "pipeline": Pipeline.from_dict(manifest["pipeline"]),
        "models": models,
        "weights": np.asarray(manifest["weights"], dtype=float),
        "feature_names": manifest["feature_names"],
        "task_type": manifest["task_type"],
    }


def predict_bundle(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    features = bundle["pipeline"].transform(raw_df)
    X = features[bundle["feature_names"]]
    task_type = bundle["task_type"]
    preds = [predict_model(m, X, task_type) for m in bundle["models"]]
    return apply_blend(preds, bundle["weights"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (3 passed). `test_predict_bundle_no_target_needed_and_consistent` proves the no-recompute-at-serve rule.

- [ ] **Step 5: Commit**

```bash
git add src/tabular_agent/pipeline.py tests/test_pipeline.py
git commit -m "feat: serializable pipeline and predictor bundle"
```

---

### Task 10: Run report (`finalize`)

**Files:**
- Create: `src/tabular_agent/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing (takes plain data).
- Produces: `report.build_report(*, task_type, metric, selection_score, fold_metrics, baseline_score, model_types, feature_names, holdout_score=None) -> str` returning Markdown; `report.write_report(run_ctx, text) -> str` writing `run_report.md` and returning its path.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import os

from tabular_agent.report import build_report, write_report
from tabular_agent.runctx import RunContext


def test_build_report_contains_key_numbers():
    text = build_report(
        task_type="regression",
        metric="rmse",
        selection_score=0.123,
        fold_metrics=[0.11, 0.12, 0.13, 0.12, 0.14],
        baseline_score=0.500,
        model_types=["lgbm", "xgb"],
        feature_names=["a", "b", "c_te"],
    )
    assert "regression" in text
    assert "rmse" in text
    assert "0.123" in text
    assert "0.500" in text  # baseline shown for honesty
    assert "lgbm" in text and "xgb" in text


def test_report_flags_when_baseline_not_beaten():
    text = build_report(
        task_type="regression",
        metric="rmse",
        selection_score=0.6,
        fold_metrics=[0.6],
        baseline_score=0.5,  # model worse than baseline
        model_types=["lgbm"],
        feature_names=["a"],
    )
    assert "WARNING" in text


def test_holdout_line_present_when_given():
    text = build_report(
        task_type="binary",
        metric="auc",
        selection_score=0.9,
        fold_metrics=[0.9],
        baseline_score=0.5,
        model_types=["lgbm"],
        feature_names=["a"],
        holdout_score=0.88,
    )
    assert "0.88" in text
    assert "holdout" in text.lower()


def test_write_report_creates_file(tmp_path):
    ctx = RunContext(str(tmp_path / "run"))
    path = write_report(ctx, "# hello")
    assert os.path.exists(path)
    assert path.endswith("run_report.md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_agent.report'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tabular_agent/report.py`:

```python
"""Human-readable run report."""

import numpy as np

from .metrics import greater_is_better


def _beats_baseline(metric: str, selection_score: float, baseline_score: float) -> bool:
    if greater_is_better(metric):
        return selection_score >= baseline_score
    return selection_score <= baseline_score


def build_report(
    *,
    task_type: str,
    metric: str,
    selection_score: float,
    fold_metrics: list[float],
    baseline_score: float,
    model_types: list[str],
    feature_names: list[str],
    holdout_score: float | None = None,
) -> str:
    fold_std = float(np.std(fold_metrics))
    lines = [
        "# Run Report",
        "",
        f"- **Task type:** {task_type}",
        f"- **Selection metric:** {metric}",
        f"- **CV score:** {selection_score:.3f} (fold std {fold_std:.3f})",
        f"- **Baseline score:** {baseline_score:.3f}",
        f"- **Models:** {', '.join(model_types)}",
        f"- **Features ({len(feature_names)}):** {', '.join(feature_names)}",
    ]
    if holdout_score is not None:
        lines.append(f"- **Holdout score:** {holdout_score:.3f}")
    if not _beats_baseline(metric, selection_score, baseline_score):
        lines += [
            "",
            "> **WARNING:** the model does not beat the trivial baseline. "
            "Suspect a bug, leakage, or an unlearnable target.",
        ]
    lines.append("")
    lines.append("## Per-fold metrics")
    for i, m in enumerate(fold_metrics):
        lines.append(f"- fold {i}: {m:.4f}")
    return "\n".join(lines)


def write_report(run_ctx, text: str) -> str:
    path = run_ctx.path("run_report.md")
    with open(path, "w") as f:
        f.write(text)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run pytest -v`
Expected: all tests pass across all modules.

```bash
git add src/tabular_agent/report.py tests/test_report.py
git commit -m "feat: run report with baseline-honesty warning"
```

---

## Self-Review

**Spec coverage** (against `2026-07-21-tabular-agent-mcp-design.md`):

- §3 tool `inspect_data` → Task 4 (logic; MCP wiring in Plan 2). ✅
- §4 stateless/file-based → Task 3 (RunContext, parquet IO). ✅
- §5.1 stateless escape-hatch transforms → **Plan 2** (needs the subprocess sandbox); Pipeline has the `type` extension point (Task 9). ✅ (deferred by design)
- §5.2 / §5.3 leakage-safe stateful encoders (fit-per-fold, persisted, unknown-category fallback) → Task 6, with the headline leakage regression test. ✅
- §7 task-type/metric auto-detect + override + RMSLE non-negative guard → Task 2 + Task 1. ✅
- §8 GBDT trio + baseline + CV (metrics, fold table, model artifacts) + blend → Tasks 7, 8. ✅
- §9 model bundle + `predict` + report → Tasks 9, 10. ✅
- §10 test-set (with/without labels) → **Plan 2** (tool-level orchestration); the primitives (`predict_bundle`, `compute_metric`) exist here. ✅ (deferred by design)
- §11 observability, §12 server instructions, §13 packaging/MCP → **Plan 2**. ✅ (deferred by design)

**Placeholder scan:** no TBD/TODO; every code and test step is complete. ✅

**Type consistency:** `TrainResult` fields, `predict_model` signature, `ENCODER_CLASSES` keys (`"target"`), bundle dict keys (`pipeline`/`models`/`weights`/`feature_names`/`task_type`), and the prediction-format contract (regression 1-D, binary 1-D prob, multiclass 2-D) are used identically across Tasks 6–10. ✅

**Deferred-to-Plan-2 (explicit):** subprocess sandbox + `transform_features`, OTel observability, MCP server + tool registration + server instructions, `fit_encoding`/`train_tabular_model`/`blend_models`/`predict`/`finalize_run` tool wrappers over these functions, test-set orchestration, packaging/entry-point.
