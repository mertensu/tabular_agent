# tabular_agent

An autonomous tabular-ML agent that turns one local table plus a prediction
contract into a **leakage-safe, replayable predictor bundle** and an audit
journal.

## Product contract

**Input**

- one training table
- a target column and metric
- optionally an ordering column
- the columns available when predictions are served

**Output**

- a predictor bundle that accepts raw rows
- a compact report
- a machine-readable journal of every candidate table and its lineage

The differentiating guarantee is structural: the agent has no shell, raw-data
reader, arbitrary model trainer, or path around the fitted/replayable feature
tools. Target-free transforms are sandboxed and stability-checked; fitted
encoders are cross-fitted for training and serialized for serving.

## Start here

```bash
uv sync --extra dev
uv run pytest
```

Fresh-machine and dataset setup: [`docs/SETUP.md`](docs/SETUP.md).
Current implementation and the one next action: [`docs/CURRENT.md`](docs/CURRENT.md).
Interactive orientation: [`notebooks/START_HERE.ipynb`](notebooks/START_HERE.ipynb).

```python
from tabular_agent.harness.agent import run
from tabular_agent.harness.session import RunSession
from tabular_agent.taskspec import TaskSpec

session = RunSession("run", "train.csv", TaskSpec(target="y"), profile="minimal")
run(session, "Build the best safe table you can, then finalize.")
```

The minimal profile exposes six product tools, uses one train/evaluation split,
and ships one fixed LGBM model without terminal CV, bootstrap uncertainty,
sealed scoring or blending. `profile="full"` remains available on the
simplification branch as a compatibility/research reference.

## Repository map

```text
src/tabular_agent/   product runtime and safety kernel
tests/               production guarantees
scripts/             reusable dataset preparation only
notebooks/           one active orientation notebook
docs/                current product documentation and immutable decisions
research/            archived runs, probes, readings, notebooks and artifacts
```

## What this is not

- a general-purpose tabular utility library
- a Kaggle winner reproduction framework
- an autonomous model/hyperparameter search engine
- a claim that one internal score estimates a distant external population

Research explores those questions under `research/`; mechanisms enter the
runtime only after a prospective cross-task result and a stable user-facing
contract.
