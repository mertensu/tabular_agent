# Setup — getting a fresh machine to a running state

What a new collaborator needs beyond `git clone`. The repo tracks the library,
the tests, the runners, and the reasoning (`docs/decisions/`, `HANDOFF.md`), but
three things are deliberately **not** in git and have to be supplied locally:

1. the Python environment,
2. an API key, and
3. the Kaggle competition data.

None of the ~11 GB of agent output in `_runs/` is needed — it is reproducible
from the archived research runners and its findings are preserved under
`research/archive/readings/` and the decision records. See
[`research/archive/readings/README.md`](../research/archive/readings/README.md).

---

## 1. Environment

Managed with [`uv`](https://docs.astral.sh/uv/) against the committed lockfile.
Python 3.11+.

```bash
uv sync                 # runtime deps
uv sync --extra dev     # + pytest
uv run pytest           # sanity check — the active product/compatibility suite
```

## 2. API key and model provider

The public default calls Anthropic directly. Set the provider's key in the
environment and select its model with `provider:model`:

```bash
export ANTHROPIC_API_KEY="..."
```

```python
import tabular_agent as ta

agent = ta.Agent()  # anthropic:claude-opus-5
```

Other native Pydantic AI providers work the same way: for example,
`OPENAI_API_KEY` with `ta.Agent(model="openai:gpt-5.2")`, or `GOOGLE_API_KEY`
with `ta.Agent(model="google:gemini-2.5-flash")`. The full provider/key table is
in the [README](../README.md#choose-an-llm-provider).

Historical research runners commonly use the
[Requesty](https://router.requesty.ai) gateway instead. A model name containing
`/` but no `:` selects that path and reads `REQUESTY_API_KEY`:

```bash
cp config.env.example config.env
# then edit config.env and paste your own Requesty key
```

```python
agent = ta.Agent(model="tensorx/deepseek-v4-flash")
```

`config.env` is gitignored (`*.env`) and is supported only by the Requesty
helper; native providers read their standard environment variables. Each
collaborator uses their own key. You only need a key to run the *agent*; the
library, tests, and `scripts/prep_*.py` do not touch the LLM.

## 3. Kaggle competition data

The datasets are **probes, not part of the library** (see `CLAUDE.md`) and are
Kaggle's to distribute, not ours — so each raw dataset directory is gitignored.
Download the ones you need with the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api) (`pip install kaggle`, then
put your `kaggle.json` token in `~/.kaggle/`). You must accept each competition's
rules on its Kaggle page once before the download is allowed.

Run each download from the repo root. `unzip` into the exact directory named
below — the prep scripts and runners look there by hardcoded path.

| Dataset | Competition slug | Target dir | Build step → outputs |
|---------|------------------|------------|----------------------|
| House prices | `house-prices-advanced-regression-techniques` | `house_prices/` | none — uses `train.csv` / `test.csv` directly. `train_no_id.csv` is `train.csv` minus the `Id` column. |
| Rossmann | `rossmann-store-sales` | `rossmann_store_sales/` | `uv run python scripts/prep_rossmann.py` → `train_joined.parquet`, `test_joined.parquet` |
| Restaurant | `recruit-restaurant-visitor-forecasting` | `restaurant_visitor/` | `uv run python scripts/prep_restaurant.py` → `train_joined.parquet`, `test_joined.parquet`, `air_reserve_stream.parquet`, `hpg_reserve_stream.parquet` |
| IEEE fraud | `ieee-fraud-detection` | `fraud_detection/` | `uv run python scripts/prep_ieee.py` → `train_joined.parquet`, `test_joined.parquet` |
| Porto Seguro | `porto-seguro-safe-driver-prediction` | `seguro_driver_prediction/` | no active prep/runner; historical runners are archived under `research/archive/runs/seguro/` |

Example, for Rossmann:

```bash
kaggle competitions download -c rossmann-store-sales -p rossmann_store_sales/
unzip -o 'rossmann_store_sales/*.zip' -d rossmann_store_sales/
uv run python scripts/prep_rossmann.py
```

### Leaderboards (needed for honest scoring)

The `*-privateleaderboard-*.csv` / `*-publicleaderboard-*.csv` files inside each
dataset dir are what let a run score itself against the real board — the
private-by-default rule of decision 0061 depends on them being present. They are
downloaded from each competition's leaderboard page (Kaggle CLI:
`kaggle competitions leaderboard -c <slug> --download`). Not every competition
exposes the private board publicly; the ones on disk are the ones we had access
to. A run without a leaderboard file still fits — it just can't grade its own
answer.

The prep scripts carry the full data dictionary and every join decision in their
module docstrings (`scripts/prep_ieee.py` is the most detailed); read those, not
this table, to understand what a row of each dataset means.

---

## Then

Read `HANDOFF.md` (what happened last, possibly on the other machine) and
`docs/decisions/README.md` (why the code has the shape it has). Start there.
