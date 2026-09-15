# tabular-agent

Point it at your table, confirm a few facts about it, and get back a
**leakage-safe, replayable predictor** plus an honest report of what it did — no
feature engineering, no model tuning, no notebook plumbing on your side.

The differentiator is not accuracy for its own sake; it is that every number it
reports is one you can trust. The agent has no shell, no arbitrary model trainer,
and no path around its fitted, replayable feature tools. Transforms are sandboxed
and target-blind; categorical encoders are cross-fitted, so nothing it builds can
quietly leak the target into your validation score.

## Install

Until it is published to a package index, install it straight from the repository —
pip clones and builds it for you:

```bash
pip install git+https://github.com/mertensu/tabular_agent.git
```

For a private repository, use SSH instead:
`pip install git+ssh://git@github.com/mertensu/tabular_agent.git`. To pin a
version, append `@<tag-or-commit>`.

Everything it needs — LightGBM, XGBoost, CatBoost, scikit-learn — comes with it as
prebuilt wheels; there is nothing to compile. Python ≥ 3.11.

## Choose an LLM provider

The API key is **not** passed to `ta.Agent(...)`. Set it in the environment; the
provider selected by `model=` reads its own variable. This keeps secrets out of
notebooks, scripts, and run artifacts.

| Provider | `model=` example | Environment variable |
|---|---|---|
| Anthropic (default) | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:gpt-5.2` | `OPENAI_API_KEY` |
| Google Gemini | `google:gemini-2.5-flash` | `GOOGLE_API_KEY` |
| Groq | `groq:<model-name>` | `GROQ_API_KEY` |
| Mistral | `mistral:<model-name>` | `MISTRAL_API_KEY` |
| OpenRouter | `openrouter:<provider/model>` | `OPENROUTER_API_KEY` |
| Fireworks | `fireworks:<model-name>` | `FIREWORKS_API_KEY` |
| DeepSeek | `deepseek:<model-name>` | `DEEPSEEK_API_KEY` |
| Requesty gateway | `<provider>/<model>` | `REQUESTY_API_KEY` |

Choose one provider, export its key, and use the matching model prefix:

```bash
# Default: Anthropic
export ANTHROPIC_API_KEY="..."
```

```python
import tabular_agent as ta

agent = ta.Agent()  # default: anthropic:claude-opus-5
```

For example, to use OpenAI or Google instead:

```bash
export OPENAI_API_KEY="..."  # when using the OpenAI example below
# export GOOGLE_API_KEY="..."  # alternatively, for the Google example
```

```python
openai_agent = ta.Agent(model="openai:gpt-5.2")
google_agent = ta.Agent(model="google:gemini-2.5-flash")
```

Native Pydantic AI providers use `provider:model`. A string containing `/` but
no `:` selects the project's OpenAI-compatible Requesty gateway instead:

```bash
export REQUESTY_API_KEY="..."
```

```python
gateway_agent = ta.Agent(model="tensorx/deepseek-v4-flash")
```

Only the Requesty path also supports a gitignored `config.env` containing
`REQUESTY_API_KEY=...`; native providers read their standard environment
variables. Advanced providers that need more than one setting can be passed as
a preconfigured Pydantic AI model object. For example, Azure additionally needs
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` (and older endpoints may need
`OPENAI_API_VERSION`). Whichever provider you choose, its model must support
reliable tool calling because the agent's entire workflow uses tools.

## Try it on your own data

You need a training table with a target column. If you also have the table you'll
later predict on (e.g. a Kaggle-style `test.csv` without the target), put both in
one folder and point at the folder — the agent reads the two schemas and works out
which columns will actually exist at prediction time.

Drop a data description (`data_description.txt`, a `.md` dictionary, …) in the same
folder and `inspect()` loads it into `plan.notes` automatically, so the agent reads
your own column explanations — worth a lot when the column names are cryptic. It is
passed through verbatim; the agent structures the free text itself.

```python
import tabular_agent as ta

agent = ta.Agent()                      # uses the default Anthropic provider
plan = agent.inspect("path/to/folder")  # profiles every file, proposes nothing final
```

`inspect()` does not decide anything — it *proposes*, and you confirm. That single
rule is the whole safety story (see "Why the confirmation step" below).
`print(plan.summary())` shows the whole checklist at a glance — each decision, its
proposal, whether it is confirmed, and the one line that sets it:

```text
DataPlan — 3 decision(s) still to confirm.

  [ ] target             not set
                         candidates: Sales, Customers
                         set: plan.target = '<column>'
  [ ] serving_columns    not set
                         present at predict time: Store, Date
                         set: plan.serving_columns = [...]
  [ ] split_on           not set
                         date/time candidates: Date
                         set: plan.split_on = '<column>' (ordered) or False (random)
  [x] identifier_columns (none)
  [x] joins              store.csv:accepted
```

Or look at the raw proposals directly:

```python
plan.target_candidates           # likely target columns, best guess first
plan.proposed_serving_columns    # columns present at prediction time
plan.proposed_identifier_columns # row ids to keep for output but hide from the model
plan.ordering_candidates         # date/time columns you could split on
plan.joins                       # static lookups it can fold in, if any
```

Confirm the handful of facts only you know, then run:

```python
plan.target = "Sales"                    # what to predict
plan.serving_columns = ["Store", "Date", "Promo"]  # what exists at predict time
plan.split_on = "Date"                   # ordered split for a forecast; or False for a random split
plan.accept_identifier_proposals()       # or reject_identifier_proposals()
plan.reject_joins()                      # or accept_joins()

run = agent.run(plan)                     # builds features, evaluates, finalizes
```

You confirm by setting these attributes on the plan — plain Python, no wizard. If
you forget one, `agent.run(plan)` refuses before doing any work and names exactly
what is still unconfirmed, so you cannot start a run on an unstated assumption.

Working from a single DataFrame in a notebook works too — pass it to `inspect(df)`,
set `serving_columns` to the columns you'll have at prediction time, and choose
`split_on`.

## What you get back

```python
print(run.report())                       # portable Markdown fallback
run.report_path()                         # self-contained run_report.html dashboard
run.report("json")                       # the same sanitized, machine-readable facts

predictions = run.bundle.predict("path/to/new_rows.csv")   # or predict(df)
run.bundle.save("my_model")                # a self-contained, replayable bundle
```

The HTML dashboard opens directly in a browser without a server or network
connection. It adds accessible glossary tooltips, model/optimism comparison,
the feature-building story, a searchable feature catalog, the confirmed serving
contract, and warnings; it never recomputes scores or selects another model.

[Open the sanitized House Prices example HTML report](docs/examples/house-prices-run-report.html)
or download it and open it locally to try the offline interactions. It uses the
recorded acceptance-run scores and representative feature lineage, but contains
no training rows, target values, fitted lookup values, or other raw data.

The bundle takes the same raw rows you started with and replays the exact feature
lineage — no separate preprocessing to reproduce. The report is written to be read
without insider knowledge: it separates the search winner from what actually
ships, and flags where a score is optimistic. An excerpt from a real House Prices
run:

```text
# Run Report
- Task type:                                regression
- Selection metric:                         rmsle
- Shipped validation score:                 0.1260
- Shipped test score (scored once):         0.1259 (+/- 0.0114)
- Baseline score (trivial model):           0.363
- Models:                                   lgbm, xgb, catboost, elasticnet
- Features (145):                           OverallQual, GrLivArea, TotalSF, ... , Neighborhood_te
- Split:                                    876 work / 292 validation / 292 test (random)

## What the search cost
The candidate the search crowned scored 0.1319 on validation and 0.1283 on 292
test rows nothing in this run could read. What shipped is not that candidate:
the precommitted, fold-spread-weighted blend scored 0.1259 on those rows — +0.0025 better.

| model                                    | validation |   test | optimism |
|------------------------------------------|-----------:|-------:|---------:|
| catboost (search's pick)                 |     0.1319 | 0.1283 |  -0.0036 |
| lgbm                                     |     0.1342 | 0.1349 |  +0.0007 |
| xgb                                      |     0.1419 | 0.1450 |  +0.0031 |
| elasticnet                               |     0.1339 | 0.1358 |  +0.0020 |
| blend(lgbm+xgb+catboost+elasticnet) ✅   |     0.1260 | 0.1259 |  -0.0001 |

> These test rows ranked nothing — the winner was picked on validation before
> they existed. Choosing between models by this column would spend the last
> honest number the run has.
```

## Explain a single prediction

Ask *why this number for this row*, feature by feature — an exact, model-native
attribution of the prediction the bundle actually ships (the weighted blend, not
one member):

```python
explanation = run.bundle.explain("path/to/new_rows.csv")
print(explanation.for_row(0))     # this row's feature pushes, largest magnitude first
print(explanation.base_value)     # the expected value the row is explained away from
```

For one House Prices house, largest push first:

```text
>>> explanation.space
'log1p(target)'
>>> explanation.base_value
12.0235
>>> explanation.for_row(0).head(6)
Neighborhood_te     0.0390
Qual_x_SF          -0.0246
SaleCondition_te   -0.0198
HouseAge           -0.0185
TotalQual           0.0179
YearRemodAdd       -0.0147
```

`base_value + contributions.sum(axis=1)` reconstructs the model output, and
`explanation.space` names the scale it is additive on: `"target"` for a plain
regression (so the reconstruction *is* the prediction), or one link away
(`"log1p(target)"`, `"log-odds"`) when a transform or a classifier is involved.
It is exact — TreeSHAP for the boosted models, closed-form for the linear one — so
there is no surrogate model or sampling noise to second-guess.

## Bring your own model — reuse the learned features

The feature pipeline is not trapped inside the shipped predictor. If you'd rather
experiment with your own model, take the exact features the agent learned:

```python
# One readable row per shipped feature: source columns, kind, pipeline step,
# a plain-English explanation, and the stored recipe. No fitted values dumped.
print(run.bundle.describe_features())

# The exact leakage-safe matrix the shipped models were fitted on, plus the target.
training = run.training_features()
from sklearn.linear_model import LinearRegression
my_model = LinearRegression().fit(
    training[run.bundle.feature_names], training[plan.target]
)

# Apply the same fitted feature recipe to genuinely new rows, then predict.
serving = run.bundle.feature_table("path/to/new_rows.csv")
my_predictions = my_model.predict(serving)

# Or keep your identifiers/raw columns and append the learned model columns.
enriched = run.bundle.feature_table("path/to/new_rows.csv", include_input=True)
```

The split is deliberate and it is a safety boundary, not an inconvenience: use
`run.training_features()` for the *training* side and `bundle.feature_table()` for
*new rows*. Replaying the bundle's final target-informed encoders over their own
fit rows would reproduce in-fold values, not the out-of-fold ones the models were
trained on — i.e. it would leak. `training_features()` returns the real
out-of-fold table; `feature_table()` is the serving half.

## Why the confirmation step

The one thing the agent will not do is silently guess a modeling decision. The
classic trap: a column that correlates beautifully with the target in training but
**won't exist when you predict** (yesterday's customer count when forecasting
tomorrow's sales). Trained on, it produces a gorgeous validation score and a
useless model — and every generic guard permits it, because no leakage rule was
broken. `inspect()` catches the common form automatically (a column in your
training table but absent from the serving table is excluded), and asks you to
confirm the rest via `serving_columns`. Requiring one confirmation is the price of
never shipping that model by accident.

## Long runs stay robust

A run executes in fresh episodes that read their state from the run journal rather
than an ever-growing context, so a long or expensive search does not balloon in
cost and can resume where it left off if interrupted. The mechanics live in the
decision records ([`docs/decisions/`](docs/decisions/), 0024–0027).

## What this is not

- a general-purpose feature library you assemble yourself
- an autonomous model/hyperparameter search engine (the search is over *features*)
- a claim that one internal score estimates a distant external population — the
  report tells you exactly how far it can and cannot see

## Contributing

Development setup, the test suite, and the design decisions that shape the code:

```bash
uv sync --extra dev
uv run pytest
```

- Fresh-machine and dataset setup: [`docs/SETUP.md`](docs/SETUP.md)
- Current state and next action: [`docs/CURRENT.md`](docs/CURRENT.md)
- Why the code has the shape it has: [`docs/decisions/`](docs/decisions/)
