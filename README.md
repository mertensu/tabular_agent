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

```bash
pip install tabular-agent
```

Everything it needs — LightGBM, XGBoost, CatBoost, scikit-learn — comes with it as
prebuilt wheels; there is nothing to compile. Python ≥ 3.11.

The agent reasons with an LLM, so set an API key once:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Try it on your own data

You need a training table with a target column. If you also have the table you'll
later predict on (e.g. a Kaggle-style `test.csv` without the target), put both in
one folder and point at the folder — the agent reads the two schemas and works out
which columns will actually exist at prediction time.

```python
import tabular_agent as ta

agent = ta.Agent()                      # uses ANTHROPIC_API_KEY
plan = agent.inspect("path/to/folder")  # profiles every file, proposes nothing final
```

`inspect()` does not decide anything — it *proposes*, and you confirm. That single
rule is the whole safety story (see "Why the confirmation step" below). Look at
what it found:

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

Working from a single DataFrame in a notebook works too — pass it to `inspect(df)`,
set `serving_columns` to the columns you'll have at prediction time, and choose
`split_on`.

## What you get back

```python
print(run.report())                       # what it tried, what it shipped, and how honest the score is

predictions = run.bundle.predict("path/to/new_rows.csv")   # or predict(df)
run.bundle.save("my_model")                # a self-contained, replayable bundle
```

The bundle takes the same raw rows you started with and replays the exact feature
lineage — no separate preprocessing to reproduce. The report is written to be read
without insider knowledge: it separates the search winner from what actually
ships, and flags where a score is optimistic.

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
