# Product / UX plan — tabular_agent as a safety-first tabular-ML library

**Status: PROPOSAL for Ulf's review (2026-08-31). Not a decision record.**
Author: Claude. Folds in Qwen's positioning + two-fix points (channel thread 4d5212).
Grounded in the current API (`RunSession`, `TaskSpec`, `build_agent`/`gateway_model`,
`load_bundle`/`predict_bundle`) and the decision records cited inline.

## 1. Positioning (decide this first — it determines what we build)

The pitch is **not** "beats your model". It is:

> **The only tabular agent that refuses to leak.**

Differentiators, all already built (docs/CURRENT.md): leakage guards, replayable
bundles, sealed scoring, an audit journal, and serving-schema enforcement. The
honest score proof stays: **top 58 % of a ~3,300-team field with zero leaderboard
feedback** (0061). Target user: someone with their **own** tabular data and **no
oracle** — precisely where "the reported score is the production score" is worth
money, and where a competition optimizer's number is meaningless.

Rossmann's job (probe what the harness cannot express) is done; the median was the
price of forbidden levers (0001/0035), not a bug. The pivot is from "win a board"
to "ship a trustworthy model".

## 2. UX contract: three inputs, one visible plan, one artifact

The user states only what cannot be inferred; everything else is **proposed and
confirmed**, never silently guessed (this is the `Customers` lesson — silent prep
automation *builds* the leak).

Confirmed by the user (shown with a proposed default):
- **target** — the column to predict.
- **serving_columns** — what exists at prediction time. This defines the leak
  boundary. Default = columns of a provided test file; else train columns minus
  target, surfaced for confirmation. (`Customers` is the canonical trap.)
- **split_on / time-order** — forecast (ordered split) vs random split.

Inferred but shown for review: dtypes, join keys across files, candidate targets,
metric (from task type), suspicious/leaky columns. Optional: `column_notes`
(the data-dictionary channel; run works without it, and then measures whether it
mattered).

```python
import tabular_agent as ta

agent = ta.Agent(model="anthropic/claude-sonnet-4-6")   # any pydantic-ai model

spec = agent.inspect("/data")     # profiles ALL files; returns a PROPOSED plan
spec.target = "Sales"             # user confirms/corrects the three inputs
spec.serving_columns = [...]
spec.split_on = "Date"

run = agent.run(spec)             # build table, search features, finalize → sealed bundle
run.report()                      # human-readable report + journal
run.bundle.predict(new_rows)      # the artifact: predictions from raw rows
```

Optional one-call sugar over the same path, for the impatient:
`ta.fit("/data", target="Sales", serving_columns=[...], split_on="Date")`.

## 3. Architecture — thin facade over the existing core, no `src/` rewrite

- `ta.Agent(model=...)` → wraps `gateway_model` + `build_agent(profile=...)`.
- `agent.inspect(path_or_df) -> Spec` → **new** folder-semantics + proposal layer:
  profiles files, proposes join/grain, candidate target, time column, and the
  serving schema (columns in train but not test are flagged), infers metric.
  Returns an **editable** `Spec` that maps onto `TaskSpec` + `split_on`.
  **It proposes; it never applies silently.**
- `agent.run(spec) -> Run` → wraps `RunSession(...).run()` + finalize.
- `Run.report()` → renders safety + replay + score + lineage from the journal.
- `Run.bundle`, `bundle.predict(folder_or_df)`, `bundle.save/load` → public wrapper
  over `load_bundle`/`predict_bundle`, accepting raw rows or a folder.
- New: `src/tabular_agent/api.py` (facade) + a folder-semantics extension to
  `inspect.py`. **Zero change to the guards or harness internals** — the guarantees
  stay exactly where they are.

## 4. Two measured fixes to land BEFORE opening (both dataset-agnostic)

1. **Blend policy (0044 / 0045).** The shipped unfiltered blend costs a
   live-measured **+0.0126** (Qwen's run: 0.14706 blend vs 0.13446 best single) —
   one member that fails *badly* (catboost, +0.051 sealed optimism) drags the mean.
   **Caveat, important:** 0045 is *accepted, deliberately not implemented* and
   **refuses** holdout-consulted member filtering. So this is **not** a free code
   tweak — it needs a **new decision record** choosing a 0045-compliant remedy:
   a *pre-fixed* rule (e.g. ship best-single when the sealed-optimism spread exceeds
   a fixed bound) or a non-fitted weighting — never "drop the worst member by
   holdout". → **Product decision for Ulf, not an implementation detail.**
2. **Model budget default (0054).** The 200-tree default underfits; an honest,
   *dataset-agnostic* budget chooser (a fixed schedule, not per-dataset tuning) is
   0035-compliant and lower-risk than #1.

## 5. Non-goals / guardrails (these are the product, not limitations to remove)

- No silent prep magic — `inspect()` proposes, the user confirms (the `Customers` lesson).
- No leaderboard-fitting (0022) — optimize the contract, not a score.
- No external data (0001); no agent hyperparameter tuning (0035).

## 6. Phasing

- **P0** (~1–2 days): `api.py` facade + `inspect()` folder semantics + `bundle.predict`
  + `Run.report()` + `examples/quickstart.py`; acceptance test = drive Rossmann
  through the new facade end-to-end as the first "outside user".
- **P1**: budget-chooser fix (0054) — decision record + implementation.
- **P2**: blend-policy successor to 0045 — decision record + implementation.
- **P3**: README/docs rewrite around the safety pitch; a *second* real dataset as an
  outside-user acceptance test.

## 7. Open questions for Ulf

1. **Scope of "few lines":** is `inspect() → confirm 3 → run` the right shape, or do
   you also want the one-call `ta.fit(...)` sugar promoted as the headline?
2. **Blend policy:** open the 0045-successor decision record now, or leave 0045 as-is
   and merely *document* the +0.0126 cost to users?
3. **Packaging:** is this headed for a published pip package (name?), or internal use?
