# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.

## 2026-07-30 (latest+1) — Block-level feedback: what the agent needs to iterate

**Machine:** MacBook Air | **Branch:** master

**Read `docs/feature_attribution_and_agent_iteration.md` first** — it records a long
discussion with Ulf whose conclusions are not obvious and which *reversed* two earlier
positions. Summary:

- Measuring every link would still not answer "what is this feature worth": a chain
  gives marginal-given-everything-before, and credit is order-dependent (redundancy
  inflates whoever goes first; interaction punishes isolation). That is the Shapley
  argument, arrived at from the data side → §18 SHAP is the tool for the *explanation*
  question, and the wrong tool for the *search* question.
- **Ulf's reframe, which set the direction:** it is less about final credit, more about
  the agent iterating well. Search guidance only has to beat nothing — and nothing is
  what it got (one scalar per experiment after adding 16 columns).
- **Ulf: "it could build a feature on top of parent giving no improvement, but the two
  together help."** Correct, and it means greedy per-feature measurement actively
  misleads. **This reverses my earlier "kitchen sink = noise" criticism**: a big batch is
  the only way interacting pairs arrive in the same fit. Real trade — one-at-a-time is
  attributable but interaction-blind; batch is the reverse.

**Built:**
- **`feature_blocks` in every training result** — importance share summed per *tool call*
  (`journal.blocks` regroups by `new_columns`; no new bookkeeping). Never per column: a
  feature that only pays off next to another scores zero alone, and per-column feedback
  is how it gets pruned before its partner exists.
- **`vs_parent_table`** — gain over the best experiment on the parent table, **`None` if
  the parent was never trained**. Deliberate: it makes measuring the previous step *pay
  off* instead of being a rule. Same shape as the advisory that moved the agent off t6
  without ever refusing it.
- **`how_to_read_blocks`** ships with every result and the wording is load-bearing: zero
  share = "no evidence it helped **in this feature set**", explicitly *not* "useless".
  Nothing auto-prunes. Tables are immutable, so abandoning a branch is free — the agent
  is told that too.
- **SYSTEM_PROMPT steer: one tool call = one idea** (dates in one, ratios in another),
  but keep plausibly-interacting features together. A steer, never enforced — refusing a
  mixed transform would be wrong.
- Notebook `step4` extended to 43 cells (new §10), all code cells smoke-tested.
  **Tests: 156 passed.**

**Verified:** a dates block scores `share=0.298`, a squares block `share=0.000`,
`unused=2`, `vs_parent_table=0.0` — the exact signal that was missing.

**Next (sequenced, Ulf agreed):**
1. `blend_models` + `predict`.
2. **Residual feedback — the missing arc.** Mapped our loop against the CoreWeave post's
   (inspect → hypothesise → build → train → score(OOF) → decide): five of six arcs exist.
   The gap is **score(OOF) → hypothesise** — we write `oof.parquet` and never show it to
   the agent, so it hypothesises from column names alone. That is why it reaches for
   `price_sq`. Needs a tool + a summary format that does not leak the target back.
3. §19 compaction; `max_tables`; exclusive-create id claim.

**Caveat worth keeping:** that post's "60 cycles overnight compounds" assumes each cycle
carries information. Our data: 19 cycles found the same score as 8. Better feedback per
cycle beats more cycles — which is the argument for (2) above.

## 2026-07-30 (latest) — Stop rule reworked, attribution gaps, and a real leakage hole

**Machine:** MacBook Air
**Branch:** master

### 1. `pd.qcut` — the guarantee had a hole (the important one)

A live run wrote `pd.qcut(df["price"], 10)` inside a good 60-line transform. **Proved
the skew:** loaded the shipped bundle, predicted the same 5 rows batched with 800 and
alone — different predictions. `qcut` takes deciles of whatever frame it is handed.

The step-3 notebook claimed the sandbox's import block means the code "cannot learn a
parameter". **That was wrong.** Blocking sklearn does nothing against `qcut`,
`rank(pct=True)`, `x / x.mean()` — pure pandas, all batch-dependent, and only the
*code* is stored, so replay recomputes them from whatever arrives.

Blacklisting calls is unwinnable (any aggregation is one), so `transform.unstable_columns`
tests the **property**: re-run the code over a random 25% subset, compare rows present in
both, reject columns that disagree. Costs one extra sandbox call at fit time, nothing at
predict (`check_stability=False` on the replay path). Verified against the real offending
code — flags exactly `['area_bin', 'price_bin']`, passes the other ~50 columns.
Docstrings in `transform.py` / `tools.py` and the step-3 notebook all corrected.

### 2. Stop rule reworked — **Ulf's call: "let it iterate and search"**

`patience=4` punished exploration. Two-level rule now:
- **`patience=10`** (run level, generous) — eight ideas that fail to find the ninth is
  the job, not a failing search.
- **`table_patience=3`** (per feature table, tight) — experiments on one table can only
  differ by model/hyper-parameters. Refuses **that table**, not the run, and names
  `transform_features`/`fit_encoding` as the next move. A redirect costs nothing; a stop
  costs the answer. Escalates to the run level only if the same dead table is demanded
  `max_refusals` times.

**This settles the deferred `SYSTEM_PROMPT` question: "features over tuning" does NOT go
in the prompt — it is now structural, in the tool.**

### 3. Attribution gaps (Ulf spotted this)

`t5` was built and immediately extended into `t6`; only `t6` was trained. So the jump
`0.316 → 0.2741` cannot be split between t5's columns and t6's — no measurement in
between. **Ulf chose option 2 of 3: report it, don't constrain the search.**
`journal.attribution()` walks the winner's ancestry; `finalize_run` returns
`unmeasured_tables`; the report gets a "How the winning feature set was built" section
plus an explicit **Attribution gap** warning. Rejected option 1 (refuse to stack on an
untrained table) for the same reason patience is generous.

**Tests: 152 passed.** Notebooks `step3` (45 cells) and `step4` (38 cells) both
regenerated and fully smoke-tested.

### Live run (deepseek, apartments) — three-way comparison

| rule | experiments | best |
|---|---|---|
| uncapped | 11 | 0.2718 |
| patience=4 | 8 | 0.2743 |
| patience=10 + table redirect | 19 | 0.2719 |

**All three found the same score** (differences ≪ fold spread 0.02). Two findings:
- **The structural refusal never fired — `refusals: 0`.** t6 hit its limit at e12, the
  *advisory* said "this table is used up", and the model went and built new features on
  its own. The guidance layer did the work; enforcement was the backstop that wasn't
  needed. Best evidence yet for the two-layer design.
- **The loosening bought nothing here: 19 experiments vs 8 for the same answer.** Open
  question, not a settled result — this dataset's signal is nearly all in one column, so
  there was never a ninth idea waiting. Needs a second dataset before judging.

**Next:**
- `blend_models` + `predict`, then §19 compaction.
- **Known gap (Ulf raised, deferred):** nothing caps table *building* — only trainings
  (`max_experiments=25`). A model could call `transform_features` forever without
  training. Not observed; the fix is `max_tables` on `StopPolicy`.
- Exclusive-create id claim (other half of the concurrency fix).

**Gotchas:**
- The model's summary claimed "city one-hot encoding" and "target encoding of listed_at
  captured it" — it did hand-roll one-hot in transform code (`is_berlin` etc., which is
  fine, stateless). Prose still drifts from the journal; the journal is the record.
- `attribution()` picks best-per-table with `max`/`min` off the metric — don't hardcode.

## 2026-07-30 (later) — Harness step 4: the stop rule (§19.3)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`harness/stop.py`** — `StopPolicy` (patience=4, min_gain_ratio=0.25,
  max_experiments=25, max_refusals=3) + `evaluate(leaderboard, metric, policy)`, a
  **pure function of the leaderboard** so the interesting histories can be unit-tested
  instead of trained. On `RunSession(..., stop_policy=...)`, since it's this run's
  budget, not a library constant.
- **The yardstick is `fold_spread`, not an absolute threshold.** A gain counts only if
  it beats `0.25 x` the incumbent's own fold-to-fold wobble. An absolute number would
  need tuning per metric and per dataset and fails silently. The incumbent tracks the
  *true best*, so a creeping series of sub-noise gains can't be laundered into progress
  by moving the comparison point.
- **Patience > 1 on purpose** — a run branches; a dip before a real improvement is the
  search working. This is early stopping, same trade-off.
- **Two layers.** Advisory: every train result carries `since_improvement`,
  `experiments_remaining` and a note, so a cooperative model never gets refused.
  Structural: past patience, `train_tabular_model` returns `{"trained": false, ...}`.
- **A refusal is a return value, not an exception** — raising would eat `TOOL_RETRIES`,
  and exhausting that ends the run and discards every completed experiment. Being asked
  to train once more is not a malformed argument.
- **Escalation:** refusal → (at `max_refusals`) **the harness finalizes the run itself**
  and returns the bundle → after that, `RuntimeError`. The bundle is the deliverable; a
  loop that ends without one is the real failure.
- `finalize_run`'s auto-baseline calls with `enforce_stop=False` — the harness training
  on its own behalf. Without it, finalizing a stalled run crashes on a missing baseline.
- **Tests: 134 passed** (8 new in `test_stop.py`, 5 in `test_harness_tools.py`).
- **Notebook `phase3_harness_step4_stopping.ipynb`** (22 cells, unexecuted, all 8 code
  cells smoke-tested in order).

**Live comparison, same prompt/model/dataset:**

| | experiments | best | refusals |
|---|---|---|---|
| uncapped (step 3) | 11 | 0.2718 | — |
| stop rule | **8** | 0.2743 | 1 |

Gave up `0.0025` against a fold spread of `0.0209` — i.e. nothing measurable — for three
fewer experiments. **The agent complied with the first refusal and finalized.**

**Next:**
- `blend_models` + `predict` — the last two tools.
- §19 **compaction**: the journal grows without bound and a long run will outlive the
  context it's read into. This is the remaining half of §19.
- Exclusive-create id claim (the other half of the concurrency fix).
- **Ulf's call, now deferred to after a few more runs:** "prefer feature work over
  hyperparameter tuning" in `SYSTEM_PROMPT`. His decision was to see whether the tuning
  spiral still happens now the run is capped — keeps the two effects separable. First
  capped run still spent e5–e8 on tuning, so the answer isn't in yet.

**Gotchas:**
- The advisory describes the run *after* the experiment just finished; the refusal is
  decided *before* the next one starts. So a model is always told once before being
  refused — don't read the off-by-one as a bug.
- The stop rule says the *search* stopped moving. It knows nothing about whether the
  features are exhausted, and a leaky run converges fast and stops early, pleased.

## 2026-07-30 — Harness step 3: `transform_features` tool + the concurrency fix

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`transform_features` is now a tool** (5th on the surface) —
  `tools.transform_features(session, code, table, timeout)`. Runs the agent's
  `df -> df` code in the existing sandbox, writes `tables/<id>.parquet` + the
  lineage sidecar, records the table in the journal. Chains onto encoded tables
  (`t3` lineage = `["transform", "encode", "transform"]`, replayed in that order).
- **The design call: the target column is dropped before the code sees it.** Two
  reasons and the *second* is the one that decides it: (1) a derived column can't
  be a function of the target if the target was never in the frame — the sandbox's
  import block does nothing against `df["cheat"] = df["y"]`, that's pure pandas;
  (2) **there is no target at predict time**, so code referencing it would crash or
  silently compute something else when the pipeline replays. Same frame both
  places or "re-run the code at predict" is a lie. A `KeyError: 'y'` from the child
  is rewritten into a message that says which of those happened — a bare `'y'`
  reads like a typo and the model just tries again.
- **The lineage stores the code, not the columns.** Feature *pipeline*, not feature
  *store*: a new row goes through identical arithmetic. Journal table rows carry
  the code too, so a resuming agent sees what was *tried*, not just what landed.
- **`SandboxError` joined `ValueError`/`FileNotFoundError` as retryable** in
  `_retryable`. Code the model wrote badly is an argument problem — the argument
  just happens to be a program, and those sandbox messages are already written to
  teach ("use fit_encoding instead").
- **`max_concurrency=1`** (the cheap concurrency fix from the last entry) — asserted
  in a test via `agent._concurrency_limiter.max_running`. **Still only half a fix:**
  it's per-`Agent`, so two processes on one run dir need the exclusive-create id claim.
- **Tests: 121 passed** (+8 tool-layer, +2 agent; tool-surface set now 5).
- **Notebook `phase3_harness_step3_transforms.ipynb`** (40 cells, ships unexecuted,
  all 15 smoke-tested code cells run in order + a real deepseek run verified separately).

**Live run** (deepseek via Requesty, apartments dataset): baseline `0.6903` → best
`0.2718`. It parsed `listed_at` on its own and used the tool hard. **0 fatal errors,
no parallel tool calls observed in any single message.**

**Next:**
- `blend_models` and `predict` — the last two tools.
- §19 compaction + harness-owned stop. The live run did **11 experiments**; the last
  six moved the score by ~0.003. It doesn't know when to stop, and nothing makes it.
- Exclusive-create id claim + write-then-rename (the other half of the concurrency fix).
- Still Ulf's call: put "prefer feature work over hyperparameter tuning" into `SYSTEM_PROMPT`?

**Gotchas:**
- **A GBDT recovers `price / area` on its own** — the first draft of the notebook was
  built on "ratios are hard for trees" and the numbers refused to cooperate (raw beat
  the derived features). Rebuilt around a **datetime column**, where the raw model is
  level with the baseline because the information is *absent*, not merely awkward.
  Worth remembering before claiming a feature helps: check it.
- The model writes **kitchen-sink transforms** — ~20 columns including `price_sq`,
  `area_sq`, `price_x_rooms`. Free noise for a tree (it can already split anywhere on
  `price`). Safe, not smart; another argument for the stop rule.
- The notebooks in the working tree are modified (executed outputs from Ulf's runs) and
  were **not** committed — the convention is that they ship unexecuted.

## 2026-07-29 (later) — Live-run hardening: everything the first real agent runs broke

Four live runs on `tensorx/deepseek-v4-flash` (Requesty) found five bugs the unit tests
could not, because they were all *environment* or *agent-behaviour* bugs.

**Fixed:**
- **`config.env` wasn't found from `notebooks/`** — a notebook's CWD is its own folder. `load_api_key` now walks up from CWD *and* from the package. My smoke test ran from the repo root, which is exactly why it passed.
- **`run_sync` in Jupyter** — "This event loop is already running". Notebook cells must `await agent.run(...)`; scripts want `run_sync`. Noted in the notebook.
- **Ordinal encoding was impossible to request.** `inspect_data` reported `n_unique: 3` and never the values, so the agent could not state a level order it had never seen — it sent an empty list and the run died. Non-numeric columns with ≤20 uniques now list their `categories`. **After this fix the agent chose `ordinal` for `size` on the very next run, having gone `frequency` three runs running.** It wasn't reluctance, it was missing data.
- **One bad argument killed a whole run** (8 completed experiments discarded). Tool `ValueError`/`FileNotFoundError` are now raised as `ModelRetry` → the agent self-corrects in one turn. Other exceptions still stop the run loudly.
- **…which then died at PydanticAI's default budget of 1 retry.** `TOOL_RETRIES = 3` on the `Agent`, and unknown table paths now list the tables that exist instead of failing deeper down.
- **`_make_model` param collisions** (see previous entry) — same class of bug: agent-supplied input meeting our defaults.

**Result — "don't tune hyperparameters, work on features" beats tuning, on this data:**
`0.2825` in 5–6 experiments (untuned) vs `0.2870` in 12 (tuned). The agent's own ablation:
CatBoost on the raw table scores `0.6282` against a `0.6905` baseline — essentially all the
signal is in the encoding. It also **branched** exactly as the lineage design assumes:
`t1 (ordinal size)` → `t2 (freq city)` **and** → `t3 (target city)`, holding `t1` fixed to
compare two encoders. Reproducible across two runs, 0 retries.

**Next:**
- **Concurrency (real, unfixed):** `next_table_id`/`next_experiment_id` read-then-write without a lock. A model can put several tool calls in **one** message, and PydanticAI runs those concurrently (sync tools go to worker threads), so two `fit_encoding` calls can both claim `t1` and write the same parquet — one corrupt-parquet crash seen, consistent with this, but **not reproduced on demand**; it is not confirmed that this model emits parallel calls. Worth checking first: count tool-call parts per message in `result.all_messages()`; >1 in a message means it happens. `e1` collisions would overwrite artifacts *silently*, which is worse than the crash.
  Two fixes, do both eventually:
  1. `Agent(..., max_concurrency=1)` — one line, serializes tool execution, kills the whole class now. Costs parallelism we don't currently benefit from (each tool takes seconds anyway). **Unverified — no live run since finding it.**
  2. Exclusive-create id claim + write-then-rename. The real fix: `max_concurrency` is per-Agent, so it cannot protect a run dir touched by two processes (a resumed run alongside the original).
- Ulf's call to make: put "prefer feature work over hyperparameter tuning" into `SYSTEM_PROMPT`? The evidence above supports it, but this dataset is friendly to that conclusion.
- Then `transform_features`.

**Gotchas:**
- The model's prose lies in small ways — it reported the ordinal codes as "small=1, medium=2, large=3" when they are 0/1/2. The journal and the lineage sidecar are the record; the summary is a claim.
- Restart the kernel after pulling, or the notebook keeps the old module.

## 2026-07-29 — Harness step 2: `fit_encoding` tool + table lineage + non-Anthropic model

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`fit_encoding` is now a tool** (4th on the agent's surface) — `tools.fit_encoding(session, columns, method, table, categories, smoothing)`. Writes `tables/<id>.parquet` (source table + encoded columns), records the table in the journal (`next_table_id`/`append_table`), returns paths + column names only. Refuses to encode the target. Chains: pass a previous tool's `table` back in.
- **Table lineage — the real design call.** Each derived table gets a `tables/<id>.pipeline.json` sidecar holding the fitted encoders that produced it. `finalize_run` now loads the lineage of *the table the winning model trained on* instead of building an empty `Pipeline`. Per-table rather than per-run because a run branches (t1, t2, …) and a run-wide pipeline would be a lie about at least one model. Generalizes to `transform_features` (`Pipeline.add_transform` already exists).
- **Provider-agnostic model wiring** — `agent.gateway_model(model_id)` → `OpenAIChatModel` over an OpenAI-compatible gateway (Requesty, `https://router.requesty.ai/v1`); `build_agent`/`run` now accept a model string *or* a `Model` object. `load_api_key` reads `$REQUESTY_API_KEY` then `config.env`. **`DEFAULT_GATEWAY_MODEL = "tensorx/deepseek-v4-flash"`.** `.gitignore` now covers `*.env`, `.secrets/`, `*.key` — `config.env` is untracked and was never committed.
- **Bug found by the live run** (`models._make_model`): caller params collided with our keyword defaults — `TypeError: got multiple values` for lgbm/xgb, and CatBoost *rejects* two spellings of one setting (`iterations` vs `n_estimators`), which killed a whole agent run. Now merged through `_resolve_params` with a CatBoost alias table. 2 tests.
- **Notebook `phase3_harness_step2_encoding.ipynb`** (40 cells, ships unexecuted, all 19 code cells smoke-tested in order **including a real agent run on deepseek**). Covers the label-encoding trap, the three encoders, OOF-vs-full-fit, lineage, and journal-vs-model-prose.
- **Tests: 103 passed** (7 new in `test_harness_tools.py`, tool-surface set updated to 4).

**Next:**
- `transform_features` (sandboxed free code) — the lineage step type already exists, so this is the same adapter pattern plus `add_transform`.
- Then `blend_models`, `predict`, `journal_append`; §19 compaction + harness-owned stop (the live run did 12 experiments long past the point of gains).

**Gotchas:**
- The live agent frequency-encoded `size` instead of using `ordinal` — safe but suboptimal. The harness enforces *can't do the unsafe thing*, not *does the smart thing*. Don't confuse the two when reading a run.
- `result.usage` is a property in pydantic-ai 2.19, not a method.
- Requesty rewrites the model id in its response (`tensorx/...` → `deepseek/...`); that's the router, not a misconfiguration.

## 2026-07-29 — `OrdinalEncoder`: ordered categoricals as a first-class encoder

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`OrdinalEncoder` in `encoders.py`** — `OrdinalEncoder(categories={"size": ["small","medium","large"]}, unknown=-1.0)`, emits `<col>_ord`, registered in `ENCODER_CLASSES` so it round-trips through `Pipeline`/`save_bundle` JSON. Uses only `X` → no folds, no leakage (same class of encoder as `FrequencyEncoder`).
  - **Order is a required parameter, never inferred.** That's the whole point: `pd.factorize` orders by first appearance, `.cat.codes` lexicographically — both encode `Large/Medium/Small` as `0/1/2` and look fine doing it.
  - **Strict at fit, lenient at predict:** a value in the fit data missing from the declared order raises (typo / forgotten level); a category first seen at predict maps to `unknown = -1`, below every level, so one split isolates it. Duplicate levels also raise.
- **`fit_encoding(..., method="ordinal", categories={...})`** — third method alongside `target`/`frequency`; errors if `categories` is absent or doesn't cover `columns`.
- **`inspect_data`'s note** now steers the agent: ordinal when the categories are genuinely ordered, frequency/target when nominal.
- **Tests:** 8 new in `test_fit_encoding.py` + a bundle round-trip in `test_pipeline.py` (reversed row order + a level absent from the predict frame → codes unchanged). **Full suite: 95 passed.**

**Why (Ulf's prompt):** he'd been label-encoding by hand and asked whether that was wrong. It's not leaky — the real risk is that an ad-hoc `factorize`/`cat.codes` mapping isn't *stateful*, so train and predict can silently disagree, and nothing lands in the pipeline. Ordered columns being common, the fix was a real encoder rather than a warning.

**Next:**
- Unchanged from below: `fit_encoding` (now with three methods) still needs its **harness tool wrapper** in `tools.py`, plus `transform_features`, `blend_models`, `predict`.

## 2026-07-28 — Harness step 1: file-based tool layer + PydanticAI loop (thin vertical slice)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **New package `src/tabular_agent/harness/`** — the constrained harness from the pivot.
  - `session.py` — `RunSession(run_dir, data_path, spec)`. Holds *pointers only*; the run dir is the state (§4). Resolves+caches the task spec (task type/metric auto-detected once, then pinned so leaderboard scores stay comparable).
  - `tools.py` — **the file-based adapter layer** (this was the real design call, see below): `inspect_data`, `train_tabular_model`, `finalize_run`. Paths in → artifacts into `run_dir/` → compact JSON summary + paths out. Plain functions, **no LLM dependency** → unit-testable and hand-drivable.
  - `journal.py` — `run_journal.json` (§19), minimal: `task`/`baseline`/`best`/`leaderboard`/`log`. `train_tabular_model` appends a leaderboard row **as a side effect of training** (§19.2) — can't be skipped. `best` re-derived from the whole leaderboard each append; baseline tracked separately so it's never shipped as the deliverable. `append_note` exists but isn't wired to a tool yet.
  - `instructions.py` — system prompt (§12 revision): only *order* + how to read results; no "please don't train outside the tool" guidance, because there's no other tool.
  - `agent.py` — thin PydanticAI loop. `DEFAULT_MODEL = "anthropic:claude-opus-5"`, `defer_model_check=True`, `deps_type=Deps(session)`. Registers exactly the three functions.
- **Tests:** `test_harness_tools.py` (10) + `test_harness_agent.py` (3, incl. **tool surface == exactly those 3** — fails loudly if anyone adds a `read_file`). **Full suite: 87 passed.**
- **Notebook** `notebooks/phase3_harness_step1.ipynb` (38 cells, ships unexecuted, all 15 code cells smoke-tested in order) — drives the whole pipeline by hand *first*, LLM only at the end. Live-API cell is commented out.
- `pydantic-ai` 2.19.0 added as a hard dep (`uv add pydantic-ai`).

**Design call made this session (Ulf's, from a 3-option question):**
- Handoff said "register exactly the existing library functions as tools" — **not literally possible**: every library fn takes/returns an in-memory DataFrame, while §4 demands stateless file-based tools. Chose **file-based paths** over the cheaper in-memory-deps option, because in-memory state dies with the process → §19 resume would have forced a rewrite. Hence `tools.py` as a real adapter layer.
- Scope: **thin vertical slice** (inspect → train → finalize) over all-7-at-once, so any adapter-shape mistake is made once.

**Next:**
- **The other four tools** — `transform_features`, `fit_encoding`, `blend_models`, `predict`. Same adapter pattern, now proven. This is where non-numeric columns (currently dropped-with-a-warning) become usable.
- Then: `journal_append` for the agent's note layer; §19 compaction + harness-owned stop; multi-provider model config.

**Gotchas:**
- `train_tabular_model` currently **drops non-numeric columns** and reports them in `dropped_non_numeric` — deliberate (a silent cast is how train/serve skew starts), and it's the hook for the `fit_encoding` slice.
- `finalize_run` **auto-trains a baseline** if the run never did, so the report can always answer "did this beat doing nothing".
- Bundles ship `final_model` (refit on all data) with weight 1.0 — matches the existing phase-1 convention in `test_pipeline.py`.
- pydantic-ai 2.19.0's `KnownModelName` literal **doesn't list `claude-opus-5` yet** (lags releases); the string passes through to the provider fine — that's what `defer_model_check=True` is for.
- Tool-surface assertions reach into `agent._function_toolset.tools` (private API). If pydantic-ai churns, that's the line that breaks.

## 2026-07-28 — Architecture pivot: owned constrained harness (not MCP-for-others)

**Machine:** MacBook Air
**Branch:** master

**Done (spec + design only — no code yet):**
- **The pivot (Ulf's call):** the "leakage-safe *by construction*" guarantee only holds if **our tools are the only path to the data**. A general-purpose MCP client (Claude Code's bash + scratchpad) makes free code the path of least resistance → around the sandbox/OOF discipline. So: **we own a thin, constrained harness; only the LLM is swappable.** MCP demoted from "the product / distribution surface" to **optional internal transport**.
- **Harness framework decision: PydanticAI** (leaning) — model-agnostic (satisfies "LLM free"), typed tools (matches `fit_encoding` et al., no hidden tools), self-managed message history (needed for §19 compaction), headless/embeddable. Explicitly **not** Claude/OpenAI agent SDKs (model lock-in) and **not** smolagents/code-first (the free-code escape *as a feature* = anti-fit). TUI deferred (interface is Axis B, not where the value is).
- **Spec updated** (`docs/superpowers/specs/2026-07-21-...md`), all as dated `Revision (2026-07-28)` blocks so original reasoning stays visible: §3 (diagram + note), §12 (instructions → harness system-prompt, boundary enforced by tool-set not guidance), §13 (ship the harness, `pydantic-ai` hard dep / `mcp` optional), §15 (decision #2 struck + superseded by rows 2′/11-13), §16 (autonomous loop-runner promoted to v1 core — the harness *is* that runner). Commit `dfc6e7b`.
- Also this session: docs `oof_vs_final_encoder.md` (why `final_enc` isn't leakage) `f1f31dd`; **spec §19** harness & resumable memory (`run_journal` = tool-written leaderboard + thin agent notes; compact on token budget; harness owns stop) `78a7fdd`.

**Next:**
- **Build the harness, step 1:** a thin PydanticAI loop that registers exactly the existing library functions (`inspect`, `transform_features`, `fit_encoding`, `train_cv`, `blend`, `predict`, `finalize`) as its *only* tools — headless, one model to start. Validates the constrained-tool-surface end-to-end before adding model-abstraction or the §19 journal.
- Then: `run_journal` writes (train side-effect + `journal_append`), §19 compaction/stop, then multi-provider model config.

**Gotchas / open:**
- Spec **title still says "MCP-Server für …"** — now stale; retitle later (didn't, to avoid breaking refs).
- Decision still a *leaning*, not locked: PydanticAI is young/moving fast; fine, but revisit if API churn bites.
- Phase-2 core lib (transform_features/fit_encoding) is done + committed; the pivot doesn't touch it — sandbox just gets *promoted* to "the agent's scratchpad".

## 2026-07-25 — Phase 2: transform_features + fit_encoding (+ notebook)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`transform_features` (§5.1)** — the stateless `df→df` escape hatch. Reuses the same sandbox: extended `sandbox.py` with a shared `_run` driver + `run_transform` (mode `"transform"`) that returns the **whole** DataFrame via pickle (not JSON-capped like `run_readonly`). Contract enforced in-child: must return a DataFrame **with the same row index** (no reorder/drop/add rows) → teaching errors otherwise. New thin wrapper `transform.py` (`transform_features(df, code) -> (df, summary)`), symmetric to `explore.py`.
- **Pipeline persistence** — `Pipeline.add_transform(code)` stores a `{"type":"transform","code":...}` step; `Pipeline.transform` re-runs it via the sandbox at predict (lazy import so predict doesn't need the sandbox unless a code step exists). Only the **code** is stored, never the produced columns → reproducible, no train/serve skew.
- **`fit_encoding` (§5.2)** — structured stateful encoding tool in `encoders.py`. Dispatches `method="target"` (needs `y`+`folds`, returns `(oof_features, final_encoder)` via existing `oof_target_encode`) vs `method="frequency"` (new `FrequencyEncoder`, X-only so no OOF, unknown category → 0). Both registered in `ENCODER_CLASSES` and serializable.
- **Tests:** `tests/test_transform.py` (7) + `tests/test_fit_encoding.py` (8). **Full suite: 74 passed** (libomp now installed — the old 7 lgbm/xgboost failures are gone).
- **Notebook** `notebooks/phase2_transforms_and_encoding.ipynb` — phase-style, low-jargon, covers **both** tools + an end-to-end transform+encode→train→predict. Ships **unexecuted**; every cell smoke-tested in order.

**Next:**
- **The MCP server itself (§3)** — none exists yet. Wrap the primitives (`inspect_data`, `transform_features`, `fit_encoding`, `train_tabular_model`, `blend_models`, `predict`, `finalize_run`) as the stateless, file-based tool surface + server-instructions + OTel spans. This is now the big remaining v1 piece.
- Optional: §18 post-hoc SHAP explainability.

**Gotchas:**
- Task-type string is **`"binary"`** (not `"classification"`) for `train_cv`/`make_folds`/`Pipeline`; `TrainResult` exposes `fold_metrics` (list), no `cv_score` — mean it yourself. (Both bit me while writing the notebook.)
- `libomp` fixed via `brew install libomp`; the `LDFLAGS`/`CPPFLAGS` export hints Homebrew prints are **compile-time only** and not needed — the pre-built lgbm/xgboost wheels just needed `libomp.dylib` present.

## 2026-07-24 — Phase 2 started: sandbox + read-only exploration tool

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **Design change (Ulf's call), recorded as §17.1 in the spec:** dropped both original §17 items — enriching `profile_dataframe` with skew/quantiles, and a narrow `group_stats` tool — in favour of **one general read-only Python tool** next to `transform_features`. Reverted the uncommitted skew/quantile work in `inspect.py`/`test_inspect.py`; profile stays a cheap structural fingerprint (dtype/missing/unique).
- **`src/tabular_agent/sandbox.py`** — the §6 default tier: fresh subprocess per call, import whitelist (blocks `lightgbm`/`xgboost`/`catboost`/`sklearn` with *teaching* errors → structurally can't train; allows pandas/numpy/scipy + stdlib), CPU/mem limits, own temp cwd, wall-clock timeout, size-capped JSON result contract.
- **`src/tabular_agent/explore.py`** — `explore_dataframe(df, code)`: the read-only tool. `explore(df) → summary`, shown to the agent then discarded. Non-persistence + size cap = the structural guard that keeps the §17.2 leakage backdoor closed (can't hand-build a target-leaky feature).
- **Tests:** `tests/test_sandbox.py` (11 total across both) + `tests/test_explore.py`. All pass.
- **Notebook `notebooks/phase2_sandbox_and_exploration.ipynb`** — phase1-style walkthrough, low-jargon. Ships **unexecuted** (run cell-by-cell) but every cell smoke-tested.

**Next:**
- `transform_features` (§5.1): same sandbox, `df→df`, persisted as a pipeline step, re-run at predict. Reuses `sandbox.py` — just a second entry point + contract (return a DataFrame) + pipeline-step persistence.
- Then `fit_encoding` as a structured tool; §18 SHAP explainability is still an *option*.

**Gotchas:**
- **libomp still missing on this Mac** → 7 pre-existing failures in `test_models.py`/`test_pipeline.py` (lightgbm/xgboost `dlopen`). Unrelated to Phase 2 work; fix with `brew install libomp`. Everything else (49 tests) passes.
- pytest is the `dev` extra: `uv run python -m pytest` (after `uv sync --extra dev`).

## 2026-07-23 — explainability option added to Phase-2 spec

**Machine:** MacBook Air
**Branch:** master

**Done:**
- Added §18 to `docs/superpowers/specs/2026-07-21-tabular-agent-mcp-design.md`: per-prediction explainability ("influencing factors") via post-hoc grouped TreeSHAP. Flagged as an **option**, not a hard v1 requirement (Ulf's call).
- Also wrote `docs/target_encoding_leakage_insight.md` (who gets lied to: model vs. metric) — companion to the tutorial notebook.

**Key design decisions (from a long discussion with Ulf):**
- Post-hoc SHAP over glass-box (EBM/GAM): decouples "strongest model" from "explain it"; no accuracy sacrifice, only explanation-exactness-under-interactions, partly recovered via opt-in pairwise interaction values.
- Attribution unit = **raw input feature** (derived columns like one-hot / `_te` folded back via encoder metadata). Multi-feature "factors" need domain knowledge → out of scope.
- Native booster `pred_contrib` → no `shap` dependency. Composes linearly over the blend, reconciles exactly to the served prediction.
- Present in target space (additive vs. multiplicative for log-target). Label output "model reasoning, not causation." Optional declared monotone constraints attack "weird prediction" complaints at the source.
- Trustworthy only on leakage-clean encoding — SHAP is faithful to leakage-inflated feature importance.

**Context:** Ulf's prior company did demand forecasting with a block-additive neural GAM (`exp(Σ ReLU-MLP blocks)`); influencing-factor attribution was per-block. tabular_agent can't hand-design blocks (genre-agnostic), hence per-raw-feature SHAP.

**Next:** Phase 2 still open (profile stats §17, `group_stats` §17, now explainability §18).

## 2026-07-23 — target-encoding tutorial notebook

**Machine:** MacBook Air
**Branch:** master

**Done:**
- New `notebooks/target_encoding_step_by_step.ipynb` (unexecuted on purpose — Ulf wants to run it cell by cell). 15-row toy dataset; sections 1–7 = core (naive vs. OOF encoding, leak traced per row), sections 8–12 = deep dive: encoder-vs-model distinction, three schemes (A shared-fold OOF = the library, B fit-on-train = sklearn-Pipeline style, C nested CV) compared mechanically and empirically (B: honest metric but degraded model, ≈0.79 vs 0.81 AUC, 16× over-weight on a noise feature; A: ≈0.52 on pure noise = second-order optimism).
- Set up Python env on this Mac: installed uv, `uv sync` from uv.lock. All notebook cells smoke-tested.

**In progress / next:**
- Ulf is working through target encoding conceptually — found the second-order material dense; keep explanations concrete/low-jargon.
- Phase 2 (profile stats, `group_stats` tool) still open.

**Gotchas / open questions:**
- XGBoost fails to load on the MacBook Air: missing libomp (`brew install libomp`). Affects `models.py`, not the new notebook (uses sklearn's HistGradientBoosting instead).
- Jupyter is not a project dep; launch via `uv run --with jupyter jupyter lab`.

## 2026-07-22 — setup

**Machine:** MacBook Air
**Branch:** master (clean, pushed through c19ebc6)

**Done:**
- Established this handoff workflow (HANDOFF.md + CLAUDE.md convention).

**In progress / next:**
- Phase 2: profile stats + `group_stats` tool (see `docs/` Phase-2 requirements, commit c19ebc6).

**Gotchas / open questions:**
- (none yet)
