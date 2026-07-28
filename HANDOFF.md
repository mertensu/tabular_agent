# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.

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
