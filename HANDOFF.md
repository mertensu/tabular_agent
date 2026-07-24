# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.

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
