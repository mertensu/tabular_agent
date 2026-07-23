# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.

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
