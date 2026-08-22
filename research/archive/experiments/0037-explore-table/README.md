# Experiment-level residual exploration (Decision 0037)

`explore_experiment` belonged to the earlier model-experiment search. Decision
0035 removed model training from the agent surface; `explore_table` now supplies
honest OOF residual exploration during table search.

The superseded wrapper and its dedicated tests are preserved here. Shared
`_residual_frame` and experiment lookup remain active because `explore_table` and
sealed scoring still use them.

Runnable full-system context: tag `research-reference-2026-08-21`.
