# Multi-geometry panel (Decisions 0077–0078)

Archived from the production package during v0.2 simplification.

The panel varied train/gap/evaluation geometries and required unanimous sign
agreement. Its registered IEEE replay accepted 1 of 24 steps and would have
turned table ranking into an early stop rule. Production wiring was reverted the
same day; `panel.py` remained unwired with only its dedicated tests importing it.

- `panel.py` — historical implementation
- `test_panel.py` — historical focused tests
- decisions: [0077](../../../../docs/decisions/0077-several-disagreeing-rulers-rather-than-one-better-one.md)
  and [0078](../../../../docs/decisions/0078-the-panel-was-a-stop-rule-and-the-gain-threshold-is-the-real-lever.md)
- runnable full-system context: tag `research-reference-2026-08-21`

This code is research evidence, not a supported runtime module.
