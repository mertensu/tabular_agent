# Random-subset feature selection (Decision 0069)

Archived from the production package during v0.2 simplification.

The method fitted many random column subsets, ranked pair ensembles, and could
write a narrowed candidate table. It was deliberately never registered on the
agent tool surface and measured approximately zero on Rossmann, Restaurant and
House Prices. Its cost and conceptual surface were therefore not justified in
the runtime.

- `implementation_excerpt.py.txt` — exact former `harness/tools.py` block
- `test_select_features.py` — historical focused tests
- archived probe: `../../scripts/probes/probe_feature_selection.py`
- decision: [0069](../../../../docs/decisions/0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md)
- runnable full-system context: tag `research-reference-2026-08-21`

This is preserved negative research, not a supported runtime capability.
