# Research archive

This directory preserves the experiments that informed `tabular_agent`; it is
not the default product surface.

## Rules

- archive by `git mv`, never by copying and deleting provenance
- keep original basenames and record original paths/commits in `MANIFEST.csv`
- one experiment directory contains its question, script, reading, optional
  notebook and raw artifact inventory
- negative and refuted experiments are preserved
- archived code may require its historical commit and is not a supported CLI
- `_runs/`, raw competition data and fitted models remain untracked

## Layout

```text
archive/
  experiments/   decision-linked single-question probes
  runs/          whole autonomous-run recipes by dataset
  scripts/       legacy utilities and submission helpers
  notebooks/     historical curriculum and implementation slices
  readings/      tracked scalar reductions from historical runs
  handoffs/      closed coordination logs
```

The active product path is documented in [`../README.md`](../README.md) and
[`../docs/CURRENT.md`](../docs/CURRENT.md).
