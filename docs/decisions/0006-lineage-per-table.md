# 0006 — Feature lineage per table, not per run

- **Date:** 2026-07-29
- **Status:** accepted
- **Commit:** `be6f1d9`

## The problem

The deliverable of a run is a **bundle**: something you can hand a raw row and
get a prediction from. A raw row has `city = "berlin"`, not `city_freq = 0.34`.
So the bundle must contain not just the fitted model but everything needed to
rebuild the model's features from raw input — the fitted encoders, in order.

Call that the **lineage**. The question is where to keep it.

The obvious answer is one pipeline per run: the run accumulates encoding steps,
and at the end the bundle gets all of them. That is how a hand-written script
usually works, because a hand-written script does one thing.

An agent run does not do one thing.

## Background: runs branch

A real run looks like this:

```
raw data
├── t1   raw + ordinal-encoded size
│   ├── t2   t1 + frequency-encoded city
│   └── t3   t1 + target-encoded city
└── t4   raw + date features
```

The agent built `t1`, then tried two different encoders for `city` on top of it —
holding `t1` fixed so the comparison is controlled — and separately tried a
different idea from scratch. It then trained models on several of these.

A single run-wide pipeline cannot describe that. If it holds the ordinal encoder,
the frequency encoder *and* the target encoder, then it is wrong about every
model: the one trained on `t2` never saw target-encoded `city`, and the one
trained on `t3` never saw the frequency encoding. Shipping that pipeline would
compute columns the model was not trained on and feed them to it anyway.

## The decision

**Each derived table gets its own lineage**, stored as a sidecar next to it:
`tables/t2.parquet` is accompanied by `tables/t2.pipeline.json`, containing the
ordered steps that produce `t2`'s columns from raw input.

Building a new table reads its source table's lineage, appends one step, and
writes the result. So a table's sidecar always contains the *whole* chain behind
it, and any table can be packaged on its own.

`finalize_run` then looks up which table the winning model was trained on, loads
*that* table's lineage, and makes it the bundle's pipeline.

## Alternatives considered

**One pipeline per run.** Rejected: it is a lie about at least one model as soon
as the run branches, which is the normal case.

**Reconstruct the chain from the journal at finalize time.** The journal does
record each table's source, so this is possible. Rejected as more fragile for no
benefit — it puts the reconstruction logic in the one place that must not get it
wrong, and it would have to reload and re-fit the encoders anyway.

## Consequences

**Good**

- The agent can branch freely, which is exactly the behaviour we want and which
  live runs do use. Comparing two encoders on a fixed base is a *controlled*
  comparison, and this is what makes it expressible.
- Abandoning a branch is free. Tables are immutable and their lineage travels
  with them, so a "failed" table remains fully packageable later. That property is
  relied on by decision 0016.
- It generalised to code transforms without change: `transform_features` appends
  a `{"type": "transform", "code": ...}` step to the same structure (0010).

**Bad**

- Sidecar files duplicate their parents' steps. `t3`'s sidecar repeats
  everything in `t1`'s. Cheap in bytes, but it means the chain is stored once per
  descendant rather than once.
- Two files must stay together. Deleting a `.parquet` without its
  `.pipeline.json` leaves a table nobody can package, and nothing enforces the
  pairing.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — `_lineage_path`, `_load_lineage`;
  written by `fit_encoding` and `transform_features`, read by `finalize_run`.
- `src/tabular_agent/pipeline.py` — `Pipeline` holds the ordered steps and
  replays them in `transform()`.
- `tests/test_harness_tools.py` — a bundle is built from a table two steps deep
  and asked to predict from raw rows.
