# 0055 — A rename in the library makes every earlier artifact unreplayable

- **Date:** 2026-08-14
- **Status:** a defect, found by accident, **not yet fixed**. Recorded now because
  it was discovered four hours after it was introduced and by luck; the next one
  will not be found so cheaply. No code changed — the repair is a design decision
  and this record is the argument that one is needed.
- **Related:** [0010](0010-store-code-not-columns.md) (the code is the artifact —
  this is the boundary of that idea), [0051](0051-one-word-for-one-idea-and-a-flag-instead-of-an-age.md)
  (the rename, whose reasoning stands),
  [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md) (the check that
  caught it, behaving correctly),
  [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  (the measurement this blocked), `src/tabular_agent/encoders.py`.

## What happened

0054 needed to sweep model capacity on `t24`, v8's best Rossmann table. Every
budget came back refused, identically:

    unscored — 6 of this table's feature columns cannot exist at prediction time:
    ['Store_Sales_last', 'Store_Sales_last_age', 'Store_Sales_lag364',
     'Store_Sales_lag364_age', 'Store_Sales_lag728', 'Store_Sales_lag728_age']

The first two guesses were both wrong: it is not the session being reconstructed
badly, and it is not `Customers`. The timestamps say what it is.

| | |
|---|---|
| `_runs/rossmann_v8/tables/t24.parquet` written | **13:39** |
| commit `9879acd` — 0051, the rename — committed | **15:26** |

0051 renamed `HistoryEncoder`'s output columns from `_last` / `_age` to
`_latest` / `_latest_ago`. The stored table holds the old names. Replaying its
stored lineage through today's code produces the new ones. So the replay is
asked whether it can produce `Store_Sales_last`, it produces
`Store_Sales_latest`, and the answer is no.

**0030's check is behaving exactly as designed.** On the evidence available to it
— a table with six columns the lineage cannot produce — refusing is correct, and
the alternative (scoring anyway) is the failure 0030 exists to prevent. Nothing
here is an argument against that check.

## Why this is worse than one lost measurement

0010 says the code is the artifact: a bundle stores the transform, not the
transformed columns, so a served row goes through the same steps a trained row
did. That is right, and this is the boundary of it. **The stored lineage is not
self-contained.** It records *that* a `HistoryEncoder` ran and with what state,
but the column names it emits live in the library, so the artifact's meaning is a
function of the working tree it is replayed in.

Consequences, none of which were visible at the time of the rename:

* **Every table and bundle built before `9879acd` that used `fit_history` is now
  unreplayable at HEAD.** That is v8's history branch — the tables 0049, 0050,
  0052 and 0053 are about.
* **A shipped bundle would fail the same way**, at predict time, on a machine that
  had pulled. The failure mode is a refusal rather than a wrong number, which is
  the good version of this, but it is still a model that cannot answer.
* **v8's own results cannot be reproduced from v8's own artifacts** — which is the
  single thing the lineage machinery exists to guarantee.

The rename itself was right and 0051's reasoning is untouched: `_last` really is
ambiguous between "final" and "previous", and 0050 had just made that distinction
load-bearing. The defect is not the rename. It is that a rename *could* do this
silently, and that nothing in the repo said so.

## What was believed, and was wrong

That a decision record plus a passing test suite was enough to make a rename safe.
The suite passes at HEAD — it builds tables with today's code and replays them
with today's code, so it cannot see this class of break at all. **Nothing in the
project tests an artifact against a later version of the library**, which is
precisely the situation every stored bundle is in.

## Options, and why none is taken here

Written down so the next session does not re-derive them, and deliberately not
chosen — this record's job is to establish that the problem is real.

1. **Version the lineage.** Stamp a schema version into the pipeline JSON and
   refuse to replay across an incompatible one. Turns a confusing refusal into a
   clear one; does not make old artifacts work.
2. **Store the emitted column names in the encoder state** and have replay honour
   them rather than recomputing from a format string. Makes old artifacts work,
   at the cost of the names no longer being derivable from the code — which is a
   real cost, since it means two sources of truth for one thing.
3. **Treat emitted column names as public API** and forbid renames outright.
   Cheapest to implement, and it would have blocked 0051, which was a good change.
4. **A replay test against a checked-in artifact.** Does not prevent the break but
   makes it loud at the commit that causes it, which is the failure mode that
   actually matters — this one was found four hours later and by accident.

Option 4 is the one that is cheap and pure gain; the rest trade against each
other. That is a recommendation, not a decision.

## What it cost here

0054's second reading. The question "does capacity still pay on the table the
agent actually built" had to be answered on a *rebuild* — the pure-code transform
step of t24's pipeline, which is version-independent — instead of on t24 itself.
The rebuilt reading is good enough to carry 0054's conclusion and is labelled in
that record as what it is. But the exact number that would have said "capacity is
worth more than v8's entire feature chain" is not available, and the artifact that
would have supplied it is sitting on disk, intact and unreadable.
