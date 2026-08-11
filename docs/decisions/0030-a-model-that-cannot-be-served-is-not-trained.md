# 0030 — A model that could never be served is refused, not scored

- **Date:** 2026-08-11
- **Status:** accepted, implemented; **validation run in flight** (`_runs/rossmann_v2`)
- **Related:** [0029](0029-the-harness-owns-the-exam.md) (whose ownership line this
  sits exactly on), [0010](0010-store-code-not-columns.md) (why the second failure
  mode exists at all), [0013](0013-refusal-as-return-value.md) (the refusal's
  shape), [0019](0019-the-leaderboard-is-a-public-leaderboard.md) (the validation
  half that confirms the leak instead of catching it), [0024](0024-the-episode-stop-is-structural.md)
  (advisory vs structural, again)

## The run that made the case

Rossmann, `_runs/rossmann_v1`. `Customers` correlates **0.895** with `Sales`, sits
in the training file, and does not exist in `test.csv` — you cannot know today's
customer count when forecasting tomorrow's sales.

The data-dictionary channel (built the same day) said so in plain words. **It
worked, and it did not matter.** At experiment 9 the agent wrote:

    # Customers is measured on the prediction day and is NOT available at predict time
    if 'Customers' in df.columns:
        df = df.drop(columns=['Customers'])

— the *guarded* form, which survives replay, without being told that mattered. It
then built an honest chain of per-store encodings taking `val_score` from 0.1698
to 0.1317.

None of it counted:

| | val_score | servable |
|---|---|---|
| shipped: `e7`, trained on `Customers` | **0.07912** | no |
| best honest: `e17` | 0.13173 | yes, verified |
| Kaggle winner, 3300 teams | 0.08932 | |

A validation score better than first place is the signature of a column nobody
else had. And because `best` ranks on `val_score` (0019), four things followed
mechanically: the leaked model stayed the incumbent; ten genuine improvements were
each recorded as *no improvement* because none beat 0.07912; the run stopped early
on patience at experiment 17 with half its budget unspent; and `finalize_run`
bundled a model that raises `KeyError` on the first real prediction.

**Every mechanism behaved exactly as specified.** No guard was violated. The
agent's own closing summary named the problem — *"the run's recorded best is an
artifact of that leak"* — and it had no way to act on it, because nothing the agent
can call decides what `best` points at.

That is the inversion worth recording: the reasoning layer diagnosed a defect the
structural layer is blind to.

## The decision

**`RunSession(serving_columns=[...])`: what will exist when a prediction is asked
for. `inspect_data` reports what is missing from it, and `train_tabular_model`
refuses — before the fit — any model whose features cannot be rebuilt from those
columns.**

`None` is the default and means "assume this table is its own serving format",
which is right for most datasets and wrong for every one where an outcome was
recorded beside the target.

**The check is a replay, not a name comparison**, and it has to be. Almost no
feature is a raw column: `Store_te`, `Year` and `CompetitionOpenMonths` are all
derived, and none of them appears in a serving schema either. So it builds a
serving-shaped probe — raw rows minus the columns that will not exist — and runs
the table's lineage over it, which is what `predict_bundle` will do later.

One mechanism, **two** failure modes, and the second is why names would not have
been enough:

1. the model trains on a column that will not exist — the leak;
2. the model does *not* use the column, but a stored transform **references** it,
   and per 0010 the code is the artifact and gets re-run on rows that never had
   it. `df.drop(columns=["X"])` fails this; `errors="ignore"` survives.

Verified on the real data: training on the raw table is refused, an unguarded drop
is refused with a message naming the guard, a guarded drop trains, and a table
*chained* on top of a guarded drop trains.

## Why this and not the alternatives

**Not advisory.** The dictionary line was advisory and the agent obeyed it — the
fifth advisory mechanism tried here, and the first to change behaviour at all. It
still produced the wrong outcome, because obedience was not the binding
constraint: the leaderboard was. Structure is what fixes a ranking problem.

**Not a warning at `finalize_run`.** Catching it at the end means a run has already
spent its budget optimising a number measured on rows that will never exist again,
and — worse — has spent it with a corrupted patience counter. The refusal has to
land before the score enters the leaderboard, which means before the fit.

**Not silently dropping the column.** That would change what the agent asked for
without telling it, and 0029 gives the agent which-columns-to-use.

**Not comparing feature names to the schema.** Would pass every derived feature
and miss both real failures.

## What it costs

**Good**

- `best` can only point at a model that could be served, so the patience counter
  measures progress rather than distance from an unreachable number.
- The one fact no profiling can recover now has a channel *and* teeth.
- Both failure modes, including the one nothing in the harness could previously
  detect, are caught by a single check.

**Bad**

- **A lineage replay on every training call.** Twenty rows, so it is cheap, but it
  is not free and it runs even when nothing is wrong.
- **The schema is only as good as what the user passes.** A wrong `serving_columns`
  refuses valid work, and nothing checks it against reality.
- **It does not catch leakage that is not a missing column.** A column present at
  serving time but recorded *after* the target — a slow-arriving field backfilled
  into the training set — passes this check untouched. The mechanism tests
  availability, not timing.
- **`serving_columns` is a fourth thing the harness owns** (after metric, split,
  folds), and each one is a decision the run's result now depends on. 0029's line
  says that is correct; it does not make the list shorter.
- **The v1 evidence is n=1**, on one model, one dataset. That the agent reasoned
  its way to the guarded drop unaided may not replicate.

## Where it lives

- `harness/session.py` — `RunSession.serving_columns`.
- `harness/tools.py` — `_unavailable_at_predict_time`, `_refuse_if_unservable`,
  `SERVING_PROBE_ROWS`; the refusal in `train_tabular_model`, the field on
  `inspect_data`.
- `harness/agent.py` — `inspect_data`'s docstring, which is where the agent learns
  the rule exists.
- `tests/test_harness_tools.py` — both failure modes, the chained-table case, and
  the no-schema default.
