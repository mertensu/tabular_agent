# Feature attribution, and what the agent actually needs to iterate

*Written 2026-07-30, from a long back-and-forth with Ulf while building the stop
rule. The conclusions were not obvious and several of them reversed positions
taken earlier in the same session, so the reasoning is recorded here rather than
compressed into a commit message.*

## The question that started it

A live run produced this chain of feature tables (a "table" is a file: the
dataset plus columns added by one tool call):

```
t4   date parts + frequency-encoded city    14 cols  -> 0.3160
t5   raw + date parts + ratios, logs, ...   23 cols  -> never trained
t6   t5 + frequency-encoded city            25 cols  -> 0.2741   (best)
```

`t5` was built and immediately extended into `t6`, and only `t6` was trained. So
the run has no measurement between `0.3160` and `0.2741`.

Ulf's observation: **the run cannot tell whether the jump came from `t5`'s pile of
ratios or from `t6`'s encoding.** It "thinks" the huge table is best when possibly
only one column in it mattered.

That is correct, and it is the start of a longer problem.

## Layer 1 — the unmeasured link

The immediate gap. `finalize_run` now walks the winning table's ancestry
(`journal.attribution`) and reports any link with no experiment on it, in the
report and in `unmeasured_tables`.

Three options were considered:

1. **Refuse to build on an untrained table.** Forces a measurement at every link.
   Rejected: it constrains the search to buy a reporting improvement, and the
   whole direction of this session was to constrain the search *less*.
2. **Report the gap, don't prevent it.** Chosen.
3. **Leave it.** Rejected: the agent writes a summary claiming some feature was
   the breakthrough, and nothing contradicts it.

The principle: *the harness does not stop you believing a number; it makes sure
you can see what the number does not cover.* Same move as reporting `fold_spread`
next to the mean.

## Layer 2 — even a measured chain under-determines credit

Measuring every link would **not** have answered the original question. A fully
measured chain `raw -> A -> A+B -> A+B+C` gives the marginal value of C *given A
and B*. It never gives the value of C.

And the marginal value is order-dependent in two opposite ways:

- **Redundancy inflates whoever goes first.** If the date parts and a
  "days-since-listing" feature carry the same signal, the first one added takes
  the entire gain and the second measures ~0. Swap the order, swap the credit.
  Neither reading is wrong — there is no fact of the matter about "which one
  mattered".
- **Interaction punishes isolation.** `price` and `area` are each nearly useless
  for this target; `price / area` is the whole thing. A feature measured alone can
  score zero and still be essential.

Adding-last measures marginal-given-everything. Adding-alone measures
marginal-given-nothing. The truth is in between and depends on which other
features exist. Complete attribution needs every subset: 2^n retrains.

**This is exactly the argument that motivates Shapley values** — average the
marginal contribution over all orders, precisely because no single order is
privileged. Spec §18 (post-hoc SHAP on the winning model) is the machinery for
this, and it is cheap because it reads a trained model instead of retraining.

Group ablation (retrain the winner with each block dropped) was considered and
rejected as a middle option: it costs k retrains and still produces an
order-dependent number that *reads* like a definitive one. Worse than saying
nothing.

## Layer 3 — the search, not the explanation

Ulf's reframe, and the one that changed the plan:

> it's less about me knowing in the end how much a feature contributed, but
> rather the agent iterating in the right way.

These are different problems and the expensive tool (Shapley) is the wrong one
for the second. Fair credit is for the human at the end. **Search guidance only
has to be better than nothing**, and nothing is what the agent currently gets: a
single scalar per experiment, with no information about which of its 23 columns
did anything. That is why it writes kitchen-sink transforms — it is generating
candidates blind, and has no channel through which to learn that `price_sq` was
dead weight.

### Greedy search is blind to joint effects

Ulf again, and this one reversed an earlier position:

> it could build a feature on top of parent which gives no improvement, but the
> two together help

Right. If C alone does nothing but C+B is the win, then measuring C against its
parent says "useless", the agent drops it, and B never gets its chance. XOR is the
textbook case; `price`/`area` is the mild version.

**Consequence: the kitchen-sink transform was partly right, and calling it noise
was partly wrong.** Twenty columns in one call guarantees interacting pairs
arrive together, and a tree can only find an interaction if both halves are
present in the same fit. One-feature-at-a-time is cleanly attributable and
*structurally unable* to discover anything joint.

| | interactions | attribution |
|---|---|---|
| one feature at a time | invisible | clean |
| big batch | discoverable | unrecoverable |

## The resolution: blocks

A **block** is the set of columns added by one tool call — already recorded as
`new_columns` on each table row, so no new bookkeeping.

Why the block is the right unit:

- **It is the unit of decision.** The agent cannot add half a tool call. Its next
  move is "extend this table or go elsewhere", so credit reported per call matches
  a decision it can act on. Per-column numbers are finer than any available action
  and invite conclusions the measurement cannot support.
- **It is big enough to hold an interaction.** Members of a block arrive together
  and are fitted together, so joint effects inside a block survive.
- **It is small enough to be one hypothesis** — *if* the agent makes it so.

That last clause is the crux. `t5` was not a coherent block: date parts, price
ratios, logs, squares and one-hots in a single call — four ideas in a trenchcoat.
Attribution to it is unactionable. So the steer is **one tool call = one
hypothesis**:

```
t7 = raw + date block        -> 0.47   (dates worth 0.22)
t8 = t7  + price-ratio block -> 0.44   (ratios add 0.03 on top)
t9 = t8  + squares block     -> 0.44   (squares add nothing)
```

Every number is then the marginal value of one idea, interactions within an idea
are preserved, and the next move is legible.

This is a **steer, not an enforcement**. Refusing a mixed transform would be
wrong — sometimes the kitchen sink is the right first move precisely because it
catches interactions.

## What gets reported, and how it is worded

1. **Per-block importance share.** Sum the booster's `feature_importances_` over
   each block. Far more stable than per-column importance and matches the unit of
   action.
2. **`vs_parent`.** When the parent table was trained, the delta against it. This
   makes training the parent *pay off for the agent* — it is the only way to get a
   marginal number — so the good behaviour becomes rewarding rather than mandatory.
   Preferred over option 1 of Layer 1 for exactly that reason.

Wording rules, which matter as much as the numbers:

- Zero importance is reported as **"no evidence it helped here"**, never "useless".
  Otherwise we actively train the agent to prune interaction partners. The strong
  claim is only safe in the other direction: a column that *was* split on did
  something.
- **Nothing auto-prunes.** No tool drops columns.
- Abandoning a branch is cheap and should be said out loud: tables are immutable,
  so a "failed" table stays on disk and can be extended later. A dead end is
  parked, not lost.

## Where this leaves the loop

Mapped against the agent loop from the CoreWeave post
(*inspect -> hypothesise -> build feature -> train -> score (OOF) -> decide*),
five arcs exist. The missing one is **score (OOF) -> hypothesise**: OOF
predictions are computed and written to `oof.parquet` and never shown to the
agent. Hypotheses currently come from column names alone.

Residual feedback is therefore the next substantial piece after
`blend_models`/`predict` — it is the step that *generates* hypotheses rather than
scoring them.

Two caveats on that post, recorded because they cut against its conclusion:

- "Sixty cycles overnight" only compounds if each cycle carries information. Our
  own evidence: 19 cycles found the same score as 8, and the extra eleven were the
  agent trying model types for want of anything better. Better feedback per cycle
  is worth more than more cycles.
- "The expensive resource became structured GPU compute" is the cloud vendor
  talking. In these runs the fits take seconds; wall clock is almost entirely LLM
  round-trips.
