# 0069 — Feature selection pays on columns that mislead, not on columns that are idle

- **Date:** 2026-08-16
- **Status:** accepted (a measurement, and a capability built but **not shipped to
  the agent**)
- **Related:** [0035](0035-the-search-is-over-features.md) (the search is over
  features; this is the first tool that searches over *which* features),
  [0044](0044-what-ships-is-the-average-not-the-argmax.md) (an average costs no
  selection luck, an argmax does — the argument this record rediscovers one level
  down), [0023](0023-the-third-split-has-evidence-now.md) (what an argmax does to
  the rows it is taken on), [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md)
  and [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md) (a block
  far from the horizon measures a different task),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  interval that diagnosed the null), [0066](0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md)
  (the leak this search half-found), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  `src/tabular_agent/harness/tools.py`, `scripts/probe_feature_selection.py`,
  `rossmann_store_sales/winner.md` + `Rossmann_nr1_doc.pdf`.

## Where this came from

Ulf added the Rossmann winner's model documentation to the repo. It names feature
selection as the device the solution turned on, and this harness has no version of
it — **every tool makes a table wider; none can make one narrower.**

> *"With all features together, the model easily overfit the train set, resulting
> in suboptimal performance on the test set. Therefore, I needed a way to reduce
> the feature set."*

One number in that document reframes more than the feature question. His
*individual, un-ensembled* xgboost models scored about **0.105 private** — rank ~3
of 3,300, against our blend's 0.13813 at ~1,901. His whole ensembling apparatus
(500 random models, ~125,000 scored pairs, 10+ members, harmonic mean) took him
from 0.105 to 0.10021. **Ensembling was worth 0.005; everything else lives in one
model.** This project has spent 0044, 0045 and part of 0068 on the blend.

## What was built

`tools.select_features`: draw random column subsets, fit the probe on each, rank
them, and emit a new table holding the winning subset — or emit nothing when no
subset beats using every column. The narrowing is expressed as a generated
`df.drop(columns=[...])` transform, so lineage, replay, servability and the journal
all come from the existing `transform_features` path rather than being rebuilt.

Ranking rows come from a block cut out of **the table itself** — never the
validation half that ranks tables, never the sealed holdout. 0023 measured an
argmax over 24 candidates taking validation optimism from +0.00206 to +0.00944; a
subset search takes an argmax over hundreds, and pointing it at `val` would destroy
the only honest ruler a run has. Note this is *stricter* than the winner, who
selected 125,000 pairs on his last six weeks with no sealed split at all and
reported being surprised it held.

## Three readings, and the first two were my errors

**Reading 1a — the selection block must be a span, not a fraction.** The block was
a fixed 20% of the table. On Rossmann's `t13` that is about **six months**, while
the run is ranked over **45 days**. So every candidate was fitted on data ending six
months before the rows that judged it, and the search answered a question nobody
asked: which columns survive a six-month-ahead forecast. It dropped `DayOfWeek` and
`Promo`, won its own block by 0.019, and **lost the validation half by 0.020**.

0043 and 0060 both state the mechanism in advance — a block far from the horizon
you are graded at measures a different task. The constant was copied from
`probe_capacity.py`'s inner tail without checking what it denominated. Fixed: the
block is cut to the **same span as the validation block**, in the ordering column's
own units.

**Reading 1b — and the search still bought nothing.** With the span-matched block:

| | dropped | validation gain | block sd | `sign_stable` |
|---|---|---|---|---|
| Rossmann `t13` | 22 / 52 | −0.00255 | 0.00497 | False |
| Restaurant `t7` | 3 / 58 | +0.00041 | 0.00226 | False |
| house-prices | 15 / 36 | −0.00731 | — | — |

0059's interval says why rather than leaving it a mystery. On the selection block
the winning candidate beat all-columns by 0.0071, and the block's own standard
deviation for that comparison is **0.0050**. **The margin being chosen by is barely
wider than the noise it is measured in.** An argmax over ~100 such candidates
selects noise — 0023's finding, one level down, inside the selection block.

**Reading 2 — the criterion was not the winner's.** Ulf pushed back: it helped
*him*, so why not us. Re-reading the document, the answer is that the tool did not
implement his method.

> *"I ran over 500 random models and systematically calculated the validation error
> on each **pair-ensemble** of models. From the best model pairs I built a larger
> ensemble... Then I got the idea to take the features from all of the selected
> models together and combine them into one model."*

He never ranked subsets by their own scores. His criterion is the error of the
*pair averaged together* — **complementarity**, which luck supplies far less
readily than a good individual score. That is 0044's argument exactly, one level
down. Rebuilt that way (cheap: cache each subset's predictions and every pair is
one array mean), changing only the criterion so the reading stays attributable
(0017).

## The finding

The pair criterion **declined** on a synthetic table that is 12 of 16 columns pure
noise:

| | selection score | columns |
|---|---|---|
| all columns | **0.46972** | 16 |
| best single subset | 0.47786 | 11 |
| majority of top 5 pairs | 0.47108 | 7 |

Dropping nine columns of pure noise moves the score by **0.0014**. And that is the
finding, not the decline:

> **A boosted tree loses almost nothing to a useless column — it simply does not
> split on it. So removing useless columns wins almost nothing back.**

Which means the framing was wrong in *both* versions. "Remove redundant or useless
columns" is not what feature selection buys on this model family, and it explains
three real-table readings of approximately zero. The winner says what his problem
actually was:

> *"I soon noted that the **spread features made overfitting easy**."*

Standard deviation, skewness and kurtosis of trailing windows: columns that **fit
the training period and do not transfer**. That is time-instability, not
uselessness — and it is expensive, because the model *does* split on such a column.

> **Feature selection pays on columns that mislead, not on columns that are idle.**

One piece of corroboration, unlooked for: on Restaurant the search found and
dropped `air_reserve_lead_days` — one of the six collapsed reservation columns 0066
measured as actively harmful at the horizon — without anyone telling it about
provenance. Instability is detectable. Idleness is not, because it costs nothing.

## A third limit, found while writing the tests

A random subset excludes all of `k` unwanted columns with probability
`keep_prob ** k`. At the default 0.5 and `k = 6` that is one draw in 64, so 40
draws contain on average **0.6 usable candidates** — and the search returned a
subset that threw away the stable signal along with the drifting columns, scoring
**0.489 worse** than its parent on a synthetic table built for it.

> **A random-subset search needs draws exponential in the number of columns that
> must be excluded *together*.** It is good at finding one bad column and bad at
> finding six.

This is why the winner ran 500 models rather than 40, and it is a second
independent reason the tool is not a substitute for the agent naming a column it
distrusts and dropping it by hand. Recorded in the tool's docstring and in the test
fixture rather than tuned away.

## The decision

**`select_features` stays in the library and is deliberately kept off the agent's
tool surface.** It is measured at approximately zero on three tables. A tool that
consumes a table from the run's budget and returns noise is worse than no tool, and
`EXPECTED_TOOLS` records the omission as intentional so the next reader does not
"fix" it. The function, its tests and `probe_feature_selection.py` stay, so the next
attempt starts from a measured thing.

**What the next attempt should change**, on this evidence: rank a column by whether
its contribution is *stable across several consecutive blocks*, not by a score on
one. That is 0059's instrument applied one level down, it costs no additional fits
(fit once on the prefix, score on k later blocks), and it is the only criterion of
the three that targets the property the winner actually describes. It was not built
here because it is a third redesign in one session and deserves its own reading.

## What this cost

Two wrong designs, both measured rather than argued away, and both caught by the
instrument this project already had. The deeper cost is the one worth carrying: I
took a phrase from a winner's writeup — "feature selection" — and built what the
phrase usually means instead of what the document said he did. The document
specifies a pair-ensemble criterion in one sentence and names the feature family he
was fighting in another, and I read past both. **A distillation that keeps the
vocabulary and drops the mechanism is not a distillation.**

Ulf's correction is what surfaced it, and it is the second time in two sessions
that a claim survived because nobody re-read the source (0061 was the first).
