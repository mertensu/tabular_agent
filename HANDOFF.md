# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.
>
> **This file is pruned (~2 weeks) and is not the record of *why* anything was
> decided.** That lives in `docs/decisions/` — see its README for the index and
> the project vocabulary. When an entry here records a decision, write the
> decision record too, or the reasoning is deleted with the entry.
>
> **Fresh machine?** Read `docs/SETUP.md` — env, API key, and the (untracked)
> Kaggle datasets.

## 2026-08-21 — Onboarding: docs/SETUP.md, config.env.example, results/

**Machine:** MacBook Air | **Branch:** master | **What:** made the repo
clone-and-run for a second person. New `docs/SETUP.md` (uv env, `config.env`
key, per-dataset Kaggle download → prep-script → outputs table). New
`config.env.example` (no key). New tracked `results/` holds the 7 small
findings-CSVs lifted out of the 11 GB gitignored `_runs/` (see
`results/README.md`). `_runs/` and the raw Kaggle dirs stay untracked by design.
**What's next:** the exact `kaggle` command used to pull the *private*
leaderboard CSVs is not confirmed in SETUP.md — fill in if known.

## 2026-08-18 — The panel was built, measured, and reverted the same day

**Machine:** MacBook Air | **Branch:** master | **593 tests pass** | **no net
`src/` change to the ruler** | **fresh context starts here.** Two records:
[0077](docs/decisions/0077-several-disagreeing-rulers-rather-than-one-better-one.md)
(what was built and why) and
[0078](docs/decisions/0078-the-panel-was-a-stop-rule-and-the-gain-threshold-is-the-real-lever.md)
(the measurement that killed it, and the three replacements). **Read 0078; it
carries the plan.**

> Ulf asked to fix the ruler with multiple disagreeing validation schemes. It was
> built, gated on the soft setting he chose, then replayed against the run it was
> built for. **It accepts 1 of 24 steps.** Reverted.

### Why it failed, in one number

Mean per-scheme agreement with the headline gain: **0.396**, where a coin flip is
0.500. `n_agree` is centred on 2 of 4 (`6/4/9/4/1` across 0..4). The four schemes
behave close to independent coin flips about a step's sign, so an all-or-nothing
vote rejects nearly everything whether the step is real or not. With the gate in
place, patience hits after **8 tables instead of 24** — a stop rule, not a ruler.

**The porting error was legible before the build.** The winners used **one**
scheme, trained on **75%** of the data, for feature decisions (notebook cells 12
and 17); several geometries only to sanity-check the whole approach; and a
separate **per-column** "time consistency" test that rejected **8% of 242
features**. Three instruments, three questions. 0077 collapsed them into one and
aimed it at the hardest. Their 8% against its 96% was the tell.

### What survives, and it is more useful than the panel

`panel.py` and its tests stay **unwired** — the machinery is what the follow-ups
need. And three of Ulf's ideas were measured against 0073's saved capacity sweeps
(`_runs/ruler_deflation_across_runs.csv`, **42 steps across four runs**, Porto
Seguro excluded per 0073's own noise screen):

    median |gain| among steps that helped at capacity   0.00704
    median |gain| among steps that did not              0.00221
    steps scored > 0.010   ->  91% helped (10 of 11)
    steps scored 0 - 0.002 ->  40% helped ( 2 of  5)

| rule | accepts | precision | recall |
|---|---|---|---|
| current — `gain > 0` | 30 | 0.73 | 0.92 |
| **0.25 x best gain so far** | 16 | **0.88** | 0.58 |
| 0.50 x best gain so far | 12 | 0.92 | 0.46 |

**Importance / SHAP is refused** — 0017 already settled it (four columns held 77%
of the splitting and contributed nothing) and the ablation gain's sign is 73-92%
accurate, so a descriptive proxy is a step down from a measured baseline.

### THE PLAN FOR THE NEXT CONTEXT — Ulf: *"tackle those two ideas in a fresh context"*

**1. Report the gain as a fraction of the best so far.** Not a gate. It costs no
fits, and the reason it must not be enforced is measured: the quarter bar
**ratchets**, and on `restaurant_v1` it accepts 1 of 16 while 7 steps really
helped, because one transformative early step raised the bar for the rest of the
run. Diminishing returns are legitimate. The agent knows it is twelve steps into a
chain; a constant does not (Ulf's own principle — the harness guides, it does not
override).

**2. The `_noise` hole, and this one is a defect not an idea.** `stop._noise`
returns `probe_se` or **0.0**, and `probe_se` is deliberately refused on ordered
splits (0059). Measured:

    ieee_v1        probe_se None on 24/24 tables | block_sd present on 24/24
    restaurant_v2  probe_se None on 13/13 tables | block_sd present on 13/13

So `min_gain_ratio` multiplies zero and **there has been no bar at all on Rossmann,
Restaurant or IEEE-CIS** — every positive gain counts as progress. 0028 left "the
bar it is a quarter of" open; nobody noticed it had evaporated entirely on exactly
the datasets that matter. `block_sd` is a noise estimate for **the difference**,
is already computed on all those rows, and gates nothing.

**Register a prediction before touching either.** That is the mistake 0077 and
0078 both exist to document, and the honest test for (2) is the same ground truth
used above: does `gain > k * block_sd` beat `gain > 0` on the 42 steps?

**Also still unbuilt and now the best-evidenced idea in the file:** the winners'
**per-column time consistency** test — one feature, first block, predict last
block, drop if either AUC is below 0.5. 8% rejection on 242 features, and it is a
different question from anything the harness measures today. 0069 predicted this
criterion and did not build it.

### Quiz debt

**None asked, no notebook, five sessions running.** CLAUDE.md is explicit that
this is the largest outstanding item. 0077/0078 is exactly the arc worth walking
through by hand — built, measured, refuted — and it has no notebook.

---

## 2026-08-17 — The fifth dataset ran, and the index was rebuilt

**Machine:** MacBook Air | **Branch:** master | **no `src/` change** |
**fresh context starts here.** Two new records:
[0075](docs/decisions/0075-every-declared-categorical-it-could-see-it-used-and-every-one-it-could-not-it-ignored.md)
(the run) and
[0076](docs/decisions/0076-a-fitted-aggregate-loses-coverage-exactly-as-its-key-becomes-worth-having.md)
(what the winners had that we cannot express). Read 0076 first — it is the one that
names an unexamined decision rather than measuring a known one.

> **Submitted: private 0.899498, public 0.928187, rank ~4354/6355 (top 69%),
> 0.011 below the private median. THIS BOARD IS NOW SPENT** — one submission, as
> planned. All six registered predictions resolved: 1 and 3 refuted, 2/4/5/6
> confirmed.

### What was built

`scripts/prep_ieee.py` and `scripts/run_ieee_v1.py`, both new. The run: 24 tables,
4 terminal fits, 93.1 min, agent finalized itself, six predictions registered in
the runner's docstring beforehand.

| | |
|---|---|
| work table, same ruler | 0.89604 |
| best table `t18` | 0.91734 |
| blend, val / **sealed** | 0.91874 / **0.94176** |
| **private / public, scored** | **0.899498 / 0.928187** |
| private board | median 0.91096, top 10% 0.92819, best 0.94588 |

### The seal is a local peak, and half its optimism is not distance

Four windows in time order: val **0.91874**, sealed **0.94176**, then the 30-day
gap, then public **0.92819**, private **0.89950**. From the seal onward it decays
monotonically with distance — 0060 on a fourth dataset, 0066's gradient shape. But
**the sealed block scores above the block before it and above everything after
it**: it is a local maximum, not a point on the curve.

    val    -> private  +0.01924
    sealed -> private  +0.04226
    difference          ~0.023   <- the seal's own easiness, not the horizon

> **Quote +0.019 as this dataset's horizon decay, not the seal's +0.042.** The
> sealed gap overstates it about twofold. The caution was registered in 0075
> *before* the upload, and this is the measurement that earns it. The mechanism —
> why the sealed block is easy — is **still not established**; it carries a 12%
> higher fraud rate and slightly *worse* entity coverage, and neither explains it.

**0061's pattern holds a third time and is nearly gone**: ours +0.02869 public →
private against the field median's +0.03031, where Rossmann and Restaurant were
30-40% smaller than the field. Two reasons not to lean on it — the field's own
decay grows as score falls (best +0.0223, median +0.0303), so decays are not
comparable across levels; and a below-median team moving up 153 places is what
regression to the mean predicts, which 0061 already named as not ruled out.

### The finding: visibility, not dtype — and it is 49 for 49

All 31 declared-categorical **strings** were encoded. The 6 declared **numerics
inside the first 100 columns** (`card1`, `card2`, `card3`, `card5`, `addr1`,
`addr2`) were encoded. The 12 declared numerics **past column 100** — `id_13`,
`id_14`, `id_17`-`id_22`, `id_24`-`id_26`, `id_32` — were touched by nothing.

`MAX_REPORTED_COLUMNS = 100` truncates the profile and column 100 here is `V45`,
so the identity block falls off. But `non_numeric_columns` is **not** truncated, so
a bare name in a refusal list — no dtype, no cardinality, no description — was
enough for the agent to act on. **The 12 it missed are invisible in both channels
at once.** Predictions 1 and 3 both refuted; 2 and 4 confirmed.

**Do not "fix" the cap before measuring what raising it changes.** Not raising it
is what produced the clean split.

### 0076, and it is the one worth not re-deriving

The winners' whole solution is one key — `card1_addr1_D1n` — and **a run could
build it today**; `transform_features` does the arithmetic and the agent already
built `c1_addr` unprompted. What we cannot do is their next line: aggregate over
`concat(train, test)`. Measured on our own files:

    card1        98.2% of test rows covered by a train-fitted aggregate
    card1_addr1  81.7%
    their uid    21.0%   -> 79% of rows get default_

**Coverage collapses in exactly the direction that makes the key worth having.**
And the sealed holdout reports that same key at **43.0%** — twice its real reach,
which is 0060's shape arriving in *coverage* rather than in score.

The argument: **0011 refuses leakage and transduction under one rule, and they are
not the same thing.** No fix is proposed and none should be read in. The one thing
recommended without reservation: **`fit_group_stats` should report its coverage.**

### Three other things from the run

- **`elasticnet` vanished without trace.** Attempted, ran ≥45 of the 93 minutes,
  discarded, and the journal contains no mention of it — the bundle has 3 models,
  `experiments/` has `e1`-`e4`. The skip is by design; the silence is 0015's
  subject unreported. Wants its own record and a one-line fix.
- **The seal came back EASIER than validation**, −0.021 to −0.033 on all four
  candidates, against 0060. Sealed carries a 12% higher fraud rate and slightly
  *worse* entity coverage; neither obviously explains it. **Left unexplained on
  purpose** — 0058 is the record of adopting a mechanism that was really composition.
- **18 of 20 comparisons `sign_stable: False`, 9 accepted as improvements.** The
  most extreme yet (Restaurant v1 was 6 of 17). 0073's registered gotcha, on time.
  Also: the three families disagree by **0.025** on the same table, more than the
  entire feature chain was worth (0.021).

### The decisions README was rebuilt, and every annotation is intact

50 of 74 records had ended up in one section whose title stopped describing it
around 0040. Now: a **"Where each thread stands"** routing table above the index
(one row per live question, naming the record that holds it and the reading order
behind it), an **"Accepted and not built"** list, and the big section split into
five. Verified by diffing sorted line sets — the only old line that does not
survive is the renamed heading.

**Checked while doing it:** 0047/0048 say `not implemented` and that is now half
stale — `TaskSpec.forecast_horizon` exists and gates window lengths
(`encoders.py:1686`) and stream leads (`tools.py:1281`). Headers left alone per the
never-edit rule; the correction lives in the routing table.

**Still open, and it is Ulf's call:** a fixed `Status:` vocabulary for *new*
records (`decision | measurement | design`, plus built-or-not). The field currently
has ~20 spellings across 76 records, so nothing can answer "is this a rule I must
honour" without being read.

### Also on disk, and worth reading before anything else here

`fraud_detection/winner.md` and `fraud_detection/xgb-fraud-with-magic-0-9600.ipynb`
— the 1st-place writeup and notebook. Beyond 0076, three things in them this
project has circling records about and no build:

1. **"Time consistency."** Train one model on **one feature**, first month, predict
   the last month; drop it if either AUC is below 0.5. 19 of their 242 features
   failed. This is exactly what 0069 said the criterion should be and did not
   build, and the same object as 0059's interval — but per column, and cheap.
2. **Never one validation strategy.** Train 4/skip 1/predict 1; 2/2/2; 1/4/1;
   GroupKFold by month; and the public LB reframed as *"just train 6, skip 1,
   predict 1 and no less valid than other holdouts."* Five sessions have gone into
   making our **one** ruler trustworthy. Their answer is not a better ruler.
3. **D-column normalisation.** `D15n = D15 - TransactionDT/86400` turns a timedelta
   that grows with time into a fixed point, so it stops drifting between train and
   test. Expressible today; our agent built `tdt_day` and never made the link.

**Stays out** (`CLAUDE.md`): their V-column PCA, the 19 named drops, the UID formula
itself.

### Quiz debt

**None asked this session.** Backlog now: this session's run and both records, the
two passes from the previous session, slice 3, 0041/0042/0044, house-prices
capacity, the three blocks from 08-16. **No notebook again** — CLAUDE.md is explicit
that this is the largest outstanding item, and it has now been outstanding for four
sessions.

---

## 2026-08-17 — The ruler is mispriced, on one dataset, and we do not know why

**Machine:** MacBook Air | **Branch:** master | **no `src/` change** |
**fresh context starts here.** The plan's measurement is **done**. New record:
[0073](docs/decisions/0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md).
Read it before anything else here — this entry is the short form.

> **The probe inflates most feature steps, not just close calls.** v7's chain is
> worth **0.104** as the ruler tells it and **0.032** at capacity. Two thirds of
> what this project's feature search appears to have bought is capacity the ruler
> did not have.

Two scripts, both control-passed (the 200-tree arm reproduces every journal score
to six decimals): `probe_ruler_budget_rank.py` (v7's 14 tables x 5 budgets) and
`probe_ruler_deflation_across_runs.py` (four datasets, six runs, 55 steps).

### The three things worth not re-deriving

**1. No budget is correct, so "raise the constant" was never available.** Rounds
and learning rate are not interchangeable at matched shrinkage (`1000 @ 0.10` is
worse than both 3000 and 5000 @ 0.03 on six of fourteen tables), and more capacity
is not truer — `t3`, `t5`, `t7` **overfit** between 3000 and 5000. Each table has
its own optimum, and the optimum depends on the table, which is the thing being
ranked. That circularity is the answer to *"it is so hard to get a proper signal"*.

**2. Two mechanisms are clean and name no dataset.** A **row filter's cost is
training data**, invisible to a ruler too small to use it (v7's `t2->t3` discards
17% of rows; +0.014 -> -0.009, monotone in both rate families). And **v7's largest
fork reverses**: the ruler priced target encoding at +0.046 against frequency's
+0.018, and `t4` beats `t5` at all three high budgets by 0.010-0.021. Every table
v7 shipped, including `t13`, descends from `t5`.

**3. It is one dataset.** `rossmann_v8` replicates both mechanisms (`t5->t6`:
+0.0259 -> -0.0275; 5 of 11 accepted steps do not help at capacity). house-prices
0 flips, restaurant_v1 0, restaurant_v2 0. Porto Seguro flips constantly and it is
**noise** — every gain under 0.005 auc on a 3.6%-positive target.

### What did not survive, and it is most of what I proposed

Three mechanisms offered, none standing: *target-summary tools are overvalued*
(refuted inside pass 1 — `t9->t10`, `t10->t11` are target encodings that
**appreciate**); *deflation tracks key cardinality* (refuted, wrong sign;
Restaurant's 8,242-way encodings are the corpus's most stable steps); *deflation
tracks rows-per-group* (unsupported, +0.358 against cardinality's +0.295).

**The 35% headline in the script output is not real.** Flipped steps are *smaller*
than agreeing ones (median 0.00217 vs 0.00369) — the noise signature, and the
opposite of v7. At a material threshold it is 28%, and 8% above 0.010. The
falsifier failed to screen by magnitude; that is a flaw in the falsifier, not a
pass. **"Not established" is the honest verdict.**

### The decision taken: the panel is NOT built

The design was concrete (report `probe_vs_source_table` at two budgets, never
reconcile, leave `best_table` alone per 0017) and it fits Ulf's principle. Refused
on its own cost argument: the second arm costs **~10x on every dataset** — fit time
at 3000 rounds is dominated by tree count, not rows, so it is 12s even on
house-prices' 1,168 rows — to catch an effect seen in one dataset of four.

**And the tempting claim is not made:** that the misread fork cost v7 its score.
That extrapolates a step-level reading to a chain-level outcome, which is 0066's
and 0067's recorded error.

### Found on the way, and it is nobody's decision

**No model in this project has ever been told a column is categorical.**
`_feature_frame` keeps only numeric dtypes; `_make_model` sets only
`random_state`, `n_estimators`, `verbose` — never `cat_features` /
`categorical_feature`. So CatBoost's ordered target statistics never engage (this
killed a multi-family panel argument mid-session), and **the harness made
agent-chosen encoding the only route from a categorical to a model, then measured
that route with a ruler that overpays for it.** Not settled that declaring them
helps; the probe cannot be the instrument that judges it.

### THE PLAN FOR THE NEXT CONTEXT — a fifth dataset, and Ulf has already downloaded it

> Ulf: *"see fraud_detection folder, tackle this in a fresh context"*

**Why a new dataset at all.** Both large boards are spent (Rossmann 0045,
Restaurant 0066), so the harness currently has **no way to check its own answer** —
which is a worse position than being 0.008 above a median. A finished competition
with both leaderboards on disk gives one clean read.

**`fraud_detection/` — IEEE-CIS Fraud Detection, complete.** Inspected, not yet
touched. Nothing written for it yet: **`scripts/prep_ieee.py` does not exist.**

| | |
|---|---|
| `train_transaction` | **590,540 rows x 394 cols** |
| `train_identity` | 144,233 rows x 41 cols — **only 24% of transactions**, so LEFT join |
| `test_transaction` / `test_identity` | 506,691 / 141,907 |
| target | `isFraud`, **3.50% positive** (20,663) |
| metric | **AUC — already in `SUPPORTED_METRICS`**, no metric work |
| ordering | `TransactionDT`, in **seconds** |
| train span | 182 days | 
| **gap train end -> test start** | **30.0 days** |
| test span | 183 days |
| private board | 6,355 teams, best 0.94588, top 10% 0.92819, **median 0.91096** |
| public board | best 0.96814, top 10% 0.95321, median 0.94127 |

**`TransactionDT` is a timedelta from an undisclosed reference datetime, not a
timestamp.** Two consequences worth knowing before anyone writes a date feature.
Its unit is seconds, so a *relative* weekday and hour are derivable —
`(DT // 86400) % 7` — but **which day is Monday is unknown**, and so is the
calendar date of any row. Nothing external can be joined to it, which is
convenient rather than limiting: 0001 keeps external data out anyway.

**The 30-day gap is the point, and it is not a nuisance.** `TaskSpec.forecast_horizon`
is expressed *in the ordering column's units*, so with `split_on="TransactionDT"` it
is **2,592,000**, not 30 — 0051 forbids guessing units and this is where that bites.
It also puts [0064](docs/decisions/0064-month-ahead-is-the-same-idea-and-does-not-get-the-same-tool.md)
directly under test: 0064 refused a `gap` argument on `fit_history` on the grounds
that the asymmetry belongs to the **evaluation geometry**. Here the geometry has a
real 30-day gap, so that refusal either holds or it does not.

**What this dataset is being used to probe** (it is an instrument, not a target —
`CLAUDE.md`):

1. **0073's mechanism, n=2.** `card1` and friends are thousands of distinct values
   on 590k rows — the high-cardinality identity structure the mispricing was about.
   It appeared on Rossmann and on none of the other three datasets.
2. **Today's categorical finding, in its sharpest available form.** The competition
   *declares* which columns are categorical: `ProductCD`, `card1`-`card6`, `addr1`,
   `addr2`, `P_/R_emaildomain`, `M1`-`M9` on the transaction side; `DeviceType`,
   `DeviceInfo`, `id_12`-`id_38` on the identity side. **49 columns. The harness
   handles every one of them wrongly, in two different ways** (measured, not
   assumed — see the table below):

   | | | |
   |---|---|---|
   | **31 are strings** | `ProductCD`, `card4`, `card6`, both email domains, `M1`-`M9`, `DeviceInfo` (1,786 values), `id_30/31/33/34`… | `_feature_frame` **drops** them |
   | **18 are numeric dtype** | `card1` (**13,553 values**), `card2` (500), `id_19` (522), `id_21` (490), `id_25` (341), `addr1` (332)… | **kept, and split as ordered numbers** |

   The second row is the dangerous one and it is **silent**. `_feature_frame` at
   least reports what it dropped; nothing reports that a tree is asking
   *"is `card1` > 9000?"* of an arbitrary identifier. Dropping is lossy and visible;
   this is wrong and invisible.

   **Corrected after checking the literature — the earlier framing here was wrong.**
   The 18 are not being handled *badly*: an arbitrary integer split as a number is
   **label encoding**, a legitimate GBDT treatment, and rsakata's encoding
   comparison reports it is notably **robust** to large `num_leaves` where native
   categorical handling overfits at high cardinality. So the accidental treatment is
   the capacity-robust one. **The defect is that nobody chose it and nothing can say
   otherwise** — `ProductCD` is dropped for being a string, `card1` is label-encoded
   for being an int, two fates decided by a CSV parser.

   Also corrected: the harness has **three** encodings, not one. `fit_encoding`
   takes `frequency` (default), `ordinal` and `target`, plus `fit_group_stats`
   separately. "The only route is target encoding" was overstated.

   **Ulf's call, designed and not built:**
   [0074](docs/decisions/0074-a-categorical-declaration-is-a-fitted-step-not-a-flag.md)
   — *"let the agent decide if it chooses native categorial handling and let it
   declare a column as categorical if it wants to"*. Read it before touching this;
   the short version is that it **cannot be a flag**, because pandas `category`
   codes are positions into the observed level set and a predict batch remaps them
   silently — the same defect 0011 caught when `factorize` was tried. So it is an
   encoder with a fitted, persisted level set, which also keeps 0035 intact.

   Typing in `prep_ieee.py` remains available and is sanctioned (`CLAUDE.md` names
   typing as prep's job), but it is now the *fallback* rather than the plan.
3. **0054's open item 2.** Whether the terminal fit's 200 trees is visibly part of
   why a run lands mid-table, on a board that can actually say so.
4. **0060 / 0061.** The field decays ~0.030 AUC public -> private (median 0.94127 ->
   0.91096). We decayed *less* than the field on both previous datasets; this is the
   third test of that.

**Build order:**

1. **`scripts/prep_ieee.py`**, modelled on `prep_rossmann.py`: the LEFT join on
   `TransactionID`, the typing, `ABOUT` / `COLUMN_NOTES`, `METRIC="auc"`,
   `SPLIT_ON="TransactionDT"`, and `leaderboard(private=True)` /
   `leaderboard_rank(score, private=True)` reading **both** CSVs from disk — per
   0061, defaulting to private. Dataset knowledge lives **only** here.
2. **One run at current defaults.** Register predictions in the runner's docstring
   *before* it starts.
3. **One submission. Then the board is spent** — write that down at the time. A
   finished competition returns a private score on every late submission, so the
   seal here is self-imposed discipline and nothing else. That is exactly how 0045
   happened.

**The gotcha that this session paid for:** AUC differences of ~0.002 on a
3.5%-positive target are **noise**, not findings. Porto Seguro produced 9 of 19
spurious flips that way and contaminated 0073's headline until they were screened
out. Screen every comparison on this dataset by magnitude before believing it.

**Deferred, not dropped:** pointing `probe_ruler_budget_rank.py` at `restaurant_v1`
(Rossmann ~821 rows/store vs Restaurant ~274; if capacity substitutes only once
there are enough rows per group, Restaurant should flip at a budget above
3000 @ 0.03). IEEE-CIS answers the same question on fresher evidence, so it goes
first. Also still open: step 2, the stability-check hole (tests row-locality, not
batch-independence; measure what a widened check would newly refuse *before*
changing it).

### Quiz debt

**None asked this session** — one was offered and Ulf chose to discuss the results
instead. Backlog grows: this session's two passes, slice 3, 0041/0042/0044,
house-prices capacity, and the three blocks from 08-16. **No notebook** for this
session either; CLAUDE.md is explicit that this is the largest outstanding item.

---

## 2026-08-17 — The rule landed: a feature is an average, not a row

**Machine:** MacBook Air | **Branch:** master | **576 tests pass** (was 573) |
**fresh context starts here.** Steps 0 and 1 of the plan are **done and shipped**;
step 2 is untouched. New record: [0072](docs/decisions/0072-a-feature-is-an-average-over-a-stretch-not-the-value-of-one-row.md).

> **`fit_history` emits trailing windows and nothing else** — no last-known pair,
> no lags. `windows` is required.

Ulf, arriving at it from the general direction rather than from the measurement:
*"features die auf dem wert eines tags (einer row) beruhen sollten nicht
existieren, wenn sollten es (rolling)averages sein, die sind viel stabiler."*

**The mechanism, which is why this is not merely a variance argument.** The
per-fold freeze (0048) is *correct* and therefore silent: a single stored
observation takes 4,041 distinct values across the folds a model **trains** on and
is one frozen per-group constant for every row it is **served**. The model learns
a feature and is handed a constant of the same name. On Rossmann the one draw fell
on 1 May 2015, when 1,095 of 1,115 stores were shut, so 98% of stores were served
a zero. No guard can fire: 0011 sees batch independence, 0030 sees servability,
the fold logic sees no future.

| | probe | interval | sealed |
|---|---|---|---|
| Rossmann (0071) | +0.01428 | 5/5, `sign_stable: True` | ~ |
| **Restaurant** (`scripts/probe_restaurant_latest_pair.py`) | **+0.00216** | 4/5, **not stable** | **+0.00349** |

Control passed: the `with_pair` arm reproduces v2's blend val 0.519707 / sealed
0.505713 to six decimals. **The mechanism transfers, the price is a lottery** —
Restaurant's freeze is an ordinary Saturday.

### The finding nobody was looking for, and it sharpened the rule

The offset half was measured rather than argued, and it says something better:

| key | point `364_ago` | window 364-median | obs |
|---|---|---|---|
| `Store` | 0.8292 | **0.8184** | 337 |
| `Store × DayOfWeek × Promo` | 0.8292 | **0.9500** | 27 |

> **A window is not automatically better** — on the coarse key it is *worse* than
> the lag. And the point lag is identical on both keys because 364 days is 52
> weeks: its strength is borrowed, it smuggles a grouping key in through the
> distance. So the rule is not "averages beat points"; it is **comparability
> belongs in the key**, and a lag is a key expressed as a distance.

Honest limit: correlations on served rows, not probe differences with intervals —
one level weaker than the `_latest` reading. Recorded as such.

### Two other things that contradict what this file believes elsewhere

- **Sealed came back *larger* than probe** on Restaurant (+0.0035 vs +0.0022),
  against the probe→sealed halving 0068 found for capacity and 0071 assumed.
- **`val` prefers the pair** while sealed does not (0.51971 → 0.52229 val;
  0.50571 → 0.50222 sealed). 0068's signature. No default may be read off `val`.

### Does this get Rossmann under the median? No, and the reason matters

**v7 — our best run, 0.13813 private — has zero history steps.** This change
cannot move it; there is no such column in that chain. It only affects *future*
runs that build windows. The only number that already carries the rule is the
hand-built winner rebuild: **blend sealed 0.130242**. Rossmann's own measured
sealed→private drift is +0.0019 (t13: 0.136181 sealed, 0.13813 private), so that
extrapolates to ~0.132 against a median of **0.12967** — still ~0.0025 short, from
a table our agent did not build, at a non-default budget. **And it cannot be
checked: the private board is spent (0045).**

### Where the rule lives, for whoever touches this next

The refusal is at the **fitting entry points**, not in `HistoryEncoder`. The class
still computes the pair and lags, and `from_dict` defaults `latest` to `True` when
the key is absent — because bundles written before today are shaped
`{... 'offsets': [], 'keep': 92, 'windows': [[91.0, 'median']]}` with no `latest`
key, and re-deriving it would hand a model **fewer columns than it was trained
on** (0055). Four tests pin it; mutating that one default fails three with
`KeyError` at predict time. The refusal itself is pinned as a **signature** —
neither entry point has `offsets` or `latest` as a parameter (0018's argument).

### Also done, and it is a rule worth keeping

The agent-facing docstrings no longer illustrate with a retail example. Ulf caught
*"what has this store been selling lately"* in the tool description the agent reads
on **every** dataset. The line: **dataset names belong in comments (evidence),
never in docstrings the agent reads as instruction (frame).** `agent.py` is clean;
`StreamAggregateEncoder`'s "store-day table" illustrations are untouched and open.

### THE PLAN FOR THE NEXT CONTEXT — the ruler, and Ulf asked for a fresh context

> Ulf: *"halte das fest und wir gehen das mit frischem Kontext an"*

He asked why we are above the median. The answer that survived the evidence: not
the vocabulary (0071), not capacity (0068, +0.0037 on the blend), not feature
selection (0069's null) — and those levers **substitute rather than add** (0067),
so their sum is not the gap. What is left is that **every decision a run makes
goes through a ruler that has now been measured wrong four times**, and a run
makes 15–25 of them.

**The naive fix is the wrong one, and this is the part worth not re-deriving.**
0059 built a block interval and nothing is gated on it. Making it bind is the
obvious move and it would not have caught either failure we know about:

- 0066's leaky table read `sign_stable: True` at **2.9 sd**, every block agreeing
  — and lost privately.
- 0059's own history arm had **5/5 blocks** in favour and Kaggle read −0.0301.
- Inverted today: Restaurant's +0.00216 came back `sign_stable: False` and was
  real anyway.

> **An interval measures variance. Every documented failure here was bias** — the
> ruler pointing confidently the wrong way. A wider interval does not fix a
> shifted one. (This argument deserves a record *if it survives the measurement
> below*; it is not one yet, and no decision to refuse gating has been taken.)

How often the existing interval already disagrees, counted from the journals:

    restaurant_v1   17 steps, 6 unstable, 4 of those accepted as improvements
    restaurant_v2   13 steps, 3 unstable, 3 of those accepted as improvements

**So: the probe measures a different model than the one that ships.** It is one
lgbm at 200 trees; what ships is a blend at a real budget. 0054 measured the
*sign* of the feature block's value reversing between them (−0.0152 at 200 trees,
+0.0087 at 5000@0.03). That is a systematic shift, not noise, and every table
decision this project has ever made rests on it.

#### The measurement to run first — one script, hours not days

**Re-probe every table of v7's chain at `1000 @ 0.10` and rank-correlate against
the 200-tree probe scores already in its journal.** v7 is the run whose tables
still replay (0055 broke v8's), which is why this is possible at all. Reuse
`probe_capacity_v7_chain.py`'s copy-the-split pattern.

Decisive both ways:

- **Same ranking** → the probe is fine, the cause named here is wrong, and the
  remaining candidate is architecture (0070).
- **Different ranking** → every table decision in the project's history was taken
  on an instrument measuring something else, and the open question becomes *at
  which budget does the ranking stabilise, and is that affordable per step*.

Either way it answers **0054's open item at the same time**: "where does the
budget come from" and "what is the ruler measuring" are one problem, not two.
Register the prediction in the script's docstring before running it.

#### The ambitious follow-on, deliberately not decided

0044 already settled this one level down: for **models** we ship the average, not
the argmax, because an argmax pays selection luck. Over **tables** the harness
still takes an argmax at every step, on a noisier instrument. The same argument
one level up is: fit and average the best *k* tables — which is exactly the
"ensemble over feature subsets" 0070 lists in the winner's architecture. It is the
only idea on the table that defuses the ruler *and* matches what he actually did.
Own record, own decision, after the measurement.

### Still open: step 2

**The stability-check hole** — it tests row-locality, not batch-independence, and
accepts any column that is a function of the set of distinct keys (0071's second
finding). Own notebook, own record, and **measure what a widened check would newly
refuse before changing it**, because it changes what every existing transform may
do. Not superseded, just not chosen for the next context.

**Not next, unchanged and for measured reasons:** model configuration is blocked
behind 0054's open item (0068 measured `val` choosing the budget wrongly on both
datasets — and this session is a third instance of `val` pointing the wrong way);
feature selection is 0069's null; trend extrapolation is not in the winner's main
model (0070's correction).

### Quiz debt

Two asked and answered this session (why the per-fold freeze makes the pair
*more* dangerous; what Restaurant could show that Rossmann's 5/5 reading cannot
exclude). Backlog unchanged: slice 3, 0041/0042/0044, house-prices capacity, and
the three blocks from 08-16.

---

## 2026-08-17 — the notebook, and the pair confirmed on a second dataset

**Machine:** MacBook Air | **Branch:** master | **573 tests pass** (nothing in `src/`
changed today) | **fresh context starts here.**
Steps 0 and 1 of the plan below are **done**. No decision record yet — it is waiting
on one call from Ulf, named at the bottom of this entry.

### Step 0 — the notebook, before any library code

`notebooks/phase5_the_column_nobody_asked_for.ipynb` (27 cells, 11 code, ships
unexecuted, every code cell smoke-tested in order). It drives `HistoryEncoder` by
hand on v7's split and **found the mechanism 0071 did not have**:

| `Store_Sales_latest` | distinct | zeros | age |
|---|---|---|---|
| training rows (freeze **per fold**) | 4,041 | 0.1% | 1..305 days |
| served rows (final freeze) | 21 | **98.2%** | 1..45 days |

v7's freeze lands on **1 May 2015**, a public holiday on which **1,095 of 1,115
stores were shut**. So the model trains on a genuinely informative recent-sales
column and is served a different column wearing the same name.

> **The per-fold freeze is correct, and that is exactly what makes the pair
> dangerous.** A naive single freeze would give training rows the same degenerate
> value, the column would be near-constant, and a tree would ignore it — harmless.
> Correctness makes fit and serve **procedurally consistent** and lets them drift
> **distributionally**, so no guard can fire: 0011 sees batch independence, 0030
> sees servability, the fold logic sees no future. There is no error to find. Only
> a price.

That is why it costs 0.014 where nine pure-noise columns cost 0.0014 (0069): the
pair is not idle, it is **confidently wrong**.

### Step 1 — confirmed on Restaurant, and the price does not transfer

`scripts/probe_restaurant_latest_pair.py`. v2's `t7` against the same table with
only `air_store_id_visitors_latest` / `_latest_ago` dropped; v2's split copied,
real `finalize_run`, **one table per sealed arm** so there is no argmax to go
astray (0071's gotcha). Control passed: the `with_pair` arm reproduces v2's blend
val **0.519707** / sealed **0.505713** to six decimals.

| | probe | interval | sealed |
|---|---|---|---|
| Rossmann (0071) | +0.01428 | sd 0.00576, 5/5, **stable** | ~ |
| **Restaurant** | **+0.00216** | sd 0.00205, 4/5, **`sign_stable: False`** | **+0.00349** |

Registered prediction held on all three counts (direction positive, magnitude
+0.000..+0.005, interval likely unstable).

> **The mechanism transfers; the price is a lottery.** One value per served group
> for the whole horizon is structural — the freeze does not move while the rows do.
> But Restaurant's freeze is an ordinary Saturday, train and serve nearly coincide
> (82 vs 87 distinct, no zeros), and the column is genuinely informative on the
> scored rows (corr 0.55 against 0.69 for `_30_mean`). **Rossmann's 0.014 was a
> holiday landing on the freeze.**

**Two things nobody was looking for, and both belong in the record:**

- **Sealed is *larger* than probe here** (+0.00349 against +0.00216), the opposite
  of the probe→sealed halving 0068 measured for capacity and 0071 assumed for
  Rossmann. That assumption is now unsupported on a second dataset.
- **`val` points the wrong way again.** The blend gets *worse* on val without the
  pair (0.51971 → 0.52229) and *better* sealed (0.50571 → 0.50222). 0068's
  signature. **Any default chosen off a validation score chooses wrongly here.**

### THE OPEN CALL — the default, and it is the only thing blocking the code

That the argument gets added is settled: *not declinable* is the defect, and the
pair has now failed to help on two datasets. **What the default should be is not
derivable from the numbers** — +0.002 inside its own block noise cannot carry a
default change. Two honest shapes, put to Ulf and unanswered at the time of
writing:

- **A — pure opt-out.** `latest=True` stays the default. Nothing existing changes,
  everything stays replayable. Cost: the agent has to already know about the defect
  to avoid it, and neither of these two measurements would ever arise from a run,
  because no run performs an ablation.
- **B — the pair is emitted when it *is* the request.** `latest=None` → emit only
  when neither `windows` nor `offsets` were asked for; opt-in otherwise. A bare
  `fit_history` call is asking precisely for the last-known value and 0049's
  argument stands there untouched. Cost: it changes what existing calls emit
  (0055's class of change), so `from_dict` without the key must mean `latest=True`
  or old bundles break.

**Recommended: B**, on Ulf's own principle — *"der agent ist die Intelligenz, die
harness sollte ihn nicht beschneiden, ihn nur in richtige bahnen leiten"*. Four
columns for a one-column request is a judgement the *tool* makes, that the agent
never made and cannot see without an ablation.

**Then:** the few lines, tests, bundle compatibility, and a record saying which of
0049's and 0050's conclusions survive.

**Step 2 is untouched** — the stability-check hole (see the 08-17 entry below),
its own notebook and its own record, and measure what a widened check would newly
refuse *before* changing it.

### Quiz debt

Two questions asked and answered this session (the per-fold freeze; what Restaurant
could show that Rossmann cannot exclude). The older backlog is unchanged: slice 3,
0041/0042/0044, house-prices capacity, and the three blocks from 08-16.

---

## 2026-08-17 — The gap is a column pair nobody asked for

**Machine:** MacBook Air | **Branch:** master | **573 tests pass** | **fresh context starts here.**
Written up as [0071](docs/decisions/0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md).
Read it, then the 08-16 entry below for the three records that led here.

Ulf: *"ich will nur vermeiden eine echte Lücke zu haben in unserer harness/agent."*
So the winner's **salesmodel** was rebuilt with real tool calls, on v7's copied
split, at 0068's budget, and scored on v7's sealed holdout.

| | value |
|---|---|
| capability gap — the 3 calendar windows no tool expresses | +0.00122 |
| agent gap — his salesmodel vs v7's `t13`, sealed | +0.00220 |
| **`fit_history`'s unrequested `_latest` pair** | **+0.01428** |

> **The tool defect is 6x the agent gap and 12x the capability gap**, at 5/5 blocks
> and 2.5 block sd — the first reading in three sessions that passes 0059's
> instrument convincingly. Sealed it is ~+0.008 (the same probe→sealed halving 0068
> found for capacity).

`fit_history` emits four columns where the winner's feature is one, and there is no
way to decline the `_latest` / `_latest_ago` pair. 0049 and 0050 measured the
*direction* of this; nobody had measured the **price on a table worth having**.

### THE PLAN FOR THE NEXT CONTEXT — Ulf's call, and the order is part of it

> *"ja, mach 1 und 2 — aber erst das notebook und alles aber in frischem Kontext"*

**Step 0 is the notebook, and it comes before any library code.** This is not
ceremony: nine blocks of notebook/quiz debt are owed, and every wrong turn in the
last two sessions was caught by a measurement that could have been a plot first.
Ulf said the same thing about trend extrapolation earlier — *"bislang kann ich mir
nicht vorstellen wie das umgesetzt wird"* — and the answer both times is to drive it
by hand before building it.

1. **Notebook: what the `_latest` pair does.** Drive `fit_history` by hand on
   Rossmann, show the four columns it returns for a one-column request, plot
   `Store_Sales_latest` against the target for a few stores, and show the ablation
   (0.14098 → 0.12670) as a before/after. Style of
   `notebooks/phase3_harness_step1.ipynb`: hand-driven first, ships unexecuted,
   every cell smoke-tested in order. **Then quiz Ulf on it.**
2. **Confirm on a second dataset, then fix.** Restaurant's `t7` is on disk and
   carries the same ablation in about an hour. **Do not skip this** — n=1 is the
   record this project keeps breaking, and 0069 is the freshest example. Then the
   few-line change to `fit_history` (make the pair optional), then a record.
3. **The stability-check hole** (see below). Its own notebook, its own record: it
   is about `leakage-safe by construction` being false for a whole column class,
   not about score. Widening the check changes what *every* existing transform may
   do, so measure what it would newly refuse before changing it.

**Not next, and the reasons are measured, not guessed:** model configuration is
blocked behind 0054's open item (0068 measured `val` choosing the budget wrongly on
both datasets, and any config search inherits that ruler); feature selection is
0069's null; trend extrapolation is not in the winner's main model at all (0070's
correction).

**And a standing caution.** His ~0.105 is ~0.027 beyond us and no measured lever in
this project exceeds 0.014. Chasing it means adopting `colsample_bytree=0.3` and
segment models *because they worked on Rossmann*, which is exactly what 0022 and
`CLAUDE.md` forbid. **The private median (0.12967) is ~0.002-0.003 away and is a
legitimate health check; 0.105 is not a goal.**

### Two findings nobody was looking for

- **The stability check tests row-locality, not batch-independence.** The naive
  rolling-window holiday feature was **accepted**. `transform.py` resamples
  `df.sample(n // 4)` — *rows*. With 1,115 stores per date, `0.75 ** 1115` is the
  chance a date disappears, so `groupby(Date)` is unchanged and the column looks
  stable. **Any column that is a function of the set of distinct keys passes**, and
  the failure is silent. 0011's own cases (`qcut`, batch norm) are still caught.
  Needs its own record: widening the check changes what every transform may do.
- **`Customers` is not dropped out of inability — the harness rewards dropping
  it.** 0030 refuses to score any table still carrying a column absent from
  `test.csv`, so every step before the drop is `unscored`. Dropping it first is
  rational, and it destroys the only thing it was good for (`fit_group_stats`
  explicitly supports a train-only value column). Incentive problem, not a
  capability one.

### And the answer to "why can't we replicate the winner"

His single un-ensembled model is ~0.105 private (rank ~3/3300). A faithful rebuild
of that model's feature set lands at **0.13024 sealed**, 0.0022 better than what our
agent found alone. **The distance is not the feature vocabulary.** It is
architecture: four model families, segment models on the test months,
`colsample_bytree=0.3`, 5000 rounds, an ensemble over feature subsets.

### Gotchas in the probe script

- `finalize_run` ships the argmax of the probe scores, so arm A's sealed number is
  the **ablated** table and arm B's is not. The sealed B−A difference is therefore
  mostly the `_latest` pair, not the calendar windows — the clean capability-gap
  number is the probe at the same step of both arms. The script says so in its own
  output now; the first version printed the misleading one.
- `fit_encoding(["Store", "Promo"], method="target")` encodes the columns
  **separately**. The interaction needs the key built first, and getting that wrong
  cost 0.014 and flattered the agent.
- `fit_history` runs **before** `filter_rows`: its windows count rows, and a row is
  a day only while every store-day is present.

---

## 2026-08-16 — The winner's writeup arrived, and it moved the priorities

**Machine:** MacBook Air | **Branch:** master | **fresh context starts here.**
Three new records: [0068](docs/decisions/0068-capacity-reaches-the-shipped-blend-at-half-strength-and-validation-picks-the-wrong-budget.md),
[0069](docs/decisions/0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md),
[0070](docs/decisions/0070-what-the-winner-had-that-we-cannot-express.md).
**Read 0070 first** — it is the one that changes what to do next.

### THE PLAN FOR THE NEXT CONTEXT — and it changed twice at the end of the session

Ulf's first call was **build per-group trend extrapolation** (0070 item 1, 0058
item 2): a per-group linear model fitted on a trailing window and **extrapolated
into the forecast window**, which names no dataset because a GBDT cannot
extrapolate past the range it trained on. He asked for a fresh context and for the
**notebook to come second, before the library code**, because he wants to see the
fitted line against the actual later values before anything is built.

**Then he asked whether it would reach the top 10%, and checking the answer moved
the plan.** It will not, and his own feature dict says so:

    'linmod_quarterly':[0,0,0,1],'linmod_yearly':[0,0,0,1]

The bits are `[salesmodel, customermodel, MA-models, pmd-variant]`. **The trend
features are not in his main model at all** — only in the previous-month-date
variant of the month-ahead models, the branch he calls "almost as good". 0070 had
ranked this "the largest unbuilt item"; that is corrected in the record, along with
the reason it happened.

> **So the next step should be a measurement, not a third guess.** Rebuild his
> **salesmodel** feature set by hand in `scripts/`, fit one model at a real budget,
> score it on our sealed holdout. Decisive both ways: near his ~0.105 and the gap
> is vocabulary and we learn which columns carry it; near our ~0.138 and the gap is
> **not** vocabulary, and every capability list including 0070 is beside the point.

Two rankings were attempted today (feature selection, trend extrapolation) and both
were wrong, both because a lever was sized from a description instead of measured.
Every lever this project has actually measured is 0.0037–0.0068; the gap to the
private top 10% is 0.0204.

**Also from that dict, and it cuts against this project's direction:**
`'store':[0,0,1,0]` — **his main model has no store identity.** He excluded it
deliberately, worried that competition distance would proxy for it. 0054 measured
per-entity identity as the mechanism that makes capacity pay (+0.0004 → +0.0283 on
restaurant once the model could tell shops apart). Unresolved tension, worth
holding in mind.

Trend extrapolation stays a real gap and a legitimate build — just not on the
grounds that it is the biggest one.

### Why the other two threads are closed for now

**Capacity (0068).** Measured on what actually ships — the terminal blend on the
sealed holdout, not one probe lgbm on validation.

| | single lgbm (probe) | blend (sealed) |
|---|---|---|
| Rossmann `t13` | +0.00682 | **+0.00373** |
| Restaurant `t7` | +0.00186 | **+0.00197** |

So capacity closes ~**44%** of Rossmann's gap, not the 80% the last plan assumed.
All four controls passed (both 200-tree arms reproduce their source journals to six
decimals, `elasticnet` unmoved, the probe sweep replayed 0067 digit-for-digit).

> **And the part that outlives the numbers: `val` picks the wrong budget.** On
> Rossmann `val` ranks `5000@0.03` best while the sealed rows rank `1000@0.10`
> best; on Restaurant every budget makes `val` worse and sealed better. **Every
> design for 0054's open item 2 that lets validation choose is refuted before being
> built.** Early stopping is refuted too, twice (1195 rounds → +0.0028 on `t13`;
> 1498 rounds → −0.0002 on `t7`). The three families also disagree with each other
> consistently across both datasets, so one common budget is the wrong shape.

**Feature selection (0069).** Built (`tools.select_features`), tested, measured at
~zero on three tables, and **deliberately not on the agent's tool surface** —
`EXPECTED_TOOLS` records the omission as intentional so nobody "fixes" it. Two of
my designs were wrong and both were caught by measurement:

- a selection block cut as a **fraction** is six months where the run is graded
  over 45 days → −0.020 on Rossmann (0043/0060, third instance today);
- ranking subsets **individually** is an argmax whose margin (0.0071) is thinner
  than the block's own noise (block sd 0.0050) → −0.0026, `sign_stable: False`.

> The finding is why: **a boosted tree loses almost nothing to a useless column, so
> removing one wins almost nothing back.** Dropping nine pure-noise columns moves
> the score by 0.0014. The winner's problem was different — *"the spread features
> made overfitting easy"* — columns that fit the training period and **do not
> transfer**. If this is picked up again, the criterion is stability across several
> consecutive blocks (0059 one level down), not a score on one.

### The number from the winner's writeup that reframes everything (0070)

> *"Most of the individual XGBoost models already achieve around **0.105** on the
> private leaderboard."*

That is rank **~3 of 3,300** for a single un-ensembled model. His whole ensembling
apparatus — 500 random models, ~125,000 scored pairs, 10+ members — took him from
0.105 to 0.10021, i.e. **0.005**. We are 0.033 behind his *simplest* model, and this
project has spent 0044, 0045 and part of 0068 on the blend.

**Four things we cannot express** (0070): per-group trend extrapolation; shape
statistics beyond mean/std/median (he used skew, kurtosis, harmonic mean, p10);
forward-looking calendar windows (`holidays_nextweek` — a second input known in
advance); and feature subset selection. **One thing we can express and throw away:**
`fit_group_stats` explicitly supports a value column absent at predict time, and
every Rossmann run drops `Customers` rather than building its per-store aggregate.

**Vindicated:** 0064's refusal of month-ahead models — he reports them "almost as
good", never better. **Still out:** external weather (0001), the 0.985 multiplier.
**Do not read his method as licence to loosen 0023** — he selected 125,000 pairs on
his last six weeks with no sealed split and was surprised it held.

### Gotchas

- `_validation_span` guards a `TypeError`: dates read from a **CSV** are strings,
  which sort (so the run is ordered) but do not subtract. Unguarded, that crashed
  `select_features` on a legal configuration. A test pins the fallback.
- `scripts/probe_capacity_terminal_blend.py` layers the budget **on top of** the
  params `_terminal_fits` passes. Patching the library default alone would leave
  the blend's lgbm member at 200 trees, because `PROBE_PARAMS` pins it — an arm
  that looks like a blend at budget and is not.
- Two probe scripts here copy the source run's split rather than recutting it, and
  drive the real `finalize_run`. Keep doing that; it is what made the controls
  possible.

### Owed, and the debt grew

**Quizzes for every block this session** (capacity-on-blend, the winner audit,
feature selection) on top of the five already owed, plus slice 3, 0041/0042/0044
and house-prices capacity. **No notebook was written this session** — 0062 step 6 is
still open too. CLAUDE.md is explicit that code Ulf has not followed is a failure
even with green tests, and that is now the largest outstanding item in this file.

---

## 2026-08-16 — Capacity measured on the table that ships: +0.0068, not +0.06

**Machine:** MacBook Air | **Branch:** master | **553 tests pass** | **fresh context starts here.**
Written up as [0067](docs/decisions/0067-capacity-is-worth-a-tenth-of-that-on-a-table-that-has-features.md).

Ulf asked how this project gets better. I answered "capacity, three to seven times
the gap to the median", then ran the measurement that answer rested on. **It was
wrong**, and the correction is the most important thing in this file.

### The reading

`scripts/probe_capacity_v7_chain.py` — 0061's "untested cell", v7's `t13` (the
chain that shipped) at 0054's own budgets, same grid and monkeypatch, imported not
restated. Possible now only because `t13` predates `fit_history` and so still
replays; v8's `t24` does not (0055).

| table | 200 trees | best budget | gain |
|---|---|---|---|
| baseline (raw, `Customers` dropped) | 0.21810 | 0.15749 | **+0.0606** |
| v8's feature block (code only) | 0.23331 | 0.15213 | **+0.0812** |
| **v7's `t13` — what shipped** | **0.13202** | **0.12520** | **+0.0068** |

> **Capacity and feature work substitute. Whichever you apply second is worth
> about a tenth of what it was worth first.**

They do not *converge*, though: 0.15749 / 0.15213 / **0.12520** at best budget, so
the feature chain is worth ~0.03 that capacity cannot recover.

### Why this matters more than the number

If the whole +0.0068 transferred to Kaggle — and 0056 showed capacity gains do
transfer — v7's private **0.13813 → 0.13133**, against a private median of
**0.12967**. Still short.

> **Features and capacity are both bounded now, and together they do not close the
> gap to the median on Rossmann.** The remaining gap is made of something this
> project has not identified.

**So do not open the next session by building a capacity feature.** The obvious
plan is now measured and it is not enough on its own. Untested candidates, none
endorsed: one model rather than per-segment models; no row weighting of any kind;
every hyperparameter except `n_estimators`/`learning_rate` is a library default in
every run ever made; the blend has never been compared against a tuned single
model.

### Two things that survive unchanged

- **"Do not raise the constant" (0054) is strengthened**, not weakened. The
  inner-tail early-stopping arm chose 1195 rounds on `t13` and delivered +0.0028
  against the fixed 5000@0.03's +0.0068 — it **under-shoots on a well-featured
  table**. That is evidence about 0054's open item 2 (*where does the
  early-stopping block come from*), and evidence that the obvious answer is not
  good enough.
- The reading is a **probe** score: one lgbm, one table, one dataset, and 0056
  says the probe ranks unreliably at 200 trees. Direction trustworthy, magnitude
  not a Kaggle prediction. No submission — Rossmann's board is spent (0045).

### THE PLAN FOR THE NEXT CONTEXT — Ulf: *"ich möchte gerne besser als der median werden"*

Read 0067 first, then this. **The priority did not change; the number did.**
+0.0068 is **80% of Rossmann's 0.00846 gap** and is still the largest measured
lever in the project by a factor of three. What changed is that it is no longer
expected to be *sufficient* on its own.

One fact checked while writing this and worth knowing before anything else:
**`models.py` sets `n_estimators=200` for lgbm, xgb *and* catboost, and
`PROBE_PARAMS` does the same.** So the terminal fits — the models that actually
ship — have never run at a real budget either. Every capacity number in this repo
is a *probe* reading on a single lgbm.

**A. Two measurements first. No LLM, no submissions, hours not days.**

1. **v7's `t13` → the terminal blend at a real budget**, scored on v7's own sealed
   holdout, against the same blend at 200 trees. This is "what would actually
   ship", which no reading in this repo covers. Reuse
   `probe_capacity_v7_chain.py`'s copy-the-split pattern and
   `probe_restaurant_collapsed_columns.py`'s trick of driving the real
   `finalize_run` rather than reimplementing it.
2. **The same on Restaurant's `t7`** (already on disk in `_runs/restaurant_v2`).
   Two datasets, one question, and Restaurant's gap is larger (+0.01175) so it is
   the harder test.

Register the prediction first. Mine: the blend gains **less** than the single
lgbm's +0.0068, because an equal-weight average already recovers some of what
capacity buys — but this is a guess and the point of A is that it is cheap.

**B. Then the design decision, which is the actual build: 0054's open item 2 —
where does the budget come from?**

Not a constant (0054's rule, and 0067 strengthens it). The evidence in hand says
the *obvious* answer is not good enough either: inner-tail early stopping chose
1195 rounds on `t13` and delivered +0.0028 against a fixed 5000@0.03's +0.0068. So
the question is live, it touches 0023/0029/0041/0043 (which rows may the stopping
criterion see), and per this repo's habit it wants a **decision record before the
code**. The code then lands in `probe.py`/`models.py` so every fit chooses.

**C. Only after A and B: the remaining ~20%.**

If A confirms and B ships, Rossmann sits near 0.1313 against a median of 0.12967 —
about **0.0017 short**. That last piece is not identified. The candidate with the
strongest prior and the right general framing:

> **Every hyperparameter except `n_estimators` and `learning_rate` is a library
> default in every run this project has ever made.** On 916k rows lgbm's
> `num_leaves=31` is small.

Same class as capacity ("the harness never chooses model capacity"), so it names
no dataset. **But it must be built as a mechanism for choosing, never as a
hardcoded set of winning numbers** — that is CLAUDE.md's line and 0022 one level
up. One axis at a time (0017).

**Running alongside, not after:** 0062 step 6 (the notebook) and **quizzes for all
five blocks** are owed. CLAUDE.md is explicit that code Ulf has not followed is a
failure even with green tests, and that debt is now five blocks deep plus the
older backlog (slice 3, 0041/0042/0044, house-prices capacity).

**Honest expectation:** A+B plausibly closes ~80% of Rossmann's gap and less of
Restaurant's. Beating the median is *reachable* on this evidence, not *implied* by
it. Say so in whatever gets written next rather than letting the plan read as a
forecast.

### The methodological lesson, twice in one day

Both of today's registered predictions were wrong, and both for the same reason:
**they were argued from uncontrolled comparisons.** 0066's was argued from v1
against v2 (two agent trajectories); this one from bare-table capacity readings
against a table that ships. In both cases the confound was written down first and
leaned on anyway. When a number is about to become a priority, check which
comparison produced it.

---

## 2026-08-16 — the controlled test landed: the agent was right, and every instrument here was wrong

**Machine:** MacBook Air | **Branch:** master | **553 tests pass**.
Written up as [0066](docs/decisions/0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md).
**Read this before 0065** — it confirms 0065's central claim and refutes the
prediction I registered against it.

`scripts/probe_restaurant_collapsed_columns.py`: v2's own **t7** (six collapsed
reservation columns) against **t8** (same chain, those six dropped), on identical
rows, both shipped through the real `finalize_run` path. Two controls passed — the
t7 arm reproduced v2's blend val 0.519707 / sealed 0.505713 to six decimals *and*
came back from Kaggle at 0.55124 / 0.51197, digit-identical.

| t8 − t7, by distance from the end of training | |
|---|---|
| probe (in-sample block) | +0.00849 |
| validation | +0.01122 |
| sealed, adjacent | +0.01123 |
| public (nearer half of the horizon) | +0.00228 |
| **private (further half)** | **−0.00149** |

> **The collapsed column's advantage decays monotonically with distance and
> reverses sign where the competition grades.** The gradient is the evidence, not
> the flip.

**All three internal instruments were confidently wrong in the same direction** —
probe +0.0085 (and it is the argmax that ships), 0059's block interval
`sign_stable: True`, and the **sealed holdout worst of the three at +0.0112.** So:

> A sealed score is a lower bound on a *difference* too, and where the defect is in
> the data's **provenance** rather than in the model, the seal can order two
> candidates **backwards**. No block of these rows could have caught it.

**The argmax-shipping defect 0065 named is now measured, not theoretical:** the
agent diagnosed the leak, built the clean table, and the harness shipped the leaky
one, at 0.00149 private rmsle. Ulf's principle — *"der agent ist die Intelligenz,
die harness sollte ihn nicht beschneiden, ihn nur in richtige bahnen leiten"* —
was stated **before** this measurement and now has a number behind it.

### The mistake worth carrying forward, and it is mine

My registered prediction (t7 wins privately, the trap is a null) rested on two
*uncontrolled* comparisons — 0060's decay reading and v1-against-v2. 0065 names
that confound about v1/v2 in the same breath as registering the prediction, and I
leaned on it anyway. **A confound you have written down is still a confound.**

### Limits, so the number is not over-read

Twelve ranks (~1446 → ~1434 of 2150, both top 67%), both arms far below the private
median 0.53223. n=1 with no interval on the Kaggle difference; what lifts it is the
ordered gradient. **It does not license dropping such columns as a rule** (0022, one
level up) — the general claim is about the *instruments*. **The Restaurant private
board is now spent for this question** (three submissions scored).

### Next, and the direction is settled even though the detail is not

0065's open decision now has a direction: prefer the fix that lets the agent's
correct judgement survive (a not-to-ship channel) over the one where the harness
decides for it (provenance in the serving declaration). 0062 step 6 — the notebook
— is still the only unbuilt item of that build order. Quizzes still owed for all
five blocks.

---

## 2026-08-16 — Restaurant v2 ran: the agent found the leak and lost the argument

**Machine:** MacBook Air | **Branch:** master | **553 tests pass**.
Written up as [0065](docs/decisions/0065-the-agent-found-the-leak-and-the-ruler-overruled-it.md).
0062 steps 5 and the run are done; **step 6 (notebook) is the only one left**.

`scripts/run_restaurant_v2.py` — v1 with **one change**, the streams declared. 13
tables, 5 terminal fits, 8.2 min, agent finalized itself. Five predictions
registered in the docstring before it started.

### The result, and the ordering is the finding

| table | | probe | vs parent | block sd | stable |
|---|---|---|---|---|---|
| t4 | stream aggregate, air, leads 7/14/30 | 0.55340 | **+0.00497** | 0.00287 | True |
| t5 | stream aggregate, hpg, leads 7/14 | 0.55546 | −0.00206 | 0.00091 | True |
| **t7** | target — **shipped** | **0.52135** | +0.03205 | 0.00458 | True |
| **t8** | t7 with the collapsed reservation columns dropped | 0.52984 | **−0.00849** | 0.00297 | True |

The agent used both streams, **diagnosed the cut itself off `column_notes`**, built
the honest table — and `best_table` is the argmax of the probe, so the leaky one
shipped. Its own closing words: *"I deliberately built and tested a version without
them (t8, 0.5298) to avoid that leak, but the harness's automatic selection of the
best table chose the version that includes them."*

> **Adding honest columns helped. Removing dishonest ones was punished, at 2.9
> block sd with every block agreeing.** The ruler is not undecided; it is confident
> and pointing the wrong way — because no block of these rows can measure the
> difference, every one of them sharing the cut.

Predictions: **1 confirmed** (it reached the stream at all), **2 refuted** (it chose
a *ladder* 7/14/30, not the declared horizon of 1 — the tool's own intended use,
which the prediction failed to consider), **3 confirmed in a sharper form**,
**4 confirmed** (9 honest and 6 collapsed columns shipped, of 58), **5 open**.

Sealed optimism negative on all five (−0.009 to −0.025) — v1's exact signature
before Kaggle punished it by +0.042. **A run cannot detect this from inside
itself.**

### Owed, and it is the actionable item

**`_runs/restaurant_v2/submission.csv` is written and not scored.** 0065 registers
the prediction that v2 decays *less* sealed→private than v1's +0.042, despite v2's
sealed number (0.50571) being marginally worse than v1's (0.50176). Upload it
before doing anything else with this thread — the record says which way to read
both outcomes.

### Two gotchas

- **0059's block interval was confident and wrong, for the second time.** t8 reads
  `sign_stable: True` at 2.9 sd. That is not a bug — blocking samples the periods
  it has, and all of them share the defect — but stop citing `sign_stable` as if it
  bore on honesty.
- **My summariser flattered the run.** It classified shipped columns as honest by
  testing `"_lead" in name`, which caught `air_reserve_lead_days` — a *collapsed*
  column. Printed 11 honest / 4 collapsed; truth is 9 / 6. 0050's shape in the
  reporting layer for the third time.

### The open decision, deliberately not taken

Three shapes for letting an agent's correct diagnosis survive, none chosen: a
channel to mark a table as not-to-ship (narrower than returning the terminal
choice, which 0035 removed on purpose); extending the serving declaration from
*presence* to *provenance* (the gap 0030 leaves); or an evaluation block that does
not share the cut (0041/0043). Needs its own record and probably a second dataset.

---

## 2026-08-16 — 0062 steps 1-4 built; the grain moved into the library

**Machine:** MacBook Air | **Branch:** master | **552 tests pass** (was 479).
Two new records: [0063](docs/decisions/0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md)
and [0064](docs/decisions/0064-month-ahead-is-the-same-idea-and-does-not-get-the-same-tool.md).
**Nothing has been run against real data yet** — every number below is a test, not
a measurement.

**0062's build order, steps 1-4 done, 5-6 open:**

| | | tests |
|---|---|---|
| 1 | `StreamAggregateEncoder` in `encoders.py` | 33 |
| 2 | stream registry on `RunSession`, `inspect_data`, brief | 18 |
| 3 | `fit_stream_aggregate` in `tools.py` + `agent.py` | 17 |
| 4 | `Pipeline.transform(streams=)`, `predict_bundle`, manifest | 5 |
| 5 | `prep_restaurant.py` writes the stream out | **open** |
| 6 | notebook + quiz | **open** |

### The design call 0062 left open, and Ulf overruled my first answer

A stream is stamped finer than the table it feeds (a booking at 18:00, a
store-day row). Something has to say which row a record belongs to. **I first
required the two to line up and left the aligning to `prep_restaurant.py`.** Ulf
rejected it, and the reason generalises past this dataset: *"etwas dem prep skript
überlassen ist gefährlich … es sollte so wenig wie möglich in prep stehen müssen
und alles was dort drin stehen muss, muss festgehalten werden. sonst können andere
mein tool nicht nutzen."*

He is right, and it sharpens a line CLAUDE.md draws loosely: `prep_*.py` is a home
for **dataset knowledge**, not for **library preconditions**. A rule every user
must obey that the library never states, checks or records is a trap — and getting
it wrong here yields a well-formed column of zeros, not an error.

So **`fit` measures the table's commonest row spacing, stores it as a width plus a
phase, and floors both sides onto it** (0063). Believed only if every distinct
ordering value sits on that grid; otherwise it refuses and asks for `event_grain`
to be declared. Three properties keep it defensible: the **lead keeps full
resolution** (flooring picks the row, never what was knowable), the grid is
**stored arithmetic and not a calendar** (every serving row is past the end of the
fitted table), and **raggedness does not move it** (gaps are read table-wide).

Why this is not 0051's inference: 0051 forbids guessing a *unit*. The unit is
already fixed; what is measured is the row spacing *in that unit*, and it is
checked against the data it came from. An inference that cannot be falsified is a
guess; this one is falsified by any row off the grid.

### 0064 — a capability refused after it passed every test for adding one

Ulf, reading the tool description: *"'month-ahead models' hat keinen zweiten input
oder?"* Correct. **0057 item 1 is two capabilities in one sentence** and only the
bitemporal-stream half is built.

0062 said `fit_history` "substantially does it already". **True in shape, false in
scale**: there is no `gap` argument, and the gap that does appear comes from the
fold geometry — a training row sits 0..`block_span` past its block's freeze, a
served row `horizon`..`horizon + block_span`. 0049 measured it: training ages
1-305, validation ages 1-45.

Refused anyway, and the third reason is the one that decides it: **0053 showed the
*existing* knob cannot be searched** (+0.047 at `n_splits=3`, −0.093 at 5, identical
counts), so a second free parameter enlarges only what the agent must guess. The
real objection is that the asymmetry belongs to the **evaluation geometry**
(0041/0043) — a per-feature knob lets the agent work around the exam instead of
fixing it. Discriminating test against the stream: *there, no combination of
existing tools could express the honest feature at all.*

### Gotchas for whoever picks this up

- **`probe_table` replays the lineage to check servability, so step 3 could not
  work without part of step 4.** Without the run's stream reaching
  `Pipeline.transform`, every stream table would have been marked unservable and
  **silently left unscored** — a missing number, not an error. `_run_streams()`
  and a test pin it.
- **A stale stream does not raise.** Hand the training-time copy back to
  `predict_bundle` and it predicts, with `_count` reading 0. What protects the
  caller is the count column, which is why it is never optional.
- **The manifest names `event_time` and `observed_at`.** Which of two timestamps
  is the observation time cannot be recovered by looking, and reading the wrong
  one builds the feature out of the future without failing.
- Adding an agent tool trips `test_harness_agent.py`'s pinned tool set on purpose
  (0018) — that failure is the guard working.

### Still owed

**Quizzes for all four steps** — none asked yet this session, on top of the
backlog (slice 3, 0041/0042/0044, house-prices capacity). `scripts/run_rossmann.py`
still broken since 0035. The `fit_history` train/serve age diagnostic named in
0064's cost section is not built (0058 item 3: instruments when a run needs them).

---

## 2026-08-15 — 0057 item 1 designed, build started

**Machine:** MacBook Air | **Branch:** master.
Design written up as [0062](docs/decisions/0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
**before** the code. Read that first — it is the plan, not a summary of one.

Building 0057 item 1 (lead-time-restricted stream aggregates), per 0058's revised
order. **No code written yet at the time of this entry.**

**The build order is 0062's, and its step list is the todo list:**

1. `StreamAggregateEncoder` in `encoders.py` + tests
2. stream registry on `RunSession`, surfaced in `inspect_data` + brief
3. `fit_stream_aggregate` in `harness/tools.py` + `harness/agent.py`
4. `Pipeline.transform(df, streams=...)`, `predict_bundle`, bundle manifest
5. `prep_restaurant.py` writes the stream out (collapsed column stays, 0029)
6. notebook + quiz Ulf

**Two decisions already taken, do not re-litigate them:**

- **Recipe, not materialised state.** The encoder stores only
  `(by, value, event_time, observed_at, lead, stat)`; the stream is passed in
  again at predict time. Storing the precomputed `{(key, event_time) -> value}`
  map needs no plumbing at all and was still rejected — it only works while the
  stream file already contains the future, which is true of a Kaggle download and
  false of any real serving. See 0062's "What was rejected".
- **No probe first.** Measuring "does the lead help on Restaurant" was proposed
  and dropped: that is the leaderboard-shaped question CLAUDE.md rules out. The
  library question is whether the agent can express it; 0029 says the agent
  decides whether to use it.

**Gotcha for whoever picks this up:** 0057 claims item 1 was found independently
twice. 0062 narrows that — the Rossmann winner's "month ahead models" restrict an
aggregate of the *main table's own past*, which `fit_history` largely does
already. The bitemporal-stream-at-another-grain case is one dataset of four.

---

## 2026-08-15 — we were measuring against the wrong board

**Machine:** MacBook Air | **Branch:** master.
Written up as [0061](docs/decisions/0061-we-were-measuring-against-the-wrong-board.md).

Every Rossmann script hardcoded `median: 0.11677` off the **public** board while
quoting **private** scores beside it. Found only because 0060 made the identical
mistake on Restaurant an hour earlier, so Rossmann got checked.

| board | best | top 10% | median |
|---|---|---|---|
| public | 0.08932 | 0.10294 | 0.11678 |
| **private** | 0.10021 | 0.11774 | **0.12967** |

**v7 blend: private 0.13813 = top 58%, +0.00846 from the median** — not the
+0.0214 this file has been repeating. **The gap is a third of what we thought.**
Distance to private top 10% is 0.0204.

That does not make the work better — the rank is what it always was — but it
changes what counts as worth building: **a mechanism worth 0.01 now clears the
median**, where under the old reference line nothing short of 0.021 did.

**The trap in reading that optimistically:** capacity *was* already spent, on a
bare feature block, and that arm sits at top 69% — behind v7's chain at 200 trees
(top 58%). The untested cell is still **v7's chain at a real budget**.

### Unlooked-for: we are more horizon-robust than the field, on both datasets

| | field mean, private − public | ours | Spearman |
|---|---|---|---|
| Rossmann | +0.01473 | **+0.00881** | 0.969 |
| Restaurant | +0.04464 | **+0.03097** | 0.928 |

Rank improves on both (62%→58%, 72%→64%). Either the ordered split asks the
forecasting question honestly (0029), or a mid-table model has less to lose and
this is regression to the mean. **Not resolved** — the test is whether it survives
among teams at similar public scores, one groupby, not run.

Also: Rossmann's between-board Spearman is **0.969**. Its shake-up reputation is
about rank movement at the very top, not a noisy score distribution — which means
0045 treated the private board as a cleaner seal than it credited.

### Fixed

`prep_rossmann.leaderboard(private=True)` / `leaderboard_rank(score,
private=True)` read both CSVs from disk, defaulting to private. Runner constants
deliberately **not** rewritten — they are inputs to runs that already happened.

---

## 2026-08-15 (later) — Restaurant v1 ran and was scored; the seal is a lower bound

**Machine:** MacBook Air | **Branch:** master.
Written up as [0060](docs/decisions/0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md).
**Next session: 0057 — but follow [0058](docs/decisions/0058-the-ranking-failure-was-one-feature-not-the-block.md)'s
revised order, not 0057's own.** 0057 step 1 says "fix the ranking block first";
0058 withdrew that and 0059 built the interval instead. So: **lead-time-restricted
aggregates first, then per-group trend extrapolation.**

`scripts/run_restaurant_v1.py`, Rossmann v8's config verbatim. 17 tables, 5 terminal
fits, 21.9 min, agent finalized itself. Five predictions registered before the run.

### The measurement, and the ordering is the finding

| | rmsle |
|---|---|
| validation (lgbm, selected) | 0.52099 |
| **sealed holdout, blend — shipped** | **0.50176** |
| Kaggle public (nearer half) | 0.51301 |
| **Kaggle private (further half)** | **0.54398** |

Val→sealed optimism was **negative on all five models** (−0.012 to −0.026), which
reads as "the exam is honest" — the same signature v5 produced on Rossmann. It was
not. **sealed→private is +0.042.** Four windows in time order, each fair about
itself, each flattering the next.

> **A sealed score is a lower bound on the error at the horizon you will be graded
> at, not an estimate of it.**

Ranks, both boards (2,150 teams): public ~1544 (top 72%), **private ~1369 (top
64%), only +0.0118 above the median** (0.53223). For scale, 0054 measured
store-identity + capacity on this dataset at **+0.0283** — one already-measured
mechanism clears both median and top 10%.

### A mechanism refuted, and only the private board could do it

Prediction 2 said Kaggle would punish more at the far edge. Direction right (+0.031
public→private). **Mechanism wrong:** joining both boards on team, the field's mean
private−public is **+0.04464** against our **+0.03097**. Everyone decays with
horizon and **we decay less than average** — our rank *improves* 72%→64%. So the
decay says nothing about the cut reservation stream it was predicted to expose. A
control run with those columns withheld is one script, not yet written.

Read against the public board alone the story was available and wrong.

### The other predictions

- **1 confirmed** — 6 reservation features in the shipped model (67 total).
- **3 confirmed** — **6 of 17 comparisons `sign_stable: False`**, the block mode's
  first live firing. Every unstable table has |gain| < 0.003 against a block sd 2-5x
  larger; two negative. The four real steps (+0.190, +0.034, +0.032, +0.013) are all
  5/5 stable. It separated signal from tail cleanly.
- **4 untested, still open** — the agent built 90/180/365-*day* windows on a
  **ragged** spine where `fit_history` counts **rows**. Nothing caught it, nothing
  measured it. Real gap in the tool.
- **5 confirmed** — below median.

### Unpredicted: the agent used the new instrument

Stopped at 17 of 25 permitted tables, saying gains were *"below the probe's own
standard error and not cross-block-stable … so I stopped rather than overfit."*
0059 was eight hours old. **Patience cannot explain it** — on an ordered split
`probe_se` is `None`, so the bar floors at zero and every positive gain still counts
as progress. Weigh against 0027 (three advisory steers that changed nothing). n=1.

Also: the agent read the sealed score post-finalize and narrated it. Known, open
since 0042.

### Not done, deliberately

Submission **not clipped** — one prediction of 32,019 came out at 2868.5 against a
training max of 877 (~0.0015 of rmsle). Flooring it after seeing the score is the
correction 0022 exists to refuse. A general "predictions outside the training
range" diagnostic would be legitimate and names no dataset.

### Also this session

`docs/decisions/README.md` indexed; HANDOFF pruned to the ~2-week rule and the
08-05..08-13 entries compressed to summaries (full text at `801b52d:HANDOFF.md`).

---

## 2026-08-15 — the ruler is fixed, and it would not have caught the thing it was built for

**Machine:** MacBook Air | **Branch:** master | **479 tests pass** (was 456).
Written up as [0059](docs/decisions/0059-the-interval-belongs-to-the-difference-not-the-score.md).
0036's paired bootstrap is **built** — owed since 2026-08-13 and named the blocking
item in four entries.

**Priority call:** 0058's list is capabilities; this is the instrument that measures
them. Seven feature readings had been decided on differences with no interval, and
0052/0053/0058 are a trail of mechanisms withdrawn afterwards. Lead-time-restricted
aggregates (0058 item 1) is still next; it now has a yardstick to be measured with.

### What was built

`src/tabular_agent/compare.py`. Every table-vs-parent comparison now carries an
interval around the *difference*, not two point scores against a pooled `val_se`.

- **Random split → paired bootstrap over rows.** One resample, both candidates
  scored on it. That requires both prediction vectors, so `probe_table` now stores
  each table's validation predictions (`run_dir/probe_preds/<id>.npy`, one float64
  vector, 400KB on Rossmann, bounded by `max_tables`).
- **Ordered split → the difference recomputed across consecutive blocks.** A row
  bootstrap is refused there for the reason `val_se` already was: the rows are one
  *period*, and resampling inside it cannot see which period it is.
- **The mode is chosen by passing the ordering column, not a flag** — a flag lets a
  caller say "ordered" without supplying the order, the one case that cannot be
  served.
- `_score_on_held_out` returns a 4-tuple now (predictions alongside the score); six
  call sites unpack one more value.

### The bug the design was written around

**The validation half is not stored in its ordering column's order.** `RunSession`
picks the val positions in time order then writes them with `np.sort(val_pos)` —
which sorts the *positions*, so the file comes back in the input's row order.
Rossmann's raw file is **reverse** chronological (`ship_rossmann_median.py` was
already bitten by this). Blocking on file order would have cut five arbitrary
slices and reported a reassuring number. The ordering column is a required argument
and the sort happens inside the instrument.

### The measurement, and my prediction half failed

`scripts/probe_difference_interval.py`, against the history arm Kaggle already
refuted (t9 vs t8 in `_runs/ship_rossmann_median`; **−0.0301 private**).

| instrument | reading |
|---|---|
| pooled gain, t9 over t8 | **+0.02028** |
| paired bootstrap over rows | se **0.00074** → **gain/se 27.3** |
| pairing ratio | **0.371** |
| blocks, 44-day val window | sd **0.01214**, `sign_stable` **True**, 5/5 |
| blocks, 90-day window (val+holdout) | sd **0.01670**, `sign_stable` **False**, 8/10 |
| Kaggle, Aug–Sep | **−0.0301** |

1. **The row bootstrap is confidently wrong** — 27 standard errors from zero for a
   difference that reverses externally.
2. **0036's pairing ratio replicated**: 0.371 inside its 0.18–0.37, on a second
   dataset, a different metric, 50,175 val rows against 292. 0036 named that as the
   thing it could not establish.
3. **Blocking helped 16x and was still not enough.** `gain/se` 27.3 → 1.7, but all
   five blocks still favoured the history table. **A run would not have caught it.**
   Registered prediction 2 said the blocks would disagree; on the window the harness
   actually uses, they did not.
4. **The window was the problem, not the block count.** At 90 days the blocks
   disagree, the pooled gain nearly halves, and the decay is direct: first five
   blocks **+0.0200**, last five **+0.0022**. Kaggle's window starts two weeks later
   and reads −0.0301 — the same line continued.

So the limitation `compare.py` states in its own docstring *before* it was measured
is the one that bit: blocking samples the periods it has. More blocks over the same
44 days would not help; reaching further out would, and that is 0043's trade-off,
not this instrument's.

### Deliberately not done

**`min_gain_ratio` is unchanged at 0.25.** 0028 left the magnitude open and 0036
proposed this as a candidate, but that is one comparison on one dataset — picking a
multiple of the block sd from it is 0034's refusal verbatim. The interval is
reported; nothing is gated on it.

Also: `tests/test_compare.py` (15), `tests/test_compare_harness.py` (8),
`notebooks/phase5_the_interval_on_a_difference.ipynb` (35 cells, 14 code, ships
unexecuted, every code cell smoke-tested in order).

### Still owed

**Quizzes**: this chunk (asked, unanswered), slice 3, 0041/0042/0044, house-prices
capacity. `scripts/run_rossmann.py` still broken since 0035. The request timeout is
still only patched in one driver.

---

## 2026-08-14 (night, newest) — four submissions, and the ruler is the problem

**Machine:** MacBook Air | **Branch:** master.
[0056](docs/decisions/0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
(the measurements) and
[0057](docs/decisions/0057-the-capabilities-the-agent-does-not-have.md)
(what to build, nothing built yet).

| arm | val | holdout | Kaggle priv | Kaggle pub |
|---|---|---|---|---|
| bare block, 200 trees | 0.23331 | 0.24228 | 0.23048 | 0.23420 |
| bare block, 4,144 trees | 0.16000 | 0.15766 | **0.15166** | **0.14853** |
| full pipeline + history, blend | **0.13276** | 0.17678 | 0.18438 | 0.17444 |
| full pipeline no history, blend | 0.14793 | 0.15841 | 0.15429 | 0.16193 |

1. **Capacity confirmed externally**: +0.0788 private against a predicted +0.0846.
2. **`val` ranks at Spearman +0.20 against Kaggle; the sealed holdout ranks at
   +1.00.** The holdout is the six weeks adjacent to the test window. **We rank
   every table on the wrong block** — 0043's trade-off priced the other way round.
3. **History refuted externally**, −0.0301. Fifth negative, first external. The
   competition winner independently reports recent data barely mattering to him.
4. **The encoding pipeline is worth nothing once the budget is right** — one lgbm
   on a bare feature block beats the three-way blended pipeline.

**We did not beat v7 (0.13813 private / 0.12932 public) and are not near the
median (0.11677).** And an error worth not repeating: the rebuild targeted v8's
chain because `t24` had the best *probe* score, but v7 is the better run — the
rebuild target was chosen with the instrument 0056 shows is unreliable.

**Finding 2 was then withdrawn the same night — see
[0058](docs/decisions/0058-the-ranking-failure-was-one-feature-not-the-block.md).**
Ulf asked whether it was Rossmann-specific. It was: drop the one history arm and
`val` ranks fine (its only error is a pair separated by 0.0026); that arm decays
with distance from its 2015-05-01 freeze, which is 0049/0050 not a block problem;
`val`'s window is the odd one on *composition* (20.8% closed vs test's 14.6%, a
May-holidays accident) rather than on recency; and a perfect ordering of four
happens 1 time in 24 by chance. **Findings 1, 3 and 4 stand** — they rest on
differences 10-30x larger.

**Next, in this order** (0058's revision of 0057):
1. **Lead-time-restricted aggregates** — the only capability found independently
   twice, and `restaurant_visitor` already carries a natural experiment for it.
2. **Per-group trend extrapolation** — justified by GBDTs not extrapolating at all,
   which names no dataset.
3. Two *instruments*, when a run needs them, not pre-emptively: a composition
   diagnostic comparing the evaluation block against `serving_columns`, and
   ranking stability across several ordered blocks (which would have caught the
   history arm with no external seal).
4. `fit_history` should leave the vocabulary in its current form.

**Do not** swap which block ranks, and do not hardcode a block choice. 0043 stands.

Not done: 0036's paired bootstrap, still. Every number above is a difference with
no interval. Quizzes still owed.

---

## 2026-08-14 (evening) — the capacity plan was run, and it was wrong

**Machine:** MacBook Air | **Branch:** master.
Written up as [0054](docs/decisions/0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
and [0055](docs/decisions/0055-a-rename-makes-every-earlier-artifact-unreplayable.md).
**The START HERE plan below is now superseded on its main point — read this first.**

Ulf: *prepare Recruit Restaurant as a new dataset, then check the n_estimators
hypothesis everywhere, it is most probably wrong.* Both done, and he was right.

### New dataset: Recruit Restaurant Visitor Forecasting

`scripts/prep_restaurant.py`, `notebooks/phase5_restaurant_dataset.ipynb`,
public leaderboard in `restaurant_visitor/` with `leaderboard_rank()`.
252,108 store-days, 829 stores, rmsle, ordered on `visit_date`, 5%/5% like
Rossmann. Six files joined; two of them at a finer grain, which is the join the
harness has no tool for.

**It came with a trap worth the dataset.** The reservation stream is cut at the
moment the forecast is issued, so 30% of *training* reservations were booked
same-day while **no** forecast-window row can have one — and the floor grows to 39
days at the far edge. No guard we have sees it: no fold crossed, no encoder
misfitted, column present at serve time, and validation *confirms* it because
validation shares the uncut stream. Kept and documented, per the `Customers`
precedent (0029). The general capability it asks for — *aggregate a related stream
restricted to what was knowable a given lead time ahead* — does not exist.

Also: the spine is **ragged** (a store has no row on days it did not report), so
"last 364 rows" ≠ "last 364 days" here. Every window feature from 0052/0053
assumed they were the same, because on Rossmann they are.

### The capacity hypothesis is refuted; the mechanism is identity, not rows

`scripts/probe_capacity.py`, `_runs/capacity_probe.csv`,
`notebooks/phase5_model_capacity.ipynb`. Prediction registered before running.

| dataset | rows | best fixed budget | early stop |
|---|---|---|---|
| house-prices | 1,168 | −0.0034 (all lose) | 162 → **+0.0062** |
| restaurant | 227,139 | +0.0004 | 1,018 → +0.0003 |
| Porto Seguro | 476,170 | −0.0188 (all but one lose) | 181 → **+0.0049** |
| Rossmann | 915,744 | **+0.0573** | 16,926 → **+0.0606** |

Restaurant at 227k gains nothing, so it is not rows. The confound — nearly written
up as the conclusion — is that Rossmann's `Store` is an *integer* and survives
`_feature_frame` while restaurant's `air_store_id` is a *string* and is dropped.
Give restaurant an arbitrary integer store code and the same rows go from **+0.0004
to +0.0283**. It is per-entity effects: 200 trees cannot reach 829 shop effects.

### The thing nobody was looking for

On Rossmann, v8's own feature block measured against the same split:

| rmspe | baseline | + feature block | features worth |
|---|---|---|---|
| **200 trees (today)** | 0.21810 | 0.23331 | **−0.0152** |
| 5000 @ lr 0.03 | 0.16083 | 0.15213 | **+0.0087** |

**The sign reverses.** The ruler that ranks every table in every run has been
changing the sign of what it measures. `best_table` is that ruler's argmax.

### What to do next, in order

1. **Do not raise the constant.** Early stopping asked for 162 / 181 / 1018 / 3430
   / 16926 — two orders of magnitude. Any constant big enough for Rossmann wrecks
   house and Seguro; picking Rossmann's is a leaderboard-fit (0022, one level up).
2. **Design where the early-stopping block comes from.** This is the open decision.
   The arm here carves the tail 10% off the work half — fine for an ordered split,
   needs an argument otherwise, and it touches 0023/0029/0041/0043. **Not** an
   argument for giving params back to the agent (0035).
3. **Re-read `_runs/` with suspicion.** Every table comparison was made at 200
   trees. 0052 and 0053 are about features measured by this ruler.
4. **0055 first if anything is to be shipped:** v8's tables cannot be replayed at
   HEAD. Cheapest useful repair is a replay test against a checked-in artifact, so
   the *next* rename is loud at the commit rather than four hours later by luck.

### Owed, unchanged

0036's paired bootstrap still not wired in — every number above is still a
difference with no interval around it. Quizzes owed: slice 3, 0041/0042/0044, and
the one asked this session (house-prices capacity, unanswered).
`scripts/run_rossmann.py` still broken since 0035.

---

## SUPERSEDED — the old plan, kept only so its refutation is legible

> **Do not follow this section.** Its central claim — that model capacity matters
> in proportion to *rows* — was tested on 2026-08-14 and **refuted**
> ([0054](docs/decisions/0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)):
> a 227k-row dataset gained nothing while a 916k-row one gained 0.057, and the
> real mechanism is high-cardinality *identity*, not size. Its second guess — that
> capacity would dwarf the feature search — turned out to be right for a different
> reason than it gives.
>
> **The current plan is the newest entry at the top of this file.** This block is
> left intact rather than deleted because the decision records cite it as what we
> believed beforehand, and rewriting it would erase the prediction that was made
> before the measurement.

Written 2026-08-14 at Ulf's request, so a fresh context can begin without
re-deriving anything. Read this, then `docs/decisions/README.md`, then the entries
below for detail.

### Where we actually are

Sixteen runs, two datasets. **Rossmann's best is ~rank 2,200/3,303 — below
median.** House-prices reached top 9%, Porto Seguro top 57%. The harness is sound
(leakage-safe, sealed holdout, replayable bundles) and the *scores* are mediocre
on the forecasting task. Ulf: *"somehow we need to get better than the median, do
not know what we miss still."*

### My best guess at what we are missing, and it is not features

**Capacity, not vocabulary.** Every fit in this harness runs at a **fixed, modest
budget that nothing ever tunes**. Checked, so the next session does not have to:

    probe.py:110      PROBE_PARAMS = {"n_estimators": 200}
    models.py:419-432 lgbm / xgb / catboost defaults: n_estimators=200
    learning_rate     never set anywhere — whatever the library ships (lgbm: 0.1)

So it is 200 trees at lr 0.1, on 850k rows, for every model in every run. That is
not absurd, but competitive solutions on a dataset this size run thousands of
trees at a low learning rate, and **the harness cannot express that at all** —
there is no tuning step, no early stopping, and 0035 removed the agent's ability
to touch model params on purpose.

This is not a new idea — **0039 named it and left it open deliberately**: *"the
terminal phase enumerates library defaults, so it measures the libraries. A fair
comparison needs a common budget or early stopping — left open."* It has been
open since. Meanwhile three sessions went into feature vocabulary, and 0052/0053
just spent a whole slice finding that the newest feature is worth less than one
we already had.

**The cheapest decisive test, and it should be the first thing next session:**
take v8's best table and fit the probe estimator twice — the current 200 trees at
lr 0.1 versus a real budget (lr 0.03, a few thousand trees, early stopping on the
val block) — and read the difference. Register the prediction first. If capacity
is worth more than ~0.02 rmspe it dwarfs everything the feature search has ever
bought and the project's priorities are wrong.

It also explains the *pattern* of our results: house-prices (1,168 rows) reached
top 9% where 200 trees is roughly right, while the two large datasets (850k, 476k)
land mid-table — which is exactly where a fixed small budget would bite.

It is also **general** — "the harness never chooses model capacity" names no
dataset, which is the test CLAUDE.md sets.

Caveat to respect: 0035 removed the model tools deliberately, and 0029 says the
harness owns the exam. **This is not an argument for handing hyperparameters back
to the agent** — it is an argument for the terminal phase fitting each family at a
defensible budget instead of at whatever the library ships.

### `min_periods` — worth building, as an instrument

Ulf asked. The literature's standard answer to "rows that cannot fill a window"
is `min_periods` (skforecast's `RollingFeatures` drops such rows; a
wildfire-forecasting paper excludes its first three years for a 36-month window).

**Do not ship it as a guard.** 0053 showed `w=728` has identical count
distributions at `n_splits=3` and `5` with opposite outcomes, so a threshold on
counts cannot repair a difference that exists between two same-count
configurations. That would be the fourth guard this session built on an unverified
story.

**Do build it as a discriminating experiment.** Raggedness — the per-fold ramp
20/39/59/78/93 — is the last untested candidate mechanism, and `min_periods` is
exactly the knob that removes it. Set it high, re-run the n=3/n=5 pair:

- gap collapses ⇒ raggedness is the mechanism, and `min_periods` becomes a fix
  *on evidence*;
- gap survives ⇒ raggedness is dead too, and the cause is elsewhere.

Implementation is nearly free: NaN the statistic where `_count < min_periods`.
Blank rather than drop rows — a harness table's row count is load-bearing, and
NaN is already a first-class "no estimate" (0046/0047).

### A second forecasting dataset — yes, and pick one that breaks assumptions

Rossmann's private LB is **spent** for the blend-membership question (0045), and
every conclusion about ordered splits rests on one dataset. Candidates, with what
each would stress that Rossmann does not:

- **M5 / Walmart (hierarchical, intermittent).** Many series, lots of true zeros,
  hierarchical aggregation. Would stress everything this harness assumes about a
  single flat table. Heaviest lift, and its metric is complex — mind 0022, the
  yardstick must measure the ranked quantity.
- **Corporación Favorita.** Closest to Rossmann in shape, much bigger. Best for
  *comparability* — a second reading of the same claims — worst for finding new
  gaps.
- **Recruit Restaurant Visitor Forecasting.** Genuine multi-table join, different
  horizon, moderate size. A good middle option.

**My recommendation: Recruit Restaurant or Favorita**, because the immediate
question is "why are we below median on a task we understand", and answering it
needs a *comparable* second reading, not a harder problem. Save M5 for when the
capacity question is settled.

Whichever it is: `scripts/prep_*.py` holds the join, typing, metric and split —
nothing dataset-specific enters `src/` (CLAUDE.md).

### Distilling a winner writeup — the guardrail

Ulf wants to re-read the Rossmann winner (Gert Jacobusse) and distill. That is
sanctioned design input — it is where slice 3's argument came from — but the
distillation has to come out as **capabilities stated without naming the dataset**,
or it is leaderboard-fitting one level up (0022, CLAUDE.md).

The test that already worked once: *"per-group aggregates of a column, fitted once
and replayed"* passes; *"exclude rows where the store was closed"* does not — its
general form is *"some rows should not be trained on"*, and **that** is what gets
built.

Questions worth putting to a writeup, none of which name a dataset:

- What did they do about **model capacity and early stopping**? (Suspected biggest
  gap — see above.)
- Did they **weight or filter training rows** (recency weighting, dropping
  structurally-zero rows)? The harness has `filter_rows` but no weighting at all.
- How did they handle the **target transform** and any post-hoc correction?
  (0021 owns the transform; the 0.985 multiplier stays out — its own author reports
  the effect was inside CV noise.)
- Did they fit **one model or many** (per segment)? The harness can only fit one.
- What did their **validation** look like? Ours is one period with no error bar,
  which 0036 was supposed to fix.

### The blocking item behind all of it

**0036's paired bootstrap is still not wired in.** Ordered split ⇒ val is one
period, `val_se` is `None`, n=1 in the unit that varies. 0052 and 0053 are both
built on differences nobody can put an interval around, and the capacity
experiment above will produce another one. Fix the ruler before trusting the next
number.

### Also owed

- **Quizzes**: slice 3 (one question asked and answered), plus 0041/0042/0044 from
  earlier sessions.
- `scripts/run_rossmann.py` broken since 0035 (`StopPolicy(max_experiments=)`
  renamed to `max_tables`).
- Request timeout still only patched in one driver, not lifted into the library.
- `_latest` still emitted unconditionally by `fit_history`, so every history table
  carries the pair that destroyed v8's t7.
- `to_periods` turns every date into days; a second-resolution log is unusable.
  Fix is a *declared* unit (0051), not an inferred one.

---

## 2026-08-14 (night, last) — Ulf's question refuted 0052's mechanism

**Machine:** MacBook Air | **Branch:** master | suite green.
Written up as [0053](docs/decisions/0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md).

Ulf, reading 0052: *"but the 136 for rossmann are only due to how we split the
data into folds no?"* Yes — fold 0's prefix is `span/(n_splits+1)`. Testing it
killed the explanation.

Same sweep at `n_splits=3` and `5`, everything else identical, both reproducing
to the digit:

| feature | n=5 | n=3 | swing |
|---|---|---|---|
| all-time mean of `Store_DOW` (0046) | +0.06048 | +0.06004 | **0.0004** |
| window 364 | +0.04458 | +0.05338 | 0.0088 |
| window 546 | −0.00030 | +0.04078 | 0.0411 |
| window 728 | **−0.09287** | **+0.04654** | **0.1394** |

**w=728's train/serve counts are identical in both** (median 59 against a served
101, 1.71x). The quantity 0052 blamed did not move while the outcome reversed.
Across pairs the ratio is *anti*-correlated with the score, and the two-level
guard I had just built fires in **both** configurations — 728 exceeds every prefix
at n=3 too (204/408/612). A guard that fires on the good case and the ruinous one
alike is uninformative.

**What survives.** 0052's measurements (all reproduced) and its conclusion: the
all-time mean beats every window under *both* fold counts, so slice 3 still buys
nothing over 0046. And a new statement about the feature class —

> a trailing window's measured worth is substantially an artifact of the fold
> count, sensitivity growing with length, while the unbounded aggregate it loses
> to is invariant.

So **window length cannot be searched over here and the number believed** (0022's
problem, arriving in a feature instead of a metric).

**No replacement mechanism is offered**, deliberately. Candidates considered and
untested: fold-dependent NaN share (25% at n=3 vs 17% at n=5), number of coverage
regimes in the trained column, interaction of the ramp with the target's time
trend. **Third failed story about these columns in one session** (0050's
`summarise_history` verdict, 0052's coverage count, 0052's prefix mechanism).

**Code:** prefixes still reported (the agent cannot get them otherwise), warnings
demoted to notes that describe the geometry and explicitly decline to predict a
cost.

**From the literature** (searched at Ulf's request): the two standard answers are
fixed-size sliding-window CV (`TimeSeriesSplit(max_train_size=...)`, skforecast's
"fixed training size" backtesting) and **`min_periods` — skforecast's
`RollingFeatures` drops training rows that cannot fill the window**, and a
wildfire-forecasting paper excludes its first three years for a 36-month window.
`min_periods` is the better fit here and is nearly free (NaN where `_count <
min`), but **not built**: 0053 says the mechanism is unknown, and building a fix
for an unidentified failure is how the last two guards got written. Switching CV
schemes stays out too — it changes the exam for every feature and makes all 16
prior runs incomparable, which is 0041's argument for opt-in.

**Next:** 0036's paired bootstrap. Two records now rest on differences nobody can
put an interval around; it is the blocking item for this whole area.

## 2026-08-14 (night, latest) — slice 3 built, and the measurement says it lost

**Machine:** MacBook Air | **Branch:** master | **suite green.**
Written up as [0052](docs/decisions/0052-a-window-is-a-worse-estimator-of-a-better-idea.md).

Windows are in `fit_history(windows=[[364, "mean"]])`. Row-anchored `[row - w, row)`,
strict right endpoint, `_count` per length, `WINDOW_STATS = (mean, std, median)`.
Then three hand-driven Rossmann probes said it was not worth building.

### The result, in the order it has to be read

**1. The key is the gain, not the averaging.** Same tool, same 364 days, same stat:

| key | probe | vs baseline |
|---|---|---|
| `Store` | 0.22072 | −0.00263 |
| `Store`+`DayOfWeek` | 0.17352 | **+0.04458** |

**2. And that +0.045 is mostly 0046, not slice 3.** The control nobody had run:

| | probe | vs baseline |
|---|---|---|
| **all-time target mean of `Store_DOW` (0046)** | **0.15762** | **+0.06048** |
| trailing 364 mean, `Store`+`DayOfWeek` (slice 3) | 0.17352 | +0.04458 |
| both | 0.16944 | +0.04866 |

**Slice 3 bought −0.01590 over what 0046 already had**, and stacking it on top of
the all-time mean is *worse than the all-time mean alone*. The winner's *"sales
for this store on a typical Monday over the last 6 months"* is two halves and
**the half we were missing was the periodic key** — built in 0046, paid for in v7.

`docs/slice3_windows.md` pre-registered exactly this: "if a window adds nothing,
that is a publishable result and the slice should stop."

### The new failure mode, and it is general

Length sweep at a fixed key. Registered prediction was **monotone**; wrong:

| window | probe | obs |
|---|---|---|
| 91 | 0.23365 | 3 |
| **364** | **0.17352** | 39 |
| 546 | 0.21840 | 58 |
| 728 | 0.31097 | **59** |

546 and 728 hold the *same* number of observations and are 0.093 apart, so
coverage is not the mechanism. It is:

> **A window longer than the shortest fold's training prefix cannot be filled in
> training, while a served row fills it completely.**

Work half 850 days, fold 0's prefix **136**. So w=728 trained at median 59 obs and
served at **101** (1.71x); w=364 ran 39 against 49 (1.26x). Train/serve skew
through a door 0048 did not cover — not *when* history stops but *how much* there
was to summarise.

### The gotcha, and it is mine again

**`window_observations`, added this session to catch a badly-chosen window, rated
the catastrophic one higher than the good one** (59 vs 39). The damage is in the
*ratio between the training and serving sides* and the count only sees one side.
0031's shape in the reporting layer for the second time in three records (0050
caught it in `summarise_history`).

Fixed generally: `fit_history` now returns **`shortest_training_prefix`** and warns
when a window exceeds it, saying outright that the count is no guide there.

**The guard over-fires and says so.** Rossmann's shortest prefix is 136, so the
warning flags **364 — the best window in the sweep** — alongside the ruinous 728.
The prefix marks where the mechanism starts, not where it starts costing. Warning
on the *ratio* instead needs a threshold, and there are two points on that curve
(1.26x fine, 1.71x ruinous); picking a number between them is 0034's refusal to
fit a constant to three observations. Conservative and honest about it, rather
than tuned and sharper-looking.

Also: the training-side clipping is **not uniform** — fold 0 holds ≤19 weekly obs
in a 728 window, fold 4 up to ~101, serving gives 97–104 to everyone. The trained
column is a *mixture* of differently-truncated windows under one name. Less data
everywhere would be a noisier feature; differing amounts across the seam is a
different feature.

### Decisions taken with Ulf

- **Keep the code, record the loss.** One dataset is one reading, and Rossmann is
  unusually stable per store-weekday — the feature was tested where it had least
  to offer. A negative result that stays retestable is worth more than one that
  does not.
- **Keep `median`, explicitly unvalidated.** 0.0003 from the mean on the good key,
  worse on `Store`. Dropping a stat because one dataset did not need it is the
  mirror of adding one because it did.

### Carry forward

- **None of these gaps has a yardstick.** Ordered split ⇒ val is one period,
  `val_se` is `None`, n=1 in the unit that varies (0022/0036/0041). Directions are
  consistent across three comparisons; the numbers have no error bars. **0036's
  paired bootstrap is still not wired in and is what would fix this.**
- **Offsets `[364,728]` scored −0.0229 here against +0.0217 in v8.** Likely because
  v8's agent had filtered `Open == 1` first, so a 364-day lookup could not land on
  a closed day. Plausible, untested, named so it is not later assumed.
- **Unexplained:** on the `Store` key, offsets alone −0.0229 and window alone
  −0.0026 combined to **+0.0458**. Recorded as an oddity, not narrated.
- Scripts: `probe_rossmann_windows.py`, `..._windows_control.py`,
  `..._window_length.py`. All register predictions in their docstrings.
- **Notebook** `phase5_history_and_the_horizon.ipynb` now 55 cells, sections 6–11,
  ships unexecuted, every code cell smoke-tested in order.
- **Quiz for slice 3 still owed** (0041/0042/0044's too).

## 2026-08-14 (night, later) — naming changed, flag replaces lag age, slice 3 planned

**Machine:** MacBook Air | **Branch:** master | **426 tests pass.**

Acting on 0050 plus Ulf's review of the column names.

### Columns renamed (breaking — the v8 bundle no longer replays)

| was | now |
|---|---|
| `<base>_last` | `<base>_latest` |
| `<base>_last_age` | `<base>_latest_ago` |
| `<base>_lag364` | `<base>_364_ago` |
| `<base>_lag364_age` | **`<base>_364_ago_exact`** |

**One word for one idea: `_ago`.** In `<base>_364_ago` the distance is a literal
in the name; in `<base>_latest_ago` it is unknown until you look, so it becomes a
column. Same word, same meaning — one fixed, one measured. ("history" was
considered and rejected: it suggests the column *contains* history, when it holds
a single distance.)

**No unit in the names, deliberately.** Ulf's `_days_old` read better but the
ordering column is not always days — an event log may be in seconds — and a name
saying "days" over a series counted in something else is precisely the
confidently-wrong label this area keeps producing.

`_runs/rossmann_v8/bundle` still stores `feature_names` with the old names, so it
will not replay. Accepted rather than versioned: nothing depends on it.

### The lag age became a flag

For an offset the age is near-constant — it *equals* the offset whenever the
lookup worked — so all its information is in the failures. `_exact` is 1/0/NaN and
carries the same signal legibly. **`_latest_ago` stays a real measured distance**, because there
it genuinely varies per row (1–45 days on Rossmann).

### `fit_history` now reports what the agent could not know

Ulf's catch, and it was a real gap: the warning said *"use offsets longer than the
period you are predicting"* — but the agent **has no way to learn that number**.
`inspect_data` reports `n_val` as a *row count*; nothing converts it to a span. So
the tool computes it and states it:

- **`block_span`** — how wide a scored block is, in the ordering column's units.
- **`offsets_answerable`** — `{"7": 0.112, "364": 0.805}` per offset, plus a
  WARNING in the note below 50%.

That is the line that would have made v8's `lag7` mistake obvious at a glance
instead of after an investigation.

### Known limitation, now documented in `to_periods`

**Dates always become days, whatever their granularity.** Fine for a daily panel,
unusable for a second-resolution log (age `0.00347`, an hour as offset `0.0417`).
Correct arithmetic, unusable numbers. Fix is a *declared* unit, not an inferred
one; a numeric ordering column already sidesteps it. Not built.

**Next: slice 3, in a fresh context — see `docs/slice3_windows.md`**, which carries
the full argument, the design constraints, and the two open questions
(should `_latest` still be unconditional; the time unit).

## 2026-08-14 (night) — Rossmann v8: offsets are the win, `_last` is the disaster

**Machine:** MacBook Air | **Branch:** master | first agent run with `fit_history`.
Written up as [0050](docs/decisions/0050-the-anchor-is-what-matters-not-the-history.md).
25 tables in 10.8 min, `_runs/rossmann_v8/`.

**The four best tables in the run are history tables. So is the worst.** Same tool,
same key, same value — the whole spread is one parameter:

| table | offsets | probe | vs parent |
|---|---|---|---|
| t24 | `[364, 728]` | **0.13669** | +0.0217 |
| t22 | `[364]` | 0.13915 | +0.0192 |
| t20 (best non-history) | | 0.15840 | |
| **t7** | **`[]` last-only** | **0.46396** | **−0.2639** |

**Why: the anchor.** `_last` anchors at the *freeze boundary* — one date shared by
every served row, and one date is never typical. 0049 caught it as zeros. v8 shows
the general form: the agent had already filtered `Open == 1`, so **no zeros at
all**, and `_last` was still ruinous — the last open day before a holiday is a
**pre-holiday spike, median 9,697 vs 6,337 overall (+53%)**. The filter fixed the
symptom, not the disease. An offset anchors at *the row*, so each row draws
independently.

**New rule 0047 lacked:** the declared floor makes an offset *legal*; the **span of
the serving block** makes it *useful*. Val block is 44 days, so `lag7` was actually
honoured for **11.2%** of val rows (median age delivered: 24) — the rest degraded
into `_last`. That is why `[7,30,364]` was the worst of the four good ones.

**Sealed:** blend (what ships, 0044) val 0.14138 → **0.17773**, beating the argmax
lgbm's 0.18595 — 0044 holds on a second dataset. Optimism +0.036, so v7's
registered "optimism ≤ 0" fails; drift and selection stay unseparated (0041).

### Two things to carry forward

- **`_age` measured to the observation *actually read* is what made this
  diagnosable.** `lag7` reported median age 24 and told the truth about failing.
  Had age been the offset *requested*, it would have read 7 everywhere and been
  silently wrong.
- **`forecast_horizon` was nearly left undeclared.** Without it offsets are refused
  and the run's best table cannot exist.

### The gotcha, and it was mine

`summarise_history` printed *"the feedback loop is decorative"* — a **hardcoded
verdict that assumed its own premise** (that the probe had called these tables
worse) while the deltas printed directly above it said 4 of 5 were better. Fixed
to read the deltas. 0031's shape in the reporting layer: a check that looks like
it checks and only asserts. **Do not trust a conclusion a script prints about its
own run without reading the numbers beside it.**

**Next:** slice 3's argument has changed — offsets already deliver recency *and*
periodicity (the agent picked 364/728 to keep the weekday aligned, which is the
winner's feature shape). A window would add only the averaging. Possibly worth
less than making `_last` optional and reporting each offset's answerable share.

## 2026-08-14 (evening) — Rossmann says no, and the reason is worth more than the number

**Machine:** MacBook Air | **Branch:** master | **425 tests still pass** (no code
changed as a result of this).

Ran slice 2 against Rossmann via `scripts/probe_rossmann_history.py` — not an
agent run, `fit_history` driven by hand against v7's split so the number moves for
one reason. **The feature made the probe substantially worse.** Written up as
[0049](docs/decisions/0049-the-last-observation-is-a-sample-of-size-one.md).

- baseline (Customers dropped) **0.21810** → + Store history of Sales **0.35054**
  (−0.13244). Every variant worse; `Store+DayOfWeek` −0.09720, `lag7` −0.13659,
  history of `Customers` −0.14111.
- The **naive** lag (val rows reading their own labels, built outside the harness
  because 0011 refuses to build it) scored **0.09999** — better than baseline by
  +0.11811.

**The cause:** the work half ends **2015-05-01**, a public holiday with **98.21%
of stores shut**. Training columns are healthy (median `_last` 6488, 0.1% zeros)
because forward chaining freezes at five different fold boundaries. But **every
validation row reads one freeze date**, and that one was a holiday — median
`_last` 0, **98.2% zeros**. Not this split's bad luck either: **16.8% of the 851
candidate work-end dates** have >90% of stores selling nothing.

**Control** (`probe_rossmann_history.py 0.06`, boundary moved to an ordinary
Wednesday, 0% shut): baseline 0.22460 → 0.23602, **−0.01142**. So the holiday is
~92% of the damage and the mechanism is not broken — **but the feature still hurts
on a clean boundary.**

**One key survives.** On that clean boundary `Store`+`DayOfWeek` history is
**+0.00272**, the only positive of the four. "What did this store sell the last
time it was this weekday" puts the closed days in **their own groups**, so a
Sunday's zero can only be read by another Sunday. Small and directional only, but
it establishes the mechanism can help at all. General form: **a group's last
observation is usable only if the group is homogeneous enough for one member to
stand for the rest** — the same requirement a group mean has (0046), from the
other side. Hence the fix is a narrower key or a wider window; both cut the
variance of what is read.

### The general statement, and the two things 0047/0048 missed

**The most recent observation is a sample of size one.** `_age` reports how *old* a
value is; **nothing reports how *typical* it is**. Honest about staleness, silent
about representativeness.

Second, quieter: **the age ranges don't match.** Training ages run 1–305 (median
74, fold blocks ~150 days), validation ages 1–45 (median 23). 0048's serving
symmetry holds in *shape* but not in *scale* — block width comes from `n_splits`,
the horizon from the task. Structural, not incidental.

### The gotcha for whoever picks this up

**Do not fix this by making Rossmann better.** Skipping closed days in history, or
taking the last *non-zero* observation, each improves the number and neither can
be defended without naming the dataset — 0022 one level up. 0049 says so
explicitly.

**Next:** slice 3 (rolling/expanding windows) is now **the fix, not a bonus**. A
7-day mean over the same stored tail gives **0%** of stores a zero (median 5771.7)
— robust to exactly this failure, because it is not a sample of size one.

### Input for slice 3, from outside: what the competition winner used

Ulf points out the Rossmann winner's writeup leans on features like *"sales for
this store on a typical Monday, averaged over the last 6 months / 1 year"*. That
is **a narrow periodic key plus a trailing window** — the two fixes 0049 named,
combined. Three things follow, and they change slice 3's framing:

- **The key half is already built and already paid.** `fit_encoding(method=
  "target")` on a composite `Store_DOW` key *is* "average sales for this store on
  a Monday", and it was v7's **t8, +0.0243, the largest gain of that run** (0046).
  Not a gap.
- **The missing half is recency.** That aggregate runs over *all* work-half
  history; the winner's is limited to a trailing period. A store whose trade level
  moved a year ago is described badly by an all-time mean and well by a trailing
  one. **That is slice 3, and it is a better argument for it than 0049's
  structural-zeros one.**
- **It argues `_last` is not the valuable part of history.** The winner used
  windowed aggregates, not a last-known value — which is what 0049's numbers say
  too. Worth stating before slice 3 rediscovers it.

**Where the line actually is — and it is not where a first reading suggests.** The
*agent* must be able to build this feature. `fit_history(by=["Store","DayOfWeek"],
value="Sales", window=180, stat="mean")` **is** the winner's feature, assembled by
the agent out of general parts, and that is the goal rather than the thing to
avoid. What stays out of `src/` is a *default*, a blessed list of window lengths,
or a docstring that names the dataset. The capability is general; the choice is
the agent's (0029).

Concretely, for slice 3:

- **`window` is a free number in the ordering column's units**, not a menu. A
  menu would be us picking the windows, which is the agent's call and would
  encode "180 and 365 are the good ones" — true of one dataset.
- **`keep` has to cover the longest window asked for.** `MAX_KEEP` is 1000, so a
  365-day trailing window is expressible; a multi-year one on daily data is not,
  and that ceiling should be a stated refusal rather than a silent truncation.
- **Watch the storage.** `Store`×`DayOfWeek` over a year is ~52 observations ×
  7,805 groups; `Store` over a year is 365 × 1,115. Both ~400k floats — bounded,
  which is the property that matters, but no longer small. Worth measuring before
  it goes in a bundle.

## 2026-08-14 (later still) — §3.4 settled, and slice 2 is built

**Machine:** MacBook Air | **Branch:** master | **425 tests pass** (was 387).

### 0048 — the last open question, and its premise was wrong

`docs/time_aware_features.md` §3.4 said a target lag "is target encoding's near
relative and needs the same care: out-of-fold construction". It is not. Two
failures were wearing one word:

- **Self-inclusion** (the row's own target in its own feature) is what OOF removes
  for `TargetEncoder`/`GroupStatsEncoder`. A strictly backward lag **cannot commit
  it** — the row at `t` only reads `t-k`, at any group size, under any folds. OOF
  buys nothing.
- **Availability skew** (reading a label that will not exist at serving time) is
  the real risk, and OOF does not fix it. The fix is 0047's frozen horizon,
  mirrored per fold.

So the fold loop looks like `fit_group_stats`'s and exists for an unrelated
reason. **This is the gotcha to carry forward:** fitting each fold's history on
"everything outside the validation fold" satisfies the older rule and is a
straight future leak here, because *last known* is defined relative to the fitting
set — put a later block in it and the newest observation comes from the future.
A strictly backward lag survives that by luck; the last-known lookup 0047 chose
does not.

Also decided: **target history is refused under random k-fold** (0013-style, not a
warning). Checked as a property of the index pairs, not a trusted flag.

### Slice 2 built

`HistoryEncoder` + `fit_history` in `encoders.py`, `history` in `ENCODER_CLASSES`
so `Pipeline.transform` replays it, `TaskSpec.forecast_horizon`, and the
`fit_history` harness tool. Emits **pairs**: `<by>_<value>_last` and `..._last_age`,
plus `_lag<k>`/`_lag<k>_age` per offset.

- **Age = row − the observation actually read**, not the offset asked for. So an
  unanswerable `lag7` degrades into `_last` and reports the real age rather than a
  stale number under a fresh name.
- **Offsets require `forecast_horizon`** and must be ≥ it. Undeclared → last-known
  pair only, which is honest without a declaration.
- Storage bounded by `keep` per group; encoder size never depends on table length.
- `value` **may be the target** here, unlike `fit_group_stats` — a target lag can
  only come from stored history, never from a served frame.

### Two things that bit, worth knowing

- **`pandas` hands back `datetime64` in s/ms/us/ns** depending on how the column
  was built. Scaling from the integer representation put every age out by 1000x.
  `to_periods` now divides by a `timedelta` instead.
- **Dates arrive as strings**, because harness tables come from CSV. Unit tests
  used an integer `day` and missed it entirely; caught only when the notebook ran
  the real tool. `to_periods` accepts **ISO 8601 only** — `01/02/2015` is refused
  rather than guessed, since guessing wrong silently transposes day and month.

**Notebook:** `notebooks/phase5_history_and_the_horizon.ipynb` (36 cells, ships
unexecuted, every cell smoke-tested in order). Its section 6 is the payload: the
naive lag scores 0.973 within-store correlation against the frozen one's 0.77 —
**that gap is the size of the lie, not the size of the loss**, and 0048 says so
explicitly so it is not later "fixed".

**Next:** slice 3 — rolling/expanding windows over the stored tail. `keep` already
carries the `L + W` bound. Still unvalidated by any real run; nothing here has been
scored on Rossmann.

## 2026-08-14 (later) — The two blocking questions are answered, and Gap A is closed

**Machine:** MacBook Air | **Branch:** master | **387 tests pass** (was 363).

Worked `docs/time_aware_features.md`. Both questions it named as blocking are
settled with Ulf, and slice 1 is built.

### 0047 — the horizon, settled (accepted, **not implemented**)

Ulf's answer to §3.3 collapsed both questions into one. The real forecasting
contract is: history to `T`, predict `T+1 … T+k`. For `T+1` you know yesterday's
value; for `T+2` you do not. **So availability is not a property of the served
set, it is each row's distance from the end of history** — which is what §3.2 was
asking. One mechanism answers both:

- **Frozen history + a per-row staleness column.** `T+1` gets the last known
  value at staleness 1 (that *is* yesterday's value), `T+5` the same value at
  staleness 5. Degrades visibly instead of lying quietly.
- **`TaskSpec.forecast_horizon` optional, and a *minimum*.** The refinement that
  is easy to get wrong: `h` varies per row inside one predict batch, so a single
  declared horizon cannot gate lags globally. It is the floor across the block;
  staleness stays per row.
- **No recursion** (predictions not fed back as lags) — sequential prediction,
  compounding error, and structural train/serve skew. Rejected-for-now, reopenable.
- **Predict rows never read their own batch's past.** This is the one that costs
  a guarantee: a lag reading the batch's earlier rows *is* 0011's batch-dependence.
  Frozen history keeps batch-independence by construction.

Two pre-work checks that changed the framing:

- **0030's serving schema does not already express §3.3.** It is column-level;
  §3.3 is temporal. And since the target is excluded from `serving_columns` by
  construction, a target lag can never come from a served frame at all — so §3.3
  is really "is stored history ever *extended*?", a new declaration.
- **0011's check passes a stored-history step as-is** (`unstable_columns` only
  ever runs on sandbox code), *because* history is frozen. That equivalence is
  exactly what the permissive answer would have spent.

### 0046 — slice 1 built: a group is more than its average

`fit_encoding(method="target", stats=[...])`, `TARGET_STATS = (mean, std,
median, count)`. `mean` keeps the bare `<col>_te` name; the rest get
`<col>_te_<stat>`. Default is `["mean"]`, so nothing changes unasked.

- **One tool, not two.** Lifting `fit_group_stats`'s target refusal for the
  non-mean stats is less code and produces an indefensible seam — a group's
  target *mean* from one tool, its target *median* from another.
- **Only the mean is smoothed.** The weighted-average formula is a statement
  about means; shrinking a median by a count weight invents an estimator and
  gives it a standard one's name. **`count` is emitted instead** — same
  information, learned rather than baked into an untuned constant.
- **`min`/`max`/`sum` off the menu.** An extreme of the target in a group is
  essentially one row's target and survives OOF only by luck of fold assignment;
  `sum` is `count * mean`.
- Everything is out-of-fold, including `count`.

### The gotcha worth carrying forward

**A leak test written for one statistic does not transfer to another statistic of
the same column.** The mean's OOF test correlates the encoding with `y`. Applied
to `std` it *fails* — and not because `std` is safe: a row inflates its group's
spread whether its target is high **or** low, so the leak lands on `|y - mean|`
and sits near zero against `y`. Replaced with a direct perturbation test (move
one row's target, watch that row's own feature) which covers all four with one
mechanism.

Also: **the serialized target encoder changed shape** — `mapping` is now nested
by statistic. `from_dict` detects the old flat shape by the absent `stats` key
and loads it mean-only, so bundles on disk keep predicting what they predicted.
One existing test read the flat shape and was updated.

### Also

- `tests/test_target_stats.py` (15) + `tests/test_target_stats_harness.py` (9).
- `notebooks/phase4_target_distribution.ipynb` ships unexecuted, all 11 code
  cells executed in order from the shipped file.
- `fit_group_stats`'s target refusal now names `stats=` instead of only saying
  the mean needs smoothing.

### Next

- **Slice 2 — the `history` lineage step.** Unblocked now; 0047 is the spec.
  Bounded storage (last `L + W` per group), replay at predict, and §3.4 (lagging
  the *target* specifically) still to settle.
- **No run has used 0046.** 0032 is the precedent: `fit_group_stats` was the tool
  the agent asked for and then never called once in 69 calls.
- Everything from the entry below is still open: fix the ruler, a non-Rossmann
  sealed run for 0044's claims, the request timeout, `scripts/run_rossmann.py`.

### Gotchas

- **Quiz for 0046/0047 done** (0041/0042/0044's are still owed). It surfaced a
  caveat 0046 does not state and should have: **`count`-instead-of-smoothing only
  works for a model that can express an interaction.** A GBDT splits on
  `cat_te_count < 5` and learns the discount; `elasticnet` fitted on both columns
  just adds them, so for the linear member of the terminal ballot `count` is
  strictly weaker than smoothing and smoothing is still doing the real work.
  0046 is accepted and stays as written — carry this into slice 2's record, or
  its own if slice 2 does not touch it.
- `count` under an ordered split is a *partial* count (folds preceding the row),
  while the stored encoder's count covers all rows — so the trained column and
  the served column measure subtly different things. Already true of the mean;
  noted in 0046, not fixed.

## 2026-08-14 — Kaggle confirmed the blend, so the blend is what ships now

**Machine:** MacBook Air | **Branch:** master | **363 tests pass** (was 353).

### 0042 claim 1 passed on an external seal

Rossmann v7's submissions, scored on Kaggle:

| | private | public |
|---|---|---|
| `submission.csv` — argmax winner (xgb) | 0.14356 | 0.13960 |
| `submission_blend_lgbm_xgb_catboost.csv` | **0.13813** | **0.12932** |

**+0.00543 private**, on rows nothing in this project has ever touched. Not a
fresh run — same models, same table, only the judging rows are new — but it rules
out the specific worry that stopped 0042 implementing: that the effect lived in
v7's sealed tenth alone.

### 0044 — implemented

`finalize_run` **ships an equal-weight average of every terminal fit that
trained**. No weights fitted, no candidate dropped, nothing chooses. Verified
against v7: the new library path scores the four-way blend at **0.13618** on v7's
sealed rows, matching 0042's recorded number exactly.

- **Every candidate, no filtering.** "Keep the trees, drop the linear model" is
  refuted by house v9, where elasticnet was the best model on the board. And on
  v7 four-way vs three-way differs by 0.00008.
- **Residual risk, named in the record:** Kaggle backed the *three-GBDT* blend;
  what ships is the four-way.
- **`selected` vs `shipped`** are now separate flags on the holdout rows.
  `selected` still means "what the search crowned" so 0039/0040's optimism series
  stays comparable; `shipped` is what the bundle predicts with. Their difference
  is 0042's A/B, **recomputed by every run from now on** instead of by a script.
- **The seal still picks nothing.** Members are fixed before any sealed row is
  read; `_score_on_holdout` receives the finished blend as an argument.

### A defect found on the way — and it would have shipped an unmeasured estimator

`predict_bundle` combined models **on the model's scale and inverted once at the
end**. Under `log1p` that is a *geometric* mean of the target. Every number in
0042, and both Kaggle submissions, were arithmetic means (`blend_submission.py`
inverts first). So passing k models into the old bundle would have shipped an
estimator never scored against anything, quoting a number belonging to a
different one. `predict_bundle` now inverts per model, then averages.

**Nothing on disk changed:** for a single-model bundle the two orders are
identical, and until now every library-written bundle had one model.

### Also

- `tests/test_blend_ships.py` — 10 tests. `notebooks/phase4_blend_ships.ipynb`
  ships unexecuted, all 10 code cells executed in order from the shipped file.
- Report drops the CV score line, the fold table, and the "gap to the CV score"
  sentence when a blend ships — it was never cross-validated.
- `scripts/blend_submission.py` docstring updated; it now enumerates *subsets*,
  which the harness still does not.

### 0045 — the four-way blend scored, and 0044's cost estimate was 44x wrong

Scored after 0044 was committed:

| submission | private | vs argmax |
|---|---|---|
| `submission.csv` — argmax (xgb) | 0.14356 | — |
| `submission_blend_all.csv` — **what ships** | **0.14172** | **+0.00184** |
| `submission_blend_lgbm_xgb_catboost.csv` | 0.13813 | +0.00543 |

**0044's decision holds** — the average beats the argmax on an external set. **Its
cost section does not:** including elasticnet costs **0.00359**, where v7's seal
said 0.00008. Two thirds of the benefit of blending, eaten by the weakest member.

**No code changed, deliberately.** Filtering on this number is choosing a rule by
consulting a holdout — 0042's refusal, and the private LB is now spent for this
question. House v9 still refutes the category rule ("drop linear models" deletes
that run's winner). 0045 registers a hypothesis with a mechanism instead: the
seal may understate a linear model's drag because it measures at six weeks where
Kaggle measures at four months (0043), **or** 0.00008 was never an effect at all,
just an unresolved difference quoted as if the smallness were the finding.

### The bigger finding this session: we are out of feature vocabulary

Went through v7's 15 tables. **Ulf's hypothesis — per-group target means like
"average sales for this store on a Monday" — is already there and was the
biggest win in the run:** `t7` built `Store_DOW`, `t8` target-encoded it, probe
0.1587 → 0.1344, **+0.0243**. Then `t9`–`t15` built nine more composite keys the
same way for **+0.0006 total**, three negative. Half the run re-drilling one hole.

Two gaps behind that, both general:

- **Only the mean.** `fit_group_stats` correctly refuses the target as its value
  column, and the target encoder only makes a smoothed mean — so per-group
  **std / median / count** of the target cannot be written at all.
- **No feature can look at another row in time.** No lag, no rolling, no
  days-since/until. `transform_features` *could* write `groupby().shift()` and
  0011's stability check would rightly kill it; and at predict time the test
  period is later than every training row. **A lag needs the harness to carry a
  time series across the fit/predict seam, and it can only carry a mapping.**

Stated generally: *"a feature whose value depends on other rows, ordered by a
known key, whose history is stored at fit time and replayed at predict time."*
No dataset in the sentence.

### Next

- **Time-aware features — start here, and start by reading
  `docs/time_aware_features.md`.** Written this session precisely so a fresh
  session does not rebuild the argument. It has the evidence, the general
  statement of the gap, a three-slice plan, and **two blocking questions that
  need Ulf's answer before any code**: (3.2) what a lag serves when the predict
  rows are past the end of stored history, and (3.3) whether validation rows may
  use their own past. Everything else follows from those two.
- **Fix the ruler** (blocked/rolling-origin CV, or paired-bootstrap-of-difference
  from 0036) — still necessary, but it is a multiplier on a search that currently
  has nothing new to search over. After lags, not before.
- 0044's registered claims need a **non-Rossmann** sealed run to test.
- Still open from last session: lift the request timeout into the library;
  `scripts/run_rossmann.py` still broken since 0035.

### Gotchas

- **No quiz done** for 0041/0042 yet, and none for 0044 — questions asked at the
  end of this session are unanswered.
- Rossmann private LB is 3,303 teams and 0.13813 is roughly **rank ~2,200** —
  interpolated from HANDOFF's own score→rank mapping, **not** from the real
  leaderboard file. Below median; worth pinning down properly.
- **Rossmann's private LB is now spent for the blend-membership question.** All
  three relevant submissions are scored; there is no unconsulted external set
  left to judge a filtering rule on this dataset. The next test has to be a
  different dataset.

> **Entries below are compressed summaries** (done 2026-08-15). Each one's full
> argument is in the decision records it cites; the long-form text is in git at
> `801b52d:HANDOFF.md` if a detail is ever needed. Only findings *not* carried by a
> record are stated at length here.

## 2026-08-13 (later) — the sealed split is in, and it caught us shipping the worst model

**MacBook Air | master | 353 tests.** Records [0041](docs/decisions/0041-the-sealed-split-scores-the-ballot-not-the-winner.md), [0042](docs/decisions/0042-the-terminal-argmax-shipped-the-worst-model.md), [0043](docs/decisions/0043-on-ordered-data-the-seal-costs-the-freshest-rows.md).

`RunSession(holdout_fraction=)` built, opt-in and off by default, a fraction of the
whole input, ordered splits cutting `work | val | holdout` in time order, every
terminal candidate scored. Rossmann v7 then **shipped the worst GBDT**: xgb won val
at 0.12980 and scored 0.15253 sealed, against catboost's 0.14113 — optimism +0.0227,
the worst ever measured. Every equal-weight blend beat every single model
(lgbm+catboost 0.13432), which became 0042 and later 0044. 0043: on ordered data the
seal takes the *freshest* tenth, so v7 forecasts 3–5 months out where the field
forecasts six weeks, and v5's Kaggle 0.135 is not a fair control.

**Live gotchas:** a hung request has **no timeout anywhere in the library** — v7
attempt 1 sat 105 minutes on one gateway POST; patched only in that driver.
`scripts/run_rossmann.py` broken since 0035. The agent can read the sealed number
after `finalize_run` — harmless by call ordering, not by structure.

## 2026-08-13 — 44% of the search budget went to a set we could have enumerated

**MacBook Air | master | 329 tests.** Records [0034](docs/decisions/0034-the-tiebreak-was-tested-and-rejected.md), [0035](docs/decisions/0035-the-search-is-over-features.md), [0036](docs/decisions/0036-the-pairing-asymmetry-was-not-there.md), [0037](docs/decisions/0037-residuals-need-folds-the-ranking-does-not.md), [0038](docs/decisions/0038-the-first-feature-only-run.md), [0039](docs/decisions/0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md), [0040](docs/decisions/0040-seguro-v4-a-near-copy-of-the-ruler-is-not-the-ruler.md).

Counted across 16 runs: 222 experiments, **97 (44%) varied only the model or its
params**, and the winner was a model variation in 10 of 16. So the agent's action
space became **tables only**, ranked by a fixed probe model (`probe.py`, one lgbm at
library defaults, one fit per table); the terminal phase enumerates families
afterwards. 0034 tested the cv tiebreak (1 win, 1 loss, 1 null) and rejected it.
0036 refuted 0035's pairing-asymmetry claim by paired bootstrap. 0037: ranking needs
one fit, residuals need folds — `explore_table` refits on request and reads the
**work half only**.

House v9 then ran feature-only: val *worse*, Kaggle *better* (0.12037, top 9% vs
v8's top 20%), and ablation showed the gain was the feature search, not the terminal
enumeration. Seguro v3/v4 found the same error at two grains — first the probe's
*family* missing from the terminal ballot (cost 0.0256 gini), then its
*configuration* (seed 42 vs 0) — hence 0040's rule: **an invariant about a specific
measurement cannot be enforced by reproducing the category it belonged to.**

**Live:** the lgbm-vs-selection confound is unresolved (every selected/unselected
pair so far is lgbm vs elasticnet). Terminal fairness open — enumerating library
defaults measures the libraries; this is what 0054 eventually cashed in.

## 2026-08-12 — first classification run; the diagnostic model was unrunnable at scale

**MacBook Air | master | 309 tests.** Record [0033](docs/decisions/0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md).

Porto Seguro, 595k rows, given to the agent **untouched** — it dropped all 20
`ps_calc_*` columns and target-encoded the `_cat` columns in two tool calls,
unprompted. Kaggle private gini 0.272, top 66%; optimism +0.0128.

The defect: `_LINEAR_MAX_ITER = 20000` served both coordinate descent and saga,
**which do not share a unit** — one fit projected at 77 min, the search at ~510
hours. Split by solver, plus `_PenaltySearchThenRefit` (search sees ≤20k rows, refit
sees all). 510 hours → 4.6 min, bit-identical below the cap. Convergence turned out
to be a property of the *penalty*: at the `C=0.01` the search selects, saga
converges in 118 epochs.

v2 then showed **`cv` ranked all three models correctly and `val` put the best one
last**, with the inflation specifically in the models the argmax crowned (`e8` and
`e11` differ on val by a twentieth of one `val_se`, and their val errors differ 3x).
That is the winner's curse with the confounds stripped out, and it is what 0023/0041
were built for.

**Live gotchas:** the wall-clock bound checks *between* folds, so a single runaway
fold outruns it entirely — refuted premise, guard unchanged. `python script.py > log`
buffers; use `python -u`. A killed run loses `transcript_N.json` but keeps the
journal.

## 2026-08-12 — thinking changed the behaviour and not the score; the ceiling is diversity

**MacBook Air | master | 306 tests.** **No decision record — `elasticnet` and
`fit_seconds` are still untaught and unrecorded, and the elasticnet-as-diagnostic
argument is the part a newcomer cannot reverse-engineer.**

First runs on a reasoning route (`fireworks/deepseek-v4-flash-0731`). **Behaviour
changed a lot, the score did not:** house v8 hit the same Kaggle 0.123 as run 3 from
a val score 0.0018 better — i.e. optimism went *up* on *fewer* experiments. Thinking
was cheap (fewer tokens, half the tool calls). What it did change was tool use:
`fit_group_stats` went from 0 calls to the second call of the run, `filter_rows` was
used for outlier removal, and the data-dictionary channel finally paid — v8 encoded
21 columns as `ordinal`, an order that can only have come from the dictionary.

**Blending is not our gap, measured twice** (Rossmann OOF-weighted: worse;
house 3-GBDT: +0.0002). The reason is GBDT-vs-GBDT residual correlation of
**0.86–0.88**. The gap is diversity, which is why `elasticnet` exists — on v8's
winning table it *matches* a tuned catboost, so by the TabArena rule the signal is
in the features. Its preprocessing lives **inside the estimator** so `train_cv`
refits per fold; in a feature table it would leak across the fold boundary in the
one place 0011 cannot see.

**Live:** a gateway's `supports_reasoning` flag describes the model *family*, not the
route — only a real call tells you. `ElasticNetCV` has no `scoring` parameter (noted
in code, not fixed). And an unacted finding worth keeping: **on ordered splits the
last fold ranks models far better than the mean of folds** (Spearman 0.960 vs 0.864),
a candidate correctness fix to what `cv_score` means under `split_on`.

## 2026-08-11 (latest+7) — both gaps closed; the run used the wrong one, and it did not help

**MacBook Air | master | 291 tests.** Record [0032](docs/decisions/0032-fitted-aggregates-and-a-step-that-is-not-replayed.md).

`fit_group_stats(by, value, stat)` — per-group statistics of a **non-target** column,
fitted once and looked up at predict, **always out-of-fold with no opt-out** (the
tool cannot tell whether the value column proxies the target). `filter_rows(code)` —
returns a mask, rows dropped once at table-write time, lineage step carries
`train_only: True` so `Pipeline.transform` skips it. Two exemptions follow from
"evaluated once, never replayed": the filter may read the target and may be
batch-dependent.

Rossmann v5 finished, finalized itself, wrote the first `submission.csv`, and
**refuted the premise the filter was prioritised on**: dropping the 17% closed-store
rows cost 0.0034 on a fold spread of 0.07 — because `Open` is a feature and the model
already learns "closed → 0". The capability is still right on generality. And
`fit_group_stats` — the tool the agent had *asked for* — got **zero calls in 69**,
the fourth time changing what the agent is told failed to change what it does.

Kaggle scored it 0.135 and **our exam was pessimistic for the first time**
(val 0.13760), which is 0029 paying off: an ordered split plus forward chaining asks
the competition's question, so there is no interpolation premium to hand back.

## 2026-08-11 (latest+6) — a decorative guard, a corrupted run, and the tool the agent keeps asking for

**MacBook Air | master | 255 tests.** Record [0031](docs/decisions/0031-the-guard-was-decorative.md).

v3 died holding the best servable model any run had produced, on a corrupt parquet:
two `fit_encoding` calls computed the same table id and wrote the same file at once.
**0008 set `max_concurrency=1` to prevent exactly this, and that parameter limits
concurrent agent *runs*, not tool calls — it never did anything**, and the test
asserted the constructor argument had been passed rather than the property it was
named for. Fixed in the primitives: `O_CREAT|O_EXCL` id claims, `os.mkdir`,
tmp→fsync→`os.replace`.

Also found: the agent reached unprompted for per-store aggregates of another column,
0011 refused them correctly, the refusal message pointed at a tool that **cannot do
this**, and three refusals exhausted the retry budget and killed the run. That gap is
what `fit_group_stats` closed the next session.

## 2026-08-11 (latest+5) — Rossmann ran three times; the leak is refused, and we audited what we cannot express

**MacBook Air | master | 254 tests.** Record [0030](docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md).

v1's agent spotted the `Customers` leak in prose, dropped it unprompted, and built an
honest chain to 0.13173 — **which counted for nothing**, because the leaked model at
val 0.07912 (better than first place, the signature of a column nobody had) stayed
incumbent, ten genuine improvements were each recorded as no-improvement, and
finalize bundled a model that raises `KeyError` on the first real prediction. Fixed
by `serving_columns` + refusal at train time; v2's agent read
`columns_not_available_at_predict_time` off `inspect_data` and guarded it in its
*first* table.

**The capability audit against the winner's solution** named three gaps: no row
filtering, no time-windowed group aggregates, no blending — all three since built
(0032, slice 2/3, 0044). Deliberately not gaps: external data (0001) and the 0.985
prediction correction, whose own author reports it was inside the CV noise.

## 2026-08-11 (latest+4) — Rossmann runner written, NOT run; a third failure mode

**MacBook Air | master | 250 tests.** Dry-running the machinery without the LLM found
that dropping a serving-absent column has **two** failure modes, and the middle one
is the trap: `df.drop(columns=['Customers'])` gives an honest score and still cannot
be served, because per 0010 a transform is stored as *code* and replayed on raw rows
that never had the column. `errors='ignore'` is the only submittable form. **Nothing
in the harness can catch it** — `_score_on_val` replays over validation rows drawn
from the training file, which still have the column. There is no row anywhere in a
run shaped like a real serving row. Generalises: any transform referencing a
predict-time-absent column breaks the bundle silently.

## 2026-08-11 (latest+3) — the harness owns the exam; CV forecasts when the run does

**MacBook Air | master | 250 tests.** Record [0029](docs/decisions/0029-the-harness-owns-the-exam.md).

**The harness owns the evaluation protocol; the agent owns the modelling.** So
`Customers` went back into the Rossmann data — dropping it was the harness doing the
agent's work — and whether the agent uses it became a measured result. Three fixes to
an exam that was subtly wrong: `make_folds(order=)` builds forward-chaining folds
(random k-fold on a panel is interpolation, not forecasting); the split cuts on a
*value*, not a row count; a table dropping the ordering column is refused rather than
silently falling back to random folds.

**Live:** under forward chaining ~1/(n_splits+1) of training rows get no out-of-fold
prediction at all, and out-of-fold target encoding leaves that same block NaN with
the missingness time-correlated — a tree can learn it as "early period". Not fixed;
every alternative computes the value from the row's own future or from itself.

## 2026-08-11 (latest+2) — Rossmann: the data dictionary now has a channel

**MacBook Air | master | 236 tests.** `TaskSpec.column_notes={col: text}`, rendered
per-column inside `inspect_data`'s profile; run-level `notes` returns as
`about_this_data`. An undocumented dataset costs exactly nothing.

The finding that mattered more than the feature: **`Customers` is a leak this harness
cannot see** — 0.895 correlated with `Sales`, absent from `test.csv`, present in our
validation half because that is drawn from `train.csv`. No fold crossed, no encoder
misfitted, every guard holding. Nothing computable from the training file reveals it;
the only evidence is the competition's data dictionary. `scripts/prep_rossmann.py`
carries the join, typing and dictionary.

## 2026-08-11 (latest+1) — `val_se` shipped; the bar it gates is deliberately still open

**MacBook Air | master | 233 tests.** Record [0028](docs/decisions/0028-the-yardstick-is-fixed-the-bar-is-not.md), **which corrects 0022**.

`_score_on_val` bootstraps the validation score over its own rows (200 resamples,
seeded). It lands on the leaderboard row, `meta.json`, the payload, and the report.
**The correction is the important part:** 0022 claims the fix would have told run 3
its 0.008 gain sat inside the noise. It would not — the enforced bar is a *quarter*
of the yardstick, so rejecting that gain needs `min_gain_ratio > 0.66`. 0022 fixed
the category error and left the magnitude error untouched.

`min_gain_ratio` stays at 0.25, still open. An advisory note fires when a gain is
inside one SE — shipped as the experiment, since advisory guidance had already failed
three times. **`val_se` understates uncertainty under an ordered split** (the
bootstrap assumes i.i.d. rows); inherited from 0022, unaddressed, and it is why 0036's
paired bootstrap is still the blocking item.

## 2026-08-11 (latest) — v7 ran: the brief does not cut the orientation tax

**MacBook Air | master | 226 tests, no library code changed.** Record [0027](docs/decisions/0027-the-brief-did-not-cut-the-orientation-tax.md), **which supersedes 0026's central claim**.

Orientation per resumed episode: v6 without a brief **41%**, v7 with one **40%**. No
change. The *composition* changed completely while the size did not — `inspect_data`
calls fell exactly as designed and the agent spent the freed calls on
`explore_journal` instead. So **orientation is not an information deficit that can be
paid off; it is roughly a constant fraction of whatever episode it is given.** Third
advisory fix in a row to fail. Whatever solves it will be structural.

Not claimed: v7's better score (inside one `val_se`) or its lower call count (episode
1 is configured identically in both runs and differed by 8 calls — that is the noise
floor). General lesson from the same session: **a baseline not computed with the same
code is not a baseline.**

## 2026-08-11 (earlier) — the derived brief is built; it is not yet measured

**MacBook Air | master | 226 tests.** Record [0026](docs/decisions/0026-the-brief-is-rendered-not-remembered.md) (implementation stands, hypothesis refuted by 0027 above).

`harness/brief.py` renders a ~170-token `RUN STATE` block from the journal, wired as a
*dynamic* system prompt so it renders when the episode starts. Returns `None` on a
fresh run, deliberately, so a first episode is byte-identical to a run that never had
episodes — which is what made v6 and v7 comparable.

**The load-bearing test is the flatness one:** `test_the_brief_does_not_grow_with_the_run`
fakes 151 leaderboard rows and asserts the brief moves by <20 chars. It is the only
thing standing between this design and someone adding "recent experiments", which is
the natural next request and the thing that breaks it.

## 2026-08-10 (latest+1) — `explore_journal` is taught, and writing it found a bug

**MacBook Air | master | 215 tests.** No record — nothing decided.
`notebooks/phase3_harness_step6_journal.ipynb` (46 cells) teaches the tool and target
transforms together, since `target_transform` is a column in `lb` and "did log1p pay"
*is* a journal query.

**The bug:** `explore_journal`'s `tables` frame reported **zero columns added, for
every table, in every run since it was written** — it read `t.get("columns")` where
journal rows carry `new_columns`. Not a crash; a confident `0` to the one question
that frame exists to answer. Six unit tests covered the tool and all passed either
way, because none asserted a count they had independent reason to expect. **First time
the notebook convention paid in a bug rather than in prose.**

**Live gotcha:** the `notes` frame comes from `journal["log"]`, which no agent-facing
tool writes, and unlike `lb`/`tables` its columns are **not declared** — so it is a
`(0, 0)` frame and `notes["hypothesis"]` raises `KeyError`. Harmless while always
empty; a trap the day a note tool lands.

## 2026-08-10 (latest) — episodes work; the agent will not end one by itself

**MacBook Air | master | 214 tests.** Records [0024](docs/decisions/0024-the-episode-stop-is-structural.md), [0025](docs/decisions/0025-episodes-buy-feasibility-not-cost.md).

Asked in plain terms to follow one idea then stop, the agent ran all 25 experiments
in a single episode. **It may own the episode stop; it will not exercise it.** With a
structural cap, episodes work: peak context per episode 9k–12k against a single run's
climb to 43,691 — linear becomes sawtooth. A resumed agent orients itself unprompted,
opening `inspect_data → explore_journal` with no mention of the tool in its prompt.

0025 corrected two of 0024's claims: the orientation tax is ~1/3 of every episode at
**both** caps (given room the agent orients *more*, so a bigger cap is not the fix),
and episodes cost **more** per experiment at this scale — they buy **feasibility, not
cost**. Crossover ~35 experiments; single-run impossible ~60–80.

## 2026-08-10 (earlier) — three live Kaggle runs measured the leaderboard problem

**MacBook Air | master | 214 tests.** Records [0021](docs/decisions/0021-the-agent-picks-the-target-scale.md), [0022](docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md), [0023](docs/decisions/0023-the-third-split-has-evidence-now.md).

`explore_journal` (read-only code over the run's own records), target transforms over
a fixed menu (a per-experiment argument makes it an *experiment*, not a project
assumption — the agent immediately ran a controlled pair and switched), and
`harness/usage.py`. Three house-prices runs, top 20–22%.

**The big one: our val said the target transform bought 0.00808; Kaggle says
0.00070.** 91% was optimism, and optimism grew 4.6x for 1.4x the experiments.
Bootstrapping val over its own 292 rows gives sd ~0.010 — **the gain was smaller than
one standard error of the instrument that measured it.** That is 0022: 0019 moved
ranking onto `val_score` and left the yardstick on `fold_spread`, a bar four times
too small. Also: **`Id` was a feature in run 1** (train 1–1460, test 1461–2919,
entirely disjoint), inert in-sample and damaging only at serve time — no CV, no
holdout drawn our way, and no ablation can see it.

## 2026-08-10 (later) — the agent can see its own mistakes; the RLM kernel was declined

**MacBook Air | master | 197 tests.** Record [0020](docs/decisions/0020-the-sandbox-has-no-kernel.md).

`explore_experiment(exp_id, code)` — read-only code over one experiment's rows plus
`oof_pred`, `residual`, `fold` — closes the `score → hypothesise` arc. Free, no
leaderboard row, works after finalize. **The rejection is the real decision:** a
persistent kernel adds only names surviving between calls, which is run state not in
`run_dir`, dies at the episode seam anyway, and saves re-pickling ~100KB.
`test_nothing_survives_between_calls` holds the line.

**Live:** the leak that remains is **through the agent's reasoning** — an agent
picking features by staring at out-of-fold errors is doing selection informed by
held-out predictions. Every data-path guard holds; no guard catches this. That is the
price of 0020, and it is live, not hypothetical.

## 2026-08-10 — the input is split; the leaderboard is a public leaderboard

**MacBook Air | master.** Record [0019](docs/decisions/0019-the-leaderboard-is-a-public-leaderboard.md).

`RunSession` splits the input once, before any tool runs, into `work.parquet` +
`val.parquet` — time-ordered on `split_on`, seeded-random otherwise, idempotent and
re-read on resume. Every experiment is scored on the validation half by replaying the
trained table's lineage over raw validation rows, and `best`/`baseline`/`vs_baseline`/
`vs_source_table`/attribution/stop all moved onto `val_score`. `cv_score` stays on
every row: the gap between them is the run's record of a gain that lived only in the
folds. `resolve_table` became an allowlist (it used to accept **any absolute path on
disk**), and `finalize_run` became genuinely once.

**Live:** the reported number is the one the agent optimised against — 150
experiments ranked on the same rows is Kaggle's public leaderboard, and nothing in a
run measures how much the search gamed it. Deliberate; the fix is the third split,
which 0023 then argued for and 0041 built.

## 2026-08-09 — credit by ablation; the payload and the prompt cut hard

**MacBook Air | master | 170 tests.** Records [0017](docs/decisions/0017-credit-by-ablation.md), [0018](docs/decisions/0018-the-payload-is-a-budget.md); **0016 superseded** (its unit of credit survived, its measure did not).

`importance_share` is gone; `explain_table` fits once with every column and once per
contributing table held out. Importance decomposes one model's *attention*; the
question is a counterfactual, and the two come apart routinely — on the motivating
case the date columns score `+1.16` and the four original columns `-0.06`, and the
latter held **77% of the splitting**. One sign convention everywhere (`_gain`, positive
is better), after a `logloss` run reported the same improvement as `-0.19` and `+0.19`
in one result dict. Payload 21 → 13 fields, system prompt 924 → 229 words, under three
subtractive rules: say only what is not derivable, not tool-specific, not run-specific.

**Live:** `explain_table` understates every member of a redundant group — two copies
of a signal both read ~0, and it does not test subsets. The prompt guards are
keyword-based and did **not** fire on a real paraphrased duplication Ulf caught by
reading; that rule still needs a human.
`docs/feature_attribution_and_agent_iteration.md` is stale.

## 2026-08-06 — two journal fixes shipped; metadata-kernel + episodes parked

**MacBook Air | master.** `journal.save` became atomic (tmp → fsync → `os.replace`);
`vs_parent` and `feature_blocks` are now persisted rather than returned to the model
and dropped, so the two numbers that say whether an *idea* worked stop dying at the
next context reset.

Three parked designs were written up here at length and **none has a decision
record**: a metadata kernel over the journal (RLM-style, no path to `tables/`), the
episode loop, and a **SQLite schema for the journal**. The first two were overtaken —
`explore_journal` and episodes were built instead. The SQLite sketch was never
built and never recorded; if it is ever picked up, the full schema is at
`801b52d:HANDOFF.md` and **it needs a record of its own**.

The caveat that outlived all three: **target-conditional statistics must not reach a
metadata path.** Distributional facts about features are safe; a per-column
correlation with the target computed on the full dataset is feature selection on all
rows — selection leakage arriving through metadata while every data-path guard holds.

## 2026-08-05 — orientation notebook: the whole agent, one call at a time

**MacBook Air | master.** Nothing built. `notebooks/phase3_agent_map.ipynb` (109
cells) drives every tool by hand and prints a `tree()` of the run dir after each call,
so the folder filling up is the spine of the explanation. A wiki page was explicitly
**not** written — a notebook breaks loudly when it drifts.

Three findings, all from writing prose against real numbers rather than expected ones:
lgbm on the raw table beats the baseline by 20x under the noise; importance share and
marginal gain point **opposite ways in both directions**; and **the noise floor
accepted noise** — `city` is `rng.choice`, independent of the target by construction,
and it cleared the `min_gain_ratio` bar, was recorded as progress, and shipped as
`best`. The stop rule is a defence against *sustained* self-deception, not a
per-experiment lie detector. That finding is what put the holdout on the path to 0023.

---

*Entries before 2026-08-01 pruned on 2026-08-15 (~2-week retention). Their reasoning lives in `docs/decisions/` 0001–0033 and the commit log.*
