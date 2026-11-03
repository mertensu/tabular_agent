# freMTPL2: rows observed for different lengths of time

Opened 2026-11-03. Status: **references, r1 and r2 run 2026-11-03; readings below.**

## Question

Some tabular problems count events over an observation period that differs by row:
claims over the months a policy was in force, sales over the days a product was
listed, failures over a machine's hours. The textbook model fits the rate with the
period as an offset (or, equivalently for a Poisson loss, the rate weighted by the
period). The harness has a Poisson objective but no offset and no row weights, so the
period can only enter as an ordinary column the trees split on.

How much does that cost, and does the agent find a way around it? freMTPL2freq is the
standard public table for this: 678,013 French motor policies, claim counts, exposure
in years, with published GLM and GBM results.

## Design

- Data: OpenML 41214 into `fremtpl2/` (gitignored); `scripts/prep_fremtpl2.py` only
  strips the quotes OpenML puts around `VehGas`. The literature's caps (exposure at 1,
  claims at 4) are not applied: what counts as an error is the run's to decide. The
  notes are the CASdatasets documentation, transcribed.
- Test rows: 20% of policies (seed 0, stratified on having a claim), never seen by a
  run or used to fit a reference. Score: mean Poisson deviance of the predicted claim
  counts, lower is better. For a count predicted as rate × exposure this equals the
  exposure-weighted deviance of the rate, so the offset and the column formulations
  are scored on one scale.
- `references.py` (no LLM): intercept-only, a Poisson GLM with offset (scikit-learn
  tutorial preprocessing), and early-stopped LightGBM Poisson twice: **Exposure as a
  column** (the best the harness can express) and **log(Exposure) as an offset**.
  Paired bootstrap against the column version.
- `runner.py`: one run of `ta.Agent()`, minimal profile with 0098 (early-stopped fit
  ships), objective Poisson, metric `poisson_deviance`, agent model `gpt-6.1-terra@eu`
  at medium effort. IDpol is offered as an identifier and accepted.

## Prediction (written before running)

- The offset beats Exposure-as-a-column in LightGBM, with a paired 95% interval
  excluding zero, by at least 0.001 in mean deviance (about 0.3%). Trees can
  approximate a monotone exposure effect, but only as a step function, and every
  split spent on it is one not spent on risk factors.
- Both LightGBM arms beat the GLM by more than that.
- The agent's run lands within 0.001 of `lgbm_feature`: its features (age bands,
  density logs, interactions) add little that trees cannot find, and it has no
  way to express an offset. It may build something like log(Exposure) as a column,
  which a tree is indifferent to.

## Decision rule

If the offset's gain over the column is clear of zero and at least 0.001: a row's
observation period is a gap any count problem can hit, and an offset (or a weight)
belongs in `TaskSpec` as an owner's declaration, like `objective`. Otherwise the trees
absorb it and the harness needs nothing.

## Readings

Mean Poisson deviance of the claim counts on the 135,603 test policies (lower is
better); paired bootstrap intervals against `lgbm_feature`. Uncapped data (the prep
script was first written with the literature's caps; results were the same within
0.0003).

| arm | deviance | vs `lgbm_feature` |
|---|---|---|
| null (frequency × exposure) | 0.3279 | worse by 0.040 |
| GLM, offset | 0.3116 | worse by 0.024 |
| LightGBM, log(Exposure) as offset | 0.2999 | worse by 0.012 [0.0105, 0.0137] |
| LightGBM, offset **and** Exposure as a column | 0.2879 | level |
| **LightGBM, Exposure as a column** | **0.2879** | — |
| agent r2 (IDpol an identifier) | 0.2884 | worse by 0.0005 [0.0002, 0.0008] |
| agent r1 (IDpol used) | 0.2291 | not a reading, see below |
| LightGBM, Exposure and IDpol as columns | 0.2307 | not a reading |

- **Against the prediction: the offset is worse, by 0.012.** Exposure here is not
  observation time in the offset's sense. Claims per policy-year are 0.46 below 0.1
  years of exposure and 0.07 at a full year; claims per policy grow only from 0.027
  to 0.070. Short exposures are plausibly policies that ended early, some because of
  a claim. An offset forces the count to be proportional to exposure; a column lets
  the trees learn the relation as it is. With the column present, the offset adds
  nothing.
- **Against the decision rule: not met.** On this table the harness needs no offset
  or weight; exposure as a column is the better model. This does not show the gap
  is absent where the period really is pure observation time.
- **The agent (r2) is level with the raw table**, 0.0005 behind (interval clear of
  zero, but small). 14 tables in under 2 minutes: target and native categorical
  encodings, exposure transforms (log, months, short/full-year flags), age and
  bonus bands, interactions. None beat the raw columns by more than noise. It kept
  Exposure as a column and did not filter the large exposures or claim counts.
- **r1 mined the ID.** The harness did not propose `IDpol` as an identifier (the
  name check in `inspect.propose_identifier_columns` misses a leading `ID`), and
  the agent built buckets of it (`IDpol // 1000`, ...) while noting in its
  reasoning that it might overfit. Policies with claims are numbered first in the
  source, so the ID carries the target; a random test split shares the artefact,
  so no number the run reports can catch it. r2 declared `IDpol` an identifier, as
  an owner who knows the column would, and the run dropped it before the agent saw
  anything.
