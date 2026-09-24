# 0080 — A blend weight a member cannot inflate, decided before the seal

- **Date:** 2026-09-01
- **Status:** accepted (2026-09-01). Supersedes 0045's "not implemented" stance
  for the blend *weighting*; 0045's refusal to select a member by a holdout still
  stands.
- **Related:** [0045](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md)
  (which registered the prediction this record acts on, and which refused to act
  on the earlier evidence), [0044](../../../docs/decisions/0044-what-ships-is-the-average-not-the-argmax.md)
  (equal-weight average, whose weighting this changes), [0042](0042-the-terminal-argmax-shipped-the-worst-model.md),
  [0022](0022-*.md) (variance must not come from `val`), [0034](0034-*.md)
  (a rule with a constant in it is a smell), [0068](0068-*.md) (val prefers a
  budget the seal punishes)

## What happened

0045 refused to touch the unfiltered blend and **registered a prediction instead**,
before the run that would test it existed:

> On the next sealed run, the gap between the full blend and the blend without its
> worst-scoring member is larger than the seal suggests.

That run now exists — the Rossmann final run (`origin/chad/rossmann-final`,
git-verified). It **confirms the registration**:

| | value |
|---|---|
| shipped four-way blend, sealed | 0.14706 |
| best single member (lgbm), sealed | 0.13446 |
| **cost of shipping the blend** | **+0.01260** |
| seal's own advance estimate of that gap | +0.0128 |
| the member that failed | catboost, **+0.051 sealed-vs-val optimism** |

The mechanism 0045 named is exactly what happened: an equal-weight member
contributes `1/k` of every prediction with no bound on how wrong it can be, and
the blend has no way to notice a member that fails *badly* rather than *poorly*.
The seal estimate (+0.0128) and the live gap (+0.0126) agree, and the failure
moved from `elasticnet` (house v9) to `catboost` (Rossmann) — so it is a property
of the *weighting*, not of a model family. That is the thing a category rule
could never fix and 0045 was right to refuse.

**What is now different from 0045's day:** the effect is no longer "unresolved."
0045 could not separate horizon from noise on a difference of 0.00008; here the
gap is 0.0126 with an advance estimate that matched it. The registration is
resolved in favour of the horizon reading. Acting on it now is not "choosing a
rule by consulting a holdout" — the rule below is fixed on first principles and
in the run's own yardstick, and Rossmann is its *first evidence*, not the set it
was tuned on.

## The decision

Replace the equal-weight blend with a weight a member cannot inflate by being
optimistic — one part, decided before any held-out set is read:

**Weight members by inverse out-of-sample instability, not equally.** Each
member's weight is `w_i ∝ 1 / (s_i + ε)`, where `s_i` is the member's
**cross-fitting fold-spread** — the fold-to-fold dispersion of its OOF score,
computed by the harness and **not** derived from `val` (0022) or from the seal.
A member that is unstable across the ordered folds — the in-sample signature of a
member that will be optimistic out of sample — is down-weighted automatically.
`ε` defaults to the median spread so a zero/unknown-spread member cannot take all
the weight. No constant, no category, no holdout consulted (0034). This is the
"weights that are not equal but are also not fitted" option 0045 licensed, with
the variance estimator named: the fold spread, which 0022 permits for a *relative*
comparison between members (its use here reads only ratios, never a level — 0022's
objection was to fold-spread as an absolute bar). **Every member still ships; this
reweights, it does not select** (0044 stands).

## What is rejected

- **A `val`-based safety net ("ship the best single member if the weighted blend
  does not beat it on `val`").** This was in the accepted draft and was cut during
  implementation, because it is wrong: the best member is *by construction* the one
  that scored best on `val`, so a blend beats it on `val` only when the members are
  diverse enough — often they are not, and then the net fires and ships the argmax.
  That is exactly the selection 0042/0044 refuted (the val-argmax shipped the worst
  model on the seal). Demonstrated: on a clean synthetic fixture the net fired and
  shipped one model of four. A net that reverts to the refuted rule on healthy data
  is not insurance, it is the bug. The catastrophic-member guard (a member collapsed
  to a constant, a scale error) is therefore **left open**, exactly as 0045 left it —
  it needs a signal that is neither the seal nor the val-argmax, and none is measured.
- **Dropping a member by its seal/holdout score** (0045's refusal stands): the seal
  is the unconsulted judge; spending it to pick blend membership leaves nothing to
  judge the result.
- **A hard exclusion threshold in fold-spread units** (`exclude if s_i > k·median`).
  Defensible, but it reintroduces the constant `k` that 0034 refused twice.
  Inverse-variance weighting achieves the same down-weighting continuously with no
  constant to justify, so it is preferred.
- **Reading any parameter off the Rossmann sweep.** The weighting scheme and the
  variance estimator are fixed here, a priori. Rossmann is the confirming evidence,
  not the tuning set (0022; Qwen's check).

## What it costs, and what is registered

- **Cost:** a member's fold-spread must be available to the finalizer where blend
  weights are set (`save_bundle` receives `weights`). If it is not already threaded
  there, that is the whole implementation surface — the guards, tools, and bundle
  format are untouched.
- **Registered, before the next unconsulted run exists:**
  1. On the next sealed run on a *new* dataset, the inverse-fold-spread blend's
     sealed score is **≤** the equal-weight blend's, and its gap to the best single
     member is smaller than the equal-weight blend's gap. (Rossmann is consulted
     now, so it can only illustrate, not confirm — this is the honest first test.)

## Ordering consequence

The product facade (product-ux-plan.md, P0) promises "what ships is honest." That
promise is only true once this weighting exists. So: **this record accepted →
implement the inverse-fold-spread weighting with a regression test → then P0.** No facade code before the
gate.
