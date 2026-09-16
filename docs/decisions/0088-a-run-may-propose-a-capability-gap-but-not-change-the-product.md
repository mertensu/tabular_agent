# 0088 — A run may propose a capability gap, but not change the product

- **Date:** 2026-09-16
- **Status:** accepted and implemented
- **Related:** [0023](0023-the-seal-is-one-number-seen-once.md) (the internal
  test ranks nothing), [0029](0029-the-harness-owns-the-exam.md) (the harness
  owns evaluation), [0057](0057-the-capabilities-the-agent-does-not-have.md)
  (the first manual capability inventory), [0070](0070-what-the-winner-had-that-we-cannot-express.md)
  and [0071](0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md)
  (capability gap, agent gap, and tool defect are different claims),
  [0084](0084-one-finalized-run-has-one-report-model-and-three-views.md)
  (one sanitized report model)

## What happened

Every important product improvement so far came from a manual postmortem. The
Rossmann records separated operations the tools could not express from operations
the agent simply failed to ask for, and from a ruler defect that made good work
look bad. The Otto run supplied the opposite result: its anonymous, already
aggregated columns gave the feature agent little leverage, while the model family
and blend dominated the error. Calling both outcomes “we need another feature
tool” would erase the most useful distinction the experiments produced.

That analysis existed only in conversations and decision records. Nothing in a
finalized run made the categories explicit, required evidence, or let recurring
claims be grouped across datasets.

## The decision

**A capability review is a separate, explicit post-finalization pass.**
`Agent.review_capabilities(run)` gives a tool-less structured-output reviewer an
allow-listed account of the finalized run. It can classify a limitation as:

1. a missing generic capability;
2. an agent failure to use an existing capability;
3. a tool or default defect;
4. a model/selection limitation;
5. a data-contract limitation; or
6. no demonstrated gap.

Every finding must cite IDs from the supplied evidence, name the desired
capability without a dataset name, consider the current tools, state leakage or
serving risk, give a falsifiable hypothesis, and name an alternative explanation.
Unknown evidence IDs are rejected before an artifact is written.

The input is built by an allow-list. It contains finalized aggregate scores,
split and serving contracts, feature-attempt summaries, experiment summaries,
warnings, and refusal counts. It excludes rows, target values, transform code,
fitted lookups, artifact paths, and transcripts. The reviewer has no tools, so it
cannot fit, score, select, or ship anything. Its result is written once to
`capability_review.json` under a cross-process lock and appears in all three run
report views as **“What limited this run?”**

`capability_backlog(runs)` groups the deterministic fingerprints of multiple
reviews. One occurrence is `proposed`; two independent runs make it `recurring`.
Those are the only automatic states. `preregistered`, `tested on unseen run`, and
`accepted` remain owner decisions backed by a separate experiment. Recurrence is
reason to test a claim, not proof that it is true.

The pass is opt-in rather than hidden inside `Agent.run()`. It spends another LLM
call, and the user should know when that cost and external metadata transfer
happen. Re-reading a reviewed run returns the persisted result rather than asking
again, preventing repeated prompts from fishing for a preferred diagnosis.

## Why the internal test may be visible

The once-scored internal test is valuable for diagnosing validation optimism and
selection defects after the run. Hiding it would prevent the reviewer from seeing
the exact evidence this mechanism exists to collect. But the review has no action
surface: it cannot re-rank a model, modify a blend, or launch a fit. The score is
therefore evidence about a completed process, never an input to that process.

## What was rejected

- **Let the feature agent review itself during the run.** It would rationalize
  its own choices, consume search context, and could steer toward the internal
  test once that score exists.
- **Give the reviewer feature or shell tools.** Then “postmortem” would be another
  search phase and the test would become a second validation set.
- **Send the complete journal.** Transform code, filesystem paths, fitted state,
  and other unnecessary details widen both privacy exposure and prompt-injection
  surface without adding diagnostic evidence.
- **Automatically add or change tools from one finding.** A stochastic opinion
  about one dataset is not a product specification. Even repeated findings only
  earn a preregistered test.
- **Run the review automatically.** That hides provider cost and egress behind a
  completed training call. Explicit invocation keeps both visible.

## What it cost

One optional LLM call per finalized run, a small JSON artifact, and a deterministic
aggregation pass. The reviewer can still be wrong; structured evidence and
recurrence make its claims auditable, not true. Fingerprints currently rely on a
normalized capability key, so semantically identical findings with different keys
may remain separate until an owner consolidates them.
