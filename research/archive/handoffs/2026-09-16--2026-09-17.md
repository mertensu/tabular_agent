# Handoff

## Latest session (2026-09-16)

Built an evidence-bound capability-review V1 on branch
chad/capability-review-v1, based on origin/master at 8bcf0e3.

- Agent.review_capabilities(run) performs one explicit post-finalization LLM
  review with structured output and no tools
- the reviewer receives allow-listed aggregate run evidence, never raw rows,
  target values, fitted lookups, transform code, artifact paths, or transcripts
- findings distinguish capability, agent, tool, model/selection, and data-contract
  gaps from a run that demonstrates no generic gap
- every finding must cite real evidence IDs, name a generic capability, state an
  alternative cause and leakage/serving risk, and propose a falsifiable test
- unknown evidence references are rejected before persistence
- capability_review.json is write-once under a cross-process lock; repeat calls
  return the persisted result instead of spending another model call
- Markdown, sanitized JSON, and offline HTML all gain a “What limited this run?”
  section; dynamic prose is escaped in rendered surfaces
- tabular_agent.capability_backlog(runs) groups deterministic fingerprints as
  proposed or recurring; it can never accept a product change

The architectural boundary and rejected alternatives are recorded in
docs/decisions/0088-a-run-may-propose-a-capability-gap-but-not-change-the-product.md.
The previous handoff is archived verbatim under
research/archive/handoffs/2026-09-15--2026-09-16.md.

## Verification so far

- focused reviewer/dashboard/API suite: 21 tests pass
- full repository suite: 617 tests pass, exit 0 (6:50)
- unknown-evidence, prompt allow-list, write-once, report injection, and recurring
  backlog contracts have dedicated regression tests
- Wheel and sdist build successfully with uv build
- git diff --check passes

## Next action

Review and merge chad/capability-review-v1. After several independent real
runs have reviews, inspect the grouped backlog and preregister one recurring
hypothesis on an unseen dataset; do not treat recurrence itself as acceptance.

## Blocking gotchas

- capability review is deliberately opt-in because it is an additional paid LLM
  call and transfers sanitized run metadata to the configured provider
- the internal test score may support retrospective diagnosis, but the reviewer
  has no fit/score/select/ship action; do not add one
- do not replace the allow-list with the full journal: it contains transform code
  and filesystem paths that the reviewer does not need
- the lock file is durable coordination state; as with the journal lock, flock
  is attached to the descriptor and is released automatically
- recurring means “worth an owner-preregistered test,” never “build this tool”
