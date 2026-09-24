# 0084 — One finalized run has one report model and three views

- **Date:** 2026-09-14
- **Status:** accepted and implemented
- **Related:** [0015](0015-report-attribution-gaps.md),
  [0023](0023-the-third-split-has-evidence-now.md),
  [0083](../../../docs/decisions/0083-learned-features-are-a-reusable-artifact.md)

## Problem

The Markdown report is portable and auditable, but a long linear document makes
model comparison, optimism, feature lineage, and feature definitions harder to
scan than they need to be. The journal contains machine-readable facts, while
the public feature catalog separately knows what each shipped model column
means. A UI assembled independently from those sources could drift, recompute a
score, expose fitted lookup values, or tempt a reader to choose a model by the
internal test after finalization.

## Decision

A finalized run writes three sibling artifacts from one sanitized report view
model:

- `run_report.md` remains the portable text and compatibility surface.
- `run_report.json` is the versioned, machine-readable view model.
- `run_report.html` is a self-contained offline dashboard rendered from that
  JSON-shaped model with inline CSS, JavaScript, and no remote assets.

The view model contains finalized aggregate scores, split facts, shipped model
roles and weights, block-level attribution, safe feature recipes, the confirmed
serving contract, and warnings. It contains no raw rows, targets, fitted lookup
mappings, or estimator objects. Bundle and dashboard feature explanations share
one catalog function so the two public views cannot disagree.

HTML is a renderer only. It cannot score, fit, rank, or change what ships, and
the internal test has no selection action. Glossary help works on hover and
keyboard focus; the feature catalog is searchable without a server. Markdown
remains the default return of `Run.report()`; `Run.report_path()` points to
HTML by default and `Run.report("html"|"json")` reads the other views.

## Rejected

- **React for each run.** It adds Node and a build/runtime surface to an artifact
  that should open by double-click while offline. React remains appropriate for
  a later multi-run viewer that consumes several report JSON files.
- **Rendering directly from the complete journal.** The journal is internal run
  state and may contain more detail than a shareable report should expose.
- **Dumping encoder mappings into the feature catalog.** Exact recipes are
  useful; learned category-to-target values may disclose sensitive statistics.
- **Selecting or recommending a model from test-score interactions.** That
  spends the independent check after the fact and contradicts 0023.

## Cost and evidence

Every finalization writes two additional small files and maintains a versioned
view-model schema. Tests require the HTML to have no external URLs, escape
feature content, support keyboard glossary access, retain metric direction, and
carry the same finalized scores and confirmed serving contract as JSON.
