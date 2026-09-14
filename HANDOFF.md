# Handoff

## Latest session (2026-09-14)

Implemented the first offline HTML run dashboard on branch
`chad/html-report-v1`, based on `origin/master` at `1d2b5ec`.

- every finalization now writes `run_report.md`, versioned
  `run_report.json`, and a self-contained `run_report.html`
- the dashboard shows outcome cards, split purpose, shipped members and weights,
  validation/test/optimism comparison, feature-block history, searchable feature
  definitions, serving contract, and warnings
- glossary help works on hover and keyboard focus; HTML content is escaped,
  print/mobile/dark-mode styles are embedded, and there are no remote assets
- `Run.report()` remains the Markdown-compatible default;
  `Run.report_path()` points to HTML and `Run.report("html"|"json")` reads the
  other views
- bundle and dashboard feature definitions now share
  `feature_catalog.describe_feature_rows`; fitted lookup mappings remain absent
- the facade enriches the finalized dashboard with the owner-confirmed target,
  serving columns, identifiers, and static joins

Decision record:
`docs/decisions/0084-one-finalized-run-has-one-report-model-and-three-views.md`.
The previous handoff is archived verbatim under
`research/archive/handoffs/2026-09-10--2026-09-14.md`.

## Verification so far

- full repository suite: 609 tests pass
- focused dashboard/API/report/minimal suite: 38 tests pass
- full-profile blend/holdout/finalization regression suite passes
- tests prove offline/no-external-URL output, HTML escaping, keyboard glossary
  access, metric-direction-correct baseline gain, safe feature recipes, confirmed
  serving-contract enrichment, and public Run access
- wheel and sdist build successfully with `uv build`

## Next action

Review and merge `chad/html-report-v1`. A useful post-merge acceptance check is
to open the generated HTML from one fresh real facade run and assess the visual
hierarchy with a human reader; the semantic and offline contracts are already
covered automatically.

## Blocking gotchas

- the per-run dashboard is intentionally plain HTML/CSS/JavaScript. React belongs
  to a later multi-run viewer consuming several `run_report.json` files.
- the HTML renderer never scores or selects; do not add test-driven model actions
  to it, because the internal test ranked nothing.
- report JSON is a sanitized projection, not the complete journal. Do not expose
  raw rows, targets, fitted encoder mappings, or estimator objects through it.
