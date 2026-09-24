# 0085 — A data description in the folder is context, not a decision

- **Date:** 2026-09-15
- **Status:** accepted (2026-09-15)
- **Related:** [0029](0029-the-harness-owns-the-exam.md) (the harness must not
  silently make the owner's modeling decisions), [0026](0026-the-brief-is-rendered-not-remembered.md)
  (the brief is O(1); documentation reaches the agent through `inspect_data`, not
  the brief body), the `Customers` gotcha (a silently-made *modeling* decision is
  the failure to avoid)

## What happened

`inspect()` read only the tables. A data dictionary the owner had dropped in the
same folder (`data_description.txt`, a `.md` dictionary — the norm on Kaggle) was
ignored, so the agent saw cryptic column names (`MSSubClass`, `BsmtFinSF1`) with no
meaning attached. The description channel existed (`TaskSpec.notes` →
`inspect_data` returns it as `about_this_data`) but had to be filled by hand.

## The decision

**`inspect()` auto-detects a description file in the folder and loads its text
into `plan.notes`, verbatim.** The agent structures the free text itself; we parse
nothing. Detection prefers an explicitly named dictionary over a generic `README`,
caps very large files, and records `notes_source` plus every candidate so
`summary()` shows what was picked and how to override it.

**Why this is auto-attached and not a confirmation step.** The five owner
decisions `run()` gates on — target, serving columns, split, identifiers, joins —
each *change what is modeled*, so guessing one is the 0029 failure (the `Customers`
leak). A description does not change what is modeled: it is context the agent
reads, exactly the material the owner would otherwise type into `notes`. Attaching
it silently cannot ship a wrong model; at worst it adds text the agent ignores. So
it is loaded, not proposed — while the *modeling* decisions stay strictly
propose-and-confirm.

## What was rejected

- **An LLM pre-parser that turns the description into per-column `column_notes` and
  flags leaks.** Deferred: the descriptions are unstructured prose, and the agent
  is already an LLM that reads `about_this_data` — a second parse would duplicate
  it and add a surface to keep correct. If it returns, its leak flags must be
  *proposals* the owner confirms, because *that* would be a modeling decision.
- **Requiring the owner to confirm the picked-up description.** That would treat
  context like a modeling decision and add friction to the common single-file
  case for no safety gain; the override path in `summary()` is enough.
