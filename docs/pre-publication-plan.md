# Pre-publication plan

Six design points from the 2026-09-25 review, with Ulf's calls on each (revised
the same day: no sealed split; the explore tools stay). Suggested
order is at the end. Items that change a product contract get a decision record
(next free number: 0092).

---

## 1. The reported score is the score the search selected on

**Problem.** `ta.Agent()` runs the minimal profile: one work/evaluation split.
The agent sees `evaluation_score` for every table it builds and keeps what scores
higher, and the final report prints that same score. The report says so
honestly ("an internal search score, not an independent estimate"), but the
README sells "three splits" and "every number it reports is one you can trust".
The minimal profile rejects `holdout_fraction` outright (`session.py`).

**Decision.** Keep the single work/evaluation split. Give the agent a coarser
signal instead: **round every score and gain it sees to 3 decimals** — tool
results (`_agent_score_fields` in `tools.py`) and the rendered brief
(`harness/brief.py`). Journal and report keep full precision. This slows how
fast the evaluation rows wear out; it does not make the final score an
independent estimate, and the report keeps saying so.

*Caveat:* fixed decimals depend on the metric's scale. On AUC or log-RMSE,
3 decimals is about the noise level; on RMSE in raw target units (sales in the
thousands) it hides nothing. If that bites, the same-effort variant is
3 *significant* figures.

A sealed third split was considered and not taken: it costs rows (on ordered
data, the freshest ones — 0043), and the report already labels the score as a
search score.

**Steps.**
1. Round in `_agent_score_fields` and `render_brief`; test that no agent-facing
   payload carries more than 3 decimals and that the journal still does.
2. Make the README match the default profile: drop "Three splits" from the
   AutoML comparison, and replace (or label) the House Prices report excerpt —
   its validation/test/optimism table comes from a full-profile run with a
   sealed split, which `ta.Agent()` does not have.

**Done when** the agent never sees a score finer than 3 decimals, and nothing in
the README promises a split the default run does not make.

---

## 2. The ruler is what the agent optimizes against

**Problem.** Tables are ranked by a fixed, untuned 200-tree model. 0054, 0067
and 0073 (archived) show it ranks features differently from a model with more
capacity, and the agent iterates against it.

**Decision.** Keep it. Searching has to be fast, and no better ruler has been
found. The trade-off is accepted, not solved.

**Steps.**
1. No code change.
2. State it where users read: one sentence in the README and the report — tables
   are ranked by a fast fixed model; the shipped models are fitted at full
   strength, so the ranking can disagree with what finally ships.
3. An external comparison (raw GBM vs this vs AutoGluon) is what would catch
   it when the ruler is wrong.

---

## 3. Leftover vocabulary, and a hint pointing at a missing tool

**Problem.**
- The sandbox's teaching hint says *"use train_tabular_model instead"*
  (`sandbox.py`, `_TRAINING_LIBS`). No profile exposes that tool to the agent,
  so the hint meant to stop wasted turns causes them.
- The product glossary still defines **experiment** (a CV-trained model per
  table), which 0035 replaced with the ruler; the journal docstring describes
  `append_experiment` being called by that tool; `append_note` is "not wired
  into a tool yet".

**Steps.**
1. Hint: say model training is not available to agent code and point at the
   fitted tools that exist (`fit_encoding`, `fit_group_stats`). Test the message
   names only registered tools.
2. Product-facing vocabulary: run, table, lineage, bundle. Remove "experiment"
   from README-level docs; leave decision records alone (they are history).
3. Journal docstring: describe what is written today. Delete `append_note` or
   wire it — do not keep a writer nothing calls.

---

## 4. It does not work on Windows

**Problem.** `import tabular_agent` loads `fcntl` (Unix-only) through
`harness/journal.py`, `harness/tools.py` and `capability_review.py`. On Windows
the import fails on line one.

**Decision.** Do not port it now. Say so.

**Steps.**
1. README, under Install: "macOS and Linux. Windows is not supported at the
   moment (the run journal uses Unix file locks); WSL works."
   *(Verify the WSL claim before writing it, or leave it out.)*
2. `pyproject.toml`: classifiers `Operating System :: MacOS` and
   `Operating System :: POSIX :: Linux`.
3. Optional: a clear error instead of `ModuleNotFoundError: fcntl` — import the
   lock lazily, or catch it at package import and say "Windows is not supported".

---

## 5. Every step writes a full copy of the table

**Problem.** Each table is a full parquet copy; `_runs/zomato` is 606 MB.
Storage grows with steps × dataset size.

**Decision.** Deferred. Fine at the sizes this is used on. The fix when needed:
store only a table's new columns plus lineage, and materialize on read.

---

## 6. One clear way of presenting the results to the user

**Problem.** At the end of a run the user meets many surfaces: `run.report()`
(Markdown), `run.report_path()` (HTML dashboard), `run.report("json")`,
`bundle.describe_features()`, `run.training_features()`, `bundle.explain()`,
and the optional capability review under "What limited this run?". 0084 already
makes the three report formats renderings of one view model, so the facts agree;
what is missing is **one obvious place to start and one fixed reading order**.
The README also shows a full-profile report excerpt, so the report a new user
sees does not match the one they read about (see 1, step 2).

**Proposal.** The HTML report is *the* result; the others are exports.

- `agent.run(...)` ends by printing a short summary (score, baseline, number of
  features, where the report is) and the path to the HTML report.
- The report answers the user's questions in a fixed order, one section each:
  1. **How good is it, and how far can I trust that?** Evaluation score next to
     the trivial baseline, labelled as a search score, with the interval where
     one exists.
  2. **What did it build?** The feature story: which tables led to what shipped,
     in plain words, from the journal.
  3. **What ships and how do I use it?** Models in the blend, the confirmed
     serving contract, the two lines to predict and save.
  4. **What limited it?** Refusals, warnings, and the capability review if run.
- Markdown and JSON stay as renderings of the same view model; the
  feature-catalog, training-features and explanation calls are linked from
  section 3 rather than being separate entry points in the README.

**Open question for Ulf:** what bothers you most about the current end of a
run — too many places to look, the report's content or order, or the gap
between what the README shows and what a default run produces? The answer
decides whether this is a README-and-ordering change (small) or a rework of the
report sections (larger).

**Steps** (once that is answered).
1. Look at a real minimal-profile report end to end and list what a newcomer
   cannot answer from it.
2. Reorder or rewrite the report sections to the four questions above.
3. Make `agent.run` print the summary and path.
4. Rewrite the README's "What you get back" around the one report, with an
   excerpt from a real `ta.Agent()` run.

---

## Suggested order

| # | item | size |
|---|---|---|
| 1 | 4 — Windows note and classifiers | minutes |
| 2 | 3 — fix the hint, clean the vocabulary | an hour |
| 3 | 1 — round agent-facing scores; README matches the default profile | an hour or two |
| 4 | 2 — one sentence on the ruler in README and report | minutes |
| 5 | 6 — one clear results presentation | small to medium; needs Ulf's answer first |
| — | 5 — table storage | deferred |
