# Pre-publication plan

Six design points from the 2026-09-25 review, with Ulf's calls on each. Suggested
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

**Decision.** Two changes that work together:

- **(a) Coarser signal to the agent: round scores to 3 decimals.** Every score
  and gain the agent sees — tool results (`_agent_score_fields` in `tools.py`)
  and the rendered brief (`harness/brief.py`) — is rounded before it reaches the
  model. Journal and report keep full precision.
  *Caveat to keep in mind:* fixed decimals depend on the metric's scale. On AUC
  or log-RMSE, 3 decimals is about the noise level; on RMSE in raw target units
  (sales in the thousands) it hides nothing. If that bites, the same-effort
  variant is 3 *significant* figures. Better options (a better/same/worse
  verdict from the difference interval, the Ladder) are noted in the chat
  record, not planned.
- **(b) A sealed split in the product run.** Carve it at run start (the full
  profile already has the machinery: `holdout_fraction`, `data/holdout.parquet`),
  keep it away from every tool, score the shipped bundle on it exactly once at
  finalize, and make that the headline number. The evaluation score is then
  labelled as the search score.

**Steps.**
1. Round in `_agent_score_fields` and `render_brief`; test that no agent-facing
   payload carries more than 3 decimals and that the journal still does.
2. Allow `holdout_fraction` in the minimal profile with a default (e.g. 0.2),
   and handle small data (below some row count: warn, or report the interval).
3. Finalize scores the bundle on the sealed rows once; report and README lead
   with it.
4. Decision record: the product run keeps one sealed split.

**Done when** the report's headline number comes from rows no tool and no score
shown to the agent ever touched, and the README's "three splits" claim is true
for the default profile.

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
3. The sealed split (1b) and an external comparison (raw GBM vs this vs
   AutoGluon) are what catch it when the ruler is wrong.

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

## 6. One clear way of presenting the findings

**Problem.** The agent learns where the run stands through three channels that
overlap: the rendered brief (`render_brief`, every turn), `explore_journal`
(agent writes pandas over its own journal), and `explore_table` (agent writes
pandas over the data). The user learns it through a fourth, the report
(0084's view model). Ulf is not happy with the explore tools.

**Proposal.** One findings view model, built from the journal, with two
renderings:

- **To the agent:** the brief, every turn. One row per table: id, parent, what
  it did (tool + columns, one line), result (rounded score and gain vs parent),
  and refusals with their reason. Plus the current best and what is left of the
  budget.
- **To the user:** the report's search-history section, from the same view model
  (full precision there).

What the agent believed during the run is then exactly what the user reads
afterwards. `explore_journal` goes: the brief is the answer to "what have I
tried". The full profile's extra tools go with the profile (see below).

**Open question for Ulf: what replaces `explore_table`?** The agent does need to
look at data to have feature ideas.
- *Option A:* fixed views in `inspect_data` — per-column profile, relation to
  the target on work rows, group sizes for candidate keys. Predictable, cheap in
  tokens, no free-form code.
- *Option B:* keep one read-only code tool, renamed and capped. Most flexible,
  but it is the second free-form code path and the least auditable (exploration
  code is not recorded).
My lean is A, with B only if runs show the agent starving for a view A lacks.

**Tied to it: one profile.** `ForecastingAgent` hardcodes the full profile
(`api.py`) and the forecast workflow requires it (`build_agent`). Settling the
findings view is the moment to move forecasting onto minimal plus its forecast
tools and retire the full profile, so there is one tool surface to publish.

**Steps.**
1. Decide Option A or B.
2. Build the view model; render the brief from it; point the report's history
   section at it.
3. Remove `explore_journal` (and `explore_table` if A).
4. Move forecasting onto the minimal profile; retire the full profile.
5. Decision record: one findings view for agent and user.

---

## Suggested order

| # | item | size |
|---|---|---|
| 1 | 4 — Windows note and classifiers | minutes |
| 2 | 3 — fix the hint, clean the vocabulary | an hour |
| 3 | 1a — round agent-facing scores to 3 decimals | an hour |
| 4 | 2 — one sentence on the ruler in README and report | minutes |
| 5 | 1b — sealed split in the product run | a day |
| 6 | 6 — findings view, explore tools, one profile | a few days; needs the A/B call first |
| — | 5 — table storage | deferred |
