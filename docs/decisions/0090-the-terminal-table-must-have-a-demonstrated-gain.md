# 0090 — The terminal table must have a demonstrated gain: the probe-score argmax keeps descending into the noise floor

- **Date:** 2026-09-17
- **Status:** proposed
- **Proposed by:** Qwen, after the Rossmann ForecastingAgent repeats; number
  assigned by Claude (0089 taken on `claude/episodic-runner`)
- **Related:**
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  block instrument this record finally lets decide something),
  [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (whose open bar this
  supplies, in the search's last step),
  [0078](0078-the-panel-was-a-stop-rule-and-the-gain-threshold-is-the-real-lever.md)
  (the same gain-over-noise idea, as a *stop* rule; this is the same idea as a
  *selection* rule),
  [0012](0012-harness-owns-the-stop.md) (the harness owns the stop, and the
  terminal table choice is stop territory),
  [0042](0042-the-terminal-argmax-shipped-the-worst-model.md) (what an
  ungoverned terminal choice did before),
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) (the seal
  is still read once, at the end — this gate never touches it)

## The problem, stated as a count

The terminal basis — the table every terminal fit trains on — is chosen by
`stop.best_table`: the argmax of `probe_score` over all tables, one fixed ruler.
Across the four completed runs on this box, **the shipped terminal basis had a
gain inside one block standard deviation of its source in 3 of 4 runs**:

| run | shipped basis | gain | block_sd | sign_stable | blocks favouring |
|---|---|---|---|---|---|
| forecast repeat_01 | t9 | +0.00073 | 0.00282 | False | 3/5 |
| forecast repeat_02 | t14 | +0.00221 | 0.00249 | False | 4/5 |
| forecast repeat_03 | t12 | +0.00162 | 0.00107 | False | 4/5 |
| rossmann_final | t8 | +0.00582 | 0.00622 | False | 4/5 |

All four are *gains*, but three are inside their own interval and all four have
a disagreeing block. The instrument says: **on the rows that were measured, it
is not demonstrated that this table beats the one it came from.** The argmax
took it anyway, because each noise-level step still moved `probe_score` a hair
in the favourable direction on one validation window.

The mechanism is a ratchet. Every step in a lineage adds a small positive
movement on val, so `probe_score` keeps improving past the point where the
block instrument stopped agreeing. repeat_02's basis walked five steps from the
lineage's first demonstrated gain (t3, +0.011, 5/5 blocks) to t14 (+0.0022,
4/5); the last two steps (t11, t14) were both inside noise. repeat_01's basis walked two
noise-level steps past its last demonstrated gain (t7). The argmax chases
movements the interval says are not demonstrated, and the terminal fits inherit
exactly that drift: repeat_02's winner optimism was **+0.0215** (val 0.1364 →
holdout 0.1579), against +0.006/+0.007 in the two sibling repeats.

The instrument already exists, is already computed for every table
(`probe_vs_source_interval`, 0059), and is already stored in the journal. It
decides nothing: `sign_stable` is surfaced to the agent in the leaderboard view
and consulted nowhere — not by the stop rule, not by the terminal choice. This
record is the 0028 bar arriving at the one place it has been missing: **the
last selection event of the run.**

## The rejected fixes, and why

Two alternatives were considered first and measured away:

1. **Prune tables with non-positive marginal gain at build time** (Claude's
   proposal from the same session). Measured against all three forecast runs:
   **17 of 17 negative-gain tables already had no children** — the agent never
   builds on a table it can see has lost. The rule would change nothing, and
   repeat_02's basis had no negative step in it: every ancestor of t14 was
   positive, just not *demonstrated*.
2. **Prune the agent's search** (refuse to build on a non-significant parent).
   The search is the agent's (0035); a noise-level step may still be the road
   to a later real one, and a gate that forbids building changes search
   behaviour for a problem that shows up only at selection time.

## The decision

**The terminal basis is the argmax of `probe_score` among tables whose gain
over their source is demonstrated: `gain > block_sd` — strictly beyond the
pairing's one-standard-deviation noise bar, the yardstick's own bar.**
Tables without an interval (no-op transforms, failed probes) are not candidates.
If no table passes, the basis falls back to the work table — shipping what the
data had before the search, the same "an interrupted search still ships" floor
the run already uses. Tie-break is unchanged (earlier table wins: fewer steps,
less lineage to go wrong at predict time).

Three properties this keeps deliberately:

1. **It is a selection rule, not a search rule.** The agent still builds
   whatever it wants; the gate acts once, at `_terminal_fits` time, in
   `stop.best_table` — harness territory (0012), a pure function of the
   journal's table list, unit-testable like `evaluate`. The sealed split is
   not consulted, at build time or at gate time; the seal is still read once,
   at finalize (0041).
2. **The threshold is the yardstick's own bar, not a new constant.**
   `gain > block_sd` is the same one-standard-deviation bar the stop rule
   already uses to call a table "inside the noise" (`min_gain_ratio`, 0078).
   Nothing here is fitted to the four runs above: the rule was derived from
   *which quantity to gate on*, not from *which constant would have excluded
   t14*. A constant chosen to fix repeat_02 would be exactly the
   holdout-shaped fitting 0034 forbids.
3. **It downgrades the argmax, it does not replace the ruler.** The ruler is
   still the fixed probe (0028); the gate only restricts which tables the
   ruler's ranking is allowed to pick from. This is the 0078 lesson applied
   where 0078 stopped: the panel was rejected as a *vote over schemes*; a
   single instrument's own interval gating a single selection event is a
   different, cheaper thing.

## What is registered before the measurement

Replay the gate offline over the four completed runs (journals only, no
refit): which table the gate picks, versus what shipped. The claim under test
is not "the gate improves these four runs" — that would be fitting — but that
**the gated basis differs from the shipped basis in exactly the runs where the
shipped basis was undemonstrated, and that on any future run the gate's choice
is decided before its holdout score is looked at.** The sealed scores of
gated-basis terminal fits (which requires an offline refit on the chosen
ancestors) are read once, at the end, and compared against the shipped ones.
If the gate's picks lose on the seal, the gate loses, and the record says so.

## Open questions the measurement must answer

- **Fallback pressure:** on runs where the search builds twenty tables and the
  gate accepts three, is the gated argmax the right table, or does the gate
  need to look only at the *tail* of the lineage (the demonstrably-improving
  stretch)? The replay answers this per run.
- **`gain == 0` tables:** no-op transforms read `gain = 0, sd = 0,
  sign_stable = True` and would satisfy any naive "stable" filter — the
  offline replay already shows this is real: filtering on `sign_stable` alone
  picks a degenerate no-op table in 2 of the 4 runs. The gate therefore
  requires strictly positive gain over a non-zero interval, which excludes
  them; the replay confirms it matters in practice.
