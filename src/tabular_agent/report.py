"""Human-readable run report."""

import numpy as np

from .metrics import greater_is_better


def _beats_baseline(metric: str, selection_score: float, baseline_score: float) -> bool:
    if greater_is_better(metric):
        return selection_score >= baseline_score
    return selection_score <= baseline_score


def _gain(score: float, reference: float, metric: str) -> float:
    """How much better ``score`` is than ``reference``, positive-is-better always.

    The same sign convention as every ``vs_*`` field the tools write, restated
    here so the report does not have to import from the harness package.
    """
    return score - reference if greater_is_better(metric) else reference - score


def selected_holdout(holdout: dict | None) -> dict | None:
    """The sealed score belonging to the candidate the *search* crowned.

    Not necessarily the one that ships — since 0044 a blend of every terminal
    fit does, and since 0080 that blend is weighted inversely to each member's
    cross-fitting fold spread. Kept distinct from
    :func:`shipped_holdout` because the optimism of the *selected* candidate is
    the quantity 0039 has been tracking across runs, and folding the two together
    would break that series at the run where the blend arrived.
    """
    for entry in (holdout or {}).get("scores") or []:
        if entry.get("selected") and entry.get("holdout_score") is not None:
            return entry
    return None


def shipped_holdout(holdout: dict | None) -> dict | None:
    """The sealed score of what the bundle actually predicts with.

    Falls back to the selected candidate for runs written before blending, where
    the argmax winner *was* what shipped.
    """
    for entry in (holdout or {}).get("scores") or []:
        if entry.get("shipped") and entry.get("holdout_score") is not None:
            return entry
    return selected_holdout(holdout)


def _holdout_section(holdout: dict, split: dict | None, metric: str) -> list[str]:
    """The gap, which is the whole reason the third split is worth its rows (0023).

    Printed as a table over every model scored, not just the winner: the run
    selected one of these on the validation half, and seeing what the sealed rows
    made of the others is what tells a reader whether the *selection* was the
    expensive part or the model was.
    """
    scores = holdout.get("scores") or []
    selected = selected_holdout(holdout)
    shipped = shipped_holdout(holdout)
    lines = ["", "## What the search cost", ""]
    if selected is not None and selected.get("optimism") is not None:
        gap = selected["optimism"]
        # Signed like every other `vs_*` number in the harness — positive is
        # "better" — which here means positive is *bad news about the headline*.
        # Spelled out in words rather than left to the sign, because a reader who
        # gets the direction backwards reads a flattered run as an honest one.
        reading = (
            "the validation half overstated this model by that much"
            if gap > 0
            else "the sealed rows scored it *better* than the validation half did, "
            "so the headline was not inflated"
        )
        lines += [
            f"The candidate the search crowned scored **{selected['val_score']:.4f}** "
            f"on the validation half it ranked by, and "
            f"**{selected['holdout_score']:.4f}** on {holdout.get('n_rows')} sealed "
            f"rows nothing in this run could read — a gap of **{gap:+.4f}**, meaning "
            f"{reading}. That is how much of the headline was the search finding "
            "signal rather than finding luck.",
            "",
        ]
    # The A/B 0042 registered, produced by every sealed run from now on rather
    # than by a script afterwards. Since 0080 every terminal fit still ships,
    # but its pre-seal weight is inverse to cross-fitting fold spread.
    if (
        shipped is not None
        and selected is not None
        and shipped is not selected
        and shipped.get("holdout_score") is not None
    ):
        delta = _gain(shipped["holdout_score"], selected["holdout_score"], metric)
        verdict = "better than" if delta > 0 else "worse than" if delta < 0 else "level with"
        lines += [
            f"**What shipped is not that candidate.** The precommitted "
            f"inverse-fold-spread-weighted blend scored "
            f"**{shipped['holdout_score']:.4f}** on the same sealed rows — "
            f"**{delta:+.4f}**, {verdict} the argmax winner. Every terminal fit "
            "still ships; the weights come only from cross-fitting fold "
            "dispersion, not from validation or sealed scores. Whether that "
            "precommitted reweighting is worth the loss of the best single model "
            "is exactly what this number answers, once per run.",
            "",
        ]
    lines += [
        "| model | experiment | validation | sealed holdout | optimism |",
        "|---|---|---|---|---|",
    ]
    for e in scores:
        mark = " (shipped)" if e is shipped else " (search's pick)" if e.get("selected") else ""
        val = "n/a" if e.get("val_score") is None else f"{e['val_score']:.4f}"
        if e.get("holdout_score") is None:
            sealed, gap = f"not scored: {e.get('holdout_unavailable')}", ""
        else:
            sealed = f"{e['holdout_score']:.4f}"
            gap = "" if e.get("optimism") is None else f"{e['optimism']:+.4f}"
        lines.append(
            f"| `{e.get('model_type')}`{mark} | `{e['experiment_id']}` | {val} | {sealed} | {gap} |"
        )
    lines += [
        "",
        "> **These rows ranked nothing.** The winner was already picked, on the "
        "validation half, before any of them existed. Choosing between them by "
        "this column would spend the third split the same way the second one was "
        "spent, and the next run would have no honest number at all.",
    ]
    if split and split.get("holdout_is_thin"):
        lines += [
            "",
            f"> **WARNING:** {split['n_holdout']} sealed rows is thin. The gap above "
            "has its own standard error and a small one should not be read as a "
            "clean bill of health.",
        ]
    if split and split.get("split_on"):
        lines += [
            "",
            f"> **The gap is not purely selection here.** Split on "
            f"`{split['split_on']}`, the sealed rows are a *later period* than the "
            "validation rows, which are themselves later than the rows the model "
            "trained on. So the difference mixes how much the search gamed the "
            "validation half with how much the world moved between the two "
            "periods, and this run cannot separate them.",
        ]
    return lines


def build_report(
    *,
    task_type: str,
    metric: str,
    selection_score: float,
    fold_metrics: list[float],
    baseline_score: float,
    model_types: list[str],
    feature_names: list[str],
    cv_score: float | None = None,
    split: dict | None = None,
    attribution: list[dict] | None = None,
    val_se: float | None = None,
    holdout: dict | None = None,
) -> str:
    # Empty when a blend ships: an average of k fitted models has no folds of its
    # own, and inventing them from its members' would report a spread for a thing
    # that was never cross-validated.
    fold_std = float(np.std(fold_metrics)) if len(fold_metrics) else float("nan")
    # The headline number's own uncertainty, printed beside it (0022). A score
    # reported bare invites the reader to compare it against other runs digit by
    # digit; +/- one standard error is what says how many of those digits are real.
    headline = f"{selection_score:.3f}"
    if val_se is not None:
        headline += f" (+/- {val_se:.3f})"
    lines = [
        "# Run Report",
        "",
        f"- **Task type:** {task_type}",
        f"- **Selection metric:** {metric}",
        f"- **Validation score:** {headline}",
        f"- **Baseline score:** {baseline_score:.3f}",
        f"- **Models:** {', '.join(model_types)}",
        f"- **Features ({len(feature_names)}):** {', '.join(feature_names)}",
    ]
    if cv_score is not None:
        lines.insert(5, f"- **CV score:** {cv_score:.3f} (fold std {fold_std:.3f})")
    shipped = shipped_holdout(holdout)
    if shipped is not None:
        sealed = f"{shipped['holdout_score']:.3f}"
        if shipped.get("holdout_se") is not None:
            sealed += f" (+/- {shipped['holdout_se']:.3f})"
        # Beside the validation score, never instead of it (0023). The pair is
        # the point: one number is what the search steered towards, the other is
        # what it was worth, and the reader needs both to read the third.
        lines.insert(5, f"- **Holdout score (sealed, scored once):** {sealed}")
    if split:
        where = f"ordered by `{split['split_on']}`" if split.get("split_on") else "random"
        sealed_rows = (
            f" / {split['n_holdout']} sealed holdout" if split.get("n_holdout") else ""
        )
        lines.append(
            f"- **Split:** {split['n_work']} work / {split['n_val']} validation"
            f"{sealed_rows} ({where})"
        )
        lines += [
            "",
            "> **Read the validation score as a public leaderboard.** Every experiment in "
            "this run was ranked against these same rows, so the winner is the best of "
            f"{split['n_val']} rows seen many times over — the score is the one the search "
            "was steering towards, not an independent check on it."
            # Only when there is one to point at. A blend has no folds, and a
            # sentence sending the reader to a number the report does not print
            # is worse than the silence it replaces.
            + (
                " The gap to the CV score says how far the two came apart."
                if cv_score is not None
                else ""
            ),
        ]
        if val_se is None and split.get("split_on"):
            lines += [
                "",
                "> **No uncertainty is reported for the validation score.** It is "
                "estimated by resampling the validation rows, which assumes they "
                "are exchangeable draws from one population. Split on "
                f"`{split['split_on']}` they are a later period, chosen precisely "
                "because it differs from the training rows — so a resample would "
                "measure the wobble *inside* that period and miss the thing that "
                "actually varies, which is the period itself. It would be "
                "confidently too small.",
            ]
        if split.get("val_is_thin"):
            lines += [
                "",
                f"> **WARNING:** {split['n_val']} validation rows is thin enough that the "
                "ranking is largely noise. Differences between the top experiments here "
                "should not be believed.",
            ]
    if not _beats_baseline(metric, selection_score, baseline_score):
        lines += [
            "",
            "> **WARNING:** the model does not beat the trivial baseline. "
            "Suspect a bug, leakage, or an unlearnable target.",
        ]
    if holdout:
        lines += _holdout_section(holdout, split, metric)
    if len(fold_metrics):
        lines.append("")
        lines.append("## Per-fold metrics")
        for i, m in enumerate(fold_metrics):
            lines.append(f"- fold {i}: {m:.4f}")

    if attribution:
        lines += ["", "## How the winning feature set was built", ""]
        # Oldest first: the chain reads as the story of the features.
        for step in reversed(attribution):
            added = ", ".join(step["added"]) or "(no new columns)"
            if step["measured"]:
                scored = f"best {step['best_score']:.4f} over {step['n_experiments']} experiment(s)"
            else:
                scored = "**never trained on**"
            lines.append(f"- `{step['table']}` — {step['method']}: {added} — {scored}")

        unmeasured = [s for s in attribution if not s["measured"]]
        if unmeasured:
            names = ", ".join(f"`{s['table']}`" for s in unmeasured)
            lines += [
                "",
                f"> **Attribution gap:** {names} was built and then extended without ever "
                "being trained on. The winning score therefore cannot be split between "
                "the columns that table added and the ones added after it — there is no "
                "measurement in between. The model is valid; the explanation of *which "
                "features earned the score* is not available from this run.",
            ]
    return "\n".join(lines)


def write_report(run_ctx, text: str) -> str:
    path = run_ctx.path("run_report.md")
    with open(path, "w") as f:
        f.write(text)
    return path
