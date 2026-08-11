"""Human-readable run report."""

import numpy as np

from .metrics import greater_is_better


def _beats_baseline(metric: str, selection_score: float, baseline_score: float) -> bool:
    if greater_is_better(metric):
        return selection_score >= baseline_score
    return selection_score <= baseline_score


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
) -> str:
    fold_std = float(np.std(fold_metrics))
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
    if split:
        where = f"ordered by `{split['split_on']}`" if split.get("split_on") else "random"
        lines.append(
            f"- **Split:** {split['n_work']} work / {split['n_val']} validation ({where})"
        )
        lines += [
            "",
            "> **Read the validation score as a public leaderboard.** Every experiment in "
            "this run was ranked against these same rows, so the winner is the best of "
            f"{split['n_val']} rows seen many times over — the score is the one the search "
            "was steering towards, not an independent check on it. The gap to the CV score "
            "says how far the two came apart.",
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
