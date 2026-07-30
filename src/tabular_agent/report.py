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
    holdout_score: float | None = None,
    attribution: list[dict] | None = None,
) -> str:
    fold_std = float(np.std(fold_metrics))
    lines = [
        "# Run Report",
        "",
        f"- **Task type:** {task_type}",
        f"- **Selection metric:** {metric}",
        f"- **CV score:** {selection_score:.3f} (fold std {fold_std:.3f})",
        f"- **Baseline score:** {baseline_score:.3f}",
        f"- **Models:** {', '.join(model_types)}",
        f"- **Features ({len(feature_names)}):** {', '.join(feature_names)}",
    ]
    if holdout_score is not None:
        lines.append(f"- **Holdout score:** {holdout_score:.3f}")
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
