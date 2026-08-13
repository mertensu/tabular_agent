"""Rebuild a submission from **any** experiment in a run, not just the winner.

`finalize_run` packages exactly one model — the one `best` points at, which is
`argmax(val_score)`. Every other experiment stays on disk fully intact (its final
model, its metadata, the lineage of the table it was trained on) and has no way to
be turned back into predictions. This script is that way.

It exists because of a measurement rather than a preference. On Porto Seguro the
selected model scored 0.26522 on Kaggle and an unselected one from the same run
scored **0.27724**, having lost the selection by 0.0004 of `val_score` — a
twentieth of one `val_se`. Deciding whether that is a rule or a coincidence means
scoring models the harness declined to package, on datasets where the truth is
already known.

**This is a diagnostic tool, not part of a run.** It reads a finished run
directory and writes a CSV; it never touches the journal, and nothing in `src/`
depends on it. Which is the point — if second-guessing the selection ever becomes
something the *harness* does, that is a decision record and a change to
`finalize_run`, not a script.

    uv run python scripts/rebuild_submission.py _runs/seguro_v1 e3 \\
        --test seguro_driver_prediction/test.csv --id id --target target

The rebuild is verified before it is written: the reconstructed model is replayed
against the run's own validation half and the score must match what the journal
recorded for that experiment. A mismatch means the reconstruction is wrong — most
likely the wrong lineage — and is worth more than a submission file.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from tabular_agent.metrics import compute_metric
from tabular_agent.pipeline import Pipeline, predict_bundle


def load_experiment(run_dir: str, exp_id: str) -> dict:
    """A bundle-shaped dict for one experiment, assembled from what is on disk.

    The same four things `load_bundle` returns, sourced per-experiment: the model
    fitted on all work rows, the feature list it was trained with, the task type,
    and the replayable lineage of its source table. `predict_bundle` cannot tell
    the difference, so the prediction path is the real one rather than a
    reimplementation of it.
    """
    exp_dir = os.path.join(run_dir, "experiments", exp_id)
    with open(os.path.join(exp_dir, "meta.json")) as f:
        meta = json.load(f)

    from tabular_agent.harness import journal

    j = journal.load(run_dir)
    row = next((r for r in j.get("leaderboard") or [] if r["id"] == exp_id), None)
    if row is None:
        raise SystemExit(f"{exp_id} is not on {run_dir}'s leaderboard")

    source = row.get("source_table") or ""
    if source.startswith("tables/"):
        lineage_path = os.path.join(run_dir, source.replace(".parquet", ".pipeline.json"))
        with open(lineage_path) as f:
            lineage = json.load(f)
    else:
        # Trained on the work table itself, so there is nothing to replay: the
        # features are the raw columns.
        lineage = {"task_type": meta["task_type"], "steps": []}

    return {
        "pipeline": Pipeline.from_dict(lineage),
        "models": [joblib.load(os.path.join(exp_dir, "final_model.joblib"))],
        "weights": np.array([1.0]),
        "feature_names": meta["feature_names"],
        "task_type": meta["task_type"],
        "target_transform": meta.get("target_transform", "none"),
        "_meta": meta,
        "_source": source or "data/work.parquet",
    }


def verify(bundle: dict, run_dir: str, target: str) -> float:
    """Replay on the run's own validation half and insist on the journalled score.

    The check that makes the output trustworthy. A wrong lineage still produces a
    plausible-looking CSV — the columns exist, the numbers are in range — and
    would be indistinguishable from a real submission until Kaggle scored it.
    """
    val = pd.read_parquet(os.path.join(run_dir, "data", "val.parquet"))
    y = val[target]
    preds = predict_bundle(bundle, val.drop(columns=[target]))
    meta = bundle["_meta"]
    got = compute_metric(meta["metric"], y, preds)
    want = meta["val_score"]
    if abs(got - want) > 1e-6:
        raise SystemExit(
            f"reconstruction does not reproduce the journal: {meta['metric']} "
            f"{got:.6f} replayed vs {want:.6f} recorded — the lineage is wrong"
        )
    return got


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("experiment")
    p.add_argument("--test", required=True, help="the competition's test file")
    p.add_argument("--id", required=True, help="id column, kept out of the features")
    p.add_argument("--target", required=True, help="name of the target column")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    bundle = load_experiment(a.run_dir, a.experiment)
    meta = bundle["_meta"]
    print(f"{a.experiment}: {meta['model_type']} on {bundle['_source']}, "
          f"{len(bundle['feature_names'])} features, "
          f"transform {bundle['target_transform']}")
    print(f"  journal: {meta['metric']} val {meta['val_score']:.5f} "
          f"cv {meta['cv_score']:.5f}")

    got = verify(bundle, a.run_dir, a.target)
    print(f"  replayed val {got:.5f} — matches")

    test = pd.read_csv(a.test)
    ids = test[a.id]
    # The id is dropped only if the model was not trained with it. Some runs keep
    # it as a feature; that is their business and the replay has to match.
    features = test if a.id in bundle["feature_names"] else test.drop(columns=[a.id])
    preds = predict_bundle(bundle, features)

    bad = int((~np.isfinite(preds)).sum())
    if bad:
        print(f"  !! {bad} non-finite predictions")
    if meta["task_type"] == "binary":
        outside = int(((preds < 0) | (preds > 1)).sum())
        if outside:
            print(f"  !! {outside} predictions outside [0,1] from a classifier")
    elif (preds < 0).any():
        # Reported, not clipped: flooring here would hide a defect in the fit.
        print(f"  note: {int((preds < 0).sum())} predictions below zero, left as they are")

    out = a.out or os.path.join(a.run_dir, f"submission_{a.experiment}.csv")
    pd.DataFrame({a.id: ids, a.target: preds}).to_csv(out, index=False)
    print(f"  wrote {out}  ({len(preds)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
