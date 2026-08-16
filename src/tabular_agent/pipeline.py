"""Serializable feature pipeline + predictor bundle. predict replays stored
parameters and never needs the target column."""

import json
import os

import joblib
import numpy as np
import pandas as pd

from .blend import apply_blend
from .encoders import ENCODER_CLASSES
from .models import predict_model
from .target import invert_target


class Pipeline:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.steps: list[dict] = []

    def add_encoder(self, encoder) -> None:
        self.steps.append({"type": "encode", "encoder": encoder.to_dict()})

    def add_transform(self, code: str, func_name: str = "transform_features") -> None:
        """Persist a stateless df->df sandbox transform (§5.1) as an ordered step.

        Only the code is stored; it is re-run in the sandbox at ``transform``
        time (i.e. at predict), never the produced columns — that is what keeps
        it reproducible and train/serve-skew-free."""
        self.steps.append({"type": "transform", "code": code, "func_name": func_name})

    def add_filter(self, code: str, func_name: str = "keep_rows") -> None:
        """Record a row filter — the one step that is applied at fit and never replayed.

        Kept in the lineage rather than left out of it entirely, because the
        lineage is also the *record* of how a table was built and a table with
        fewer rows than its parent has to say why. It carries ``train_only`` so
        :meth:`transform` knows to skip it; see :mod:`tabular_agent.rows` for what
        breaks if it does not.
        """
        self.steps.append(
            {"type": "filter", "code": code, "func_name": func_name, "train_only": True}
        )

    def stream_requirements(self) -> dict[str, dict]:
        """``{stream name: what its file must contain}`` (0062).

        A stream aggregate stores the *recipe* and recomputes from the file, so
        the file has to come back at predict time. Read off the steps rather than
        tracked separately: a requirement kept in a second place is a requirement
        that can disagree with what the steps actually need.

        The columns are named because whoever calls ``predict_bundle`` a year
        later has the file and no idea which of its timestamps this bundle read as
        the observation time — and reading the wrong one builds the feature out of
        the future without erroring.
        """
        needed: dict[str, dict] = {}
        for step in self.steps:
            if step["type"] != "encode":
                continue
            enc = step["encoder"]
            if enc["kind"] != "stream_aggregate":
                continue
            needed.setdefault(enc["stream"], {
                "event_time": enc["event_time"],
                "observed_at": enc["observed_at"],
                "columns": [],
            })
            for col in [*enc["by"], enc["value"]]:
                if col not in needed[enc["stream"]]["columns"]:
                    needed[enc["stream"]]["columns"].append(col)
        return needed

    def required_streams(self) -> list[str]:
        """Named second inputs this pipeline cannot be replayed without."""
        return sorted(self.stream_requirements())

    def transform(self, df: pd.DataFrame, streams: dict | None = None) -> pd.DataFrame:
        """Replay every step. ``streams`` supplies the second inputs by name.

        Default ``None``, so every pipeline built before streams existed replays
        exactly as it did. A pipeline that *needs* one and is handed none fails
        loudly rather than producing the column from nothing — see
        :class:`~tabular_agent.encoders.StreamAggregateEncoder`, where the refusal
        lives, because only the encoder knows which stream it wanted.
        """
        streams = streams or {}
        out = df
        for step in self.steps:
            # Train-only steps are history, not instructions. Replaying a row
            # filter here would drop validation rows out from under the scorer and
            # refuse predictions to real serving rows — the two callers of this
            # method are exactly `_score_on_val` and `predict_bundle`.
            if step.get("train_only"):
                continue
            if step["type"] == "encode":
                enc = ENCODER_CLASSES[step["encoder"]["kind"]].from_dict(step["encoder"])
                # Only the encoders that declare a stream are handed one, so the
                # other four keep their one-argument signature and nothing about
                # them had to change to make room for this.
                if step["encoder"]["kind"] == "stream_aggregate":
                    out = enc.transform(out, streams.get(step["encoder"]["stream"]))
                else:
                    out = enc.transform(out)
            elif step["type"] == "transform":
                # Imported lazily so predict does not require the sandbox unless
                # a code step is actually present.
                from .sandbox import run_transform
                out, _ = run_transform(out, step["code"], func_name=step["func_name"])
            else:
                raise ValueError(f"unknown step type: {step['type']}")
        return out

    def to_dict(self) -> dict:
        return {"task_type": self.task_type, "steps": self.steps}

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        p = cls(d["task_type"])
        p.steps = d["steps"]
        return p


def save_bundle(run_ctx, pipeline: Pipeline, models: list, weights, feature_names, task_type,
                target_transform: str = "none") -> str:
    bundle_dir = run_ctx.path("bundle", "_")
    bundle_dir = os.path.dirname(bundle_dir)
    model_paths = []
    for i, model in enumerate(models):
        mp = os.path.join(bundle_dir, f"model_{i}.joblib")
        joblib.dump(model, mp)
        model_paths.append(f"model_{i}.joblib")
    manifest = {
        "pipeline": pipeline.to_dict(),
        "model_paths": model_paths,
        "weights": np.asarray(weights, dtype=float).tolist(),
        "feature_names": list(feature_names),
        "task_type": task_type,
        # The models were fitted on f(y); without this the bundle would predict
        # in log-dollars and look plausible doing it.
        "target_transform": target_transform,
    }
    # Derived from the steps, never passed in. A bundle that needs a second input
    # has to say so where somebody will read it — the alternative is a file that
    # predicts only when the caller happens to know, and a `streams=` argument
    # nobody knew to pass is indistinguishable from one that was not needed.
    requirements = pipeline.stream_requirements()
    if requirements:
        manifest["streams"] = requirements
    with open(os.path.join(bundle_dir, "pipeline.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return bundle_dir


def load_bundle(bundle_dir: str) -> dict:
    with open(os.path.join(bundle_dir, "pipeline.json")) as f:
        manifest = json.load(f)
    models = [joblib.load(os.path.join(bundle_dir, mp)) for mp in manifest["model_paths"]]
    return {
        "pipeline": Pipeline.from_dict(manifest["pipeline"]),
        "models": models,
        "weights": np.asarray(manifest["weights"], dtype=float),
        "feature_names": manifest["feature_names"],
        "task_type": manifest["task_type"],
        # Bundles written before target transforms existed have no such key and
        # must keep predicting exactly as they did.
        "target_transform": manifest.get("target_transform", "none"),
        # {name: the columns that file must carry}. Absent on every bundle
        # written before streams existed, which is what "needs nothing extra"
        # looks like.
        "streams": manifest.get("streams", {}),
    }


def predict_bundle(bundle: dict, raw_df: pd.DataFrame,
                   streams: dict | None = None) -> np.ndarray:
    """Replay the lineage, predict with every model in the bundle, combine.

    ``streams`` supplies the second inputs a stream aggregate needs, by name
    (0062). It is the caller's job to hand in the file **as it stands now** rather
    than as it stood at fit time — that is the entire reason the encoder stores a
    recipe instead of the numbers, and a bundle that shipped its own copy would be
    frozen the moment it was written. A bundle that requires one and is given none
    is refused here, with the columns its file has to carry.

    **Each model's target transform is inverted before the weights are applied**,
    so the combination is an average of predicted *targets*. The other order —
    combine on the model's scale, invert once — is a different estimator whenever
    the transform is non-linear: under ``log1p`` it is a geometric mean of sales
    rather than an arithmetic one, and the two disagree by more the more the
    models disagree.

    This order is the one that has been measured. 0044's blend was scored on
    Rossmann v7's sealed rows and on Kaggle's private set with the inversion
    first; combining on the log scale has never been scored against anything. For
    a single-model bundle the two are identical, which is every bundle this
    library wrote before 0044.
    """
    streams = streams or {}
    # Checked here rather than left to the encoder, which knows only its own name.
    # One message naming everything that is missing, and what each file must hold,
    # beats discovering them one raised exception at a time.
    requirements = bundle["pipeline"].stream_requirements()
    absent = [name for name in sorted(requirements) if name not in streams]
    if absent:
        needs = "; ".join(
            f"{name}: event time {requirements[name]['event_time']!r}, observation "
            f"time {requirements[name]['observed_at']!r}, plus "
            f"{requirements[name]['columns']}"
            for name in absent
        )
        raise ValueError(
            f"this bundle was built on stream(s) {absent} and cannot predict without "
            f"them: pass streams={{name: frame}}. Each needs — {needs}. Hand in the "
            "file as it stands now, not a copy from training time: the whole point of "
            "storing the recipe is that the records which have arrived since are the "
            "ones a served row should be reading."
        )
    features = bundle["pipeline"].transform(raw_df, streams=streams)
    X = features[bundle["feature_names"]]
    task_type = bundle["task_type"]
    transform = bundle.get("target_transform", "none")
    preds = []
    for m in bundle["models"]:
        p = predict_model(m, X, task_type)
        if transform != "none":
            p = invert_target(transform, p)
        preds.append(p)
    return apply_blend(preds, bundle["weights"])
