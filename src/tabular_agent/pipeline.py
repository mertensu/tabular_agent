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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
    }


def predict_bundle(bundle: dict, raw_df: pd.DataFrame) -> np.ndarray:
    features = bundle["pipeline"].transform(raw_df)
    X = features[bundle["feature_names"]]
    task_type = bundle["task_type"]
    preds = [predict_model(m, X, task_type) for m in bundle["models"]]
    blended = apply_blend(preds, bundle["weights"])
    # Inverted after blending, not before: the models all share one target scale,
    # so averaging on that scale and inverting once is the same operation the
    # cross-validated scores were computed with.
    transform = bundle.get("target_transform", "none")
    if transform != "none":
        blended = invert_target(transform, blended)
    return blended
