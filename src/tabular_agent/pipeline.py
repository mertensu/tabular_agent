"""Serializable feature pipeline + predictor bundle. predict replays stored
parameters and never needs the target column."""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from .blend import apply_blend
from .encoders import ENCODER_CLASSES
from .models import predict_model
from .target import invert_reference_target, invert_target, normalize_reference_target


class Pipeline:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.steps: list[dict] = []

    def add_encoder(self, encoder) -> None:
        self.steps.append({"type": "encode", "encoder": encoder.to_dict()})

    def add_transform(
        self,
        code: str,
        func_name: str = "transform_features",
        new_columns: list[str] | None = None,
    ) -> None:
        """Persist a stateless df->df sandbox transform (§5.1) as an ordered step.

        The code is re-run in the sandbox at ``transform`` time (i.e. at
        predict). Output names may be stored as explanation metadata, but their
        values are never stored or trusted — that is what keeps replay
        reproducible and train/serve-skew-free."""
        step = {"type": "transform", "code": code, "func_name": func_name}
        if new_columns is not None:
            # Explanation metadata only. Replay still executes the stored code;
            # trusting remembered output would reintroduce train/serve skew.
            step["new_columns"] = list(new_columns)
        self.steps.append(step)

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


def input_dtypes(df: pd.DataFrame, exclude=()) -> dict[str, str]:
    """``{column: dtype name}`` of the rows a lineage was built from."""
    return {c: str(t) for c, t in df.dtypes.items() if c not in set(exclude)}


def conform_input_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Give serving columns the numeric type they had in training, when no value changes.

    A lineage replays code written against the training rows, and that code can
    depend on a column's type as well as its values: ``astype(str)`` on a key
    makes ``"1"`` in training and ``"1.0"`` at serving, and an encoder then reads
    a key it never saw. A serving file stores the same values with a different
    type easily: one missing value turns a column of whole numbers into floats.

    Only lossless numeric conversions are made: floats holding whole numbers back
    to integers (a nullable integer when values are missing), integers or
    booleans to floats, and 0/1 back to booleans. Anything else is left for the
    lineage to handle or refuse as it would have. Returns the frame and one line
    per converted column.
    """
    out, notes = df, []
    for col, want in dtypes.items():
        if col not in df.columns or str(df[col].dtype) == want:
            continue
        have = df[col]
        try:
            target = np.dtype(want)
        except TypeError:
            continue
        if not pd.api.types.is_numeric_dtype(have) and not pd.api.types.is_bool_dtype(have):
            continue
        values = have.dropna()
        if target.kind in "iu":
            if not np.all(np.mod(values.astype(float), 1) == 0):
                continue
            new = have.astype(want if len(values) == len(have) else
                              ("Int64" if target.kind == "i" else "UInt64"))
        elif target.kind == "f":
            new = have.astype(want)
        elif target.kind == "b":
            if len(values) != len(have) or not values.isin([0, 1]).all():
                continue
            new = have.astype(bool)
        else:
            continue
        if out is df:
            out = df.copy()
        out[col] = new
        notes.append(f"{col}: {have.dtype} -> {new.dtype}")
    return out, notes


def save_bundle(run_ctx, pipeline: Pipeline, models: list, weights, feature_names, task_type,
                target_transform: str = "none",
                reference_target=None,
                training_table: str | None = None,
                objective: str | None = None,
                fitted_on: dict | None = None,
                input_dtypes: dict[str, str] | None = None) -> str:
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
        # Fit used reference-forward and then f(y); prediction must reverse both.
        "target_transform": target_transform,
        "reference_target": normalize_reference_target(reference_target),
        # The fitted models already carry their loss, so prediction needs nothing
        # from this. It is recorded for `explain`, whose attribution space it sets.
        "objective": objective,
    }
    # Retain the exact fitted table so Run.training_features can return its OOF
    # values instead of leaking through final target-informed encoders.
    if training_table is not None:
        manifest["training_table"] = training_table
    # Which rows the models and the lineage's fitted state came from. A run ranks
    # on work-only fits and ships a refit on work + validation; whoever loads the
    # bundle later should not have to read the journal to know which they hold.
    if fitted_on is not None:
        manifest["fitted_on"] = dict(fitted_on)
    # The type of every input column the lineage was written against, so a
    # serving file storing the same values differently is put back first.
    if input_dtypes is not None:
        manifest["input_dtypes"] = dict(input_dtypes)
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
        # Missing fields are the direct-target defaults for old bundles.
        "target_transform": manifest.get("target_transform", "none"),
        "reference_target": manifest.get("reference_target"),
        # Absent on bundles written before objectives existed: squared error.
        "objective": manifest.get("objective"),
        # {name: the columns that file must carry}. Absent on every bundle
        # written before streams existed, which is what "needs nothing extra"
        # looks like.
        "streams": manifest.get("streams", {}),
        # Optional: bundles written before reusable features remain valid.
        "training_table": manifest.get("training_table"),
        # Absent on bundles written before the refit: fitted on work rows only.
        "fitted_on": manifest.get("fitted_on"),
        # Absent on bundles written before it was recorded: rows replay as given.
        "input_dtypes": manifest.get("input_dtypes") or {},
    }


def feature_table_bundle(bundle: dict, raw_df: pd.DataFrame,
                         streams: dict | None = None) -> pd.DataFrame:
    """Replay lineage and return the exact matrix handed to every shipped model.

    This is the reusable half of :func:`predict_bundle`: a custom estimator
    receives the same ordered columns and fitted encoder values on new rows as
    the bundle's models. It deliberately stops before model-specific
    preprocessing (for example an estimator's imputer or scaler), because those
    parameters belong to that model rather than to the learned feature pipeline.

    Stream requirements are checked here so feature export and prediction fail
    in the same way when a time-varying second input is absent.
    """
    streams = streams or {}
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
            f"them or build features: pass streams={{name: frame}}. Each needs "
            f"— {needs}. Hand in the file as it stands now, not a copy from training "
            "time: the whole point of storing the recipe is that the records which "
            "have arrived since are the ones a served row should be reading."
        )
    raw_df, converted = conform_input_dtypes(raw_df, bundle.get("input_dtypes") or {})
    if converted:
        warnings.warn(
            "serving columns converted to their training type before replay: "
            + "; ".join(converted),
            stacklevel=2,
        )
    features = bundle["pipeline"].transform(raw_df, streams=streams)
    missing = [name for name in bundle["feature_names"] if name not in features]
    if missing:
        raise ValueError(
            "the learned feature pipeline did not produce the model columns "
            f"{missing}; the bundle cannot be replayed"
        )
    X = features.loc[:, bundle["feature_names"]].copy()
    # A nullable integer restored above reaches the models as float with NaN,
    # the form a missing number takes everywhere else in the matrix.
    for col in X.columns:
        if isinstance(X[col].dtype, pd.api.extensions.ExtensionDtype) and \
                pd.api.types.is_numeric_dtype(X[col].dtype):
            X[col] = X[col].astype("float64")
    return X


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
    X = feature_table_bundle(bundle, raw_df, streams=streams)
    task_type = bundle["task_type"]
    transform = bundle.get("target_transform", "none")
    reference_target = bundle.get("reference_target")
    preds = []
    for m in bundle["models"]:
        p = predict_model(m, X, task_type)
        if transform != "none":
            p = invert_target(transform, p)
        p = invert_reference_target(p, X, reference_target)
        preds.append(p)
    return apply_blend(preds, bundle["weights"])
