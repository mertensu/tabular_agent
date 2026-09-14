"""Safe, serializable descriptions of the feature pipeline a bundle ships."""

from __future__ import annotations

import json

import pandas as pd

from .encoders import ENCODER_CLASSES


def _encoder_description(stored: dict) -> tuple[list[str], str, str]:
    kind = stored["kind"]
    if kind == "target":
        sources = list(stored["columns"])
        recipe = {
            "columns": sources,
            "stats": stored.get("stats", ["mean"]),
            "smoothing": stored["smoothing"],
        }
        return (
            sources,
            "Out-of-fold target statistics by category; the saved lookup is used on new rows.",
            json.dumps(recipe, sort_keys=True),
        )
    if kind == "frequency":
        sources = list(stored["columns"])
        unit = "share" if stored["normalize"] else "count"
        return (
            sources,
            f"Training-set category {unit}; unseen values become zero.",
            f"{unit} by {sources}",
        )
    if kind == "ordinal":
        sources = list(stored["categories"])
        recipe = {"categories": stored["categories"], "unknown": stored["unknown"]}
        return (
            sources,
            "Declared category order converted to numeric ranks.",
            json.dumps(recipe, sort_keys=True),
        )
    if kind == "group_stats":
        sources = [*stored["by"], stored["value"]]
        recipe = {key: stored[key] for key in ("by", "value", "stat")}
        return (
            sources,
            "Out-of-fold group statistic fitted on training rows and looked up on new rows.",
            json.dumps(recipe, sort_keys=True),
        )
    if kind == "history":
        sources = list(dict.fromkeys([*stored["by"], stored["value"], stored["order"]]))
        recipe = {
            key: stored.get(key)
            for key in ("by", "value", "order", "offsets", "windows", "latest", "keep")
        }
        return (
            sources,
            "Past-only group history with stored state and explicit offsets/windows.",
            json.dumps(recipe, sort_keys=True),
        )
    if kind == "stream_aggregate":
        sources = list(dict.fromkeys([
            *stored["by"], stored["value"], stored["event_time"],
            stored["observed_at"], stored["order"],
        ]))
        recipe = {
            key: stored.get(key)
            for key in (
                "stream", "by", "value", "event_time", "observed_at", "order",
                "leads", "stats", "event_grain", "grain", "origin",
            )
        }
        return (
            sources,
            "Observation-time-guarded aggregate from a named event stream.",
            json.dumps(recipe, sort_keys=True),
        )
    return [], f"Learned {kind!r} encoder output.", json.dumps({"kind": kind})


def describe_feature_rows(
    pipeline,
    feature_names: list[str],
    *,
    input_columns: list[str] | None = None,
) -> list[dict]:
    """Describe ordered features without exposing fitted lookup mappings."""
    provenance: dict[str, dict] = {}
    transform_steps = []
    for step_number, step in enumerate(pipeline.steps, start=1):
        if step["type"] == "transform":
            transform_steps.append((step_number, step))
            for output in step.get("new_columns", []):
                provenance[output] = {
                    "kind": "row_transform",
                    "source_columns": [],
                    "pipeline_step": step_number,
                    "description": "Created by agent-written row-local transform code.",
                    "definition": step["code"],
                }
            continue
        if step["type"] != "encode":
            continue
        stored = step["encoder"]
        if stored["kind"] == "frequency":
            outputs = [f"{column}_freq" for column in stored["columns"]]
        elif stored["kind"] == "ordinal":
            outputs = [f"{column}_ord" for column in stored["categories"]]
        else:
            encoder = ENCODER_CLASSES[stored["kind"]].from_dict(stored)
            outputs = getattr(encoder, "output_columns", None)
            if outputs is None:
                outputs = [encoder.output_column]
            elif callable(outputs):
                outputs = outputs()
        source_columns, description, definition = _encoder_description(stored)
        for output in outputs:
            provenance[str(output)] = {
                "kind": stored["kind"],
                "source_columns": source_columns,
                "pipeline_step": step_number,
                "description": description,
                "definition": definition,
            }

    inputs = set(input_columns or [])
    rows = []
    for feature in feature_names:
        info = provenance.get(feature)
        if info is None and feature in inputs:
            info = {
                "kind": "input",
                "source_columns": [feature],
                "pipeline_step": 0,
                "description": "Numeric input column used directly by the models.",
                "definition": feature,
            }
        elif info is None and transform_steps:
            info = {
                "kind": "row_transform",
                "source_columns": [],
                "pipeline_step": None,
                "description": (
                    "Derived by stored row-local transform code; this legacy bundle "
                    "did not record which transform introduced it."
                ),
                "definition": "\n\n".join(step["code"] for _, step in transform_steps),
            }
        elif info is None:
            info = {
                "kind": "input_or_legacy",
                "source_columns": [],
                "pipeline_step": None,
                "description": "Shipped feature; detailed lineage is unavailable.",
                "definition": "",
            }
        rows.append({"feature": feature, **info})
    return rows


def describe_features_frame(
    pipeline,
    feature_names: list[str],
    *,
    input_columns: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        describe_feature_rows(pipeline, feature_names, input_columns=input_columns),
        columns=[
            "feature", "kind", "source_columns", "pipeline_step",
            "description", "definition",
        ],
    )
