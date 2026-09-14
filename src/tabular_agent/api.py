"""Public, safety-first facade over the existing tabular-agent harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import pandas as pd

from .encoders import ENCODER_CLASSES
from .harness.agent import gateway_model, run as run_agent
from .harness.session import RunSession
from .harness.tools import finalize_run
from .inspect import (
    profile_collection,
    profile_dataframe,
    propose_identifier_columns,
    table_paths,
)
from .explain import Explanation, explain_bundle
from .io import load_table, save_parquet
from .pipeline import feature_table_bundle, load_bundle, predict_bundle
from .taskspec import TaskSpec


@dataclass
class JoinProposal:
    """A static many-to-one join which remains inert until explicitly accepted."""

    right_table: str
    left_on: str
    right_on: str
    columns: list[str]
    decision: str = "proposed"

    def accept(self) -> JoinProposal:
        self.decision = "accepted"
        return self

    def reject(self) -> JoinProposal:
        self.decision = "rejected"
        return self


@dataclass
class DataPlan:
    """Editable inspection result. Proposed facts are not modeling decisions."""

    source: str | None
    tables: dict[str, dict]
    training_table: str
    serving_table: str | None
    target_candidates: list[str]
    ordering_candidates: list[str]
    training_only_columns: list[str]
    proposed_serving_columns: list[str] | None
    proposed_identifier_columns: list[str] = field(default_factory=list)
    joins: list[JoinProposal] = field(default_factory=list)
    target: str | None = None
    serving_columns: list[str] | None = None
    identifier_columns: list[str] | None = field(default_factory=list)
    split_on: str | bool | None = None
    task_type: str | None = None
    metric: str | None = None
    notes: str | None = None
    column_notes: dict[str, str] | None = None
    n_splits: int = 5
    seed: int = 42
    target_transform: str | None = None
    forecast_horizon: float | None = None
    _frame: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def from_profile(cls, profile: dict) -> DataPlan:
        return cls(
            source=profile["source"],
            tables=profile["tables"],
            training_table=profile["training_table"],
            serving_table=profile["serving_table"],
            target_candidates=profile["target_candidates"],
            ordering_candidates=profile["ordering_candidates"],
            training_only_columns=profile["training_only_columns"],
            proposed_serving_columns=profile["serving_columns"],
            proposed_identifier_columns=list(profile.get("identifier_columns") or []),
            # With no proposal there is no fifth decision to make. When a
            # plausible identifier exists, keep the role unconfirmed just like
            # the serving schema: inspection may propose, only the owner decides.
            identifier_columns=([] if not profile.get("identifier_columns") else None),
            joins=[JoinProposal(**join) for join in profile["joins"]],
        )

    def accept_identifier_proposals(self) -> DataPlan:
        self.identifier_columns = list(self.proposed_identifier_columns)
        return self

    def reject_identifier_proposals(self) -> DataPlan:
        self.identifier_columns = []
        return self

    def accept_joins(self) -> DataPlan:
        for join in self.joins:
            join.accept()
        return self

    def reject_joins(self) -> DataPlan:
        for join in self.joins:
            join.reject()
        return self

    def summary(self) -> str:
        """A readable checklist of the owner decisions and what is still open.

        Display only — it changes nothing and enforces nothing. It exists so the
        confirmation step reads as a guided list rather than five attributes you
        have to remember: for each decision it shows whether it is confirmed, the
        value or the proposal, and the one call that sets it. :meth:`validate` (run
        by :meth:`Agent.run`) remains the actual gate and refuses on anything still
        open, so this can never let an unconfirmed run start — it only makes the
        open items easy to see first.
        """

        def _preview(items: list, limit: int = 8) -> str:
            items = [str(item) for item in items]
            if not items:
                return "(none)"
            shown = ", ".join(items[:limit])
            return shown if len(items) <= limit else f"{shown}, … (+{len(items) - limit} more)"

        lines: list[str] = []
        open_count = 0

        def _row(confirmed: bool, name: str, value: str, *hints: str) -> None:
            nonlocal open_count
            if not confirmed:
                open_count += 1
            mark = "[x]" if confirmed else "[ ]"
            lines.append(f"  {mark} {name:<18} {value}")
            for hint in hints:
                lines.append(f"                         {hint}")

        _row(
            bool(self.target),
            "target",
            str(self.target) if self.target else "not set",
            *(() if self.target else (f"candidates: {_preview(self.target_candidates)}",
                                      "set: plan.target = '<column>'")),
        )
        _row(
            self.serving_columns is not None,
            "serving_columns",
            _preview(self.serving_columns) if self.serving_columns is not None else "not set",
            *(() if self.serving_columns is not None else (
                f"present at predict time: {_preview(self.proposed_serving_columns or [])}",
                "set: plan.serving_columns = [...]")),
        )
        _row(
            self.split_on is not None,
            "split_on",
            ("random split (False)" if self.split_on is False else str(self.split_on))
            if self.split_on is not None else "not set",
            *(() if self.split_on is not None else (
                f"date/time candidates: {_preview(self.ordering_candidates)}",
                "set: plan.split_on = '<column>' (ordered) or False (random)")),
        )
        _row(
            self.identifier_columns is not None,
            "identifier_columns",
            _preview(self.identifier_columns) if self.identifier_columns is not None else "not set",
            *(() if self.identifier_columns is not None else (
                f"proposed: {_preview(self.proposed_identifier_columns)}",
                "set: plan.accept_identifier_proposals() or .reject_identifier_proposals()")),
        )
        pending = [join.right_table for join in self.joins if join.decision == "proposed"]
        if not self.joins:
            joins_value = "(none proposed)"
        else:
            decided = [f"{join.right_table}:{join.decision}" for join in self.joins]
            joins_value = _preview(decided)
        _row(
            not pending,
            "joins",
            joins_value,
            *(() if not pending else (
                f"undecided: {_preview(pending)}",
                "set: plan.accept_joins() or .reject_joins()")),
        )

        header = (
            "DataPlan — all decisions confirmed."
            if open_count == 0
            else f"DataPlan — {open_count} decision(s) still to confirm."
        )
        footer = (
            "Pass it to agent.run(plan)."
            if open_count == 0
            else "Confirm the [ ] items, then agent.run(plan) (it refuses on anything left)."
        )
        return "\n".join([header, "", *lines, "", footer])

    def validate(self) -> None:
        train_columns = set(self.tables[self.training_table]["columns"])
        accepted_columns = {
            column
            for join in self.joins
            if join.decision == "accepted"
            for column in join.columns
        }
        available = train_columns | accepted_columns
        if not self.target:
            raise ValueError(
                "target is unconfirmed; choose one of target_candidates or set it explicitly"
            )
        if self.target not in available:
            raise ValueError(f"target {self.target!r} is not in the prepared training table")
        if self.serving_columns is None:
            raise ValueError(
                "serving_columns are unconfirmed; state exactly what exists at prediction time"
            )
        if self.identifier_columns is None:
            raise ValueError(
                "identifier_columns are unconfirmed; accept or reject the proposed "
                f"row identifiers {self.proposed_identifier_columns}"
            )
        unknown_identifiers = [
            column for column in self.identifier_columns if column not in available
        ]
        if unknown_identifiers:
            raise ValueError(f"identifier_columns are not available: {unknown_identifiers}")
        if self.target in self.identifier_columns:
            raise ValueError("the target cannot also be an identifier")
        if isinstance(self.split_on, str) and self.split_on in self.identifier_columns:
            raise ValueError(
                "split_on cannot also be an identifier; it is removed before modeling"
            )
        if self.target in self.serving_columns:
            raise ValueError("serving_columns must not contain the target")
        if self.split_on is None:
            raise ValueError(
                "split_on is unconfirmed; set a column for ordered splitting or False "
                "to explicitly choose a random split"
            )
        if isinstance(self.split_on, str) and self.split_on not in available:
            raise ValueError(f"split_on column {self.split_on!r} is not available")
        undecided = [join.right_table for join in self.joins if join.decision == "proposed"]
        if undecided:
            raise ValueError(
                f"join proposals are unconfirmed for {undecided}; accept or reject them"
            )
        missing = [column for column in self.serving_columns if column not in available]
        missing_join_keys = [
            join.left_on for join in self.joins
            if join.decision == "accepted" and join.left_on not in self.serving_columns
        ]
        if missing_join_keys:
            raise ValueError(f"accepted join keys are absent at serving: {missing_join_keys}")
        if missing:
            raise ValueError(f"serving_columns are not available after accepted joins: {missing}")

    def _training_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame.copy()
        return load_table(self.tables[self.training_table]["path"])

    def _apply_joins(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame
        for join in self.joins:
            if join.decision != "accepted":
                continue
            right = load_table(self.tables[join.right_table]["path"])
            keep = [join.right_on, *join.columns]
            before = len(result)
            result = result.merge(
                right[keep],
                how="left",
                left_on=join.left_on,
                right_on=join.right_on,
                validate="many_to_one",
            )
            if len(result) != before:
                raise ValueError(f"join with {join.right_table!r} changed row count")
            if join.right_on != join.left_on:
                result = result.drop(columns=[join.right_on])
        return result

    def materialize(self, run_dir: str | os.PathLike) -> tuple[str, dict]:
        self.validate()
        run_path = Path(run_dir)
        prepared = run_path / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        frame = self._apply_joins(self._training_frame())
        # Identifiers are transport metadata: accepted joins may need them first,
        # but the feature agent and every fitted encoder must never see them.
        frame = frame.drop(columns=list(self.identifier_columns or []), errors="raise")
        if isinstance(self.split_on, str):
            parsed = pd.to_datetime(frame[self.split_on], errors="coerce")
            if parsed.notna().all():
                frame[self.split_on] = parsed
        path = save_parquet(frame, str(prepared / "train.parquet"))
        manifest = {
            "training_table": self.training_table,
            "serving_table": self.serving_table,
            "target": self.target,
            "serving_columns": list(self.serving_columns or []),
            "identifier_columns": list(self.identifier_columns or []),
            "split_on": self.split_on,
            "joins": [asdict(join) for join in self.joins if join.decision == "accepted"],
        }
        with open(prepared / "api_plan.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        return path, manifest


def _describe_encoder(stored: dict) -> tuple[list[str], str, str]:
    """Summarise a serialized encoder without exposing its fitted lookup values."""
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


class Bundle:
    """A deployable predictor that accepts the same raw rows the facade inspected."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(Path(path).resolve())
        self._bundle = load_bundle(self.path)
        plan_path = Path(self.path) / "api_plan.json"
        with open(plan_path) as handle:
            self._plan = json.load(handle) if plan_path.exists() else {}

    @classmethod
    def load(cls, path: str | os.PathLike) -> Bundle:
        return cls(path)

    def save(self, destination: str | os.PathLike) -> str:
        destination = str(Path(destination).resolve())
        if os.path.exists(destination):
            raise ValueError(f"destination already exists: {destination}")
        shutil.copytree(self.path, destination)
        return destination

    def _raw_frame(self, rows: pd.DataFrame | str | os.PathLike) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            return rows.copy()
        path = Path(rows).expanduser().resolve()
        if path.is_dir():
            serving = self._plan.get("serving_table")
            if not serving:
                paths = table_paths(path)
                if len(paths) != 1:
                    raise ValueError("bundle has no serving-table name; pass a dataframe or file")
                path = paths[0]
            else:
                path = path / serving
        return load_table(str(path))

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame
        for join in self._plan.get("joins", []):
            missing = [column for column in join["columns"] if column not in result.columns]
            if not missing:
                continue
            right = load_table(str(Path(self.path) / "static" / join["right_table"]))
            keep = [join["right_on"], *missing]
            before = len(result)
            result = result.merge(
                right[keep],
                how="left",
                left_on=join["left_on"],
                right_on=join["right_on"],
                validate="many_to_one",
            )
            if len(result) != before:
                raise ValueError(f"bundle join with {join['right_table']!r} changed row count")
            if join["right_on"] != join["left_on"]:
                result = result.drop(columns=[join["right_on"]])
        split_on = self._plan.get("split_on")
        if isinstance(split_on, str) and split_on in result:
            parsed = pd.to_datetime(result[split_on], errors="coerce")
            if parsed.notna().all():
                result[split_on] = parsed
        hidden = [
            *list(self._plan.get("identifier_columns") or []),
            *([self._plan["target"]] if self._plan.get("target") else []),
        ]
        result = result.drop(columns=hidden, errors="ignore")
        return result

    @property
    def feature_names(self) -> list[str]:
        """The exact ordered columns consumed by the shipped models."""
        return list(self._bundle["feature_names"])

    def feature_table(
        self,
        rows: pd.DataFrame | str | os.PathLike,
        streams: dict | None = None,
        *,
        include_input: bool = False,
    ) -> pd.DataFrame:
        """Apply learned feature lineage to new rows without running the models.

        By default the result is the exact ordered matrix the bundle's models
        consume. ``include_input=True`` adds those model-ready columns to a copy
        of the raw table, preserving identifiers and a target column if the user
        supplied one. A raw column also used as a model feature is replaced by
        its replayed value, so selecting ``bundle.feature_names`` from the
        enriched table always recovers the exact matrix.

        The feature pipeline stops before estimator-specific preprocessing such
        as an individual model's imputer or scaler. Streams have the same
        requirements and semantics as :meth:`predict`.
        For fitting another estimator, use :meth:`Run.training_features` for
        the training side. Replaying a final target/group-statistics lookup over
        rows it was fitted on would not reproduce the out-of-fold values the
        agent trained on and can leak; this method is the serving/new-row half
        of that workflow.
        """

        raw = self._raw_frame(rows)
        matrix = feature_table_bundle(
            self._bundle, self._prepare(raw.copy()), streams=streams
        )
        if len(matrix) != len(raw):
            raise ValueError(
                "the learned feature pipeline changed the number of rows; "
                "features cannot be aligned to the input table"
            )
        matrix.index = raw.index
        if not include_input:
            return matrix
        enriched = raw.copy()
        for name in matrix.columns:
            # Assign by position: static joins may replace an arbitrary source
            # index with a RangeIndex while preserving row order and count.
            enriched[name] = matrix[name].to_numpy()
        return enriched

    def describe_features(self) -> pd.DataFrame:
        """Return one readable lineage row for every shipped model feature.

        ``definition`` is the exact stored recipe where one exists. New bundles
        record which columns each row-local transform introduced; legacy bundles
        still list every feature but may only identify it as a transform output.
        No fitted category-to-value mappings are dumped into this summary.
        """
        pipeline = self._bundle["pipeline"]
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
            source_columns, description, definition = _describe_encoder(stored)
            for output in outputs:
                provenance[str(output)] = {
                    "kind": stored["kind"],
                    "source_columns": source_columns,
                    "pipeline_step": step_number,
                    "description": description,
                    "definition": definition,
                }

        input_columns = list(self._plan.get("serving_columns") or [])
        for join in self._plan.get("joins", []):
            input_columns.extend(join.get("columns") or [])
        rows = []
        for feature in self.feature_names:
            info = provenance.get(feature)
            if info is None and feature in input_columns:
                info = {
                    "kind": "input",
                    "source_columns": [feature],
                    "pipeline_step": 0,
                    "description": "Numeric input column used directly by the models.",
                    "definition": feature,
                }
            elif info is None and transform_steps:
                # Old pipeline manifests stored executable transform code but
                # not the outputs it introduced. Stay useful without pretending
                # an exact per-step attribution can be recovered after the fact.
                info = {
                    "kind": "row_transform",
                    "source_columns": [],
                    "pipeline_step": pd.NA,
                    "description": (
                        "Derived by stored row-local transform code; this legacy "
                        "bundle did not record which transform introduced it."
                    ),
                    "definition": "\n\n".join(s["code"] for _, s in transform_steps),
                }
            elif info is None:
                info = {
                    "kind": "input_or_legacy",
                    "source_columns": [],
                    "pipeline_step": pd.NA,
                    "description": (
                        "Shipped feature; detailed lineage is absent in this legacy bundle."
                    ),
                    "definition": "",
                }
            rows.append({"feature": feature, **info})
        return pd.DataFrame(
            rows,
            columns=[
                "feature", "kind", "source_columns", "pipeline_step",
                "description", "definition",
            ],
        )

    def predict(
        self,
        rows: pd.DataFrame | str | os.PathLike,
        streams: dict | None = None,
    ):
        return predict_bundle(self._bundle, self._prepare(self._raw_frame(rows)), streams=streams)

    def explain(
        self,
        rows: pd.DataFrame | str | os.PathLike,
        streams: dict | None = None,
    ) -> Explanation:
        """Per-feature attribution of the shipped prediction for each row.

        Takes the same raw rows as :meth:`predict` and prepares them identically
        (joins, identifier drop, split-column typing); see
        :func:`~tabular_agent.explain.explain_bundle` for the additivity and
        scale guarantees.
        """
        return explain_bundle(
            self._bundle, self._prepare(self._raw_frame(rows)), streams=streams
        )


class Run:
    """A finalized run, its report, journal and deployable bundle."""

    def __init__(self, run_dir: str | os.PathLike):
        self.run_dir = str(Path(run_dir).resolve())
        self.bundle = Bundle(Path(self.run_dir) / "bundle")

    @property
    def journal(self) -> dict:
        with open(Path(self.run_dir) / "run_journal.json") as handle:
            return json.load(handle)

    def report(self) -> str:
        with open(Path(self.run_dir) / "run_report.md") as handle:
            return handle.read()

    def training_features(self, *, include_target: bool = True) -> pd.DataFrame:
        """Return the exact leakage-safe table used to fit the shipped models.

        Unlike replaying the final bundle over its own training rows, this reads
        the selected run table, preserving out-of-fold values for target and
        group-statistics encoders. The result contains model columns in their
        exact order and, by default, the target as the last column.

        This table is the training half of a custom-model workflow;
        ``run.bundle.feature_table(new_rows)`` is its serving half.
        """
        source = self.bundle._bundle.get("training_table")
        if not source:
            raise ValueError(
                "this run predates reusable training features; its bundle can still "
                "build features for new rows, but the exact out-of-fold training "
                "table was not recorded"
            )
        root = Path(self.run_dir).resolve()
        path = (root / source).resolve()
        if root not in path.parents:
            raise ValueError("bundle training-table reference leaves the run directory")
        table = load_table(str(path))
        columns = self.bundle.feature_names
        missing = [column for column in columns if column not in table]
        if missing:
            raise ValueError(f"recorded training table is missing shipped features {missing}")
        if include_target:
            target = self.bundle._plan.get("target") or (
                self.journal.get("task") or {}
            ).get("target")
            if not target or target not in table:
                raise ValueError("recorded training table does not contain the run target")
            columns = [*columns, target]
        return table.loc[:, columns].copy()


class Agent:
    """Thin public facade; every safety decision remains enforced by RunSession."""

    def __init__(
        self,
        model: Any = "anthropic:claude-opus-5",
        *,
        profile: str = "minimal",
        run_root: str | os.PathLike = "runs",
        val_fraction: float = 0.2,
        holdout_fraction: float | None = None,
        fit_time_budget_s: float | None = None,
    ):
        self.model = model
        self.profile = profile
        self.run_root = Path(run_root).expanduser().resolve()
        self.val_fraction = val_fraction
        self.holdout_fraction = holdout_fraction
        self.fit_time_budget_s = fit_time_budget_s

    def inspect(self, source: pd.DataFrame | str | os.PathLike) -> DataPlan:
        if isinstance(source, pd.DataFrame):
            profile = {
                "source": None,
                "tables": {"dataframe": profile_dataframe(source)},
                "training_table": "dataframe",
                "serving_table": None,
                "target_candidates": sorted(
                    list(source.columns),
                    key=lambda column: (
                        str(column).lower() not in {"target", "label", "response", "outcome", "y"},
                        str(column).lower(),
                    ),
                ),
                "ordering_candidates": [
                    column for column in source.columns
                    if pd.api.types.is_datetime64_any_dtype(source[column])
                    or any(token in str(column).lower() for token in ("date", "time", "timestamp"))
                ],
                "serving_columns": None,
                "identifier_columns": propose_identifier_columns(source),
                "joins": [],
                "training_only_columns": list(source.columns),
            }
            plan = DataPlan.from_profile(profile)
            plan._frame = source.copy()
            return plan
        return DataPlan.from_profile(profile_collection(source))

    def _resolved_model(self):
        if isinstance(self.model, str) and "/" in self.model and ":" not in self.model:
            return gateway_model(self.model)
        return self.model

    def run(self, plan: DataPlan, prompt: str | None = None) -> Run:
        plan.validate()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.run_root / f"{stamp}-{uuid4().hex[:8]}"
        data_path, manifest = plan.materialize(run_dir)
        spec = TaskSpec(
            target=str(plan.target),
            task_type=plan.task_type,
            metric=plan.metric,
            notes=plan.notes,
            column_notes=plan.column_notes,
            n_splits=plan.n_splits,
            seed=plan.seed,
            target_transform=plan.target_transform,
            forecast_horizon=plan.forecast_horizon,
        )
        session = RunSession(
            str(run_dir),
            data_path,
            spec,
            val_fraction=self.val_fraction,
            holdout_fraction=self.holdout_fraction,
            split_on=plan.split_on if isinstance(plan.split_on, str) else None,
            serving_columns=plan.serving_columns,
            fit_time_budget_s=self.fit_time_budget_s,
            profile=self.profile,
        )
        run_agent(
            session,
            prompt or "Inspect the data, build honest useful features, and finalize the run.",
            model=self._resolved_model(),
        )
        bundle_path = run_dir / "bundle"
        if not (bundle_path / "pipeline.json").exists():
            finalize_run(session)
        static_dir = bundle_path / "static"
        for join in plan.joins:
            if join.decision != "accepted":
                continue
            static_dir.mkdir(exist_ok=True)
            shutil.copy2(plan.tables[join.right_table]["path"], static_dir / join.right_table)
        with open(bundle_path / "api_plan.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        return Run(run_dir)
