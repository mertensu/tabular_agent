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

from .harness.agent import gateway_model, run as run_agent
from .harness.session import RunSession
from .harness.tools import finalize_run
from .inspect import profile_collection, profile_dataframe, table_paths
from .io import load_table, save_parquet
from .pipeline import load_bundle, predict_bundle
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
    joins: list[JoinProposal] = field(default_factory=list)
    target: str | None = None
    serving_columns: list[str] | None = None
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
            joins=[JoinProposal(**join) for join in profile["joins"]],
        )

    def accept_joins(self) -> DataPlan:
        for join in self.joins:
            join.accept()
        return self

    def reject_joins(self) -> DataPlan:
        for join in self.joins:
            join.reject()
        return self

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
            "split_on": self.split_on,
            "joins": [asdict(join) for join in self.joins if join.decision == "accepted"],
        }
        with open(prepared / "api_plan.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        return path, manifest


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
        return result

    def predict(
        self,
        rows: pd.DataFrame | str | os.PathLike,
        streams: dict | None = None,
    ):
        return predict_bundle(self._bundle, self._prepare(self._raw_frame(rows)), streams=streams)


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
