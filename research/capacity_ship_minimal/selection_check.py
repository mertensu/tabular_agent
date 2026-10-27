"""Does the selection rule throw away chains of small real gains? (no LLM)

    uv run python research/capacity_ship_minimal/selection_check.py r4 r10 r11

0090 ships the best table whose gain over its *immediate source* exceeds the
spread of that gain across validation blocks. In r11 the probe score fell from
0.1426 (t8, shipped) to 0.1250 (t24) in steps each inside its own spread, so
nothing after t8 could ship.

The alternative judged here: a table passes when its gain over its *nearest
passing ancestor* (its immediate source while none has passed) exceeds the
spread of that comparison. A table one noise-level step past the last passing
one still fails, which is what 0090 is for; a chain that adds up to a gain
beyond the spread passes.

Per run, from the stored validation predictions (`probe_preds/`):
  1. the table each rule ships;
  2. for both picks, the run's ruler (200-tree LightGBM probe) fitted on the
     training rows and scored on our own test days (`audit.parquet`, never seen
     by the run), plus the blockwise comparison between the two picks there.

Writes `selection_check_rossmann.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import HERE, OUT  # noqa: E402

from tabular_agent.compare import compare_predictions  # noqa: E402
from tabular_agent.harness import stop  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, _load_lineage  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.probe import fit_probe, probe_predict  # noqa: E402

TARGET, METRIC, ORDER = "Sales", "rmspe", "Date"


def _id(rel: str | None) -> str | None:
    return Path(rel).stem if rel else None


def chain_pick(tables: list[dict], preds: dict, y, order) -> tuple[dict | None, dict]:
    """The alternative rule: gain over the nearest passing ancestor > its spread."""
    by_id = {t["id"]: t for t in tables}
    passed, verdicts = {}, {}
    for t in tables:  # creation order: ancestors come first
        if t["id"] not in preds or t.get("probe_score") is None:
            continue
        anchor, node = None, _id(t.get("source_table"))
        while node in by_id:
            if passed.get(node):
                anchor = node
                break
            node = _id(by_id[node].get("source_table"))
        anchor = anchor or _id(t.get("source_table"))
        if anchor not in preds:
            continue
        iv = compare_predictions(METRIC, y, preds[t["id"]], preds[anchor], order_values=order)
        ok = stop._has_demonstrated_gain({"probe_vs_source_interval": iv})
        passed[t["id"]] = ok
        verdicts[t["id"]] = {"anchor": anchor, "gain": iv and iv["gain"],
                             "block_sd": iv and iv["block_sd"], "passes": ok}
    rows = [t for t in tables if passed.get(t["id"])]
    return (min(rows, key=lambda t: t["probe_score"]) if rows else None), verdicts


def test_predict(root: Path, table_id: str, audit: pd.DataFrame, task: dict) -> np.ndarray:
    path = root / "tables" / f"{table_id}.parquet"
    table = pd.read_parquet(path)
    X, _ = _feature_frame(table, TARGET)
    fit = fit_probe(X, table[TARGET], "regression", METRIC,
                    task.get("target_transform"), task.get("objective"))
    Xa = _load_lineage(str(path), "regression").transform(audit.drop(columns=[TARGET]))
    return np.asarray(probe_predict(fit, Xa[list(X.columns)], "regression"), dtype=float)


def main(*tags: str) -> int:
    audit = pd.read_parquet(OUT / "rossmann" / "audit.parquet")
    y_test, order_test = audit[TARGET].to_numpy(), audit[ORDER].to_numpy()
    out = {}
    for tag in tags:
        root = Path(json.loads((HERE / f"result_rossmann_{tag}.json").read_text())["run_dir"])
        j = json.loads((root / "run_journal.json").read_text())
        tables = j["tables"]
        val = pd.read_parquet(root / "data" / "val.parquet")
        y, order = val[TARGET].to_numpy(), val[ORDER].to_numpy()
        preds = {p.stem: np.load(p) for p in (root / "probe_preds").glob("*.npy")}

        current = stop.best_table(tables, METRIC)
        chain, verdicts = chain_pick(tables, preds, y, order)
        row = {
            "val_days": int(val[ORDER].nunique()),
            "current": current and {"id": current["id"], "val_probe": current["probe_score"]},
            "chain": chain and {"id": chain["id"], "val_probe": chain["probe_score"]},
            "best_probe": min((t for t in tables if t.get("probe_score") is not None),
                              key=lambda t: t["probe_score"])["id"],
            "chain_passing": [k for k, v in verdicts.items() if v["passes"]],
            "verdicts": verdicts,
        }
        picks = {k: row[k]["id"] for k in ("current", "chain") if row[k]}
        test = {pid: test_predict(root, pid, audit, j["task"]) for pid in set(picks.values())}
        row["test_probe"] = {k: round(compute_metric(METRIC, y_test, test[pid]), 6)
                             for k, pid in picks.items()}
        if len(test) == 2:
            row["test_chain_vs_current"] = compare_predictions(
                METRIC, y_test, test[picks["chain"]], test[picks["current"]],
                order_values=order_test)
        out[tag] = row
        print(tag, json.dumps({k: v for k, v in row.items() if k != "verdicts"},
                              default=str), flush=True)

    (HERE / "selection_check_rossmann.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(sys.argv[1:] or ["r4", "r10", "r11"])))
