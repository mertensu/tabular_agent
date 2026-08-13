"""house_prices run 9: the search only builds features.

The control is **run 8** (`_runs/house_prices_v8`), and this is a pre-registered
A/B: same data file, same prompt, same model, same route, same dictionary. The
only thing that differs is the harness.

    run 8   14 experiments   val 0.11171  ->  Kaggle 0.12298   optimism +0.0113

## What changed, and it is one thing

`0035`: the agent no longer chooses a model. It builds tables; each is scored on
creation by a fixed probe (`0036`); `finalize_run` fits xgb/catboost/elasticnet on
the best table and packages the winner. The stop rule counts tables, not
experiments.

Everything else is held: `fireworks/deepseek-v4-flash-0731` with thinking on at
high effort, `train_no_id.csv`, the same two-sentence prompt, the same
`data_description.txt` through `column_notes`.

## The claim being registered, before the submission is scored

Stated here so it cannot be adjusted afterwards:

1. **Kaggle score comparable to or better than run 8's 0.12298.** The point of
   0035 is not a better score — it is a search whose comparisons mean something —
   so *comparable* is a pass and *better* is a bonus. Materially worse is a
   result, and it goes in the record.
2. **Less optimism than +0.0113.** Fewer selection events on the ranked number
   should mean a reported score closer to the truth. This is the prediction most
   likely to be wrong, because the ranked number is now `probe_score` on the same
   validation half, and nothing about the third split (`0023`) has been built.
3. **Fewer, more informative comparisons.** Run 8 spent 5 of 14 experiments on
   model variations. This run cannot: every measurement it makes is a table.

## What it cannot settle

Both numbers it reports are measured on rows the search selected against. If the
Kaggle score improves, that is evidence; if only the *val* score improves, that is
not evidence of anything, and 0034 is the record of why. The instrument that would
settle it is 0023, still unbuilt, and deliberately not built first: a third split
takes rows out of training and would make this run incomparable to run 8.
"""
import json
import os
import re
import sys
import time
from collections import Counter

import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModelSettings

from tabular_agent.harness import journal
from tabular_agent.harness.agent import Deps, build_agent, gateway_model
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.usage import context_profile, format_profile
from tabular_agent.taskspec import TaskSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "house_prices_v9")
DATA = os.path.join(REPO, "house_prices", "train_no_id.csv")
TEST = os.path.join(REPO, "house_prices", "test.csv")
DICT = os.path.join(REPO, "house_prices", "data_description.txt")
LEADERBOARD_CSV = os.path.join(
    REPO, "house_prices",
    "house-prices-advanced-regression-techniques-publicleaderboard-2026-08-10T10:53:09.csv",
)

MODEL = "fireworks/deepseek-v4-flash-0731"
MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
}

#: 1460 rows, so a fit is seconds and this will never fire. Set anyway, because a
#: backstop that is only present on the dataset that already blew up is not a
#: backstop.
FIT_TIME_BUDGET_S = 10 * 60

#: Verbatim from runs 1, 2 and 3. Rewriting it would improve the prompt and forfeit
#: the comparison — and the dictionary is deliberately not mentioned, so using it is
#: a discovery rather than an instruction followed.
PROMPT = """Build the best model you can for this dataset of house sales.
The target is SalePrice. Work until you stop making progress, then finalize."""

ABOUT = """Residential house sales in Ames, Iowa (the classic Kaggle
house-prices task). One row is one sale. Most columns describe the property --
size, quality ratings, year built, neighbourhood -- and many of the quality and
condition ratings are genuinely *ordered* categories rather than nominal ones;
each such column's description lists its levels. The target is the sale price in
dollars and is strongly right-skewed."""

#: The dictionary names two columns differently from the data. Fixed here rather
#: than left to silently drop two descriptions -- the dictionary is the data
#: owner's document and reconciling it with the actual header is prep work, which
#: is what a script like this is for.
DICT_RENAMES = {"Bedroom": "BedroomAbvGr", "Kitchen": "KitchenAbvGr"}


def column_notes(path: str = DICT, columns=None) -> dict[str, str]:
    """Parse `data_description.txt` into `{column: description}`.

    The file is `Name: prose` at the left margin followed by indented
    `LEVEL<tab>meaning` lines. The levels matter as much as the prose: they are
    where the *order* of an ordinal column lives, which `fit_encoding` requires as
    an explicit argument and can never infer.
    """
    notes: dict[str, str] = {}
    col: str | None = None
    desc = ""
    levels: list[str] = []

    def flush() -> None:
        if col is None:
            return
        text = desc
        if levels:
            text += " Levels: " + "; ".join(levels) + "."
        notes[DICT_RENAMES.get(col, col)] = text.strip()

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip():
                continue
            header = re.match(r"^(\S+):\s*(.*)$", line)
            if header:
                flush()
                col, desc, levels = header.group(1), header.group(2).strip(), []
            elif col is not None and line[0] in " \t":
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                if len(parts) >= 2:
                    levels.append(f"{parts[0]}={' '.join(parts[1:])}")
                elif parts:
                    levels.append(parts[0])
    flush()
    if columns is not None:
        notes = {k: v for k, v in notes.items() if k in set(columns)}
    return notes


def write_submission(run_dir: str) -> str | None:
    """Predict the real test set so Kaggle can supply the honest number.

    Run 3's submission was made by hand; doing it here means the optimism figure
    this run exists to measure is reproducible rather than remembered.
    """
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never finalized")
        return None

    test = pd.read_csv(TEST)
    try:
        preds = predict_bundle(load_bundle(bundle_dir), test.drop(columns=["Id"]))
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        return None

    negative = int((preds < 0).sum())
    if negative:
        # Reported, not clipped: flooring here would hide a defect in the fit.
        print(f"   note: {negative} predictions below zero, left as they are")
    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"Id": test["Id"], "SalePrice": preds}).to_csv(path, index=False)
    return path


def summarise_thinking(run_dir: str) -> dict:
    """Evidence that the run actually thought, and what it cost.

    A run with zero thinking parts is a non-reasoning run under a different model
    name, and every comparison drawn from it would be wrong the same way.
    """
    path = os.path.join(run_dir, "transcript.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    kinds = Counter(
        p.get("part_kind") for m in raw for p in m.get("parts", [])
    )
    usage = [m.get("usage") or {} for m in raw]
    return {
        "thinking_parts": kinds.get("thinking", 0),
        "text_parts": kinds.get("text", 0),
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
    }


def summarise_dictionary(run_dir: str, notes: dict) -> dict:
    """Did the dictionary change what got built?

    The measurable trace is `fit_encoding(method="ordinal")`: run 3 used frequency
    and target encodings throughout, and `ordinal` needs a level order that is
    exactly what the dictionary supplies and the data cannot.
    """
    j = journal.load(run_dir)
    methods = Counter(t.get("method") for t in j.get("tables") or [])
    return {
        "columns_described": len(notes),
        "table_methods": dict(methods),
        "used_ordinal_encoding": methods.get("ordinal", 0) > 0,
    }


def kaggle_rank(score: float) -> str:
    if not os.path.exists(LEADERBOARD_CSV):
        return ""
    s = pd.read_csv(LEADERBOARD_CSV)["Score"].sort_values().reset_index(drop=True)
    rank = int((s < score).sum()) + 1
    return f"rank ~{rank}/{len(s)} (top {rank / len(s):.0%})"


def main() -> int:
    for p in (DATA, TEST, DICT):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    columns = pd.read_csv(DATA, nrows=1).columns
    notes = column_notes(columns=columns)
    undescribed = [c for c in columns if c not in notes and c != "SalePrice"]
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"data dictionary: {len(notes)} of {len(columns) - 1} feature columns described"
          + (f" | undescribed: {undescribed}" if undescribed else ""))

    session = RunSession(
        RUN_DIR, DATA,
        TaskSpec(target="SalePrice", notes=ABOUT, column_notes=notes),
        fit_time_budget_s=FIT_TIME_BUDGET_S,
    )
    agent = build_agent(gateway_model(MODEL))

    t0 = time.time()
    result = agent.run_sync(
        PROMPT, deps=Deps(session=session),
        model_settings=OpenAIChatModelSettings(**MODEL_SETTINGS),
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(result.output)
    print("=" * 70)

    with open(os.path.join(RUN_DIR, "transcript.json"), "wb") as f:
        f.write(result.all_messages_json())
    messages = json.loads(result.all_messages_json())
    calls = [
        p["tool_name"] for m in messages for p in m.get("parts", [])
        if p.get("part_kind") == "tool-call"
    ]

    j = journal.load(RUN_DIR)
    rows = j.get("leaderboard") or []
    if rows and not j["stop"]["finalized"]:
        from tabular_agent.harness import tools
        print("\nthe run ended without finalize_run — packaging its best model")
        tools.finalize_run(session)
        j = journal.load(RUN_DIR)
        rows = j["leaderboard"]

    profile = context_profile(messages, experiments=len(rows))
    with open(os.path.join(RUN_DIR, "context_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    tables = j.get("tables") or []
    scored = [t for t in tables if t.get("probe_score") is not None]
    print(f"\n{'=' * 70}")
    print(f"{len(tables)} tables ({len(scored)} scored) and {len(rows)} terminal "
          f"fits in {elapsed / 60:.1f} min")
    print(f"{len(calls)} tool calls: {dict(Counter(calls).most_common())}")
    if j.get("best"):
        best = j["best"]
        print(f"best {best['experiment_id']}: val {best['val_score']:.5f} "
              f"(baseline {j['baseline']['val_score']:.5f})")
        print(f"  run 8 for comparison: val 0.11171 -> Kaggle 0.12298 "
              f"(optimism +0.0113), 14 experiments of which 5 were model variations")
        if j.get("work_probe"):
            print(f"  work table, same ruler: {j['work_probe']['probe_score']:.5f} "
                  f"— what the features had to beat")

    ranked = sorted(
        ((t["id"], t.get("method"), t["probe_score"]) for t in scored),
        key=lambda r: r[2],
    )
    print("\ntables by probe_score (the search, in full):")
    for tid, method, score in ranked:
        print(f"  {tid:>4s}  {method or '':<12s} {score:.5f}")

    costs = [r["fit_seconds"] for r in rows if r.get("fit_seconds")]
    if costs:
        print(f"\nfit cost: {sum(costs) / 60:.1f} min over {len(costs)} experiments "
              f"| worst {max(costs) / 60:.1f} min")

    print(f"\nthinking: {json.dumps(summarise_thinking(RUN_DIR), indent=2)}")
    print(f"\ndictionary: {json.dumps(summarise_dictionary(RUN_DIR, notes), indent=2)}")
    print()
    print(format_profile(profile, "context"))

    path = write_submission(RUN_DIR)
    if path:
        print(f"\nsubmission: {path}")
        if j.get("best"):
            print(f"  our val says {j['best']['val_score']:.5f} — {kaggle_rank(j['best']['val_score'])} "
                  f"if Kaggle agrees, which it will not. Score it to get the optimism.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
