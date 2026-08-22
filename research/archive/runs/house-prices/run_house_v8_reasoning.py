"""house_prices run 8: the reasoning route, and the competition's own data dictionary.

The control is **run 3** (`_runs/house_prices_v3`), chosen over the later episode
runs because it is the one with a Kaggle number, which makes it the only house
run that can answer the question this one is really for.

    run 3   24 experiments   val 0.11354   Kaggle 0.12298   optimism +0.00944

That optimism figure is the evidence behind `0023`: our validation half overstated
generalisation by nearly a point of rmsle, and the overstatement grew 4.6x between
17 and 24 experiments. So the question is not only "does a thinking model score
better" but **"does it lie to itself less"** — and unlike Rossmann, this dataset
can settle it, because Kaggle holds ground truth for `test.csv`.

## Three things differ from run 3, and that is two more than I would like

1. **The route serves thinking.** Every earlier run went through
   `tensorx/deepseek-v4-flash-0731`, which serves these weights with thinking off.
   Same model, same gateway, different provider — see the Rossmann v6 notes.
2. **The data dictionary reaches the agent.** `data_description.txt` ships with the
   competition and has never been wired in; run 3's agent recovered the ordinal
   orders (`Po<Fa<TA<Gd<Ex`) from pretraining instead. It now arrives through
   `TaskSpec.column_notes`, which is the sanctioned channel for dataset knowledge.
3. **`MAX_REPORTED_COLUMNS` went 60 -> 100**, because with 80 columns the profile
   was truncated and took a quarter of the dictionary with it, including four
   columns whose notes carry a level order.

**So a difference in outcome is not attributable to either change alone.** This is
deliberately a "does the combination help" run, not a controlled ablation. If it
does help, the ablation is the next run and it is cheap: same script, dictionary
withheld.

Everything else is byte-identical to run 3: the same data file, the same prompt,
the same single conversation with no episode loop, the same defaults.

## What it measures

1. **Kaggle vs our own number.** The one that matters. Less optimism than run 3's
   +0.00944 would be the first evidence that a thinking model games the validation
   half less, which is `0023`'s whole subject.
2. **Does the dictionary change what gets built?** Run 3 recovered the ordinals
   unaided, so the dictionary's value here is not obvious — the interesting outcome
   is `fit_encoding(method="ordinal")` appearing where run 3 used frequency, or
   nothing changing at all.
3. **Cost.** Rossmann v6 thought for *fewer* output tokens than the non-thinking
   run spent. Worth confirming on a second dataset.
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
RUN_DIR = os.path.join(REPO, "_runs", "house_prices_v8")
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

    print(f"\n{'=' * 70}")
    print(f"{len(rows)} experiments in {elapsed / 60:.1f} min")
    print(f"{len(calls)} tool calls: {dict(Counter(calls).most_common())}")
    if j.get("best"):
        best = j["best"]
        print(f"best {best['experiment_id']}: val {best['val_score']:.5f} "
              f"(baseline {j['baseline']['val_score']:.5f})")
        print(f"  run 3 for comparison: val 0.11354 -> Kaggle 0.12298 "
              f"(optimism +0.00944)")

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
