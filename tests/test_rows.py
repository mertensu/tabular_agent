"""Row filtering: the step that is applied once and never replayed."""

import pandas as pd
import pytest

from tabular_agent.pipeline import Pipeline
from tabular_agent.rows import SandboxError, select_rows


@pytest.fixture
def df():
    return pd.DataFrame(
        {"open": [1, 1, 0, 1, 0, 1], "sales": [10.0, 12.0, 0.0, 9.0, 0.0, 11.0]}
    )


def test_it_keeps_the_rows_the_predicate_marks(df):
    code = "def keep_rows(df):\n    return df['open'] == 1\n"

    kept, summary = select_rows(df, code)

    assert summary == {"n_rows": 6, "n_kept": 4, "n_dropped": 2}
    # The original index survives, so anything already aligned to these rows
    # still lines up.
    assert kept.index.tolist() == [0, 1, 3, 5]


def test_the_predicate_may_read_the_target(df):
    """Unlike `transform_features`, and safely: nothing here is replayed.

    Outlier removal by target value is ordinary modelling. It cannot leak because
    the frame this runs on is the work half and the validation half never passes
    through a filter at all.
    """
    code = "def keep_rows(df):\n    return df['sales'] > 0\n"

    kept, summary = select_rows(df, code)

    assert summary["n_kept"] == 4


def test_a_batch_dependent_predicate_is_allowed(df):
    """The one piece of agent code that needs no stability check.

    `transform_features` must reject `x > x.mean()` because the stored code is
    recomputed on whatever rows arrive at predict. A filter's code is evaluated
    once, against one frame, and never again — so there is no second run for it
    to disagree with.
    """
    code = "def keep_rows(df):\n    return df['sales'] > df['sales'].mean()\n"

    kept, summary = select_rows(df, code)

    assert summary["n_kept"] == 4


def test_returning_rows_instead_of_a_mask_is_refused(df):
    code = "def keep_rows(df):\n    return df[df['open'] == 1]\n"

    with pytest.raises(SandboxError, match="boolean pandas Series"):
        select_rows(df, code)


def test_a_non_boolean_mask_says_what_a_null_would_mean(df):
    code = "def keep_rows(df):\n    return df['sales'] / df['sales']\n"

    with pytest.raises(SandboxError, match="boolean"):
        select_rows(df, code)


def test_a_mask_built_from_already_filtered_rows_is_refused(df):
    """The subtle wrong answer: a mask that is right but shorter than the frame."""
    code = (
        "def keep_rows(df):\n"
        "    sub = df[df['open'] == 1]\n"
        "    return sub['sales'] > 0\n"
    )

    with pytest.raises(SandboxError, match="indexed like the frame"):
        select_rows(df, code)


def test_the_sandbox_still_blocks_model_libraries(df):
    code = "import sklearn\ndef keep_rows(df):\n    return df['open'] == 1\n"

    with pytest.raises(SandboxError, match="sklearn"):
        select_rows(df, code)


# ------------------------------------------------------------------ pipeline


def test_a_filter_step_is_recorded_but_not_replayed(df):
    """The load-bearing property. If this reverses, scoring and serving both break.

    `Pipeline.transform` has exactly two callers — `_score_on_val`, which lines
    predictions up against held-back targets, and `predict_bundle`, which owes an
    answer for every row it is given. A filter running in either is a silent
    misalignment.
    """
    p = Pipeline("regression")
    p.add_filter("def keep_rows(df):\n    return df['open'] == 1\n")

    out = p.transform(df)

    assert len(out) == len(df)
    # Still in the lineage, because a table with fewer rows than its parent has
    # to say why.
    assert p.to_dict()["steps"][0]["train_only"] is True


def test_a_filter_survives_a_round_trip_still_train_only(df):
    import json

    p = Pipeline("regression")
    p.add_filter("def keep_rows(df):\n    return df['open'] == 1\n")
    revived = Pipeline.from_dict(json.loads(json.dumps(p.to_dict())))

    assert len(revived.transform(df)) == len(df)
