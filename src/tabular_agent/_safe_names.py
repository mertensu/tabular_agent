"""Gradient-boosted models that accept any column name.

LightGBM refuses a feature name containing ``" , : [ ] { }`` and XGBoost one
containing ``[ ] <``. Column names come from the data and from agent code: a
one-hot of a category value ``"Jan,Apr,Jul,Oct"`` is an ordinary column, and a
table the model refuses is a table the run cannot score at all. So the model
renames its input on the way in, by a fixed function of the column list, and
everything outside the model keeps the names it had.

Imported lazily by ``models._make_model``, so the two libraries load only when a
model of theirs is built. The classes live at module level because fitted models
are pickled into experiments and bundles.
"""

import re

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

_UNSAFE = re.compile(r'[\[\]{}":,<>]')


def safe_feature_names(columns) -> list[str]:
    """``columns`` with the characters either library refuses replaced by ``_``.

    Deterministic in the column list, so fit and predict agree; a name the
    replacement makes collide with an earlier one gets its position appended.
    """
    out, seen = [], set()
    for i, name in enumerate(str(c) for c in columns):
        safe = _UNSAFE.sub("_", name)
        if safe in seen:
            safe = f"{safe}__{i}"
        seen.add(safe)
        out.append(safe)
    return out


def _safe_frame(X):
    if not isinstance(X, pd.DataFrame):
        return X
    names = safe_feature_names(X.columns)
    return X if names == [str(c) for c in X.columns] else X.set_axis(names, axis=1)


class _SafeNames:
    def fit(self, X, y, **kwargs):
        eval_X = kwargs.get("eval_X")
        if isinstance(eval_X, tuple):
            kwargs["eval_X"] = tuple(_safe_frame(a) for a in eval_X)
        elif eval_X is not None:
            kwargs["eval_X"] = _safe_frame(eval_X)
        return super().fit(_safe_frame(X), y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(_safe_frame(X), **kwargs)

    def predict_proba(self, X, **kwargs):
        return super().predict_proba(_safe_frame(X), **kwargs)


class LGBMRegressor(_SafeNames, lgb.LGBMRegressor):
    pass


class LGBMClassifier(_SafeNames, lgb.LGBMClassifier):
    pass


class XGBRegressor(_SafeNames, xgb.XGBRegressor):
    pass


class XGBClassifier(_SafeNames, xgb.XGBClassifier):
    pass
