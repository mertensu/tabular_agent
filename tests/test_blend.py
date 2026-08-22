import numpy as np
import pytest

from tabular_agent.blend import apply_blend


def test_apply_blend_rejects_mismatched_weight_length():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    with pytest.raises(ValueError):
        apply_blend(preds, np.array([1.0]))
