"""The constrained harness (spec §3 revision, §12, §13, §19).

The harness owns the agent loop and registers *only* the leakage-safe tools in
``tools.py``. There is no bash and no scratchpad. Agent-written code runs in two
places and neither is a path *to* the data: ``transform_features`` is row-local
and target-free (0009), and ``explore_experiment`` cannot persist what it
computes. That is what makes "leakage-safe by construction" a structural
property rather than a prompt-level suggestion.

Only the LLM is swappable (``agent.py``).
"""

from .session import RunSession

__all__ = ["RunSession"]
