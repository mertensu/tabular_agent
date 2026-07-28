"""The constrained harness (spec §3 revision, §12, §13, §19).

The harness owns the agent loop and registers *only* the leakage-safe tools in
``tools.py``. There is no bash, no scratchpad, no free-code path to the data —
that is what makes "leakage-safe by construction" a structural property rather
than a prompt-level suggestion.

Only the LLM is swappable (``agent.py``).
"""

from .session import RunSession

__all__ = ["RunSession"]
