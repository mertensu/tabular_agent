"""tabular_agent: leakage-safe tabular-ML core."""

from .api import Agent, Bundle, DataPlan, JoinProposal, Run
from .capability_review import capability_backlog
from .explain import Explanation

__all__ = ["Agent", "Bundle", "DataPlan", "Explanation", "JoinProposal", "Run", "capability_backlog"]
__version__ = "0.1.0"
