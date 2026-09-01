"""Minimal public-facade example.

Inspection proposes facts; the owner confirms target, serving boundary, split,
and every proposed static join before a run may start.
"""

import tabular_agent as ta

agent = ta.Agent(model="anthropic:claude-sonnet-4-6")
plan = agent.inspect("/path/to/data")

print(plan.target_candidates)
print(plan.ordering_candidates)
print(plan.joins)

plan.target = "target"
plan.serving_columns = ["feature_a", "feature_b"]
plan.split_on = False
plan.reject_joins()

run = agent.run(plan)
print(run.report())
predictions = run.bundle.predict("/path/to/serving.csv")
