"""Minimal public-facade example.

Inspection proposes facts; the owner confirms target, serving boundary, split,
and every proposed static join before a run may start.
"""

import tabular_agent as ta

agent = ta.Agent(model="anthropic:claude-sonnet-4-6")
plan = agent.inspect("/path/to/data")

print(plan.target_candidates)
print(plan.ordering_candidates)
print(plan.proposed_identifier_columns)
print(plan.joins)

plan.target = "target"
plan.serving_columns = ["feature_a", "feature_b"]
plan.accept_identifier_proposals()  # or reject_identifier_proposals()
plan.split_on = False
plan.reject_joins()

run = agent.run(plan)
print(run.report())
predictions = run.bundle.predict("/path/to/serving.csv")

# Inspect exactly what the agent learned.
print(run.bundle.describe_features().to_string(index=False))

# Train your own estimator on the exact OOF-safe feature table.
training = run.training_features()
X_train = training[run.bundle.feature_names]
y_train = training[plan.target]

# Build the matching feature matrix for genuinely new rows.
X_serving = run.bundle.feature_table("/path/to/serving.csv")
serving_with_features = run.bundle.feature_table("/path/to/serving.csv", include_input=True)
