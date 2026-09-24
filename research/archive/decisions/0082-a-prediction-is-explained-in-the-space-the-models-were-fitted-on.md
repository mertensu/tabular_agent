# 0082 — A prediction is explained in the space the models were fitted on

- **Date:** 2026-09-04
- **Status:** accepted (2026-09-04)
- **Related:** [0044](../../../docs/decisions/0044-what-ships-is-the-average-not-the-argmax.md) (invert
  each member, *then* blend — the order this attribution must respect),
  [0080](0080-a-blend-weight-a-member-cannot-inflate-decided-before-the-seal.md)
  (the shipped weights, which this attributes rather than an equal average),
  [0021](0021-the-agent-picks-the-target-scale.md) /
  [0033](0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md) (a number reported in
  the wrong unit is worse than no number), [0081](0081-random-classification-preserves-classes-identifiers-are-transport.md)
  (identifiers are transport, never features — so they get no contribution)

## What happened

The facade could say *what* a bundle predicts and, through the run report, *which
features earned their place across the table*. It could not say why a bundle
produced **one particular number for one particular row** — the question a person
asks first when a served prediction looks wrong. `bundle.explain(rows)` adds that:
per-row, per-feature contributions plus a base value, so that
`base + Σ contributions` reconstructs the prediction.

## The decision

**Attribute the shipped blend, with model-native contributions, additive in the
space the models were fitted on — and name that space.**

- **The blend, not a member.** The bundle ships the 0080-weighted combination
  (0044/0080). The attribution is the same weighted combination of each member's
  attribution: `base = Σ wᵢ baseᵢ`, `φ = Σ wᵢ φᵢ`. Explaining the search winner or
  an equal-weight average would explain a number the bundle never emits.
- **Model-native and exact — no surrogate, no background set.** TreeSHAP for the
  gradient-boosted members (`pred_contrib` / `pred_contribs` / `ShapValues`) and
  the closed-form `coefⱼ·zⱼ` for the elasticnet, read off the *fitted, standardised*
  features so the centring is the model's own. Nothing is sampled and nothing is
  approximated by a second model, so there is no Monte-Carlo variance and no
  surrogate-fidelity caveat to attach to the answer.
- **Additive in the fitted space, which is named.** SHAP decomposes the model
  *output*. For a plain regression (`target_transform="none"`) that output *is*
  the prediction, so `base + Σφ` equals `predict` to numerical precision and
  `space == "target"`. Under `log1p` the output is on the log scale; for a
  classifier it is log-odds. There the decomposition is still exactly additive —
  on that scale — and `Explanation.space` says so (`"log1p(target)"`,
  `"log-odds"`) rather than letting the sum masquerade as euros or a probability.

## What was rejected

- **Reporting everything in target/probability units to look additive.** This is
  the 0033 failure one level down: to land the contributions on the euros under
  `log1p`, or on the probability for a classifier, you either blend on the wrong
  scale (the exact estimator 0044 measured against) or spread a non-linear link
  across the features approximately and call it exact. A number in the wrong unit
  is not a friendlier number; it is a wrong one. We report the honest scale and
  name it.
- **A `shap`/`captum` dependency.** Every member already computes its own exact
  contributions; a general explainer would add a heavy dependency, a background-set
  choice, and sampling noise to reproduce — worse — what the libraries hand back
  in closed form.
- **Multiclass, for now.** v1 covers the two shapes the facade builds (regression,
  binary). A multiclass member returns a per-class block; rather than invent a
  per-class story nobody has scored, `explain` raises. The general form ("one
  attribution per class, additive per class in log-odds") is written down here so
  the refusal is a scope line, not a gap.

## What it cost

One module (`explain.py`), one `Bundle.explain` method mirroring `predict`'s row
preparation, and a dependency on each member exposing native contributions — true
of every model type the library ships. The load-bearing risk is the *space*: an
attribution that looked additive on the wrong scale would mislead exactly the
person who trusted it most, which is why the scale is a named, tested field and
not a footnote.
