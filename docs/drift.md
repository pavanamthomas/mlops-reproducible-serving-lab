# Drift

Two different objects are easy to conflate.

**Covariate shift** is a change in \(P(X)\). Labels may still follow the original \(P(Y \mid X)\).  
**Concept drift** is a change in \(P(Y \mid X)\). \(P(X)\) may look stable.

Neither is identified from a single accuracy number on a mixed window.

## Covariate shift in this laboratory

`mlserv.failures.data_drift` multiplies income by \(\exp(\delta)\) and computes:

- Population Stability Index (PSI) on quantile bins of the reference sample
- Two-sample Kolmogorov–Smirnov statistic and p-value

These are descriptive two-sample measures on a batch pair. They are not a sequential detector, not multiplicity-adjusted, and not a reason by themselves to retrain.

Common 0.1 / 0.25 PSI thresholds are rules of thumb. They are not theorems of this DGP. Empty bins are clipped with a small epsilon; that clip is numerical, not a model of rare categories.

## Concept drift in this laboratory

`mlserv.failures.concept_drift` changes the credit-score slope in the logistic DGP (default flip from \(+2.4\) to \(-2.4\)). The same fitted Pipeline is scored on the new labels.

Labels can be delayed. `delayed_accuracy` computes accuracy only on rows whose labels have arrived. Accuracy on the complementary set is not identified.

A stable prediction histogram does not identify a stable concept. The script `scripts/run_all.py` prints both the flipped-label accuracy and the prediction histogram so that they can disagree in interpretation.

## What this is not

This is not Evidently, WhyLabs, or a Prometheus stack. Limits: `docs/monitoring_limits.md`.
