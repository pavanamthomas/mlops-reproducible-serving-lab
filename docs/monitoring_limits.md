# Monitoring limits

The functions in `mlserv.monitoring` summarise **batches**. They are not a monitoring platform.

What is computed:

- feature location and spread (`feature_summaries`)
- missingness rates, including absent columns (`missingness`)
- PSI and KS on numeric features versus a reference window (`distribution_change`)
- predicted probability histogram summaries (`prediction_distribution`)
- accuracy on rows whose labels have arrived (`delayed_performance`)

Explicit limits:

1. No streaming, no sliding windows with error-spending, no multiple-testing control across features.
2. KS p-values assume iid samples within each window. They are not valid if rows are dependent.
3. PSI depends on the reference quantile bins and on the empty-bin epsilon. Changing either changes the number.
4. A shift in the prediction histogram is not concept drift and not covariate shift. It is a shift in \(\hat g(X)\).
5. Delayed accuracy is unidentified for labels that have not arrived. Filling them with zero is a different, worse estimator.
6. No alerting, no on-call, no SLO. No dashboards are shipped.
7. No claim that these statistics would catch training-serving skew. Skew can move predictions without a large PSI on raw features (feature swap of two in-range columns is the example).

Use `docs/drift.md` for the distinction between covariate shift and concept drift, and `FLAGSHIP_TRAINING_SERVING_SKEW.md` for the serving-transform failure that monitoring summaries can miss.
