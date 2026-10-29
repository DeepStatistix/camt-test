# Changelog

## 0.1.0

First release.

- `best_of_k_test`: patient-clustered, selection-adjusted test of the best of K models against a baseline (AUC, or a metric at a
  fixed operating point through `threshold_influence`), with a pruned variant, comparison analyses and warnings for small classes and
  very unequal cluster sizes.
- `design_effect`: design effect and within-patient correlation of the AUC influence function.
- `audit_claim`, `sensitivity_table`, `hanley_mcneil_se`, `rho_from_patient_level_ci`: summary-statistics audit of published claims.
- Unit tests, a quick-start example and a simulation of the size of the test.
