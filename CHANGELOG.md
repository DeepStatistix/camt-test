# Changelog

## 0.1.1

Project renamed to CAMT (clustered AUC maximum test); the distribution is `camt-test` and the import name is `camt`.
Repository and issue links added to the package metadata. No change to the statistical code.

## 0.1.0

First release.

- `camt_test`: patient-clustered, selection-adjusted test of the best of K models against a baseline (AUC, or a metric at a
  fixed operating point through `threshold_influence`), with a pruned variant, comparison analyses and warnings for small classes and
  very unequal cluster sizes.
- `design_effect`: design effect and within-patient correlation of the AUC influence function.
- `audit_claim`, `sensitivity_table`, `hanley_mcneil_se`, `rho_from_patient_level_ci`: summary-statistics audit of published claims.
- Unit tests, a quick-start example and a simulation of the size of the test.
