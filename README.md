# camt: the clustered AUC maximum test (CAMT)

Tests of the claim that **the best of K models has a higher AUC than a baseline** on a test set that contains several images or
records per patient, and a **summary-statistics audit** of published claims of this kind.

Two errors act together when the winner among K candidate models is compared with a baseline on one shared test set:

1. observations from one patient are correlated, so a test that treats every image as independent understates the uncertainty;
2. the winner is chosen among K candidates, so its apparent advantage is inflated even if all models are equally good.

`camt` implements CAMT, a test that accounts for both. Labels may vary within a patient. The method is described in the accompanying
manuscript (in preparation).

Status: research software, version 0.1.0.

## Installation

```bash
pip install camt-test      # from PyPI; the import name is `camt`
pip install .            # or from the repository root; requires Python >= 3.9, numpy and scipy
pip install ".[test]"    # also installs pytest
```

## Quick start

```python
from camt import camt_test

# scores:   array (n_images, 1 + K), one column per model; column 0 is the baseline
# labels:   array (n_images,) of 0/1 image-level labels
# patients: array (n_images,) of patient identifiers (the clusters)
result = camt_test(scores, labels, patients, baseline=0, n_boot=4000, seed=1)
print(result.summary())
```

```
best of 5 candidates vs baseline (column 0); best = column 1
  AUC best 0.841 vs baseline 0.782 (difference +0.058)
  500 images, 100 patients (47 with positives, 53 with negatives, 0 with both)
  p (patient-clustered, adjusted for K) = 0.1685
  p (pruned) = 0.1585 | clustering only = 0.0490 | selection only = 0.0310 | naive = 0.0068
```

In this simulated example no candidate is truly better, and the naive analysis reports p = 0.007. A complete script is in
`examples/quickstart.py`.

### Which p-value to report

| attribute | meaning |
|---|---|
| `p_value` | **the proposed test**: patient-clustered, adjusted for K, with the small-class calibration. Use this one. |
| `p_value_pruned` | as above, but candidates that are clearly worse than the baseline are dropped from the null distribution. More power when many candidates are poor; valid pointwise only. |
| `p_naive` | what an image-level test of the best candidate would give (for comparison). |
| `p_clustering_only` | patient-clustered, but no adjustment for K. |
| `p_selection_only` | adjusted for K, but images treated as independent. |

All p-values are one-sided; the null hypothesis is that no candidate is better than the baseline.

## Functions

| function | purpose |
|---|---|
| `camt_test(scores, labels, patients, baseline=0, n_boot=4000, seed=None, influence_fn=None)` | the test; returns a `CamtResult` |
| `design_effect(scores, labels, patients, baseline=0)` | Kish cluster size, design effect and within-patient correlation of the AUC influence function, for each model and for each difference from the baseline |
| `threshold_influence(thresholds, kind="balanced")` | influence function for sensitivity, specificity or balanced accuracy at operating points fixed in advance; pass it as `influence_fn` |
| `audit_claim(p_naive, patients, images, K, rho=...)` | adjust a published p-value for clustering and for K, using only summary numbers |
| `sensitivity_table(...)` | `audit_claim` over a range of within-patient correlations |
| `hanley_mcneil_se`, `rho_from_patient_level_ci` | helpers for the audit |

### Metrics at a fixed operating point

For balanced accuracy at thresholds chosen in advance (for example on the training patients):

```python
from camt import camt_test, threshold_influence
result = camt_test(scores, labels, patients, influence_fn=threshold_influence(thresholds, kind="balanced"))
```

Thresholds estimated on the test sample itself are not covered.

### Auditing a published claim

```python
from camt import audit_claim
a = audit_claim(p_naive=0.0004, patients=40, images=3900, K=10, rho=0.1)
a["p_bonferroni"]   # adjusted p-value if the within-patient correlation is 0.1 and 10 models were tried
a["rho_star"]       # correlation above which the clustered p-value alone exceeds 0.05
```

The within-patient correlation cannot be recovered from summary data, so report the result over a range of values
(`sensitivity_table`) and treat `rho_star` as the quantity of interest. The audit is valid if the correlation is not understated and
K is the true number of models tried.

## Assumptions and limits

- **K must be the true number of candidates examined**, and the baseline must be fixed in advance. Pre-specifying one primary model
  (K = 1) removes the selection cost.
- The target is the **fitted models** on patients like the test patients, not the training algorithm. Test patients must be disjoint
  from the training data.
- The test assumes independent patients, a bounded number of images per patient and a moderate number of candidates.
- The test can be **liberal** when fewer than about 20 patients belong to the smaller class and many candidates are screened, and
  when a few patients contribute most of the images. The function emits a warning in both situations. The cut-off for the second
  warning (fewer than 100 effective clusters, that is, images divided by the Kish cluster size) is a heuristic and has not been
  validated.
- Candidates whose scores give a zero-variance difference from the baseline are dropped.
- Metrics whose threshold is estimated on the test sample are not covered.

## Reproducing the behaviour of the test

```bash
pytest                                          # unit tests, about 25 seconds
python examples/reproduce_size.py               # false-positive rates in simulation (a few minutes)
python examples/reproduce_size.py --reps 200    # quick version
```

`examples/reproduce_size.py` simulates test sets in which all models are equal and prints the false-positive rate of the naive,
clustering-only, selection-only and proposed procedures for K = 1, 5 and 20, at a nominal level of 5%.

## Citation

If you use this software, please cite it (see `CITATION.cff`) together with the accompanying manuscript:

> Gul A. Valid inference for "best of K" model comparisons on patient-clustered diagnostic data. Manuscript in preparation.

The archived release and its DOI are given in `CITATION.cff` once available.

## Licence

MIT. See `LICENSE`.
