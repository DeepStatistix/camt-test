"""Run the test on simulated data: 100 patients with 5 images each and 5 candidates, first with no better candidate, then with one."""
import numpy as np

from camt import camt_test, design_effect

rng = np.random.default_rng(1)
P, m, K = 100, 5, 5
patients = np.repeat(np.arange(P), m)
labels = (rng.random(P) < 0.5).astype(int)[patients]


def scores(edge):
    shared = rng.normal(0, 0.6, P)[patients][:, None]
    per_model = rng.normal(0, 0.6, (P, K + 1))[patients]
    noise = rng.normal(0, 0.8, (len(labels), K + 1))
    mean = 1.5 + np.r_[0.0, edge, np.zeros(K - 1)]
    return mean[None, :] * labels[:, None] + shared + per_model + noise


for edge in (0.0, 0.8):
    S = scores(edge)
    print(f"--- true edge of candidate 1: {edge}")
    print(camt_test(S, labels, patients, seed=1).summary())
print("within-patient correlation of the AUC differences:", design_effect(S, labels, patients)["rho_difference"].round(3))
