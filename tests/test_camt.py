import warnings

import numpy as np
import pytest
from helpers import simulate

from camt import audit_claim, camt_test, design_effect, sensitivity_table, threshold_influence


def test_reference_values():
    """Deterministic quantities are fixed; bootstrap p-values are Monte Carlo estimates and are checked with a tolerance."""
    S, y, pid = simulate(seed=3)
    r = camt_test(S, y, pid, n_boot=4000, seed=11, warn=False)
    assert r.best == 5
    assert (r.n_images, r.n_patients) == (320, 80)
    np.testing.assert_allclose(r.auc, [0.8193139098, 0.8127741228, 0.8387374687, 0.8297305764, 0.7961701128, 0.8412437343], atol=1e-9)
    assert r.se_clustered[0] == pytest.approx(0.0343591316, abs=1e-9)
    assert r.se_image[0] == pytest.approx(0.0265136743, abs=1e-9)
    assert r.p_naive == pytest.approx(0.20525702527235268, abs=1e-12)
    assert r.p_clustering_only == pytest.approx(0.28082648001716126, abs=1e-12)
    assert r.p_value == pytest.approx(0.613, abs=0.04)
    assert r.p_selection_only == pytest.approx(0.497, abs=0.04)
    assert r.kish_cluster_size == pytest.approx(4.0)
    d = design_effect(S, y, pid)
    np.testing.assert_allclose(d["deff_difference"], [1.679363, 1.87628, 1.987149, 2.466094, 2.01072], atol=1e-5)


def test_detects_a_large_edge_and_not_a_null():
    S, y, pid = simulate(P=200, edge=0.8, seed=1)
    assert camt_test(S, y, pid, n_boot=1000, seed=0, warn=False).p_value < 0.05
    S0, y0, pid0 = simulate(P=200, edge=0.0, seed=1)
    r0 = camt_test(S0, y0, pid0, n_boot=1000, seed=0, warn=False)
    assert r0.p_value >= r0.p_clustering_only            # adjusting for K can only raise the p-value


def test_size_is_near_nominal_under_the_null():
    reps = 120
    rej = sum(camt_test(*simulate(P=100, K=5, seed=100 + s), n_boot=500, seed=s, warn=False).p_value < 0.05 for s in range(reps))
    assert rej / reps < 0.13                              # nominal 5%, Monte Carlo SE about 2 points


def test_balanced_accuracy_influence_runs():
    S, y, pid = simulate(P=120, edge=0.8, seed=5)
    r = camt_test(S, y, pid, n_boot=500, seed=0, influence_fn=threshold_influence(np.median(S, axis=0)), warn=False)
    assert 0 <= r.p_value <= 1


def test_input_validation():
    S, y, pid = simulate()
    with pytest.raises(ValueError):
        camt_test(S[:, :1], y, pid)
    with pytest.raises(ValueError):
        camt_test(S, y + 2, pid)
    S_bad = S.copy()
    S_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        camt_test(S_bad, y, pid)
    with pytest.raises(ValueError):
        camt_test(S, np.zeros_like(y), pid)          # no positives


def test_warns_for_a_small_minority_class():
    S, y, pid = simulate(P=60, K=5)
    y = np.where(pid < 5, 1, 0)                           # five positive patients only
    with pytest.warns(UserWarning):
        camt_test(S, y, pid, n_boot=200, seed=0)


def test_no_warning_for_a_well_posed_problem():
    S, y, pid = simulate(P=200, K=3, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        camt_test(S, y, pid, n_boot=200, seed=0)


def test_design_effect_is_one_for_one_image_per_patient():
    S, y, pid = simulate(P=300, m=1)
    d = design_effect(S, y, pid)
    assert d["kish_cluster_size"] == pytest.approx(1.0)
    assert np.allclose(d["deff_auc"], 1.0, atol=0.25)


def test_audit_worked_example():
    a = audit_claim(0.0004, 40, 3900, 10, rho=0.1, pos_patients=28, neg_patients=12)
    assert a["m"] == pytest.approx(97.5)
    assert a["p_cluster"] > 0.0004 and a["p_bonferroni"] >= a["p_cluster"]
    tab = sensitivity_table(0.0004, 40, 3900, 10)
    assert all(tab[i]["p_cluster"] <= tab[i + 1]["p_cluster"] for i in range(len(tab) - 1))   # p rises with rho


def test_audit_two_sided_halves_p():
    one = audit_claim(0.01, 100, 100, 1)
    two = audit_claim(0.02, 100, 100, 1, sided="two")
    assert one["z_naive"] == pytest.approx(two["z_naive"])
