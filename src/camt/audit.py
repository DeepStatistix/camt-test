"""Summary-statistics audit of a published "best of K models beats the baseline" claim (see the accompanying manuscript).

Inputs are the winner's naive one-sided p-value, the numbers of patients and images, and the number of models tried, K.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats import t as tdist

__all__ = ["audit_claim", "sensitivity_table", "hanley_mcneil_se", "rho_from_patient_level_ci"]


def audit_claim(p_naive, patients, images, K, rho=0.2, pos_patients=None, neg_patients=None, sided="one", alpha=0.05, image_sq_sum=None):
    """Adjust a published p-value for clustering (given rho) and for the number of models tried.

    ``sided="two"`` halves a two-sided p-value. ``image_sq_sum`` is the sum of squared images per patient if known (unequal
    cluster sizes); otherwise equal cluster sizes are assumed. Passing the numbers of positive and negative patients switches
    on the small-class t-calibration.
    """
    z = norm.isf(p_naive if sided == "one" else p_naive / 2)
    m = images / patients if image_sq_sum is None else image_sq_sum / images
    nu = None if pos_patients is None else max(min(pos_patients, neg_patients) - 1, 1)
    sf = (lambda x: tdist.sf(x, nu)) if nu else norm.sf
    crit = tdist.isf(alpha, nu) if nu else norm.isf(alpha)
    deff = 1 + (m - 1) * rho
    p_c = float(sf(z / np.sqrt(deff)))
    rho_star = max(((z / crit) ** 2 - 1) / (m - 1), 0.0) if m > 1 else np.inf
    return dict(z_naive=float(z), m=float(m), deff=float(deff), p_cluster=p_c, p_bonferroni=min(1.0, K * p_c), p_independent=1 - (1 - p_c) ** K,
                rho_star=float(rho_star), K_star_bonferroni=int(alpha // p_c) if p_c > 0 else np.inf,
                K_star_independent=int(np.log(1 - alpha) // np.log(1 - p_c)) if 0 < p_c < alpha else (0 if p_c >= alpha else np.inf))


def sensitivity_table(p_naive, patients, images, K, rhos=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), **kw):
    """List of :func:`audit_claim` results over a range of within-patient correlations."""
    return [dict(rho=r, **audit_claim(p_naive, patients, images, K, rho=r, **kw)) for r in rhos]


def hanley_mcneil_se(auc, n_pos_images, n_neg_images):
    """Image-level (independence) standard error of a single AUC from summary numbers (Hanley and McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos_images - 1) * (q1 - auc ** 2) + (n_neg_images - 1) * (q2 - auc ** 2)) / (n_pos_images * n_neg_images)
    return float(np.sqrt(var))


def rho_from_patient_level_ci(auc, ci_low, ci_high, n_pos_images, n_neg_images, patients, images, image_sq_sum=None):
    """Within-patient correlation of a SINGLE model's AUC implied by a reported patient-level 95% interval.

    On BreakHis this overstates the correlation by about 1.5-fold, because the Hanley-McNeil standard error is smaller than
    the DeLong one, and the correlation for a paired difference is usually lower than that of a single AUC. Use it as a loose
    upper bound, not as an estimate.
    """
    se_rep = (ci_high - ci_low) / (2 * 1.96)
    deff = (se_rep / hanley_mcneil_se(auc, n_pos_images, n_neg_images)) ** 2
    m = images / patients if image_sq_sum is None else image_sq_sum / images
    return float(max((deff - 1) / (m - 1), 0.0)), float(deff)
