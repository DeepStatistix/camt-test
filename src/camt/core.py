"""Patient-clustered, selection-adjusted test of "the best of K models beats a baseline" (see the accompanying manuscript)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import t as tdist

__all__ = ["CamtResult", "camt_test", "auc_influence", "threshold_influence", "design_effect"]

MIN_MINORITY_PATIENTS = 20          # below this, with many candidates, the test is liberal (paper, Section 5.3)
MIN_EFFECTIVE_CLUSTERS = 100        # heuristic flag for very unequal cluster sizes (paper, Section 6.2); not a validated threshold


def auc_influence(scores, labels):
    """AUC of every column of ``scores`` and the per-image influence function (n x M). Ties receive average ranks."""
    S = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r_all = rankdata(S, axis=0)
    V = np.empty_like(S, dtype=float)
    V[pos] = (r_all[pos] - rankdata(S[pos], axis=0)) / n0
    V[~pos] = 1 - (r_all[~pos] - rankdata(S[~pos], axis=0)) / n1
    theta = V[pos].mean(0)
    infl = np.where(pos[:, None], (V - theta) / n1, (V - theta) / n0)
    return theta, infl


def threshold_influence(thresholds, kind="balanced"):
    """Influence-function builder for a metric at operating points fixed in advance (one threshold per model column).

    ``kind`` is "balanced" (mean of sensitivity and specificity), "sensitivity" or "specificity". The thresholds must not be
    estimated from the test sample; thresholds chosen on the test sample add a quantile term that is not covered.
    Pass the returned function as ``influence_fn`` to :func:`camt_test`.
    """
    thr = np.asarray(thresholds, dtype=float)

    def fn(S, y):
        pos = y == 1
        n1, n0 = pos.sum(), (~pos).sum()
        flag = S > thr[None, :]
        sens = flag[pos].mean(0)
        spec = (~flag[~pos]).mean(0)
        w1, w0 = {"balanced": (0.5, 0.5), "sensitivity": (1.0, 0.0), "specificity": (0.0, 1.0)}[kind]
        theta = w1 * sens + w0 * spec
        infl = np.zeros_like(S, dtype=float)
        infl[pos] = w1 * (flag[pos] - sens) / n1
        infl[~pos] = w0 * ((~flag[~pos]) - spec) / n0
        return theta, infl

    return fn


def _max_pvalue(G, se, z_obs, rng, n_boot, chunk=500):
    """P(max_k sum_c xi_c G_ck / se_k >= z_obs) for independent standard normal xi_c."""
    Gs = G / se
    hits, done = 0, 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        hits += ((rng.standard_normal((b, G.shape[0])) @ Gs).max(1) >= z_obs).sum()
        done += b
    return (hits + 1) / (n_boot + 1)


@dataclass
class CamtResult:
    """Result of :func:`camt_test`. All p-values are one-sided (H0: no candidate is better than the baseline)."""

    p_value: float                      # patient-clustered, adjusted for the number of candidates, small-class calibration
    p_value_pruned: float               # same, with clearly inferior candidates dropped from the null (pointwise valid only)
    p_naive: float                      # images treated as independent, best candidate, no adjustment for K
    p_clustering_only: float            # patient-clustered standard error, best candidate, no adjustment for K
    p_selection_only: float             # adjusted for K, but images treated as independent
    best: int                           # column index of the candidate with the largest clustered z-statistic
    auc: np.ndarray                     # AUC (or other metric) of every column
    baseline: int
    candidates: list                    # column indices of the candidates that entered the test
    difference: np.ndarray              # D_k for the candidates
    se_clustered: np.ndarray
    se_image: np.ndarray
    n_images: int
    n_patients: int
    n_positive_patients: int
    n_negative_patients: int
    n_mixed_patients: int
    kish_cluster_size: float
    effective_clusters: float
    se_ratio: float
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"best of {len(self.candidates)} candidates vs baseline (column {self.baseline}); best = column {self.best}",
                 f"  AUC best {self.auc[self.best]:.3f} vs baseline {self.auc[self.baseline]:.3f} "
                 f"(difference {self.auc[self.best] - self.auc[self.baseline]:+.3f})",
                 f"  {self.n_images} images, {self.n_patients} patients ({self.n_positive_patients} with positives, "
                 f"{self.n_negative_patients} with negatives, {self.n_mixed_patients} with both)",
                 f"  p (patient-clustered, adjusted for K) = {self.p_value:.4f}",
                 f"  p (pruned) = {self.p_value_pruned:.4f} | clustering only = {self.p_clustering_only:.4f} | "
                 f"selection only = {self.p_selection_only:.4f} | naive = {self.p_naive:.4f}"]
        lines += ["  WARNING: " + w for w in self.warnings]
        return "\n".join(lines)


def _check_inputs(scores, labels, patients, baseline):
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError("scores must be a 2-D array (images x models) with the baseline and at least one candidate")
    y = np.asarray(labels)
    if y.shape != (S.shape[0],) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be a 1-D array of 0/1 values, one per image")
    pid = np.asarray(patients)
    if pid.shape != (S.shape[0],):
        raise ValueError("patients must have one id per image")
    if not np.isfinite(S).all():
        raise ValueError("scores contain NaN or infinite values")
    if not 0 <= baseline < S.shape[1]:
        raise ValueError("baseline must be a column index of scores")
    return S, y.astype(int), pid


def camt_test(scores, labels, patients, baseline=0, n_boot=4000, seed=None, influence_fn=None, warn=True) -> CamtResult:
    """Test whether the best of K candidate models has a higher AUC than the baseline.

    Parameters
    ----------
    scores : array (n_images, 1 + K). One column per model; ``baseline`` is the index of the baseline column.
    labels : array (n_images,) of 0/1 image-level labels. Labels may vary within a patient.
    patients : array (n_images,) of patient identifiers (the clusters).
    n_boot : number of multiplier-bootstrap draws (at least 4,000 is recommended).
    seed : seed or ``numpy.random.Generator``.
    influence_fn : optional replacement for the AUC influence function, for example from :func:`threshold_influence`.
    """
    S, y, pid = _check_inputs(scores, labels, patients, baseline)
    rng = np.random.default_rng(seed)
    _, cid = np.unique(pid, return_inverse=True)
    P, n = cid.max() + 1, len(y)
    pos = y == 1
    has_pos = np.zeros(P, bool)
    has_pos[cid[pos]] = True
    has_neg = np.zeros(P, bool)
    has_neg[cid[~pos]] = True
    n1c, n0c = int(has_pos.sum()), int(has_neg.sum())
    mixed = int((has_pos & has_neg).sum())
    if min(n1c, n0c) < 2:
        raise ValueError("need at least 2 patients with images of each class")

    theta, infl = (influence_fn or auc_influence)(S, y)
    cand = [j for j in range(S.shape[1]) if j != baseline]
    D = theta[cand] - theta[baseline]
    ID = infl[:, cand] - infl[:, [baseline]]
    g_pos = np.zeros((P, len(cand)))
    g_neg = np.zeros((P, len(cand)))
    np.add.at(g_pos, cid[pos], ID[pos])
    np.add.at(g_neg, cid[~pos], ID[~pos])
    gs = np.sqrt(n1c / (n1c - 1)) * g_pos + np.sqrt(n0c / (n0c - 1)) * g_neg
    se_c = np.sqrt((gs ** 2).sum(0))
    se_i = np.sqrt(n / (n - 1) * (ID ** 2).sum(0))
    ok = (se_c > 1e-12) & (se_i > 1e-12)
    if not ok.any():
        raise ValueError("no candidate differs from the baseline (zero-variance differences)")
    if not ok.all():
        cand = [c for c, keep_ in zip(cand, ok) if keep_]
        D, ID, gs, se_c, se_i = D[ok], ID[:, ok], gs[:, ok], se_c[ok], se_i[ok]
    z_c, z_i = D / se_c, D / se_i
    ratio = tdist.ppf(0.95, max(min(n1c, n0c) - 1, 1)) / norm.ppf(0.95)       # small-class calibration

    p_sel = _max_pvalue(ID, se_i, z_i.max(), rng, n_boot)
    p_main = _max_pvalue(gs, se_c, z_c.max() / ratio, rng, n_boot)
    keep = z_c >= -np.sqrt(2 * np.log(np.log(max(P, 3))))
    p_pruned = _max_pvalue(gs[:, keep], se_c[keep], z_c.max() / ratio, rng, n_boot) if keep.any() else 1.0

    sizes = np.bincount(cid)
    kish = float((sizes ** 2).sum() / sizes.sum())
    eff = float(n / kish)
    notes = []
    if len(cand) > 1 and min(n1c, n0c) < MIN_MINORITY_PATIENTS:
        notes.append(f"only {min(n1c, n0c)} patients in the smaller class with {len(cand)} candidates: the test is liberal in this regime; "
                     "treat the result as exploratory")
    if eff < MIN_EFFECTIVE_CLUSTERS:
        notes.append(f"effective number of clusters is about {eff:.0f} (Kish cluster size {kish:.0f}); with very unequal cluster sizes the "
                     "test was liberal in the OCT example of the paper; treat the result with caution")
    if warn:
        for w in notes:
            warnings.warn(w, stacklevel=2)
    best = int(np.argmax(z_c))
    return CamtResult(
        p_value=float(p_main), p_value_pruned=float(p_pruned), p_naive=float(norm.sf(z_i.max())), p_clustering_only=float(norm.sf(z_c.max())),
        p_selection_only=float(p_sel), best=cand[best], auc=theta, baseline=baseline, candidates=cand, difference=D, se_clustered=se_c,
        se_image=se_i, n_images=n, n_patients=int(P), n_positive_patients=n1c, n_negative_patients=n0c, n_mixed_patients=mixed,
        kish_cluster_size=kish, effective_clusters=eff, se_ratio=float(np.mean(se_c / se_i)), warnings=notes)


def design_effect(scores, labels, patients, baseline=0):
    """Design effect and within-patient correlation of the AUC influence function.

    Returns a dict with the Kish cluster size ``m`` and, for the AUC of each model and for the difference of each model from the
    baseline, the design effect (clustered variance divided by image-level variance) and rho = (DEFF - 1) / (m - 1).
    """
    S, y, pid = _check_inputs(scores, labels, patients, baseline)
    _, cid = np.unique(pid, return_inverse=True)
    P, n = cid.max() + 1, len(y)
    sizes = np.bincount(cid)
    m = float((sizes ** 2).sum() / sizes.sum())
    pos = y == 1
    h1, h0 = np.zeros(P, bool), np.zeros(P, bool)
    h1[cid[pos]] = True
    h0[cid[~pos]] = True
    a1, a0 = h1.sum() / (h1.sum() - 1), h0.sum() / (h0.sum() - 1)
    _, infl = auc_influence(S, y)
    out = []
    others = [j for j in range(S.shape[1]) if j != baseline]
    for I in (infl, infl[:, others] - infl[:, [baseline]]):
        gp, gn = np.zeros((P, I.shape[1])), np.zeros((P, I.shape[1]))
        np.add.at(gp, cid[pos], I[pos])
        np.add.at(gn, cid[~pos], I[~pos])
        se_c2 = ((np.sqrt(a1) * gp + np.sqrt(a0) * gn) ** 2).sum(0)
        se_i2 = n / (n - 1) * (I ** 2).sum(0)
        out.append(se_c2 / se_i2)
    rho = lambda d: (d - 1) / (m - 1) if m > 1 else np.full_like(d, np.nan)
    return dict(kish_cluster_size=m, deff_auc=out[0], rho_auc=rho(out[0]), deff_difference=out[1], rho_difference=rho(out[1]),
                difference_columns=others)
