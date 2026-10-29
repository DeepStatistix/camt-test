"""Small data generator shared by the tests: patients with shared random effects, K + 1 models, model 1 better by ``edge``."""
import numpy as np


def simulate(P=80, m=4, K=5, edge=0.0, seed=0):
    rng = np.random.default_rng(seed)
    yp = (rng.random(P) < 0.5).astype(int)
    pid = np.repeat(np.arange(P), m)
    y = yp[pid]
    u = rng.normal(0, 0.6, P)[pid][:, None]
    v = rng.normal(0, 0.6, (P, K + 1))[pid]
    w = rng.normal(0, 0.5, len(y))[:, None]
    e = rng.normal(0, 0.8, (len(y), K + 1))
    mu = 1.5 + np.r_[0.0, edge, np.zeros(K - 1)]
    return mu[None, :] * y[:, None] + u + v + w + e, y, pid
