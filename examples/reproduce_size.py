"""Simulation of the size of the test: all models are equal, so every rejection is a false positive.

Prints the false-positive rate (nominal level 5%) of the naive, clustering-only, selection-only and proposed procedures for
K = 1, 5 and 20 candidates, with 100 patients and 5 images per patient.

    python examples/reproduce_size.py              # 1,000 simulations per row, a few minutes
    python examples/reproduce_size.py --reps 200   # quick check (Monte Carlo SE about 1.5 points)
"""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from bestofk import best_of_k_test  # noqa: E402


def simulate(P, m, K, seed):
    rng = np.random.default_rng(seed)
    yp = (rng.random(P) < 0.5).astype(int)
    while not 2 <= yp.sum() <= P - 2:
        yp = (rng.random(P) < 0.5).astype(int)
    pid = np.repeat(np.arange(P), m)
    y = yp[pid]
    u = rng.normal(0, 0.6, P)[pid][:, None]
    v = rng.normal(0, 0.6, (P, K + 1))[pid]
    w = rng.normal(0, 0.5, len(y))[:, None]
    e = rng.normal(0, 0.8, (len(y), K + 1))
    return 1.5 * y[:, None] + u + v + w + e, y, pid


def one(args):
    P, m, K, boot, seed = args
    S, y, pid = simulate(P, m, K, seed)
    r = best_of_k_test(S, y, pid, n_boot=boot, seed=seed, warn=False)
    return r.p_naive < 0.05, r.p_clustering_only < 0.05, r.p_selection_only < 0.05, r.p_value < 0.05


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--patients", type=int, default=100)
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    print(f"{a.patients} patients, {a.images} images per patient, {a.reps} simulations per row, nominal level 5%")
    print(f"{'K':>4s} {'naive':>8s} {'clustering':>11s} {'selection':>10s} {'proposed':>9s}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for K in (1, 5, 20):
            out = np.array(list(ex.map(one, [(a.patients, a.images, K, a.boot, 10_000 * K + s) for s in range(a.reps)], chunksize=10)))
            rate = out.mean(0)
            print(f"{K:4d} {rate[0]:8.1%} {rate[1]:11.1%} {rate[2]:10.1%} {rate[3]:9.1%}")
    print(f"Monte Carlo standard error near 5%: {np.sqrt(0.05 * 0.95 / a.reps):.1%}")
