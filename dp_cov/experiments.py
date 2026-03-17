"""Experiment runners for DP covariance estimation comparisons."""

import time

import numpy as np

from .core import true_covariance, frobenius_error
from .data import subsample, _show_progress
from .mechanisms import (
    dp_laplace,
    dp_gaussian,
    dp_kt,
    dp_algorithm1_uniform,
    dp_algorithm1_rank_k,
    dp_trace_algo_2022,
    dp_tail_algo_2022,
)


def run_experiment_vary_n(X_full, ns, epsilon, delta, k, n_trials=5, seed=42):
    results = {alg: [] for alg in ["Laplace", "Gaussian", "KT", "Alg1"]}
    for n in ns:
        errs = {alg: [] for alg in results}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            X = subsample(X_full, n, rng)
            C_true = true_covariance(X)
            errs["Laplace"].append(frobenius_error(C_true, dp_laplace(X, epsilon, k, rng), X.shape[1]))
            errs["Gaussian"].append(frobenius_error(C_true, dp_gaussian(X, epsilon, delta, k, rng), X.shape[1]))
            errs["KT"].append(frobenius_error(C_true, dp_kt(X, epsilon, delta, k, rng), X.shape[1]))
            errs["Alg1"].append(frobenius_error(C_true, dp_algorithm1_rank_k(X, epsilon, delta, k, rng=rng), X.shape[1]))
        for alg in results:
            results[alg].append(np.mean(errs[alg]))
        print(f"  n={n}: done")
    return results


def run_experiment_vary_eps(X_full, epsilons, n, delta, k, n_trials=5, seed=42):
    results = {alg: [] for alg in ["Laplace", "Gaussian", "KT", "Alg1"]}
    for epsilon in epsilons:
        errs = {alg: [] for alg in results}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            X = subsample(X_full, n, rng)
            C_true = true_covariance(X)
            errs["Laplace"].append(frobenius_error(C_true, dp_laplace(X, epsilon, k, rng), X.shape[1]))
            errs["Gaussian"].append(frobenius_error(C_true, dp_gaussian(X, epsilon, delta, k, rng), X.shape[1]))
            errs["KT"].append(frobenius_error(C_true, dp_kt(X, epsilon, delta, k, rng), X.shape[1]))
            errs["Alg1"].append(frobenius_error(C_true, dp_algorithm1_rank_k(X, epsilon, delta, k, rng=rng), X.shape[1]))
        for alg in results:
            results[alg].append(np.mean(errs[alg]))
        print(f"  eps={epsilon}: done")
    return results


def run_experiment_vary_eps_gaussian_grid(
    X_full,
    epsilons,
    n,
    gaussian_deltas,
    k,
    n_trials=5,
    seed=42,
):
    """
    Supplement-style comparison:
    panel (a) uses Laplace / KT / IT-U / Alg1 (AD),
    panel (b) compares Alg1 against several Gaussian deltas and 2022 algorithms.

    Returns results_a, results_b, timing where timing maps algorithm name
    to total wall-clock seconds across all calls.
    """
    results_a = {alg: [] for alg in ["Laplace", "KT", "ITU", "Alg1"]}
    results_b = {f"G-{int(np.log10(delta))}": [] for delta in gaussian_deltas}
    results_b["Alg1"] = []
    results_b["Trace2022"] = []
    results_b["Tail2022"] = []

    time_totals = {alg: 0.0 for alg in ["Laplace", "Gaussian", "KT", "ITU", "Alg1", "Trace2022", "Tail2022"]}
    time_counts = {alg: 0 for alg in time_totals}

    total_runs = len(epsilons) * n_trials
    completed_runs = 0

    for eps_idx, epsilon in enumerate(epsilons, start=1):
        errs_a = {alg: [] for alg in results_a}
        errs_b = {alg: [] for alg in results_b}

        for trial in range(n_trials):
            _show_progress(
                completed_runs,
                total_runs,
                f"Experiment: epsilon {eps_idx}/{len(epsilons)}, trial {trial + 1}/{n_trials}",
            )
            rng = np.random.default_rng(seed + trial)
            X = subsample(X_full, n, rng)
            C_true = true_covariance(X)

            t0 = time.perf_counter()
            lap_est = dp_laplace(X, epsilon, k, rng)
            time_totals["Laplace"] += time.perf_counter() - t0
            time_counts["Laplace"] += 1
            errs_a["Laplace"].append(frobenius_error(C_true, lap_est, X.shape[1]))

            t0 = time.perf_counter()
            kt_est = dp_kt(X, epsilon, 1e-5, k, rng)
            time_totals["KT"] += time.perf_counter() - t0
            time_counts["KT"] += 1
            errs_a["KT"].append(frobenius_error(C_true, kt_est, X.shape[1]))

            t0 = time.perf_counter()
            itu_est = dp_algorithm1_uniform(X, epsilon, 1e-5, k, rng=rng)
            time_totals["ITU"] += time.perf_counter() - t0
            time_counts["ITU"] += 1
            errs_a["ITU"].append(frobenius_error(C_true, itu_est, X.shape[1]))

            t0 = time.perf_counter()
            alg1_est = dp_algorithm1_rank_k(X, epsilon, 1e-5, k, rng=rng)
            time_totals["Alg1"] += time.perf_counter() - t0
            time_counts["Alg1"] += 1
            errs_a["Alg1"].append(frobenius_error(C_true, alg1_est, X.shape[1]))

            for delta in gaussian_deltas:
                label = f"G-{int(np.log10(delta))}"
                t0 = time.perf_counter()
                g_est = dp_gaussian(X, epsilon, delta, k, rng)
                time_totals["Gaussian"] += time.perf_counter() - t0
                time_counts["Gaussian"] += 1
                errs_b[label].append(frobenius_error(C_true, g_est, X.shape[1]))

            errs_b["Alg1"].append(frobenius_error(C_true,
                dp_algorithm1_rank_k(X, epsilon, 1e-5, k, rng=rng), X.shape[1]))

            t0 = time.perf_counter()
            trace_est = dp_trace_algo_2022(X, epsilon, 1e-5, k, rng=rng)
            time_totals["Trace2022"] += time.perf_counter() - t0
            time_counts["Trace2022"] += 1
            errs_b["Trace2022"].append(frobenius_error(C_true, trace_est, X.shape[1]))

            t0 = time.perf_counter()
            tail_est = dp_tail_algo_2022(X, epsilon, 1e-5, k, rng=rng)
            time_totals["Tail2022"] += time.perf_counter() - t0
            time_counts["Tail2022"] += 1
            errs_b["Tail2022"].append(frobenius_error(C_true, tail_est, X.shape[1]))

            completed_runs += 1

        for alg in results_a:
            results_a[alg].append(np.mean(errs_a[alg]))
        for alg in results_b:
            results_b[alg].append(np.mean(errs_b[alg]))
        _show_progress(
            completed_runs,
            total_runs,
            f"Experiment: epsilon {eps_idx}/{len(epsilons)} complete",
        )
        print(f"  eps={epsilon}: done")

    timing = dict(time_totals)
    return results_a, results_b, timing
