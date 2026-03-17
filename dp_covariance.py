"""
Reproducing: "Differentially Private Covariance Estimation"
Experiments from Section 4: compare Algorithm 1 vs Laplace / Gaussian / KT baselines
and 2022 Trace/Tail-Sensitive algorithms on multiple datasets.

Run:
    python dp_covariance.py [--sampler-mode {simple,acg,auto}]

Package layout (dp_cov/):
    data.py        — dataset loading and preprocessing
    core.py        — covariance math, evaluation metrics, eigenvector samplers
    mechanisms.py  — all DP algorithms (Laplace, Gaussian, KT, Alg1, 2022)
    experiments.py — experiment runner functions
    plots.py       — visualization utilities
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from dp_cov.core import set_sampler_mode, true_covariance
from dp_cov.data import (
    load_adult_appendix,
    load_airfoil_appendix,
    load_breast_cancer_appendix,
    load_california_housing_appendix,
    load_digits_appendix,
    load_volkert_appendix,
    load_wine_appendix,
)
from dp_cov.experiments import run_experiment_vary_eps_gaussian_grid
from dp_cov.plots import plot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DP covariance experiments.")
    parser.add_argument(
        "--sampler-mode",
        choices=["simple", "acg", "auto"],
        default="auto",
        help="Eigenvector sampler for KT/Alg1: Monte Carlo simple, ACG-style, or auto.",
    )
    args = parser.parse_args()
    set_sampler_mode(args.sampler_mode)
    import dp_cov.core as _core

    _core._SAMPLER_DIAG = True

    print("Loading appendix-aligned datasets...")
    print(f"Sampler mode: {args.sampler_mode}")
    X_wine          = load_wine_appendix()
    X_airfoil       = load_airfoil_appendix()
    X_breast_cancer = load_breast_cancer_appendix()
    X_digits        = load_digits_appendix()
    X_california    = load_california_housing_appendix()
    X_volkert       = load_volkert_appendix()
    X_adult         = load_adult_appendix()

    print(f"Wine:             {X_wine.shape}")
    print(f"Airfoil:          {X_airfoil.shape}")
    print(f"Breast Cancer:    {X_breast_cancer.shape}")
    print(f"Digits:           {X_digits.shape}")
    print(f"California:       {X_california.shape}")
    print(f"Volkert:          {X_volkert.shape}")

    DELTA          = 1e-5
    N_TRIALS       = 50
    ADULT_TRIALS   = 50
    EPSILONS       = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    GAUSSIAN_DELTAS = [1e-16, 1e-10, 1e-3]

    datasets = [
        ("Airfoil",       X_airfoil),
        ("California",    X_california),
        ("Wine",          X_wine),
        ("Breast Cancer", X_breast_cancer),
        ("Digits",        X_digits),
        ("Adult",         X_adult),
        ("Volkert",       X_volkert),
    ]

    # Row 0: L/KT/AD panels; Row 1: G-delta/AD/2022 panels; one column per dataset
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 9))
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)

    json_output = {"epsilons": EPSILONS, "gaussian_deltas": GAUSSIAN_DELTAS, "datasets": {}}

    for col, (name, X_full) in enumerate(datasets):
        k       = X_full.shape[0]
        n_use   = X_full.shape[1]
        n_trials = ADULT_TRIALS if name == "Adult" else N_TRIALS

        # Diagnostic: λ₁(C)·log(λ₁(C)) vs log(1/δ)/ε threshold (paper Sec 3.1)
        C_diag = true_covariance(X_full)
        lam1 = float(eigh(C_diag, eigvals_only=True)[-1])
        lam1_logterm = lam1 * np.log(max(lam1, 1e-12))
        thresholds = [np.log(1.0 / DELTA) / eps for eps in EPSILONS]
        beats_gaussian = sum(1 for t in thresholds if lam1_logterm <= t)
        threshold_min, threshold_max = min(thresholds), max(thresholds)
        relation_all  = lam1_logterm <= threshold_min
        relation_some = threshold_min < lam1_logterm <= threshold_max
        print(f"\n[{name}] d={X_full.shape[0]}, n={n_use}, k={k}")
        print(f"  λ₁(C)={lam1:.4f},  λ₁(C)·log(λ₁(C))={lam1_logterm:.4f}")
        print(f"  log(1/δ)/ε range: [{threshold_min:.2f}, {threshold_max:.2f}]")
        if relation_all:
            print("  Condition status: satisfied for all tested ε values.")
        elif relation_some:
            print(f"  Condition status: satisfied for {beats_gaussian}/{len(EPSILONS)} tested ε values.")
        else:
            print("  Condition status: not satisfied for any tested ε value.")

        res_left, res_right, timing = run_experiment_vary_eps_gaussian_grid(
            X_full, EPSILONS, n_use, GAUSSIAN_DELTAS, k, n_trials=n_trials,
        )

        plot_results(EPSILONS, res_left,  "epsilon (privacy parameter)",
                     f"{name}: L / KT / AD",      axes[0, col], d=X_full.shape[0], n=n_use)
        plot_results(EPSILONS, res_right, "epsilon (privacy parameter)",
                     f"{name}: G-delta / AD / 2022", axes[1, col], d=X_full.shape[0], n=n_use)

        print(f"\n  [{name}] Total time (seconds):")
        for alg, t in timing.items():
            print(f"    {alg:12s}: {t:.2f}s")

        json_output["datasets"][name] = {
            "shape":    list(X_full.shape),
            "k":        k,
            "n_trials": n_trials,
            "results_left":  {alg: [float(v) for v in vals] for alg, vals in res_left.items()},
            "results_right": {alg: [float(v) for v in vals] for alg, vals in res_right.items()},
            "timing_seconds_total": {alg: float(t) for alg, t in timing.items()},
        }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print("\nSaved results.json")

    plt.suptitle("DP Covariance Estimation", fontsize=13)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Saved results.png")
    plt.show()
