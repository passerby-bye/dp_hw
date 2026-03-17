"""Plotting utilities for DP covariance experiment results."""

import matplotlib.pyplot as plt


COLORS = {
    "Laplace":    "tab:blue",
    "Gaussian":   "tab:orange",
    "KT":         "tab:green",
    "ITU":        "tab:purple",
    "Alg1":       "tab:red",
    "Trace2022":  "tab:cyan",
    "Tail2022":   "tab:brown",
    "G--16":      "#c17d11",
    "G--10":      "#e69f00",
    "G--3":       "#f4c542",
}

LABELS = {
    "Laplace":    "Laplace",
    "Gaussian":   "Gaussian",
    "KT":         "KT",
    "ITU":        "Algorithm 1 (IT-U)",
    "Alg1":       "Algorithm 1 (AD)",
    "Trace2022":  "Trace-Sensitive 2022",
    "Tail2022":   "Tail-Sensitive 2022",
    "G--16":      "G-16",
    "G--10":      "G-10",
    "G--3":       "G-3",
}

MARKERS = {
    "Laplace":   "o",
    "Gaussian":  "s",
    "KT":        "^",
    "ITU":       "v",
    "Alg1":      "D",
    "Trace2022": "P",
    "Tail2022":  "X",
    "G--16":     "s",
    "G--10":     "s",
    "G--3":      "s",
}


def plot_results(xs, results, xlabel, title, ax, d=None, n=None):
    for alg, vals in results.items():
        label = LABELS.get(alg, alg)
        color = COLORS.get(alg, None)
        marker = MARKERS.get(alg, "o")
        ax.plot(
            xs,
            vals,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Normalized Frobenius Error", fontsize=11)
    if d is not None and n is not None:
        title = f"{title}\n(d={d}, n={n})"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
