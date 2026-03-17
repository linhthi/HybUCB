"""
plotting.py
===========
All plotting functions for reproducing paper figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    "HybUCB-AR":               "#1f77b4",
    "HybUCB-AR (no offline)":  "#ff7f0e",
    "RUCB":                    "#2ca02c",
    "InterleavedFilter":       "#d62728",
    "HybElimUCB-RA":           "#1f77b4",
    "HybElimUCB-RA (no offline)": "#ff7f0e",
    "ETC":                     "#9467bd",
    "UCB":                     "#2ca02c",
    "ThompsonSampling":        "#d62728",
}

LINESTYLES = {
    "HybUCB-AR":               "-",
    "HybUCB-AR (no offline)":  "--",
    "RUCB":                    "-.",
    "InterleavedFilter":       ":",
    "HybElimUCB-RA":           "-",
    "HybElimUCB-RA (no offline)": "--",
    "ETC":                     "-.",
    "UCB":                     ":",
    "ThompsonSampling":        (0, (3, 1, 1, 1)),
}

ALPHA_SHADE = 0.2


def _plot_one_panel(ax, results_dict, T, title, xlabel="Round",
                    ylabel="Cumulative Regret"):
    """Plot all algorithms onto a single axis."""
    rounds = np.arange(1, T + 1)
    for name, (mean, se) in results_dict.items():
        color = COLORS.get(name, "gray")
        ls = LINESTYLES.get(name, "-")
        ax.plot(rounds, mean, label=name, color=color, linestyle=ls, linewidth=1.5)
        ax.fill_between(rounds, mean - se, mean + se,
                        color=color, alpha=ALPHA_SHADE)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 2  –  HybUCB-AR scalability
# ---------------------------------------------------------------------------

def plot_fig2_hybucbar_scalability(results, K_list, T, save_path):
    fig, axes = plt.subplots(1, len(K_list), figsize=(4 * len(K_list), 3.5))
    fig.suptitle("HybUCB-AR: Scalability (Synthetic)", fontsize=12, fontweight="bold")

    for ax, K in zip(axes, K_list):
        _plot_one_panel(ax, results[K], T, title=f"K={K}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 3  –  HybUCB-AR sensitivity
# ---------------------------------------------------------------------------

def plot_fig3_hybucbar_sensitivity(results, T, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.suptitle("HybUCB-AR: Parameter Sensitivity", fontsize=12, fontweight="bold")

    # Ni
    ax = axes[0]
    ax.set_title("Varying $N_i$", fontsize=10)
    rounds = np.arange(1, T + 1)
    for Ni, (mean, se) in results["Ni"].items():
        ax.plot(rounds, mean, label=f"$N_i$={Ni}", linewidth=1.5)
        ax.fill_between(rounds, mean - se, mean + se, alpha=ALPHA_SHADE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative Regret")

    # Delta
    ax = axes[1]
    ax.set_title("Varying $\\Delta$", fontsize=10)
    for d, (mean, se) in results["delta"].items():
        ax.plot(rounds, mean, label=f"$\\Delta$={d}", linewidth=1.5)
        ax.fill_between(rounds, mean - se, mean + se, alpha=ALPHA_SHADE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative Regret")

    # Vi
    ax = axes[2]
    ax.set_title("Varying $V_i$", fontsize=10)
    for vi, (mean, se) in results["Vi"].items():
        ax.plot(rounds, mean, label=f"$V_i$={vi}", linewidth=1.5)
        ax.fill_between(rounds, mean - se, mean + se, alpha=ALPHA_SHADE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative Regret")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4  –  HybElimUCB-RA scalability
# ---------------------------------------------------------------------------

def plot_fig4_hybelimucbra_scalability(results, K_list, T, save_path):
    fig, axes = plt.subplots(1, len(K_list), figsize=(4 * len(K_list), 3.5))
    fig.suptitle("HybElimUCB-RA: Scalability (Synthetic)", fontsize=12, fontweight="bold")

    for ax, K in zip(axes, K_list):
        _plot_one_panel(ax, results[K], T, title=f"K={K}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 5  –  HybElimUCB-RA sensitivity
# ---------------------------------------------------------------------------

def plot_fig5_hybelimucbra_sensitivity(results, T, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.suptitle("HybElimUCB-RA: Parameter Sensitivity", fontsize=12, fontweight="bold")

    rounds = np.arange(1, T + 1)
    for ax, key, label_prefix in zip(
            axes,
            ["Ni", "delta", "Vi"],
            ["$N_i$", "$\\Delta$", "$V_i$"]):
        for val, (mean, se) in results[key].items():
            ax.plot(rounds, mean, label=f"{label_prefix}={val}", linewidth=1.5)
            ax.fill_between(rounds, mean - se, mean + se, alpha=ALPHA_SHADE)
        ax.set_title(f"Varying {label_prefix}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Round"); ax.set_ylabel("Cumulative Regret")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 1  –  Main paper comparison (synthetic, K=8)
# ---------------------------------------------------------------------------

def plot_fig1_main(results_ar, results_ra, T_ar, T_ra,
                   title_ar="Synthetic Data (HybUCB-AR)",
                   title_ra="Synthetic Data (HybElimUCB-RA)",
                   save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 1: Cumulative Regret Comparison (Synthetic)",
                 fontsize=12, fontweight="bold")

    _plot_one_panel(axes[0], results_ar, T_ar, title=title_ar)
    _plot_one_panel(axes[1], results_ra, T_ra, title=title_ra)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Real-world comparison  (Figures 6 & 7)
# ---------------------------------------------------------------------------

def plot_real_world(results_ar, results_ra, T_ar, T_ra, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Real-World Experiments (MovieLens / Yelp)",
                 fontsize=12, fontweight="bold")

    _plot_one_panel(axes[0], results_ar, T_ar,
                    title="HybUCB-AR (MovieLens/Yelp)")
    _plot_one_panel(axes[1], results_ra, T_ra,
                    title="HybElimUCB-RA (MovieLens/Yelp)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Utility: combine two datasets on one figure (e.g. MovieLens + Yelp)
# ---------------------------------------------------------------------------

def plot_two_datasets(res1, res2, T, label1="MovieLens", label2="Yelp",
                      algo_label="HybUCB-AR", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{algo_label}: Real-World Datasets", fontsize=12, fontweight="bold")

    _plot_one_panel(axes[0], res1, T, title=f"{algo_label} ({label1})")
    _plot_one_panel(axes[1], res2, T, title=f"{algo_label} ({label2})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()
