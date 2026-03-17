"""
main.py
=======
Main experiment runner.
Reproduces all figures from the paper:

  "Learning Across the Gap: Hybrid Multi-armed Bandits with
   Heterogeneous Offline and Online Data"

Usage
-----
  # Fast demo (reduced rounds/trials):
  python main.py --mode demo

  # Full paper experiments (slow, matches paper settings):
  python main.py --mode full

  # Single algorithm group:
  python main.py --mode full --exp hybucbar_scalability
  python main.py --mode full --exp hybelimucbra_scalability
  python main.py --mode full --exp sensitivity
  python main.py --mode full --exp figure1
  python main.py --mode full --exp realworld

  # Provide MovieLens / Yelp data paths (optional):
  python main.py --mode full --movielens /path/to/ratings.csv --yelp /path/to/review.json
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from experiments_synthetic import (
    exp_hybucbar_scalability,
    exp_hybucbar_sensitivity,
    exp_hybelimucbra_scalability,
    exp_hybelimucbra_sensitivity,
    exp_figure1_synthetic,
)
from experiments_realworld import (
    load_movielens, load_yelp, make_synthetic_ratings,
    exp_realworld_hybucbar, exp_realworld_hybelimucbra,
)
from plotting import (
    plot_fig2_hybucbar_scalability,
    plot_fig3_hybucbar_sensitivity,
    plot_fig4_hybelimucbra_scalability,
    plot_fig5_hybelimucbra_sensitivity,
    plot_fig1_main,
    plot_two_datasets,
)

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)


# ===========================================================================
# Parameter sets
# ===========================================================================

PARAMS = {
    "demo": {
        # Very fast, for testing
        "n_trials": 5,
        "T_hybucbar": 3_000,
        "T_hybelimucbra": 2_000,
        "T_real": 500,
        "K_list": [8],
        "N_offline": 100,
        "N_per_pair": 100,
        "N_offline_abs_real": 5_000,
    },
    "medium": {
        # Balanced speed vs accuracy
        "n_trials": 20,
        "T_hybucbar": 10_000,
        "T_hybelimucbra": 8_000,
        "T_real": 2_000,
        "K_list": [8, 16],
        "N_offline": 300,
        "N_per_pair": 300,
        "N_offline_abs_real": 20_000,
    },
    "full": {
        # Paper settings
        "n_trials": 100,
        "T_hybucbar": 50_000,
        "T_hybelimucbra": 30_000,
        "T_real": 10_000,
        "K_list": [8, 16, 24, 32],
        "N_offline": 500,
        "N_per_pair": 1_000,
        "N_offline_abs_real": 100_000,
    },
}


# ===========================================================================
# Experiment runners
# ===========================================================================

def run_hybucbar_scalability(p, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: HybUCB-AR Scalability  (Figure 2)")
    print("=" * 60)
    results = exp_hybucbar_scalability(
        K_list=p["K_list"], T=p["T_hybucbar"],
        delta=0.1, bias=0.1, N_offline=p["N_offline"],
        n_trials=p["n_trials"], seed=seed,
    )
    out = os.path.join(OUTDIR, "fig2_hybucbar_scalability.png")
    plot_fig2_hybucbar_scalability(results, p["K_list"], p["T_hybucbar"], out)
    return results


def run_hybucbar_sensitivity(p, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: HybUCB-AR Sensitivity  (Figure 3)")
    print("=" * 60)
    results = exp_hybucbar_sensitivity(
        K=20, T=30_000 if p["n_trials"] >= 50 else p["T_hybucbar"],
        n_trials=p["n_trials"], seed=seed,
    )
    out = os.path.join(OUTDIR, "fig3_hybucbar_sensitivity.png")
    T_used = 30_000 if p["n_trials"] >= 50 else p["T_hybucbar"]
    plot_fig3_hybucbar_sensitivity(results, T_used, out)
    return results


def run_hybelimucbra_scalability(p, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: HybElimUCB-RA Scalability  (Figure 4)")
    print("=" * 60)
    results = exp_hybelimucbra_scalability(
        K_list=p["K_list"], T=p["T_hybelimucbra"],
        delta=0.1, bias=0.01, N_offline=p["N_offline"],
        n_trials=p["n_trials"], seed=seed,
    )
    out = os.path.join(OUTDIR, "fig4_hybelimucbra_scalability.png")
    plot_fig4_hybelimucbra_scalability(results, p["K_list"], p["T_hybelimucbra"], out)
    return results


def run_hybelimucbra_sensitivity(p, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: HybElimUCB-RA Sensitivity  (Figure 5)")
    print("=" * 60)
    results = exp_hybelimucbra_sensitivity(
        K=10, T=25_000 if p["n_trials"] >= 50 else p["T_hybelimucbra"],
        n_trials=p["n_trials"], seed=seed,
    )
    out = os.path.join(OUTDIR, "fig5_hybelimucbra_sensitivity.png")
    T_used = 25_000 if p["n_trials"] >= 50 else p["T_hybelimucbra"]
    plot_fig5_hybelimucbra_sensitivity(results, T_used, out)
    return results


def run_figure1(p, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: Figure 1 (Main Paper)  –  K=8 synthetic")
    print("=" * 60)
    results, T_ar, T_ra = exp_figure1_synthetic(
        K=8, seed=seed,
        T_ar=p["T_hybucbar"], T_ra=p["T_hybelimucbra"],
        delta=0.1, bias=0.1, N_offline=p["N_offline"],
        n_trials=p["n_trials"],
    )
    out = os.path.join(OUTDIR, "fig1_main_synthetic.png")
    plot_fig1_main(
        results["HybUCB-AR"], results["HybElimUCB-RA"],
        T_ar, T_ra,
        title_ar="HybUCB-AR (Synthetic, K=8)",
        title_ra="HybElimUCB-RA (Synthetic, K=8)",
        save_path=out,
    )
    return results


def run_realworld(p, movielens_path=None, yelp_path=None, seed=0):
    print("\n" + "=" * 60)
    print("Experiment: Real-World  (Figures 6 & 7)")
    print("=" * 60)

    # --- Load or substitute datasets ---
    if movielens_path:
        arms_ml = load_movielens(movielens_path, K=10, seed=seed)
    else:
        arms_ml = None

    if yelp_path:
        arms_yelp = load_yelp(yelp_path, K=10, seed=seed + 1)
    else:
        arms_yelp = None

    if arms_ml is None:
        print("  MovieLens data not found – using synthetic substitute.")
        arms_ml = make_synthetic_ratings(K=10, n_ratings_per_arm=5000, seed=seed)
    if arms_yelp is None:
        print("  Yelp data not found – using synthetic substitute.")
        arms_yelp = make_synthetic_ratings(K=10, n_ratings_per_arm=5000, seed=seed + 100)

    T = p["T_real"]
    n_trials = p["n_trials"]

    # --- HybUCB-AR ---
    print("\n  HybUCB-AR:")
    res_ar_ml = exp_realworld_hybucbar(
        arms_ml, K=10, T=T, n_trials=n_trials, seed=seed,
        N_offline_abs=p["N_offline_abs_real"])
    res_ar_yelp = exp_realworld_hybucbar(
        arms_yelp, K=10, T=T, n_trials=n_trials, seed=seed + 10,
        N_offline_abs=p["N_offline_abs_real"])

    out_ar = os.path.join(OUTDIR, "fig6_hybucbar_realworld.png")
    plot_two_datasets(res_ar_ml, res_ar_yelp, T,
                      label1="MovieLens", label2="Yelp",
                      algo_label="HybUCB-AR", save_path=out_ar)

    # --- HybElimUCB-RA ---
    print("\n  HybElimUCB-RA:")
    res_ra_ml = exp_realworld_hybelimucbra(
        arms_ml, K=10, T=T, n_trials=n_trials, seed=seed,
        N_per_pair=p["N_per_pair"])
    res_ra_yelp = exp_realworld_hybelimucbra(
        arms_yelp, K=10, T=T, n_trials=n_trials, seed=seed + 10,
        N_per_pair=p["N_per_pair"])

    out_ra = os.path.join(OUTDIR, "fig7_hybelimucbra_realworld.png")
    plot_two_datasets(res_ra_ml, res_ra_yelp, T,
                      label1="MovieLens", label2="Yelp",
                      algo_label="HybElimUCB-RA", save_path=out_ra)

    return {
        "ar_ml": res_ar_ml, "ar_yelp": res_ar_yelp,
        "ra_ml": res_ra_ml, "ra_yelp": res_ra_yelp,
    }


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Learning Across the Gap'")
    parser.add_argument("--mode", choices=["demo", "medium", "full"],
                        default="demo",
                        help="Experiment scale (demo=fast, full=paper-accurate)")
    parser.add_argument("--exp", default="all",
                        choices=["all", "hybucbar_scalability",
                                 "hybelimucbra_scalability",
                                 "sensitivity", "figure1", "realworld"],
                        help="Which experiment to run")
    parser.add_argument("--movielens", default=None,
                        help="Path to MovieLens ratings.csv")
    parser.add_argument("--yelp", default=None,
                        help="Path to Yelp review.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    p = PARAMS[args.mode]
    seed = args.seed

    print(f"\nRunning in [{args.mode}] mode  (n_trials={p['n_trials']})")
    print(f"Outputs will be saved to: {os.path.abspath(OUTDIR)}\n")

    if args.exp in ("all", "hybucbar_scalability"):
        run_hybucbar_scalability(p, seed=seed)

    if args.exp in ("all", "hybelimucbra_scalability"):
        run_hybelimucbra_scalability(p, seed=seed)

    if args.exp in ("all", "sensitivity"):
        run_hybucbar_sensitivity(p, seed=seed)
        run_hybelimucbra_sensitivity(p, seed=seed)

    if args.exp in ("all", "figure1"):
        run_figure1(p, seed=seed)

    if args.exp in ("all", "realworld"):
        run_realworld(p,
                      movielens_path=args.movielens,
                      yelp_path=args.yelp,
                      seed=seed)

    print(f"\nAll done.  Results saved in: {os.path.abspath(OUTDIR)}/")


if __name__ == "__main__":
    main()
