"""
experiments_synthetic.py
========================
Reproduces all synthetic data experiments from Appendix G.1:

  Figure 2 / Fig1-left  : HybUCB-AR scalability over K=8,16,24,32
  Figure 3              : HybUCB-AR parameter sensitivity (Ni, Delta, Vi)
  Figure 4 / Fig1-right : HybElimUCB-RA scalability over K=8,16,24,32
  Figure 5              : HybElimUCB-RA parameter sensitivity
  Figure 1 (main paper) : K=8 comparison for both algorithms
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from environment import (generate_synthetic_means,
                         generate_offline_absolute,
                         generate_offline_relative)
from hybucb_ar import run_hybucb_ar, run_hybucb_ar_no_offline
from hybucb_elimra import run_hybucb_elimra, run_hybucb_elimra_no_offline
from baselines import (run_rucb, run_if2, run_ucb,
                       run_etc, run_thompson_sampling_v2)


# ---------------------------------------------------------------------------
# Generic multi-trial runner
# ---------------------------------------------------------------------------

def run_trials(fn, n_trials, seed_base=42, **kwargs):
    """Run `fn(**kwargs)` for n_trials and return mean and stderr."""
    results = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        r = fn(rng=rng, **kwargs)
        results.append(r)
    arr = np.stack(results, axis=0)  # (n_trials, T)
    return arr.mean(0), arr.std(0) / np.sqrt(n_trials)


# ---------------------------------------------------------------------------
# Helper: build offline data for HybUCB-AR
# ---------------------------------------------------------------------------

def build_offline_absolute(K, mu_off, N_per_arm, seed=0):
    rng = np.random.default_rng(seed)
    return generate_offline_absolute(mu_off, N_per_arm, rng)


def build_offline_relative(K, mu_off, N_per_pair, seed=0):
    rng = np.random.default_rng(seed)
    return generate_offline_relative(mu_off, N_per_pair, rng)


def build_vi(K, mu_off, mu_on, bias):
    """Vi = bias (upper bound on |mu_off_i - mu_on_i|)."""
    return np.full(K, bias)


# ===========================================================================
# Experiment 1: HybUCB-AR scalability  (Figure 2 / Figure 1 left panels)
# ===========================================================================

def exp_hybucbar_scalability(K_list=(8, 16, 24, 32), T=50_000,
                              delta=0.1, bias=0.1, N_offline=500,
                              n_trials=100, seed=0):
    print("\n=== HybUCB-AR Scalability ===")
    results = {}

    for K in K_list:
        print(f"  K={K} ...", flush=True)
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta, bias, rng0)
        offline_data = build_offline_absolute(K, mu_off, N_offline, seed=seed+1)
        Vi = build_vi(K, mu_off, mu_on, bias)

        # HybUCB-AR with offline
        def fn_hyb(rng): return run_hybucb_ar(
            K, T, mu_on, offline_data, Vi=Vi, delta_t_mode=0.02, rng=rng)
        # HybUCB-AR without offline
        def fn_hyb_no(rng): return run_hybucb_ar_no_offline(
            K, T, mu_on, delta_t_mode=0.02, rng=rng)
        # RUCB
        def fn_rucb(rng): return run_rucb(K, T, mu_on, alpha=0.51, rng=rng)
        # IF2
        def fn_if2(rng): return run_if2(K, T, mu_on, rng=rng)

        res = {}
        for name, fn in [("HybUCB-AR", fn_hyb),
                         ("HybUCB-AR (no offline)", fn_hyb_no),
                         ("RUCB", fn_rucb),
                         ("InterleavedFilter", fn_if2)]:
            trials = [fn(rng=np.random.default_rng(seed + t))
                      for t in range(n_trials)]
            arr = np.stack(trials)
            res[name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
            print(f"    {name}: final regret = {arr[:,-1].mean():.1f} ± {arr[:,-1].std()/np.sqrt(n_trials):.1f}")

        results[K] = res

    return results


# ===========================================================================
# Experiment 2: HybUCB-AR parameter sensitivity  (Figure 3)
# ===========================================================================

def exp_hybucbar_sensitivity(K=20, T=30_000, n_trials=100, seed=0):
    print("\n=== HybUCB-AR Parameter Sensitivity ===")
    results = {}

    # Default values
    N_default, delta_default, Vi_default = 500, 0.1, 0.0
    bias_default = 0.0   # when Vi=0, bias=0

    # --- Varying Ni ---
    print("  Varying Ni ...")
    res_ni = {}
    for Ni in [100, 300, 500]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta_default, bias_default, rng0)
        offline_data = build_offline_absolute(K, mu_off, Ni, seed=seed+1)
        Vi = np.zeros(K)

        def fn(rng, _ni=Ni): return run_hybucb_ar(
            K, T, mu_on,
            build_offline_absolute(K, mu_off, _ni, seed=seed+1),
            Vi=np.zeros(K), delta_t_mode=0.02, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_ni[Ni] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    Ni={Ni}: {arr[:,-1].mean():.1f}")
    results["Ni"] = res_ni

    # --- Varying Delta ---
    print("  Varying Delta ...")
    res_delta = {}
    for delta in [0.05, 0.1, 0.2]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta, bias_default, rng0)
        offline_data = build_offline_absolute(K, mu_off, N_default, seed=seed+1)

        def fn(rng, _d=delta, _mo=mu_on.copy(), _od=offline_data):
            return run_hybucb_ar(K, T, _mo, _od, Vi=np.zeros(K),
                                 delta_t_mode=0.02, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_delta[delta] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    delta={delta}: {arr[:,-1].mean():.1f}")
    results["delta"] = res_delta

    # --- Varying Vi ---
    print("  Varying Vi ...")
    res_vi = {}
    for vi_val in [0.01, 0.05, 0.1]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta_default, vi_val, rng0)
        offline_data = build_offline_absolute(K, mu_off, N_default, seed=seed+1)
        Vi = np.full(K, vi_val)

        def fn(rng, _vi=Vi.copy(), _mo=mu_on.copy(), _od=offline_data):
            return run_hybucb_ar(K, T, _mo, _od, Vi=_vi,
                                 delta_t_mode=0.02, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_vi[vi_val] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    Vi={vi_val}: {arr[:,-1].mean():.1f}")
    results["Vi"] = res_vi

    return results


# ===========================================================================
# Experiment 3: HybElimUCB-RA scalability  (Figure 4)
# ===========================================================================

def exp_hybelimucbra_scalability(K_list=(8, 16, 24, 32), T=30_000,
                                  delta=0.1, bias=0.01, N_offline=500,
                                  n_trials=100, seed=0):
    print("\n=== HybElimUCB-RA Scalability ===")
    results = {}

    for K in K_list:
        print(f"  K={K} ...", flush=True)
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta, bias, rng0)
        offline_data = build_offline_relative(K, mu_off, N_offline, seed=seed+1)
        Vi = build_vi(K, mu_off, mu_on, bias)

        def fn_hyb(rng): return run_hybucb_elimra(
            K, T, mu_on, offline_data, Vi=Vi, delta_t_mode=0.05, rng=rng)
        def fn_hyb_no(rng): return run_hybucb_elimra_no_offline(
            K, T, mu_on, delta_t_mode=0.05, rng=rng)
        def fn_ucb(rng): return run_ucb(K, T, mu_on, delta_t=0.05, rng=rng)
        def fn_etc(rng): return run_etc(K, T, mu_on, explore_per_arm=500, rng=rng)
        def fn_ts(rng): return run_thompson_sampling_v2(
            K, T, mu_on, prior_mean=0.5, prior_var=1.0, rng=rng)

        res = {}
        for name, fn in [("HybElimUCB-RA", fn_hyb),
                         ("HybElimUCB-RA (no offline)", fn_hyb_no),
                         ("ETC", fn_etc),
                         ("UCB", fn_ucb),
                         ("ThompsonSampling", fn_ts)]:
            trials = [fn(rng=np.random.default_rng(seed + t))
                      for t in range(n_trials)]
            arr = np.stack(trials)
            res[name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
            print(f"    {name}: final regret = {arr[:,-1].mean():.1f}")

        results[K] = res

    return results


# ===========================================================================
# Experiment 4: HybElimUCB-RA parameter sensitivity  (Figure 5)
# ===========================================================================

def exp_hybelimucbra_sensitivity(K=10, T=25_000, n_trials=100, seed=0):
    print("\n=== HybElimUCB-RA Parameter Sensitivity ===")
    results = {}

    N_default, delta_default, Vi_default = 100, 0.1, 0.01

    # --- Varying Ni ---
    print("  Varying Ni ...")
    res_ni = {}
    for Ni in [100, 200, 500]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta_default, Vi_default, rng0)

        def fn(rng, _ni=Ni):
            od = build_offline_relative(K, mu_off, _ni, seed=seed+1)
            return run_hybucb_elimra(K, T, mu_on, od,
                                     Vi=np.full(K, Vi_default),
                                     delta_t_mode=0.05, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_ni[Ni] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    Ni={Ni}: {arr[:,-1].mean():.1f}")
    results["Ni"] = res_ni

    # --- Varying Delta ---
    print("  Varying Delta ...")
    res_delta = {}
    for delta in [0.05, 0.1, 0.2]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta, Vi_default, rng0)
        od = build_offline_relative(K, mu_off, N_default, seed=seed+1)

        def fn(rng, _mo=mu_on.copy(), _od=od):
            return run_hybucb_elimra(K, T, _mo, _od,
                                     Vi=np.full(K, Vi_default),
                                     delta_t_mode=0.05, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_delta[delta] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    delta={delta}: {arr[:,-1].mean():.1f}")
    results["delta"] = res_delta

    # --- Varying Vi ---
    print("  Varying Vi ...")
    res_vi = {}
    for vi_val in [0.01, 0.05, 0.1]:
        rng0 = np.random.default_rng(seed)
        mu_off, mu_on = generate_synthetic_means(K, delta_default, vi_val, rng0)
        od = build_offline_relative(K, mu_off, N_default, seed=seed+1)

        def fn(rng, _vi=vi_val, _mo=mu_on.copy(), _od=od):
            return run_hybucb_elimra(K, T, _mo, _od,
                                     Vi=np.full(K, _vi),
                                     delta_t_mode=0.05, rng=rng)

        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        res_vi[vi_val] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    Vi={vi_val}: {arr[:,-1].mean():.1f}")
    results["Vi"] = res_vi

    return results


# ===========================================================================
# Main paper Figure 1:  K=8, synthetic data comparison
# ===========================================================================

def exp_figure1_synthetic(K=8, seed=0,
                           T_ar=50_000, T_ra=30_000,
                           delta=0.1, bias=0.1, N_offline=500,
                           n_trials=100):
    """Reproduce Figure 1 (main paper), left two panels (synthetic)."""
    print("\n=== Figure 1: Synthetic K=8 ===")

    rng0 = np.random.default_rng(seed)
    mu_off, mu_on = generate_synthetic_means(K, delta, bias, rng0)
    offline_abs = build_offline_absolute(K, mu_off, N_offline, seed=seed+1)
    offline_rel = build_offline_relative(K, mu_off, N_offline, seed=seed+2)
    Vi = np.full(K, bias)

    results = {"HybUCB-AR": {}, "HybElimUCB-RA": {}}

    # HybUCB-AR panel
    for name, fn in [
        ("HybUCB-AR", lambda rng: run_hybucb_ar(
            K, T_ar, mu_on, offline_abs, Vi=Vi, delta_t_mode=0.02, rng=rng)),
        ("HybUCB-AR (no offline)", lambda rng: run_hybucb_ar_no_offline(
            K, T_ar, mu_on, delta_t_mode=0.02, rng=rng)),
        ("RUCB", lambda rng: run_rucb(K, T_ar, mu_on, alpha=0.51, rng=rng)),
        ("InterleavedFilter", lambda rng: run_if2(K, T_ar, mu_on, rng=rng)),
    ]:
        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        results["HybUCB-AR"][name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"  [HybUCB-AR] {name}: {arr[:,-1].mean():.1f}")

    # HybElimUCB-RA panel
    bias_ra = 0.01
    rng0 = np.random.default_rng(seed)
    mu_off2, mu_on2 = generate_synthetic_means(K, delta, bias_ra, rng0)
    offline_rel2 = build_offline_relative(K, mu_off2, N_offline, seed=seed+2)
    Vi2 = np.full(K, bias_ra)

    for name, fn in [
        ("HybElimUCB-RA", lambda rng: run_hybucb_elimra(
            K, T_ra, mu_on2, offline_rel2, Vi=Vi2, delta_t_mode=0.05, rng=rng)),
        ("HybElimUCB-RA (no offline)", lambda rng: run_hybucb_elimra_no_offline(
            K, T_ra, mu_on2, delta_t_mode=0.05, rng=rng)),
        ("ETC", lambda rng: run_etc(K, T_ra, mu_on2, explore_per_arm=500, rng=rng)),
        ("UCB", lambda rng: run_ucb(K, T_ra, mu_on2, delta_t=0.05, rng=rng)),
        ("ThompsonSampling", lambda rng: run_thompson_sampling_v2(
            K, T_ra, mu_on2, rng=rng)),
    ]:
        trials = [fn(rng=np.random.default_rng(seed + t)) for t in range(n_trials)]
        arr = np.stack(trials)
        results["HybElimUCB-RA"][name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"  [HybElimUCB-RA] {name}: {arr[:,-1].mean():.1f}")

    return results, T_ar, T_ra
