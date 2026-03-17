"""
experiments_realworld.py
========================
Real-world experiment runner (Appendix G.2).

Since the actual MovieLens/Yelp datasets may not be present, this module:
  1. Tries to load them if available.
  2. Falls back to a realistic synthetic substitute that mimics their
     statistical properties (K=10 arms, realistic rating distributions).

MovieLens-20M: 20M ratings [1-5], normalised to [0,1].
Yelp Academic: star ratings [1-5], normalised to [0,1].
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from environment import MovieLensEnvironment, bt_prob, sigmoid
from hybucb_ar import run_hybucb_ar
from hybucb_elimra import run_hybucb_elimra
from baselines import run_rucb, run_if2, run_ucb, run_etc, run_thompson_sampling_v2


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_movielens(path, K=10, min_ratings=100, seed=0):
    """
    Load MovieLens-20M ratings.csv and build K arm rating arrays.

    Returns dict { arm_idx -> np.ndarray of ratings in [0,1] }
    or None if file not found.
    """
    if not os.path.exists(path):
        return None

    print(f"Loading MovieLens from {path} ...")
    ratings_per_movie = {}
    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                movie_id = int(parts[1])
                rating = float(parts[2]) / 5.0   # normalise to [0,1]
                if movie_id not in ratings_per_movie:
                    ratings_per_movie[movie_id] = []
                ratings_per_movie[movie_id].append(rating)
            except ValueError:
                continue

    # Filter movies with >= min_ratings, take top 100 by count, sample K
    eligible = [(mid, np.array(r))
                for mid, r in ratings_per_movie.items()
                if len(r) >= min_ratings]
    eligible.sort(key=lambda x: -len(x[1]))
    top100 = eligible[:100]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(top100), size=min(K, len(top100)), replace=False)
    arms = {i: top100[c][1] for i, c in enumerate(chosen)}
    print(f"  Loaded {len(arms)} arms.")
    return arms


def load_yelp(path, K=10, min_ratings=100, seed=1):
    """
    Load Yelp academic dataset (yelp_academic_dataset_review.json).

    Returns dict { arm_idx -> np.ndarray of ratings in [0,1] } or None.
    """
    if not os.path.exists(path):
        return None

    import json
    print(f"Loading Yelp from {path} ...")
    ratings_per_biz = {}
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                biz = obj.get("business_id")
                stars = obj.get("stars")
                if biz is not None and stars is not None:
                    if biz not in ratings_per_biz:
                        ratings_per_biz[biz] = []
                    ratings_per_biz[biz].append(float(stars) / 5.0)
            except Exception:
                continue

    eligible = [(bid, np.array(r))
                for bid, r in ratings_per_biz.items()
                if len(r) >= min_ratings]
    eligible.sort(key=lambda x: -len(x[1]))
    top100 = eligible[:100]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(top100), size=min(K, len(top100)), replace=False)
    arms = {i: top100[c][1] for i, c in enumerate(chosen)}
    print(f"  Loaded {len(arms)} arms.")
    return arms


# ---------------------------------------------------------------------------
# Synthetic substitute  (used when real data is unavailable)
# ---------------------------------------------------------------------------

def make_synthetic_ratings(K=10, n_ratings_per_arm=5000, seed=42):
    """
    Create a synthetic rating dataset that mimics MovieLens/Yelp statistics:
    - Ratings in [0,1]
    - Each arm has a true mean in [0.4, 0.9]
    - Ratings drawn from a beta distribution around the mean
    """
    rng = np.random.default_rng(seed)
    means = np.sort(rng.uniform(0.4, 0.9, K))[::-1]  # sorted, best arm first
    arms = {}
    for i, mu in enumerate(means):
        # Beta(a,b) with mean mu, moderate variance
        a = mu * 8
        b = (1 - mu) * 8
        ratings = rng.beta(a, b, size=n_ratings_per_arm)
        arms[i] = ratings
    print(f"  Created synthetic rating dataset: K={K}, means={np.round(means,3)}")
    return arms


# ---------------------------------------------------------------------------
# Offline data builders for real-world experiments
# ---------------------------------------------------------------------------

def build_offline_abs_real(env, K, N_offline=100_000, seed=0):
    """
    Sample N_offline / K absolute ratings per arm from the environment.
    Returns dict { arm_idx -> np.ndarray }.
    """
    rng = np.random.default_rng(seed)
    n_per_arm = max(1, N_offline // K)
    return {i: env.offline_absolute(i, n_per_arm) for i in range(K)}


def build_offline_rel_real(env, K, N_per_pair=1000, seed=0):
    """
    Generate N_per_pair relative preference duels for all arm pairs.
    Returns dict { (i,j) -> np.ndarray }.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(K):
        for j in range(K):
            if i != j:
                data[(i, j)] = env.offline_relative(i, j, N_per_pair)
    return data


def estimate_bias_matrix(env, K, n_samples=100):
    """
    Estimate the pairwise bias matrix V[i,j] from a small pilot sample.
    We use V[i,j] = 0.1 as a conservative constant per the paper's setup.
    """
    return np.full((K, K), 0.1)


# ---------------------------------------------------------------------------
# Real-data experiment for HybUCB-AR  (Appendix G.2.2)
# ---------------------------------------------------------------------------

def exp_realworld_hybucbar(arms, K=10, T=10_000, n_trials=50, seed=0,
                            N_offline_abs=100_000):
    """
    Run HybUCB-AR and baselines on a real (or synthetic-substitute) dataset.
    Returns dict { algorithm_name -> (mean_regret, se_regret) }.
    """
    print(f"  HybUCB-AR real-world (K={K}, T={T}, n_trials={n_trials})")

    # True online means (from arm ratings)
    mu_on = np.array([arms[i].mean() for i in range(K)])
    best_arm = int(np.argmax(mu_on))

    # Re-index so best arm is arm 0
    order = [best_arm] + [i for i in range(K) if i != best_arm]
    arms_reindexed = {new_i: arms[old_i] for new_i, old_i in enumerate(order)}
    mu_on = np.array([arms_reindexed[i].mean() for i in range(K)])

    results = {}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        env = MovieLensEnvironment(arms_reindexed, rng=rng)

        # Build offline absolute data
        n_per_arm = max(1, N_offline_abs // K)
        offline_data = {i: env.offline_absolute(i, n_per_arm) for i in range(K)}

        # Bias matrix
        V = np.full((K, K), 0.1)

        # Run algorithms
        for name, run_fn in [
            ("HybUCB-AR", lambda rng, od=offline_data: _run_hybucbar_real(
                K, T, env, mu_on, od, V, rng)),
            ("HybUCB-AR (no offline)", lambda rng: _run_hybucbar_no_offline_real(
                K, T, env, mu_on, rng)),
            ("RUCB", lambda rng: _run_rucb_real(K, T, env, mu_on, rng)),
            ("InterleavedFilter", lambda rng: _run_if2_real(K, T, env, mu_on, rng)),
        ]:
            r = run_fn(rng=rng)
            if name not in results:
                results[name] = []
            results[name].append(r)

    # Aggregate
    out = {}
    for name, trials in results.items():
        arr = np.stack(trials)
        out[name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    {name}: final={arr[:,-1].mean():.2f}")

    return out


def _run_hybucbar_real(K, T, env, mu_on, offline_data, V, rng):
    """HybUCB-AR using real dueling feedback."""
    from hybucb_ar import HybUCB_AR
    from environment import bt_prob

    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5

    algo = HybUCB_AR(K, offline_data, V=V, delta_t_mode=0.02)
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        a1, a2 = algo.select_pair()
        outcome = env.duel(a1, a2, n_samples=3)
        algo.update(a1, a2, outcome, t)
        total_reg += (delta[a1] + delta[a2]) / 2.0
        cumreg[t - 1] = total_reg
    return cumreg


def _run_hybucbar_no_offline_real(K, T, env, mu_on, rng):
    empty = {i: np.array([]) for i in range(K)}
    return _run_hybucbar_real(K, T, env, mu_on, empty, np.zeros((K, K)), rng)


def _run_rucb_real(K, T, env, mu_on, rng):
    from environment import bt_prob
    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5

    W = np.zeros((K, K))
    T_ij = np.zeros((K, K), dtype=int)
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        # Build UCB
        UCB = np.ones((K, K))
        for i in range(K):
            for j in range(K):
                if i != j and T_ij[i, j] > 0:
                    p_hat = W[i, j] / T_ij[i, j]
                    rad = np.sqrt(0.51 * np.log(t) / T_ij[i, j])
                    UCB[i, j] = min(1.0, p_hat + rad)

        C = [i for i in range(K)
             if all(UCB[i, j] >= 0.5 for j in range(K) if j != i)]
        if not C:
            C = list(range(K))

        champ = C[rng.integers(0, len(C))]
        others = [j for j in range(K) if j != champ]
        chal = others[rng.integers(0, len(others))]

        y = env.duel(champ, chal, n_samples=3)
        T_ij[champ, chal] += 1
        T_ij[chal, champ] += 1
        W[champ, chal] += y
        W[chal, champ] += 1 - y

        total_reg += (delta[champ] + delta[chal]) / 2.0
        cumreg[t - 1] = total_reg
    return cumreg


def _run_if2_real(K, T, env, mu_on, rng):
    from environment import bt_prob
    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5

    W = np.zeros((K, K))
    T_ij = np.zeros((K, K), dtype=int)
    active = list(range(K))
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        if len(active) < 2:
            cumreg[t - 1] = total_reg
            continue
        i = active[0]
        j = active[rng.integers(1, len(active))]
        y = env.duel(i, j, n_samples=3)
        T_ij[i, j] += 1
        T_ij[j, i] += 1
        W[i, j] += y
        W[j, i] += 1 - y

        total_reg += (delta[i] + delta[j]) / 2.0
        cumreg[t - 1] = total_reg

        if T_ij[i, j] > 0:
            thr = np.sqrt(np.log(4 * K * T_ij[i, j] ** 2) / (2 * T_ij[i, j]))
            p_hat = W[i, j] / T_ij[i, j]
            if p_hat + thr < 0.5 and i in active:
                active.remove(i)
            elif p_hat - thr > 0.5 and j in active:
                active.remove(j)
    return cumreg


# ---------------------------------------------------------------------------
# Real-data experiment for HybElimUCB-RA  (Appendix G.2.3)
# ---------------------------------------------------------------------------

def exp_realworld_hybelimucbra(arms, K=10, T=10_000, n_trials=50, seed=0,
                                N_per_pair=1000):
    print(f"  HybElimUCB-RA real-world (K={K}, T={T}, n_trials={n_trials})")

    mu_on = np.array([arms[i].mean() for i in range(K)])
    best_arm = int(np.argmax(mu_on))
    order = [best_arm] + [i for i in range(K) if i != best_arm]
    arms_reindexed = {new_i: arms[old_i] for new_i, old_i in enumerate(order)}
    mu_on = np.array([arms_reindexed[i].mean() for i in range(K)])

    from environment import bt_prob
    delta_arms = np.array([bt_prob(mu_on[0], mu_on[i]) - 0.5 for i in range(K)])

    results = {}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        env = MovieLensEnvironment(arms_reindexed, rng=rng)

        # Build offline relative data
        offline_data = {}
        for i in range(K):
            for j in range(K):
                if i != j:
                    offline_data[(i, j)] = env.offline_relative(i, j, N_per_pair)

        V = np.full((K, K), 0.1)

        for name, run_fn in [
            ("HybElimUCB-RA", lambda rng, od=offline_data:
                _run_hybelimucbra_real(K, T, env, mu_on, delta_arms, od, V, rng)),
            ("HybElimUCB-RA (no offline)", lambda rng:
                _run_hybelimucbra_no_offline_real(K, T, env, mu_on, delta_arms, rng)),
            ("ETC", lambda rng: _run_etc_real(K, T, env, mu_on, delta_arms, rng)),
            ("UCB", lambda rng: _run_ucb_real(K, T, env, mu_on, delta_arms, rng)),
            ("ThompsonSampling", lambda rng:
                _run_ts_real(K, T, env, mu_on, delta_arms, rng)),
        ]:
            r = run_fn(rng=rng)
            if name not in results:
                results[name] = []
            results[name].append(r)

    out = {}
    for name, trials in results.items():
        arr = np.stack(trials)
        out[name] = (arr.mean(0), arr.std(0) / np.sqrt(n_trials))
        print(f"    {name}: final={arr[:,-1].mean():.2f}")
    return out


def _run_hybelimucbra_real(K, T, env, mu_on, delta_arms, offline_data, V, rng):
    from hybucb_elimra import HybElimUCB_RA
    algo = HybElimUCB_RA(K, offline_data, V=V, delta_t_mode=0.05)
    cumreg = np.zeros(T)
    total_reg = 0.0
    for t in range(1, T + 1):
        arm = algo.select_arm()
        reward = env.reward(arm, n_samples=30)
        algo.update(arm, reward, t)
        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg
    return cumreg


def _run_hybelimucbra_no_offline_real(K, T, env, mu_on, delta_arms, rng):
    empty = {(i, j): np.array([]) for i in range(K) for j in range(K) if i != j}
    return _run_hybelimucbra_real(K, T, env, mu_on, delta_arms, empty,
                                  np.zeros((K, K)), rng)


def _run_etc_real(K, T, env, mu_on, delta_arms, rng, explore=200):
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    cumreg = np.zeros(T)
    total_reg = 0.0
    for t in range(1, T + 1):
        if t <= explore * K:
            arm = (t - 1) % K
        else:
            arm = int(np.argmax(means))
        r = env.reward(arm, n_samples=30)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]
        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg
    return cumreg


def _run_ucb_real(K, T, env, mu_on, delta_arms, rng, delta_t=0.05):
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    cumreg = np.zeros(T)
    total_reg = 0.0
    for t in range(1, T + 1):
        if t <= K:
            arm = t - 1
        else:
            ucb_vals = means + np.sqrt(np.log(1.0 / delta_t) / (2.0 * counts))
            arm = int(np.argmax(ucb_vals))
        r = env.reward(arm, n_samples=30)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]
        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg
    return cumreg


def _run_ts_real(K, T, env, mu_on, delta_arms, rng,
                 prior_mean=0.5, prior_var=1.0):
    counts = np.zeros(K, dtype=int)
    sum_r = np.zeros(K)
    cumreg = np.zeros(T)
    total_reg = 0.0
    for t in range(1, T + 1):
        post_vars = 1.0 / (1.0 / prior_var + counts)
        post_means = post_vars * (prior_mean / prior_var + sum_r)
        theta = rng.normal(post_means, np.sqrt(post_vars))
        arm = int(np.argmax(theta))
        r = env.reward(arm, n_samples=30)
        counts[arm] += 1
        sum_r[arm] += r
        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg
    return cumreg
