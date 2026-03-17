"""
baselines.py
============
Baseline bandit algorithms used for comparison in the paper:

Dueling bandit baselines:
  - RUCB  (Zoghi et al., 2014)
  - IF2   (Yue et al., 2012)  -- Interleaved Filter 2

Stochastic bandit baselines:
  - UCB   (Lai & Robbins, 1985)
  - ETC   (Explore-Then-Commit)
  - ThompsonSampling (Agrawal & Goyal, 2013)
"""

import numpy as np
from environment import sample_duel, sample_reward


# ===========================================================================
# RUCB  --  Relative Upper Confidence Bound  (Zoghi et al., 2014)
# ===========================================================================

def run_rucb(K, T, mu_on, delta_t_val=0.02, alpha=0.51, rng=None):
    """
    RUCB algorithm for dueling bandits.

    Parameters
    ----------
    K          : number of arms
    T          : horizon
    mu_on      : true online reward means, shape (K,)
    delta_t_val: constant delta for UCB (used as fixed small value per paper)
    alpha      : confidence parameter (default 0.51 as in Zoghi et al.)
    rng        : numpy RNG

    Returns
    -------
    regret : np.ndarray of shape (T,), cumulative regret at each round
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    # True sub-optimality gaps
    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5   # delta[0] = 0 by definition (arm 0 is best)

    # Counts and sum of outcomes
    W = np.zeros((K, K))   # W[i,j] = #times i beat j
    T_ij = np.zeros((K, K), dtype=int)

    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        # Build UCB matrix
        UCB = np.ones((K, K)) * np.inf
        for i in range(K):
            for j in range(K):
                if i == j:
                    UCB[i, j] = 0.5
                    continue
                if T_ij[i, j] > 0:
                    p_hat = W[i, j] / T_ij[i, j]
                    rad = np.sqrt(alpha * np.log(t) / T_ij[i, j])
                    UCB[i, j] = min(1.0, p_hat + rad)

        # Candidate set: arms that could beat all others
        C = []
        for i in range(K):
            if all(UCB[i, j] >= 0.5 for j in range(K) if j != i):
                C.append(i)
        if len(C) == 0:
            C = list(range(K))

        # Select champion (uniform from C, RUCB picks one)
        champ = C[rng.integers(0, len(C))]
        # Challenger: highest UCB against champion (RUCB: random from non-champion)
        others = [j for j in range(K) if j != champ]
        challenger = others[rng.integers(0, len(others))]

        # Observe outcome
        y = sample_duel(champ, challenger, mu_on, rng)
        T_ij[champ, challenger] += 1
        T_ij[challenger, champ] += 1
        W[champ, challenger] += y
        W[challenger, champ] += 1 - y

        reg = (delta[champ] + delta[challenger]) / 2.0
        total_reg += reg
        cumreg[t - 1] = total_reg

    return cumreg


# ===========================================================================
# IF2  --  Interleaved Filter 2  (Yue et al., 2012)
# ===========================================================================

def run_if2(K, T, mu_on, rng=None):
    """
    Interleaved Filter 2 for dueling bandits.
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5

    W = np.zeros((K, K))
    T_ij = np.zeros((K, K), dtype=int)
    active = list(range(K))

    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        if len(active) == 1:
            # Already found best arm, no regret
            cumreg[t - 1] = total_reg
            continue

        # Pick two arms from active set; try to eliminate
        i = active[0]
        j = active[rng.integers(1, len(active))]

        y = sample_duel(i, j, mu_on, rng)
        T_ij[i, j] += 1
        T_ij[j, i] += 1
        W[i, j] += y
        W[j, i] += 1 - y

        reg = (delta[i] + delta[j]) / 2.0
        total_reg += reg
        cumreg[t - 1] = total_reg

        # Elimination check
        threshold = np.sqrt(np.log(4 * K * T_ij[i, j] ** 2) / (2 * T_ij[i, j]))
        if T_ij[i, j] > 0:
            p_hat_ij = W[i, j] / T_ij[i, j]
            if p_hat_ij + threshold < 0.5:
                if i in active:
                    active.remove(i)
            elif p_hat_ij - threshold > 0.5:
                if j in active:
                    active.remove(j)

    return cumreg


# ===========================================================================
# UCB  (Lai & Robbins 1985 style, with log-confidence)
# ===========================================================================

def run_ucb(K, T, mu_on, delta_t=0.05, rng=None):
    """
    Standard UCB for stochastic bandits.
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    # Gaps defined via BT (consistent with paper's definition)
    delta_arms = np.array([bt_prob(mu_on[0], mu_on[i]) - 0.5 for i in range(K)])

    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        # Pull each arm once initially
        if t <= K:
            arm = t - 1
        else:
            ucb_vals = np.empty(K)
            for i in range(K):
                rad = np.sqrt(np.log(1.0 / delta_t) / (2.0 * counts[i]))
                ucb_vals[i] = means[i] + rad
            arm = int(np.argmax(ucb_vals))

        r = sample_reward(arm, mu_on, rng)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]

        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg

    return cumreg


# ===========================================================================
# ETC  --  Explore-Then-Commit
# ===========================================================================

def run_etc(K, T, mu_on, explore_per_arm=500, rng=None):
    """
    Explore-Then-Commit for stochastic bandits.
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    delta_arms = np.array([bt_prob(mu_on[0], mu_on[i]) - 0.5 for i in range(K)])

    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    cumreg = np.zeros(T)
    total_reg = 0.0

    explore_total = explore_per_arm * K

    for t in range(1, T + 1):
        if t <= explore_total:
            arm = (t - 1) % K
        else:
            arm = int(np.argmax(means))

        r = sample_reward(arm, mu_on, rng)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]

        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg

    return cumreg


# ===========================================================================
# Thompson Sampling  (Gaussian prior/posterior, Agrawal & Goyal 2013)
# ===========================================================================

def run_thompson_sampling(K, T, mu_on, prior_mean=0.5, prior_var=1.0, rng=None):
    """
    Thompson Sampling with Gaussian prior and Gaussian likelihood.
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    delta_arms = np.array([bt_prob(mu_on[0], mu_on[i]) - 0.5 for i in range(K)])

    # Posterior parameters: N(mu_post, sigma_post^2)
    mu_post = np.full(K, prior_mean)
    sigma2_post = np.full(K, prior_var)  # likelihood sigma^2 = 1

    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        # Sample from posterior
        samples = rng.normal(mu_post, np.sqrt(sigma2_post))
        arm = int(np.argmax(samples))

        r = sample_reward(arm, mu_on, rng)

        # Gaussian posterior update (known sigma^2=1 likelihood)
        sigma2_post[arm] = 1.0 / (1.0 / sigma2_post[arm] + 1.0)
        mu_post[arm] = sigma2_post[arm] * (mu_post[arm] / (sigma2_post[arm] * (1.0 / sigma2_post[arm] - 1.0 + 1.0 / sigma2_post[arm]))
                                            + r)
        # Simpler equivalent:
        old_var = sigma2_post[arm]
        mu_post[arm] = old_var * (mu_post[arm] / old_var + r)  # already updated

        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg

    return cumreg


def run_thompson_sampling_v2(K, T, mu_on, prior_mean=0.5, prior_var=1.0, rng=None):
    """
    Cleaner Thompson Sampling implementation (Gaussian conjugate).
    Prior: mu ~ N(prior_mean, prior_var)
    Likelihood: X | mu ~ N(mu, 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    from environment import bt_prob
    delta_arms = np.array([bt_prob(mu_on[0], mu_on[i]) - 0.5 for i in range(K)])

    counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K)

    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        # Compute posterior parameters
        post_vars = 1.0 / (1.0 / prior_var + counts)
        post_means = post_vars * (prior_mean / prior_var + sum_rewards)
        # Sample
        theta = rng.normal(post_means, np.sqrt(post_vars))
        arm = int(np.argmax(theta))

        r = sample_reward(arm, mu_on, rng)
        counts[arm] += 1
        sum_rewards[arm] += r

        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg

    return cumreg
