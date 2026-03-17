"""
environment.py
==============
Bandit environment, data generation, and Bradley-Terry model utilities
for "Learning Across the Gap: Hybrid MAB with Heterogeneous Offline and Online Data".
"""

import numpy as np


# ---------------------------------------------------------------------------
# Bradley-Terry helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def bt_prob(mu_i, mu_j):
    """P(arm i beats arm j) under the Bradley-Terry model."""
    return sigmoid(mu_i - mu_j)


# ---------------------------------------------------------------------------
# Synthetic data generation  (Appendix G.1.1)
# ---------------------------------------------------------------------------

def generate_synthetic_means(K, delta, bias, rng=None):
    """
    Generate offline and online mean rewards.

    Parameters
    ----------
    K     : number of arms
    delta : sub-optimal gap between best and 2nd-best arm offline
    bias  : magnitude of offline-to-online distributional shift
    rng   : numpy.random.Generator (optional)

    Returns
    -------
    mu_off : shape (K,)  offline means
    mu_on  : shape (K,)  online means
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_off = np.empty(K)
    mu_off[0] = rng.uniform(0.5 + delta, 1.0)          # best arm
    mu_off[1] = mu_off[0] - delta                       # second best
    mu_off[2:] = rng.uniform(0.0, mu_off[1], size=K-2) # rest

    d = rng.choice([-1, 1], size=K)                    # random shift direction
    mu_on = mu_off + d * bias
    mu_on = np.clip(mu_on, 0.0, 1.0)

    return mu_off, mu_on


# ---------------------------------------------------------------------------
# Offline dataset generation
# ---------------------------------------------------------------------------

def generate_offline_absolute(mu_off, N_per_arm, rng=None):
    """
    Generate offline absolute (stochastic) feedback.

    Returns
    -------
    data : dict  { arm_index -> np.ndarray of shape (N_per_arm,) }
    """
    if rng is None:
        rng = np.random.default_rng()

    K = len(mu_off)
    data = {}
    for i in range(K):
        # 1-sub-Gaussian: Gaussian with std=1
        data[i] = rng.normal(mu_off[i], 1.0, size=N_per_arm)
    return data


def generate_offline_relative(mu_off, N_per_pair, rng=None):
    """
    Generate offline relative (pairwise preference) feedback
    using the Bradley-Terry model.

    Returns
    -------
    data : dict  { (i,j) -> np.ndarray of shape (N_per_pair,) with values in {0,1} }
    """
    if rng is None:
        rng = np.random.default_rng()

    K = len(mu_off)
    data = {}
    for i in range(K):
        for j in range(K):
            if i != j:
                p = bt_prob(mu_off[i], mu_off[j])
                data[(i, j)] = rng.binomial(1, p, size=N_per_pair).astype(float)
    return data


# ---------------------------------------------------------------------------
# Online feedback simulators
# ---------------------------------------------------------------------------

def sample_duel(arm_i, arm_j, mu_on, rng):
    """
    Sample one pairwise comparison outcome Y_{i,j} ~ Bernoulli(p_{i,j}).
    Returns 1 if arm_i wins, 0 otherwise.
    """
    p = bt_prob(mu_on[arm_i], mu_on[arm_j])
    return float(rng.random() < p)


def sample_reward(arm_i, mu_on, rng):
    """
    Sample one stochastic reward for arm_i: X ~ N(mu_on[arm_i], 1).
    """
    return rng.normal(mu_on[arm_i], 1.0)


# ---------------------------------------------------------------------------
# Real-data environment  (Appendix G.2)
# ---------------------------------------------------------------------------

class MovieLensEnvironment:
    """
    Wraps pre-processed MovieLens rating arrays into a bandit environment.

    Parameters
    ----------
    ratings : dict  { arm_index -> np.ndarray of all ratings in [0,1] }
    rng     : numpy.random.Generator
    """

    def __init__(self, ratings, rng=None):
        self.ratings = ratings
        self.K = len(ratings)
        self.rng = rng or np.random.default_rng()

    def duel(self, arm_i, arm_j, n_samples=3):
        """
        Simulate a duel: sample n_samples ratings for each arm,
        arm with higher mean wins (ties resolved randomly).
        Returns 1 if arm_i wins.
        """
        ri = self.rng.choice(self.ratings[arm_i], size=n_samples, replace=True)
        rj = self.rng.choice(self.ratings[arm_j], size=n_samples, replace=True)
        mi, mj = ri.mean(), rj.mean()
        if mi > mj:
            return 1.0
        elif mj > mi:
            return 0.0
        else:
            return float(self.rng.random() < 0.5)

    def reward(self, arm_i, n_samples=30):
        """
        Simulate an absolute reward: mean of n_samples sampled ratings.
        """
        r = self.rng.choice(self.ratings[arm_i], size=n_samples, replace=True)
        return r.mean()

    def offline_absolute(self, arm_i, N):
        """Sample N offline absolute ratings for arm_i."""
        return self.rng.choice(self.ratings[arm_i], size=N, replace=True)

    def offline_relative(self, arm_i, arm_j, N, n_samples=10):
        """
        Generate N offline preference duels by comparing means
        of n_samples ratings.  Returns array of {0,1}.
        """
        outcomes = np.empty(N)
        for k in range(N):
            ri = self.rng.choice(self.ratings[arm_i], size=n_samples, replace=True)
            rj = self.rng.choice(self.ratings[arm_j], size=n_samples, replace=True)
            mi, mj = ri.mean(), rj.mean()
            if mi > mj:
                outcomes[k] = 1.0
            elif mj > mi:
                outcomes[k] = 0.0
            else:
                outcomes[k] = float(self.rng.random() < 0.5)
        return outcomes
