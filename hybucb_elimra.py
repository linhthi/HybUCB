"""
hybucb_elimra.py
================
Algorithm 2: HybElimUCB-RA
Hybrid Elimination UCB for Stochastic Bandits with Offline Relative Feedback.

Reference: Section 5 of "Learning Across the Gap: Hybrid MAB with
Heterogeneous Offline and Online Data".
"""

import numpy as np
from environment import sigmoid, bt_prob, sample_reward


class HybElimUCB_RA:
    """
    HybElimUCB-RA: Stochastic bandits with offline pairwise preference data.

    Parameters
    ----------
    K             : number of arms
    offline_data  : dict { (i,j) -> np.ndarray of preference outcomes in {0,1} }
    V             : pairwise bias matrix V[i,j] >= |p_off_ij - p_ij|
                    If None, use Vi and derive V[i,j] = sigma(Vi+Vj)
    Vi            : arm-level bias vector (shape K)
    delta_t_mode  : "paper" or float
    """

    def __init__(self, K, offline_data, V=None, Vi=None, delta_t_mode=0.05):
        self.K = K
        self.delta_t_mode = delta_t_mode

        # Offline pairwise empirical means
        self.N_ij = np.zeros((K, K), dtype=float)
        self.p_off_hat = np.full((K, K), 0.5)
        for (i, j), data in offline_data.items():
            n = len(data)
            self.N_ij[i, j] = n
            self.p_off_hat[i, j] = data.mean() if n > 0 else 0.5

        # Bias matrix
        if V is not None:
            self.V = np.array(V, dtype=float)
        elif Vi is not None:
            Vi = np.array(Vi, dtype=float)
            self.V = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    self.V[i, j] = sigmoid(Vi[i] + Vi[j])
        else:
            self.V = np.ones((K, K))

        # Online state
        self.T_i = np.zeros(K, dtype=int)      # how many times arm i was pulled
        self.sum_rewards = np.zeros(K)          # sum of rewards per arm
        self.mu_hat = np.zeros(K)               # online empirical mean

        # UCBs: we track pairwise relative UCBs derived from online absolute
        # and hybrid (offline relative + online absolute derived)
        self.UCB = np.full((K, K), np.inf)     # pure online UCB on pi,j
        self.UCBhyb = np.zeros((K, K))

        # Active set
        self.active = list(range(K))

        # Initialise UCBhyb (T_i = 0)
        self._init_ucbhyb()

    # ------------------------------------------------------------------
    def _delta(self, t):
        if self.delta_t_mode == "paper":
            return 1.0 / (2.0 * self.K * (self.K + 1) * t ** 2)
        else:
            return float(self.delta_t_mode)

    # ------------------------------------------------------------------
    def _init_ucbhyb(self):
        """
        Initialise UCBhyb at t=0 (no online data yet).
        UCBhyb(ai,aj) = p_off_hat[i,j] + sqrt(log(1/delta_t) / N_ij)
        """
        delta0 = self._delta(1)
        K = self.K
        for i in range(K):
            for j in range(K):
                if i == j:
                    self.UCBhyb[i, j] = 0.5
                    continue
                n = self.N_ij[i, j]
                if n > 0:
                    rad = np.sqrt(np.log(1.0 / delta0) / n)
                else:
                    rad = np.inf
                self.UCBhyb[i, j] = self.p_off_hat[i, j] + rad

    # ------------------------------------------------------------------
    def _eff_online(self, i, j):
        """
        Effective online count for pair (i,j):
        Ti*Tj / (Ti+Tj)   [harmonic half-mean of individual counts]
        """
        ti, tj = self.T_i[i], self.T_i[j]
        s = ti + tj
        return (ti * tj / s) if s > 0 else 0.0

    # ------------------------------------------------------------------
    def _phat_online_ij(self, i, j):
        """
        Online estimate of pi,j from absolute rewards via BT model:
        phat_ij = sigma(mu_hat_i - mu_hat_j)
        """
        return sigmoid(self.mu_hat[i] - self.mu_hat[j])

    # ------------------------------------------------------------------
    def _update_ucbs(self, t):
        """Update UCB and UCBhyb for all pairs."""
        K = self.K
        delta_t = self._delta(t)

        for i in range(K):
            for j in range(K):
                if i == j:
                    self.UCB[i, j] = 0.5
                    self.UCBhyb[i, j] = 0.5
                    continue

                ti, tj = self.T_i[i], self.T_i[j]
                n_eff = self._eff_online(i, j)
                n_off = self.N_ij[i, j]

                # --- Pure online UCB  (Eq. 9) ---
                if n_eff > 0:
                    p_hat_on = self._phat_online_ij(i, j)
                    rad_on = 2.0 * np.sqrt(
                        np.log(1.0 / delta_t) / (2.0) * ((ti + tj) / max(ti * tj, 1))
                    )
                    self.UCB[i, j] = min(1.0, p_hat_on + rad_on)
                else:
                    self.UCB[i, j] = np.inf

                # --- Hybrid UCB  (Eq. 10) ---
                denom = n_eff + n_off
                if denom > 0:
                    alpha = n_off / denom
                    p_hat_on = self._phat_online_ij(i, j) if n_eff > 0 else 0.5
                    p_hyb = alpha * self.p_off_hat[i, j] + (1.0 - alpha) * p_hat_on

                    rad_hyb = np.sqrt(np.log(1.0 / delta_t) / (2.0 * denom))
                    bias_term = (n_off / denom) * self.V[i, j]
                    self.UCBhyb[i, j] = min(1.0, p_hyb + rad_hyb + bias_term)
                else:
                    self.UCBhyb[i, j] = np.inf

    # ------------------------------------------------------------------
    def _eliminate(self):
        """
        Remove arm ai from active set if there exists aj in active s.t.
        min{UCB(ai,aj), UCBhyb(ai,aj)} < 0.5  (Line 6 of Algorithm 2).
        """
        to_remove = set()
        for i in list(self.active):
            for j in list(self.active):
                if i == j:
                    continue
                if min(self.UCB[i, j], self.UCBhyb[i, j]) < 0.5:
                    to_remove.add(i)
                    break
        for i in to_remove:
            if i in self.active:
                self.active.remove(i)

    # ------------------------------------------------------------------
    def select_arm(self):
        """
        Select arm A(t) = argmin_{ai in C} Ti(t)  (Line 2 of Algorithm 2).
        """
        if len(self.active) == 0:
            return 0
        counts = [(self.T_i[i], i) for i in self.active]
        return min(counts)[1]

    # ------------------------------------------------------------------
    def update(self, arm, reward, t):
        """Record reward, update means, refresh UCBs, and run elimination."""
        self.T_i[arm] += 1
        self.sum_rewards[arm] += reward
        self.mu_hat[arm] = self.sum_rewards[arm] / self.T_i[arm]

        self._update_ucbs(t)
        self._eliminate()


# ===========================================================================
# Run HybElimUCB-RA  (full experiment loop)
# ===========================================================================

def run_hybucb_elimra(K, T, mu_on, offline_data, V=None, Vi=None,
                      delta_t_mode=0.05, rng=None):
    """
    Full run of HybElimUCB-RA, returning cumulative regret.

    Parameters
    ----------
    K            : number of arms
    T            : horizon
    mu_on        : online reward means
    offline_data : dict { (i,j) -> np.ndarray of outcomes in {0,1} }
    V            : pairwise bias matrix (K x K), optional
    Vi           : arm-level bias vector (K,), optional
    delta_t_mode : "paper" or float
    rng          : numpy RNG

    Returns
    -------
    cumreg : np.ndarray (T,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sub-optimality gaps  (BT-based, consistent with paper's definition)
    p1i = np.array([bt_prob(mu_on[0], mu_on[i]) for i in range(K)])
    delta_arms = p1i - 0.5   # delta[0] = 0

    algo = HybElimUCB_RA(K, offline_data, V=V, Vi=Vi, delta_t_mode=delta_t_mode)
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        arm = algo.select_arm()
        reward = sample_reward(arm, mu_on, rng)
        algo.update(arm, reward, t)

        total_reg += delta_arms[arm]
        cumreg[t - 1] = total_reg

    return cumreg


def run_hybucb_elimra_no_offline(K, T, mu_on, delta_t_mode=0.05, rng=None):
    """HybElimUCB-RA with no offline data (Ni,j = 0)."""
    empty_data = {(i, j): np.array([]) for i in range(K) for j in range(K) if i != j}
    return run_hybucb_elimra(K, T, mu_on, empty_data,
                             V=np.zeros((K, K)),
                             delta_t_mode=delta_t_mode, rng=rng)
