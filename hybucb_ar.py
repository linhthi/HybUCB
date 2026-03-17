"""
hybucb_ar.py
============
Algorithm 1: HybUCB-AR
Hybrid UCB for Dueling Bandits with Offline Absolute Feedback.

Reference: Section 4 of "Learning Across the Gap: Hybrid MAB with
Heterogeneous Offline and Online Data".
"""

import numpy as np
from environment import sigmoid, bt_prob, sample_duel


def compute_delta_t(t, K, mode="paper"):
    """
    Compute delta_t as used in the confidence bounds.

    Paper sets delta_t = 1 / (2*K*(K+1)*t^2).
    In experiments the paper also uses a fixed delta_t = 0.02
    (see Appendix G); we support both.
    """
    if mode == "paper":
        return 1.0 / (2.0 * K * (K + 1) * t ** 2)
    else:  # fixed
        return mode   # caller passes the float directly


class HybUCB_AR:
    """
    HybUCB-AR: Dueling Bandits with offline absolute feedback.

    Parameters
    ----------
    K        : number of arms
    mu_off   : offline reward means (shape K), used to pre-compute phat_off
    offline_data : dict { arm_index -> np.ndarray of rewards }
    V        : bias matrix V[i,j] >= |p_off_ij - p_ij|
               If None, use arm-level Vi and derive V[i,j] = sigma(Vi+Vj)
    Vi       : arm-level bias vector (shape K), used when V is None
    delta_t_mode : "paper" or a float for fixed delta_t
    """

    def __init__(self, K, offline_data, V=None, Vi=None, delta_t_mode=0.02):
        self.K = K
        self.delta_t_mode = delta_t_mode

        # Compute offline empirical means
        self.N = np.zeros(K, dtype=float)
        self.mu_off_hat = np.zeros(K)
        for i, data in offline_data.items():
            self.N[i] = len(data)
            self.mu_off_hat[i] = data.mean() if len(data) > 0 else 0.0

        # Offline pairwise preference estimates via BT model  (Eq. 2)
        self.p_off_hat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i != j:
                    self.p_off_hat[i, j] = sigmoid(
                        self.mu_off_hat[i] - self.mu_off_hat[j]
                    )
                else:
                    self.p_off_hat[i, j] = 0.5

        # Bias matrix V[i,j]
        if V is not None:
            self.V = np.array(V, dtype=float)
        elif Vi is not None:
            Vi = np.array(Vi, dtype=float)
            self.V = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    self.V[i, j] = sigmoid(Vi[i] + Vi[j])
        else:
            # No bias info -> V[i,j] = 1 (worst case)
            self.V = np.ones((K, K))

        # Online interaction state
        self.T_ij = np.zeros((K, K), dtype=float)   # comparison counts
        self.W_ij = np.zeros((K, K), dtype=float)   # win counts
        self.p_hat = np.zeros((K, K))               # online empirical preference

        # Effective offline count N_eff[i,j] = Ni*Nj/(Ni+Nj)
        self.N_eff = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                s = self.N[i] + self.N[j]
                self.N_eff[i, j] = (self.N[i] * self.N[j] / s) if s > 0 else 0.0

        # Initialise UCBs
        self.UCB = np.full((K, K), np.inf)
        self.UCBhyb = np.zeros((K, K))
        np.fill_diagonal(self.UCB, 0.5)
        np.fill_diagonal(self.UCBhyb, 0.5)
        self._init_ucbhyb()

    # ------------------------------------------------------------------
    def _init_ucbhyb(self):
        """
        Initialise UCBhyb before any online observations (t=0, T_ij=0).
        Eq (4) with T_ij=0:
          UCBhyb(ai,aj) = p_off_hat[i,j] + sqrt(log(1/delta_t) / (2*N_eff)) + V[i,j]
        """
        K = self.K
        # Use delta_t at t=1 for initialisation
        delta0 = self._delta(1)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                n = self.N_eff[i, j]
                if n > 0:
                    rad = np.sqrt(np.log(1.0 / delta0) / (2.0 * n))
                else:
                    rad = np.inf
                self.UCBhyb[i, j] = self.p_off_hat[i, j] + rad + self.V[i, j]

    # ------------------------------------------------------------------
    def _delta(self, t):
        if self.delta_t_mode == "paper":
            return 1.0 / (2.0 * self.K * (self.K + 1) * t ** 2)
        else:
            return float(self.delta_t_mode)

    # ------------------------------------------------------------------
    def _update_ucbs(self, t):
        """Recompute UCB and UCBhyb for all pairs after a new observation."""
        K = self.K
        delta_t = self._delta(t)

        for i in range(K):
            for j in range(K):
                if i == j:
                    self.UCB[i, j] = 0.5
                    self.UCBhyb[i, j] = 0.5
                    continue

                tij = self.T_ij[i, j]
                n_eff = self.N_eff[i, j]

                # --- Pure online UCB  (Eq. 3) ---
                if tij > 0:
                    p_hat_ij = self.W_ij[i, j] / tij
                    rad_on = 2.0 * np.sqrt(np.log(1.0 / delta_t) / (2.0 * tij))
                    self.UCB[i, j] = min(1.0, p_hat_ij + rad_on)
                else:
                    self.UCB[i, j] = np.inf

                # --- Hybrid UCB  (Eq. 4) ---
                denom = tij + n_eff
                if denom > 0:
                    # Hybrid estimate  (Eq. 2)
                    if tij > 0:
                        p_hat_ij = self.W_ij[i, j] / tij
                    else:
                        p_hat_ij = 0.0
                    alpha = tij / denom
                    p_hyb = alpha * p_hat_ij + (1.0 - alpha) * self.p_off_hat[i, j]

                    rad_hyb = np.sqrt(np.log(1.0 / delta_t) / (2.0 * denom))
                    bias_term = (n_eff / denom) * self.V[i, j]
                    self.UCBhyb[i, j] = min(1.0, p_hyb + rad_hyb + bias_term)
                else:
                    self.UCBhyb[i, j] = np.inf

    # ------------------------------------------------------------------
    def _candidate_set(self):
        """
        Build C_on, C_hyb, and C = C_on ∩ C_hyb  (Lines 7-8 of Alg 1).
        An arm a_i is in C_on  iff UCB(a_i, a_j) >= 0.5 for all j.
        An arm a_i is in C_hyb iff UCBhyb(a_i, a_j) >= 0.5 for all j.
        """
        K = self.K
        C_on = []
        C_hyb = []
        for i in range(K):
            in_on = all(self.UCB[i, j] >= 0.5 for j in range(K) if j != i)
            in_hyb = all(self.UCBhyb[i, j] >= 0.5 for j in range(K) if j != i)
            if in_on:
                C_on.append(i)
            if in_hyb:
                C_hyb.append(i)
        C = [i for i in C_on if i in C_hyb]
        return C, C_on, C_hyb

    # ------------------------------------------------------------------
    def select_pair(self):
        """
        Select arm pair (A1, A2) according to Lines 9-13 of Algorithm 1.
        """
        K = self.K
        C, _, _ = self._candidate_set()

        search_set = C if len(C) > 0 else list(range(K))

        best_val = -np.inf
        best_pair = (0, 1)
        for i in search_set:
            for j in search_set:
                if i == j:
                    continue
                val = min(self.UCB[i, j], self.UCBhyb[i, j])
                if val > best_val:
                    best_val = val
                    best_pair = (i, j)
        return best_pair

    # ------------------------------------------------------------------
    def update(self, i, j, outcome, t):
        """Record observation and refresh UCBs."""
        self.T_ij[i, j] += 1
        self.T_ij[j, i] += 1
        self.W_ij[i, j] += outcome
        self.W_ij[j, i] += 1 - outcome
        self._update_ucbs(t)


# ===========================================================================
# Run HybUCB-AR  (full experiment loop)
# ===========================================================================

def run_hybucb_ar(K, T, mu_on, offline_data, V=None, Vi=None,
                  delta_t_mode=0.02, rng=None):
    """
    Full run of HybUCB-AR, returning cumulative regret.

    Parameters
    ----------
    K           : number of arms
    T           : horizon
    mu_on       : online reward means
    offline_data: dict { arm_index -> np.ndarray }
    V           : pairwise bias matrix (K x K), optional
    Vi          : arm-level bias vector (K,), optional
    delta_t_mode: "paper" or float
    rng         : numpy RNG

    Returns
    -------
    cumreg : np.ndarray (T,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # True sub-optimality gaps
    p1j = np.array([bt_prob(mu_on[0], mu_on[j]) for j in range(K)])
    delta = p1j - 0.5   # delta[0] = 0

    algo = HybUCB_AR(K, offline_data, V=V, Vi=Vi, delta_t_mode=delta_t_mode)
    cumreg = np.zeros(T)
    total_reg = 0.0

    for t in range(1, T + 1):
        a1, a2 = algo.select_pair()
        outcome = sample_duel(a1, a2, mu_on, rng)
        algo.update(a1, a2, outcome, t)

        reg = (delta[a1] + delta[a2]) / 2.0
        total_reg += reg
        cumreg[t - 1] = total_reg

    return cumreg


def run_hybucb_ar_no_offline(K, T, mu_on, delta_t_mode=0.02, rng=None):
    """HybUCB-AR with no offline data (Ni=0 for all i)."""
    empty_data = {i: np.array([]) for i in range(K)}
    return run_hybucb_ar(K, T, mu_on, empty_data,
                         V=np.zeros((K, K)),
                         delta_t_mode=delta_t_mode, rng=rng)
