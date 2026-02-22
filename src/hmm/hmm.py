"""
Python module for creating, training and applying hidden
Markov models to discrete or continuous observations.
Author: Michael Hamilton,  mike.hamilton7@gmail.com

Theoretical concepts obtained from Rabiner (1989).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from einops import rearrange
from numpy import random as rand


class HMMClassifier:
    """
    A binary hmm classifier that uses two hmms: one corresponding
    to the positive activity and one corresponding to the negative
    activity.
    """

    def __init__(
        self,
        neg_hmm: "HMM | None" = None,
        pos_hmm: "HMM | None" = None,
    ) -> None:
        """
        Args:
            neg_hmm: hmm corresponding to negative activity
            pos_hmm: hmm corresponding to positive activity
        """
        self.neg_hmm = neg_hmm
        self.pos_hmm = pos_hmm

    def classify(self, sample: Sequence[int]) -> float:
        """
        Classification is performed by calculating the
        log odds for the positive activity.  Since the hmms
        return a log-likelihood (due to scaling)
        of the corresponding activity, the difference of
        the two log-likelihoods is the log odds.

        Args:
            sample: observation sequence

        Returns:
            log odds (positive - negative log-likelihood)

        Raises:
            ValueError: if pos_hmm or neg_hmm is missing
            NotImplementedError: if algorithms not yet implemented
        """
        if self.pos_hmm is None or self.neg_hmm is None:
            raise ValueError("pos/neg hmm(s) missing")

        pos_hmm = self.pos_hmm
        neg_hmm = self.neg_hmm
        pos_ll = forward(pos_hmm, sample, scaling=True)[0]
        neg_ll = forward(neg_hmm, sample, scaling=True)[0]

        return pos_ll - neg_ll

    def add_pos_hmm(self, pos_hmm: "HMM") -> None:
        """
        Add the hmm corresponding to positive
        activity.  Replaces current positive hmm, if it exists.

        Args:
            pos_hmm: the HMM to add for positive classification
        """
        self.pos_hmm = pos_hmm

    def add_neg_hmm(self, neg_hmm: "HMM") -> None:
        """
        Add the hmm corresponding to negative
        activity.  Replaces current negative hmm, if it exists.

        Args:
            neg_hmm: the HMM to add for negative classification
        """
        self.neg_hmm = neg_hmm


class HMM:
    """
    Creates and maintains a hidden Markov model.  This version assumes the every state can be
    reached DIRECTLY from any other state (ergodic).  This, of course, excludes the start state.
    Hence the state transition matrix, A, must be N X N .  The observable symbol probability
    distributions are represented by an N X M matrix where M is the number of observation
    symbols.

                  |a_11 a_12 ... a_1N|                   |b_11 b_12 ... b_1M|
                  |a_21 a_22 ... a_2N|                   |b_21 b_22 ... b_2M|
              A = | .    .        .  |               B = | .    .        .  |
                  | .         .   .  |                   | .         .   .  |
                  |a_N1 a_N2 ... a_NN|                   |b_N1 b_N2 ... b_NM|

           a_ij = P(q_t = S_j|q_t-1 = S_i)       b_ik = P(v_k at t|q_t = S_i)
        where q_t is state at time t and v_k is k_th symbol of observation sequence


    Attributes:
        N: Number of hidden states
        M: Number of observation symbols
        V: List of observable symbols
        A: Transition probability matrix (N x N)
        B: Emission probability matrix (N x M)
        Pi: Initial state distribution (N,)
        F: Fixed emission probabilities (dict of state -> array)
        Labels: State labels
        symbol_map: Mapping from symbols to indices
    """

    def __init__(
        self,
        n_states: int = 1,
        V: Sequence[int] | None = None,
        A: npt.NDArray | None = None,
        B: npt.NDArray | None = None,
        Pi: npt.NDArray | None = None,
        F: Mapping[int, npt.NDArray] | None = None,
        Labels: Sequence[int] | None = None,
    ) -> None:
        """
        Args:
            n_states: number of hidden states
            V: list of all observable symbols
            A: transition matrix (N x N)
            B: observable symbol probability distribution (N x M)
            Pi: initial state distribution (N,)
            F: Fixed emission probabilities (dict: state_idx -> numpy.array)
            Labels: state labels
        """
        if V is None:
            raise ValueError("V (observable symbols) must be provided")

        self.N = n_states

        self.V = list(V)
        self.M = len(self.V)
        self.symbol_map = dict(zip(self.V, range(len(self.V))))

        if A is not None:
            self.A = np.array(A, dtype=float)
            assert np.shape(self.A) == (self.N, self.N)
        else:
            raw_A = rand.uniform(size=self.N * self.N).reshape((self.N, self.N))
            self.A = (raw_A.T / raw_A.T.sum(0)).T
            if n_states == 1:
                self.A = self.A.reshape((1, 1))

        if B is not None:
            self.B = np.array(B, dtype=float)
            if n_states > 1:
                assert np.shape(self.B) == (self.N, self.M)
            else:
                self.B = np.reshape(self.B, (1, self.M))

            if F is not None:
                self.F = dict(F)
                for i in self.F.keys():
                    self.B[i, :] = self.F[i]
            else:
                self.F = {}
        else:
            B_raw = rand.uniform(0, 1, self.N * self.M).reshape((self.N, self.M))
            self.B = (B_raw.T / B_raw.T.sum(0)).T

            if F is not None:
                self.F = dict(F)
                for i in self.F.keys():
                    self.B[i, :] = self.F[i]
            else:
                self.F = {}

        if Pi is not None:
            self.Pi = np.array(Pi, dtype=float)
            assert len(self.Pi) == self.N
        else:
            self.Pi = np.array(1.0 / self.N).repeat(self.N)

        if Labels is not None:
            self.Labels = list(Labels)
        else:
            self.Labels = list(range(self.N))

        if F is not None:
            self.F = dict(F)
            for i in self.F.keys():
                self.B[i, :] = self.F[i]
        else:
            self.F = {}

    def __repr__(self) -> str:
        retn = ""
        retn += f"num hiddens: {self.N}\n"
        retn += f"symbols: {self.V}\n"
        retn += f"\nA:\n {self.A}\n"
        retn += f"Pi:\n {self.Pi}"
        return retn


def symbol_index(hmm: HMM, obs: Sequence[int]) -> list[int]:
    """
    Converts an observation symbol sequence into a sequence
    of indices for accessing distribution matrices.

    Args:
        hmm: the HMM model
        obs: observation sequence

    Returns:
        List of indices corresponding to observation symbols

    Raises:
        KeyError: if an observation symbol is not in the model's symbol map
    """
    obs_ind = []
    for o in obs:
        obs_ind.append(hmm.symbol_map[o])

    return obs_ind


def forward(
    hmm: HMM,
    obs: Sequence[int],
    scaling: bool = True,
) -> tuple[float, npt.NDArray, npt.NDArray] | tuple[float, npt.NDArray]:
    """
    Calculate the probability of an observation sequence, Obs,
    given the model, P(Obs|hmm).

    Implements the forward algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model
        obs: observation sequence
        scaling: whether to use scaling (recommended for numerical stability)

    Returns:
        If scaling=True: (log_prob_Obs, Alpha, c) where:
            - log_prob_Obs: log probability of the observation sequence
            - Alpha: forward variable matrix (N x T)
            - c: scaling coefficients (T,)
        If scaling=False: (prob_Obs, Alpha) where:
            - prob_Obs: probability of the observation sequence
            - Alpha: forward variable matrix (N x T)
    """
    T = len(obs)

    obs_indices = symbol_index(hmm, obs)

    c: npt.NDArray | None = None
    if scaling:
        c = np.zeros(T, dtype=float)

    alpha = np.zeros([hmm.N, T], dtype=float)

    alpha[:, 0] = hmm.Pi * hmm.B[:, obs_indices[0]]

    if scaling and c is not None:
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] = c[0] * alpha[:, 0]

    for t in range(1, T):
        alpha[:, t] = np.dot(alpha[:, t - 1], hmm.A) * hmm.B[:, obs_indices[t]]

        if scaling and c is not None:
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] * c[t]

    if scaling and c is not None:
        log_prob_Obs = -np.sum(np.log(c))
        return (log_prob_Obs, alpha, c)
    else:
        prob_Obs = np.sum(alpha[:, T - 1])
        return (prob_Obs, alpha)


def backward(
    hmm: HMM,
    obs: Sequence[int],
    c: npt.NDArray | None = None,
) -> npt.NDArray:
    """
    Calculate the probability of a partial observation sequence
    from t+1 to T, given some state t.

    Implements the backward algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model
        obs: observation sequence
        c: scaling coefficients from forward algorithm (if using scaling)

    Returns:
        Beta: backward variable matrix (N x T)
    """
    T = len(obs)

    obs_indices = symbol_index(hmm, obs)

    beta = np.zeros([hmm.N, T], dtype=float)
    beta[:, T - 1] = 1.0
    if c is not None:
        beta[:, T - 1] = beta[:, T - 1] * c[T - 1]

    for t in reversed(range(T - 1)):
        beta[:, t] = np.dot(hmm.A, (hmm.B[:, obs_indices[t + 1]] * beta[:, t + 1]))

        if c is not None:
            beta[:, t] = beta[:, t] * c[t]

    return beta


def viterbi(
    hmm: HMM,
    obs: Sequence[int],
    scaling: bool = True,
) -> tuple[list[int], npt.NDArray, npt.NDArray]:
    """
    Calculate P(Q|Obs, hmm) and yield the state sequence Q* that
    maximizes this probability.

    Implements the Viterbi algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model
        obs: observation sequence
        scaling: whether to use log scaling (recommended)

    Returns:
        (Q_star, Delta, Psi) where:
            - Q_star: optimal state sequence (list of state indices)
            - Delta: max probability matrix (N x T)
            - Psi: backpointer matrix (N x T)
    """
    T = len(obs)

    obs_indices = symbol_index(hmm, obs)

    delta = np.zeros([hmm.N, T], dtype=float)

    if scaling:
        delta[:, 0] = np.log(hmm.Pi) + np.log(hmm.B[:, obs_indices[0]])
    else:
        delta[:, 0] = hmm.Pi * hmm.B[:, obs_indices[0]]

    psi = np.zeros([hmm.N, T], dtype=int)

    if scaling:
        for t in range(1, T):
            # Use rearrange for explicit broadcasting: prev_state -> (prev_state, curr_state)
            # delta[prev_state] + log(A[prev_state, curr_state])
            nus = rearrange(delta[:, t - 1], "n -> n 1") + np.log(hmm.A)
            delta[:, t] = nus.max(0) + np.log(hmm.B[:, obs_indices[t]])
            psi[:, t] = nus.argmax(0)
    else:
        for t in range(1, T):
            nus = rearrange(delta[:, t - 1], "n -> n 1") * hmm.A
            delta[:, t] = nus.max(0) * hmm.B[:, obs_indices[t]]
            psi[:, t] = nus.argmax(0)

    q_star = [int(np.argmax(delta[:, T - 1]))]
    for t in reversed(range(T - 1)):
        q_star.insert(0, int(psi[q_star[0], t + 1]))

    return (q_star, delta, psi)


def baum_welch(
    hmm: HMM,
    obs_seqs: list[Sequence[int]],
    epochs: int = 20,
    val_set: list[Sequence[int]] | None = None,
    update_pi: bool = True,
    update_a: bool = True,
    update_b: bool = True,
    scaling: bool = True,
    graph: bool = False,
    norm_update: bool = False,
    fname: str = "ll.eps",
    verbose: bool = False,
) -> HMM:
    """
    EM algorithm to update Pi, A, and B for the HMM.

    Implements the Baum-Welch algorithm from Rabiner (1989).

    Args:
        hmm: HMM model to train
        obs_seqs: list of observation sequences to train over
        epochs: number of iterations to perform EM (default: 20)
        val_set: validation data set (optional, for early stopping)
        update_pi: flag to update initial state probabilities (default: True)
        update_a: flag to update transition probabilities (default: True)
        update_b: flag to update observation emission probabilities (default: True)
        scaling: flag to scale probabilities (default: True, recommended)
        graph: flag to plot log-likelihoods (default: False)
        norm_update: flag to use normalized update weights (default: False)
        fname: file name to save plot figure (default: "ll.eps")
        verbose: flag to print training progress (default: False)

    Returns:
        Trained HMM model
    """
    import copy

    LLs: list[float] = []
    val_LLs: list[float] = []
    best_A = copy.deepcopy(hmm.A)
    best_B = copy.deepcopy(hmm.B)
    best_Pi = copy.deepcopy(hmm.Pi)
    best_epoch = -1
    best_val_LL = float("-inf") if val_set is not None else None

    for epoch in range(epochs):
        LL_epoch = 0.0
        expect_si_all = np.zeros(hmm.N, dtype=float)
        expect_si_all_TM1 = np.zeros(hmm.N, dtype=float)
        expect_si_sj_all = np.zeros([hmm.N, hmm.N], dtype=float)
        expect_si_sj_all_TM1 = np.zeros([hmm.N, hmm.N], dtype=float)
        expect_si_t0_all = np.zeros(hmm.N, dtype=float)
        expect_si_vk_all = np.zeros([hmm.N, hmm.M], dtype=float)

        for obs in obs_seqs:
            obs = list(obs)
            forward_result = forward(hmm=hmm, obs=obs, scaling=scaling)
            log_prob_obs = forward_result[0]
            alpha = forward_result[1]
            c = forward_result[2] if scaling and len(forward_result) > 2 else None
            beta = backward(hmm=hmm, obs=obs, c=c)

            LL_epoch += log_prob_obs
            T = len(obs)

            if norm_update:
                w_k = 1.0 / -(log_prob_obs + np.log(len(obs)))
            else:
                w_k = 1.0

            obs_symbols = obs[:]
            obs_indices = symbol_index(hmm, obs)

            gamma_raw = alpha * beta
            gamma = gamma_raw / gamma_raw.sum(0)

            expect_si_t0_all += w_k * gamma[:, 0]
            expect_si_all += w_k * gamma.sum(1)
            expect_si_all_TM1 += w_k * gamma[:, : T - 1].sum(1)

            xi = np.zeros([hmm.N, hmm.N, T - 1], dtype=float)
            for t in range(T - 1):
                for i in range(hmm.N):
                    xi[i, :, t] = (
                        alpha[i, t] * hmm.A[i, :] * hmm.B[:, obs_indices[t + 1]] * beta[:, t + 1]
                    )

                if not scaling:
                    xi[:, :, t] = xi[:, :, t] / xi[:, :, t].sum()

            expect_si_sj_all += w_k * xi.sum(2)
            expect_si_sj_all_TM1 += w_k * xi[:, :, : T - 1].sum(2)

            if update_b:
                B_bar = np.zeros([hmm.N, hmm.M], dtype=float)
                for k in range(hmm.M):
                    which = np.array([hmm.V[k] == x for x in obs_symbols])
                    B_bar[:, k] = gamma.T[which, :].sum(0)

                expect_si_vk_all += w_k * B_bar

        if update_pi:
            expect_si_t0_all = expect_si_t0_all / np.sum(expect_si_t0_all)
            hmm.Pi = expect_si_t0_all

        if update_a:
            A_bar = np.zeros([hmm.N, hmm.N], dtype=float)
            for i in range(hmm.N):
                if expect_si_all_TM1[i] > 0:
                    A_bar[i, :] = expect_si_sj_all_TM1[i, :] / expect_si_all_TM1[i]
            hmm.A = A_bar

        if update_b:
            for i in range(hmm.N):
                if expect_si_all[i] > 0:
                    expect_si_vk_all[i, :] = expect_si_vk_all[i, :] / expect_si_all[i]

            hmm.B = expect_si_vk_all

            for i in hmm.F.keys():
                hmm.B[i, :] = hmm.F[i]

        LLs.append(LL_epoch)

        if epoch > 1:
            if LLs[epoch - 1] == LL_epoch:
                if verbose:
                    print("Log-likelihoods have plateaued--terminating training")
                break

        if val_set is not None:
            val_LL_epoch = 0.0
            for val_obs in val_set:
                val_obs = list(val_obs)
                val_LL_epoch += forward(hmm=hmm, obs=val_obs, scaling=True)[0]
            val_LLs.append(val_LL_epoch)

            if val_LL_epoch > (best_val_LL or float("-inf")):
                best_A = copy.deepcopy(hmm.A)
                best_B = copy.deepcopy(hmm.B)
                best_Pi = copy.deepcopy(hmm.Pi)
                best_epoch = epoch
                best_val_LL = val_LL_epoch

        if verbose:
            print(f"Finished epoch {epoch + 1}, LL: {LL_epoch}")
            if val_set is not None:
                print(f"  Validation LL: {val_LLs[epoch]}")

    if graph:
        import pylab  # type: ignore[import-untyped]

        if val_set is not None:
            pylab.figure()
            pylab.subplot(211)
            pylab.title("Training Reestimation Performance")
            pylab.xlabel("Epochs")
            pylab.ylabel(r"$\log( P ( O | \lambda ) )$")
            pylab.plot(LLs, label="Training data", color="red")
            pylab.subplots_adjust(hspace=0.4)
            pylab.subplot(212)
            pylab.title("Validation Reestimation Performance")
            pylab.plot(val_LLs, label="Validation LL", color="blue")
            pylab.xlabel("Epochs")
            pylab.ylabel(r"$\log( P ( O | \lambda ) )$")
            pylab.axvline(best_epoch, color="black", label="Best validation LL", linewidth=2)
            pylab.legend(labelsep=0.01, shadow=1, loc="lower right")
            pylab.savefig(fname)
        else:
            pylab.figure()
            pylab.title("Training Reestimation Performance")
            pylab.xlabel("Epochs")
            pylab.ylabel(r"$\log( P ( O | \lambda ) )$")
            pylab.plot(LLs, label="Training data", color="red")
            pylab.savefig(fname)

    if val_set is not None:
        hmm.A = best_A
        hmm.B = best_B
        hmm.Pi = best_Pi

    return hmm
