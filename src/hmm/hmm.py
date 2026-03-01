"""HMM model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from numpy import random as rand

from hmm.algorithms import ComputeMode, forward


class HMMClassifier:
    """
    A binary hmm classifier that uses two hmms: one corresponding
    to the positive activity and one corresponding to the negative
    activity.
    """

    def __init__(
        self,
        neg_hmm: HMM | None = None,
        pos_hmm: HMM | None = None,
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
        """
        if self.pos_hmm is None or self.neg_hmm is None:
            raise ValueError("pos/neg hmm(s) missing")

        pos_hmm = self.pos_hmm
        neg_hmm = self.neg_hmm

        pos_ll = forward(pos_hmm, sample, mode=ComputeMode.SCALED)[0]
        neg_ll = forward(neg_hmm, sample, mode=ComputeMode.SCALED)[0]

        return pos_ll - neg_ll

    def add_pos_hmm(self, pos_hmm: HMM) -> None:
        """
        Add the hmm corresponding to positive
        activity.  Replaces current positive hmm, if it exists.

        Args:
            pos_hmm: the HMM to add for positive classification
        """
        self.pos_hmm = pos_hmm

    def add_neg_hmm(self, neg_hmm: HMM) -> None:
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
        self.symbol_map: dict[int, int] = {v: i for i, v in enumerate(self.V)}

        if A is not None:
            self.A = np.array(A, dtype=float)
            assert np.shape(self.A) == (self.N, self.N)
        else:
            A_raw = rand.uniform(size=self.N * self.N).reshape((self.N, self.N))
            self.A = A_raw / A_raw.sum(axis=1, keepdims=True)
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
                for i in self.F:
                    self.B[i, :] = self.F[i]
            else:
                self.F = {}
        else:
            B_raw = rand.uniform(0, 1, self.N * self.M).reshape((self.N, self.M))
            self.B = B_raw / B_raw.sum(axis=1, keepdims=True)

            if F is not None:
                self.F = dict(F)
                for i in self.F:
                    self.B[i, :] = self.F[i]
            else:
                self.F = {}

        if Pi is not None:
            self.Pi = np.array(Pi, dtype=float)
            assert len(self.Pi) == self.N
        else:
            self.Pi = np.array(1 / self.N).repeat(self.N)

        if Labels is not None:
            self.Labels = list(Labels)
        else:
            self.Labels = list(range(self.N))

    def get_emission_probs(self, obs_t: int | npt.NDArray) -> npt.NDArray:
        """Returns emission probabilities for all states given observation obs_t.

        Args:
            obs_t: Observation symbol (must be in V) or array for continuous

        Returns:
            Array of shape (N,) with emission probabilities for each state
        """
        if not isinstance(obs_t, int):
            obs_t = int(np.squeeze(obs_t))
        return self.B[:, self.symbol_map[obs_t]]

    def get_all_emission_probs(self, obs_seq: Sequence[int] | Sequence[npt.NDArray]) -> npt.NDArray:
        """Returns emission probabilities for all states across entire observation sequence.

        Args:
            obs_seq: Observation sequence (list of symbols)

        Returns:
            Array of shape (N, T) with emission probabilities for each state and time step
        """
        indices = [self.symbol_map[o] for o in obs_seq]
        return self.B[:, indices]

    def m_step(
        self,
        obs_seqs: list[Sequence[npt.NDArray]],
        gammas: list[npt.NDArray],
        xis: list[npt.NDArray],
        update_pi: bool = True,
        update_a: bool = True,
        update_b: bool = True,
    ) -> None:
        """Perform M-step for discrete HMM."""
        if not gammas:
            return

        expect_si_t0_all = np.zeros(self.N, dtype=float)
        expect_si_all_TM1 = np.zeros(self.N, dtype=float)
        expected_transitions = np.zeros([self.N, self.N], dtype=float)
        expect_si_vk_all = np.zeros([self.N, self.M], dtype=float)

        for obs, gamma, xi in zip(obs_seqs, gammas, xis):
            obs_symbols = np.array([self.symbol_map[o] for o in obs])
            T = len(obs)

            expect_si_t0_all += gamma[:, 0]
            expect_si_all_TM1 += gamma[:, : T - 1].sum(1)
            expected_transitions += xi[:, :, : T - 1].sum(2)

            if update_b:
                B_bar = np.zeros([self.N, self.M], dtype=float)
                for k in range(self.M):
                    which = obs_symbols == k
                    B_bar[:, k] = gamma[:, which].sum(1)
                expect_si_vk_all += B_bar

        if update_pi:
            self.Pi = expect_si_t0_all / np.sum(expect_si_t0_all)

        if update_a:
            for i in range(self.N):
                if expect_si_all_TM1[i] > 0:
                    self.A[i, :] = expected_transitions[i, :] / expect_si_all_TM1[i]

        if update_b:
            for i in range(self.N):
                row_sum = expect_si_vk_all[i, :].sum()
                if row_sum > 0:
                    expect_si_vk_all[i, :] = expect_si_vk_all[i, :] / row_sum

            self.B = expect_si_vk_all

            for i in self.F:
                self.B[i, :] = self.F[i]

    def __repr__(self) -> str:
        retn = ""
        retn += f"num hiddens: {self.N}\n"
        retn += f"symbols: {self.V}\n"
        retn += f"\nA:\n {self.A}\n"
        retn += f"Pi:\n {self.Pi}"
        return retn
