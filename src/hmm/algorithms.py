"""HMM algorithms: forward, backward, viterbi, baum_welch."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from einops import rearrange
from scipy.special import logsumexp

from hmm.base import HMMProtocol

EPSILON = np.finfo(float).tiny


class ComputeMode(Enum):
    """Computation mode for HMM algorithms."""

    SCALED = "scaled"  # Rabiner's scaling coefficients
    LOG = "log"  # Log-domain with logsumexp
    UNSCALED = "unscaled"  # Original (risks underflow)


if TYPE_CHECKING:
    from hmm.continuous import GaussianHMM, MixtureGaussianHMM
    from hmm.hmm import HMM

    AnyHMM = HMM | GaussianHMM | MixtureGaussianHMM


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
    hmm: AnyHMM,
    obs: Sequence[int] | Sequence[npt.NDArray],
    mode: ComputeMode = ComputeMode.SCALED,
) -> tuple[float, npt.NDArray, npt.NDArray | None]:
    """Calculate the probability of an observation sequence, Obs, given the model.

    Implements the forward algorithm from Rabiner (1989).

    Three computation modes are supported:
    - "scaled": Use Rabiner's scaling coefficients c_t for numerical stability
    - "log": Use log-domain computations with logsumexp (modern approach)
    - "unscaled": Original unscaled computations (risks underflow for long sequences)

    Args:
        hmm: the HMM model (discrete HMM or GaussianHMM)
        obs: observation sequence (integers for discrete, arrays for continuous)
        mode: computation mode - "scaled" (default), "log", or "unscaled"

    Returns:
        For "scaled" and "log" modes: (log_prob_Obs, Alpha, None)
        For "unscaled" mode: (prob_Obs, Alpha, None)
    """
    T = len(obs)

    if mode == ComputeMode.LOG:
        log_Pi = np.log(np.maximum(hmm.Pi, EPSILON))
        log_A = np.log(np.maximum(hmm.A, EPSILON))

        log_alpha = np.zeros([hmm.N, T], dtype=float)

        log_alpha[:, 0] = log_Pi + np.log(np.maximum(hmm.get_emission_probs(obs[0]), EPSILON))

        for t in range(1, T):
            log_alpha[:, t] = logsumexp(log_alpha[:, t - 1, None] + log_A, axis=0) + np.log(
                np.maximum(hmm.get_emission_probs(obs[t]), EPSILON)
            )

        log_prob_Obs = logsumexp(log_alpha[:, T - 1])
        return (float(log_prob_Obs), log_alpha, None)

    c: npt.NDArray | None = None
    if mode == ComputeMode.SCALED:
        c = np.zeros(T, dtype=float)

    alpha = np.zeros([hmm.N, T], dtype=float)

    alpha[:, 0] = hmm.Pi * hmm.get_emission_probs(obs[0])

    if mode == ComputeMode.SCALED and c is not None:
        c[0] = 1.0 / np.maximum(np.sum(alpha[:, 0]), EPSILON)
        alpha[:, 0] = c[0] * alpha[:, 0]

    for t in range(1, T):
        alpha[:, t] = np.dot(alpha[:, t - 1], hmm.A) * hmm.get_emission_probs(obs[t])

        if mode == ComputeMode.SCALED and c is not None:
            c[t] = 1.0 / np.maximum(np.sum(alpha[:, t]), EPSILON)
            alpha[:, t] = alpha[:, t] * c[t]

    if mode == ComputeMode.SCALED and c is not None:
        log_prob_Obs = -np.sum(np.log(c))
        return (float(log_prob_Obs), alpha, c)
    else:
        prob_Obs = np.sum(alpha[:, T - 1])
        return (float(prob_Obs), alpha)


def backward(
    hmm: AnyHMM,
    obs: Sequence[int] | Sequence[npt.NDArray],
    mode: ComputeMode = ComputeMode.SCALED,
    scaling_coeffs: npt.NDArray | None = None,
) -> npt.NDArray:
    """Calculate the probability of a partial observation sequence from t+1 to T.

    Implements the backward algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model
        obs: observation sequence
        mode: computation mode - "scaled" (default), "log", or "unscaled"
        scaling_coeffs: scaling coefficients from forward algorithm (if already computed)

    Returns:
        Beta: backward variable matrix (N x T)
    """
    T = len(obs)

    if mode == ComputeMode.LOG:
        log_A = np.log(np.maximum(hmm.A, EPSILON))

        log_beta = np.zeros([hmm.N, T], dtype=float)
        log_beta[:, T - 1] = 0.0

        for t in reversed(range(T - 1)):
            log_beta[:, t] = logsumexp(
                log_A
                + log_beta[:, t + 1]
                + np.log(np.maximum(hmm.get_emission_probs(obs[t + 1]), EPSILON)),
                axis=1,
            )

        return log_beta

    c: npt.NDArray | None = scaling_coeffs
    if c is None:
        if mode == ComputeMode.SCALED:
            c = np.zeros(T, dtype=float)

    beta = np.zeros([hmm.N, T], dtype=float)
    beta[:, T - 1] = 1.0

    if mode == ComputeMode.SCALED and c is not None and scaling_coeffs is None:
        c[T - 1] = 1.0
        beta[:, T - 1] = beta[:, T - 1] * c[T - 1]
    elif mode == ComputeMode.SCALED and c is not None:
        beta[:, T - 1] = beta[:, T - 1] * c[T - 1]

    for t in reversed(range(T - 1)):
        beta[:, t] = np.dot(hmm.A, (hmm.get_emission_probs(obs[t + 1]) * beta[:, t + 1]))

        if mode == ComputeMode.SCALED and c is not None and scaling_coeffs is None:
            c[t] = 1.0 / np.maximum(np.sum(beta[:, t]), EPSILON)
            beta[:, t] = beta[:, t] * c[t]
        elif mode == ComputeMode.SCALED and c is not None:
            beta[:, t] = beta[:, t] * c[t]

    return beta


def viterbi(
    hmm: AnyHMM,
    obs: Sequence[int] | Sequence[npt.NDArray],
    mode: ComputeMode = ComputeMode.SCALED,
) -> tuple[list[int], npt.NDArray, npt.NDArray]:
    """Calculate P(Q|Obs, hmm) and yield the state sequence Q* that maximizes this probability.

    Implements the Viterbi algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model
        obs: observation sequence
        mode: computation mode - "scaled" (default), "log", or "unscaled"

    Returns:
        (Q_star, Delta, Psi) where:
            - Q_star: optimal state sequence (list of state indices)
            - Delta: max probability matrix (N x T)
            - Psi: backpointer matrix (N x T)
    """
    T = len(obs)

    delta = np.zeros([hmm.N, T], dtype=float)

    if mode in (ComputeMode.SCALED, ComputeMode.LOG):
        delta[:, 0] = np.log(np.maximum(hmm.Pi, EPSILON)) + np.log(
            np.maximum(hmm.get_emission_probs(obs[0]), EPSILON)
        )
    else:
        delta[:, 0] = hmm.Pi * hmm.get_emission_probs(obs[0])

    psi = np.zeros([hmm.N, T], dtype=int)

    if mode in (ComputeMode.SCALED, ComputeMode.LOG):
        for t in range(1, T):
            nus = rearrange(delta[:, t - 1], "n -> n 1") + np.log(np.maximum(hmm.A, EPSILON))
            delta[:, t] = nus.max(0) + np.log(np.maximum(hmm.get_emission_probs(obs[t]), EPSILON))
            psi[:, t] = nus.argmax(0)
    else:
        for t in range(1, T):
            nus = rearrange(delta[:, t - 1], "n -> n 1") * hmm.A
            delta[:, t] = nus.max(0) * hmm.get_emission_probs(obs[t])
            psi[:, t] = nus.argmax(0)

    q_star = [int(np.argmax(delta[:, T - 1]))]
    for t in reversed(range(T - 1)):
        q_star.insert(0, int(psi[q_star[0], t + 1]))

    return (q_star, delta, psi)


def baum_welch(
    hmm: HMMProtocol,
    obs_seqs: list[Sequence[npt.NDArray]],
    epochs: int = 20,
    val_set: list[Sequence[npt.NDArray]] | None = None,
    update_pi: bool = True,
    update_a: bool = True,
    update_b: bool = True,
    mode: ComputeMode = ComputeMode.SCALED,
    graph: bool = False,
    fname: str = "ll.eps",
    verbose: bool = False,
) -> HMMProtocol:
    """EM algorithm to train the HMM.

    Implements the Baum-Welch algorithm from Rabiner (1989).
    Uses the HMMProtocol to delegate M-step to the model.

    Args:
        hmm: HMM model (discrete HMM or GaussianHMM) to train
        obs_seqs: list of observation sequences to train over
        epochs: number of iterations to perform EM (default: 20)
        val_set: validation data set (optional, for early stopping)
        update_pi: flag to update initial state probabilities (default: True)
        update_a: flag to update transition probabilities (default: True)
        update_b: flag to update observation emission probabilities (default: True)
        mode: computation mode - "scaled", "log", or "unscaled" (default: scaled)
        graph: flag to plot log-likelihoods (default: False)
        fname: file name to save plot figure (default: "ll.eps")
        verbose: flag to print training progress (default: False)

    Returns:
        Trained HMM model
    """
    TOL = 1e-4

    LLs: list[float] = []
    val_LLs: list[float] = []
    best_hmm = copy.deepcopy(hmm)
    best_epoch = -1
    best_val_LL = float("-inf") if val_set is not None else None

    for epoch in range(epochs):
        LL_epoch = 0.0

        gammas: list[npt.NDArray] = []
        xis: list[npt.NDArray] = []

        for obs in obs_seqs:
            obs_list = list(obs)

            forward_result = forward(hmm=hmm, obs=obs_list, mode=mode)
            log_prob_obs, alpha, c = forward_result
            beta = backward(hmm=hmm, obs=obs_list, mode=mode, scaling_coeffs=c)

            LL_epoch += log_prob_obs

            if mode == ComputeMode.LOG:
                log_gamma_raw = alpha + beta
                gamma = np.exp(log_gamma_raw - logsumexp(log_gamma_raw, axis=0))
            else:
                gamma_raw = alpha * beta
                gamma = gamma_raw / np.maximum(gamma_raw.sum(0), EPSILON)
            gammas.append(gamma)

            B_next = np.column_stack([hmm.get_emission_probs(o) for o in obs_list[1:]])

            if mode == ComputeMode.LOG:
                log_B = np.log(np.maximum(B_next, EPSILON))
                log_alpha = alpha[:, :-1]
                log_beta = beta[:, 1:]
                log_xi = (
                    rearrange(log_alpha, "n t -> t n 1")
                    + np.log(hmm.A)
                    + log_B
                    + rearrange(log_beta, "n t -> t 1 n")
                )
                xi = np.exp(log_xi - logsumexp(log_xi, axis=(1, 2), keepdims=True))
            else:
                xi = np.einsum(
                    "it, ij, jt, jt -> ijt",
                    alpha[:, :-1],
                    hmm.A,
                    B_next,
                    beta[:, 1:],
                )
                xi /= np.maximum(xi.sum(axis=(0, 1), keepdims=True), EPSILON)

            xis.append(xi)

        hmm.m_step(
            obs_seqs=obs_seqs,
            gammas=gammas,
            xis=xis,
            update_pi=update_pi,
            update_a=update_a,
            update_b=update_b,
        )

        LLs.append(LL_epoch)

        if epoch > 1 and abs(LLs[epoch] - LLs[epoch - 1]) < TOL:
            if verbose:
                print("Log-likelihoods have converged.")
            break

        if val_set is not None:
            val_LL_epoch = 0.0
            for val_obs in val_set:
                val_obs_list = list(val_obs)
                val_LL_epoch += forward(hmm=hmm, obs=val_obs_list, mode=mode)[0]
            val_LLs.append(val_LL_epoch)

            if best_val_LL is None or val_LL_epoch > best_val_LL:
                best_hmm = copy.deepcopy(hmm)
                best_epoch = epoch
                best_val_LL = val_LL_epoch

        if verbose:
            print(f"Finished epoch {epoch + 1}, LL: {LL_epoch}")
            if val_set is not None:
                print(f"  Validation LL: {val_LLs[epoch]}")
                if epoch == best_epoch:
                    print(f"  -> New best validation at epoch {epoch + 1}")

    if graph:
        import matplotlib.pyplot as plt

        if val_set is not None:
            plt.figure()
            plt.subplot(211)
            plt.title("Training Reestimation Performance")
            plt.xlabel("Epochs")
            plt.ylabel(r"$\log( P ( O | \lambda ) )$")
            plt.plot(LLs, label="Training data", color="red")
            plt.subplots_adjust(hspace=0.4)
            plt.subplot(212)
            plt.title("Validation Reestimation Performance")
            plt.plot(val_LLs, label="Validation LL", color="blue")
            plt.xlabel("Epochs")
            plt.ylabel(r"$\log( P ( O | \lambda ) )$")
            plt.axvline(best_epoch, color="black", label="Best validation LL", linewidth=2)
            plt.legend(labelsep=0.01, shadow=1, loc="lower right")
            plt.savefig(fname)
        else:
            plt.figure()
            plt.title("Training Reestimation Performance")
            plt.xlabel("Epochs")
            plt.ylabel(r"$\log( P ( O | \lambda ) )$")
            plt.plot(LLs, label="Training data", color="red")
            plt.savefig(fname)

    if val_set is not None:
        hmm.A = best_hmm.A
        hmm.Pi = best_hmm.Pi

    return hmm
