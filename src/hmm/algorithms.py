"""HMM algorithms: forward, backward, viterbi, baum_welch."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from einops import rearrange

from hmm.base import HMMProtocol

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
    scaling: bool = True,
) -> tuple[float, npt.NDArray, npt.NDArray] | tuple[float, npt.NDArray]:
    """Calculate the probability of an observation sequence, Obs, given the model.

    Implements the forward algorithm from Rabiner (1989).

    Args:
        hmm: the HMM model (discrete HMM or GaussianHMM)
        obs: observation sequence (integers for discrete, arrays for continuous)
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

    c: npt.NDArray | None = None
    if scaling:
        c = np.zeros(T, dtype=float)

    alpha = np.zeros([hmm.N, T], dtype=float)
    tiny = np.finfo(float).tiny

    alpha[:, 0] = hmm.Pi * hmm.get_emission_probs(obs[0])

    if scaling and c is not None:
        c[0] = 1.0 / np.maximum(np.sum(alpha[:, 0]), tiny)
        alpha[:, 0] = c[0] * alpha[:, 0]

    for t in range(1, T):
        alpha[:, t] = np.dot(alpha[:, t - 1], hmm.A) * hmm.get_emission_probs(obs[t])

        if scaling and c is not None:
            c[t] = 1.0 / np.maximum(np.sum(alpha[:, t]), tiny)
            alpha[:, t] = alpha[:, t] * c[t]

    if scaling and c is not None:
        log_prob_Obs = -np.sum(np.log(c))
        return (log_prob_Obs, alpha, c)
    else:
        prob_Obs = np.sum(alpha[:, T - 1])
        return (prob_Obs, alpha)


def backward(
    hmm: AnyHMM,
    obs: Sequence[int] | Sequence[npt.NDArray],
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

    # Warn if using unscaled backward for long sequences
    if c is None and T > 50:
        warnings.warn(
            f"backward: No scaling coefficients provided for long sequence (T={T}). "
            "Unscaled backward values may underflow to zero. "
            "Consider using scaling=True in forward and passing c to backward.",
            RuntimeWarning,
        )

    beta = np.zeros([hmm.N, T], dtype=float)
    beta[:, T - 1] = 1.0
    if c is not None:
        beta[:, T - 1] = beta[:, T - 1] * c[T - 1]

    for t in reversed(range(T - 1)):
        beta[:, t] = np.dot(hmm.A, (hmm.get_emission_probs(obs[t + 1]) * beta[:, t + 1]))

        if c is not None:
            beta[:, t] = beta[:, t] * c[t]

    return beta


def viterbi(
    hmm: AnyHMM,
    obs: Sequence[int] | Sequence[npt.NDArray],
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

    delta = np.zeros([hmm.N, T], dtype=float)
    tiny = np.finfo(float).tiny

    if scaling:
        delta[:, 0] = np.log(np.maximum(hmm.Pi, tiny)) + np.log(
            np.maximum(hmm.get_emission_probs(obs[0]), tiny)
        )
    else:
        delta[:, 0] = hmm.Pi * hmm.get_emission_probs(obs[0])

    psi = np.zeros([hmm.N, T], dtype=int)

    if scaling:
        for t in range(1, T):
            nus = rearrange(delta[:, t - 1], "n -> n 1") + np.log(np.maximum(hmm.A, tiny))
            delta[:, t] = nus.max(0) + np.log(np.maximum(hmm.get_emission_probs(obs[t]), tiny))
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
    scaling: bool = True,
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
        scaling: flag to scale probabilities (default: True, recommended)
        graph: flag to plot log-likelihoods (default: False)
        fname: file name to save plot figure (default: "ll.eps")
        verbose: flag to print training progress (default: False)

    Returns:
        Trained HMM model
    """
    tiny = np.finfo(float).tiny

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

            forward_result = forward(hmm=hmm, obs=obs_list, scaling=scaling)
            log_prob_obs, alpha = forward_result[0], forward_result[1]
            c = forward_result[2] if scaling and len(forward_result) > 2 else None
            beta = backward(hmm=hmm, obs=obs_list, c=c)

            LL_epoch += log_prob_obs

            gamma_raw = alpha * beta
            gamma = gamma_raw / np.maximum(gamma_raw.sum(0), tiny)
            gammas.append(gamma)

            B_next = np.column_stack([hmm.get_emission_probs(o) for o in obs_list[1:]])

            xi = np.einsum(
                "it, ij, jt, jt -> ijt",
                alpha[:, :-1],
                hmm.A,
                B_next,
                beta[:, 1:],
            )

            if not scaling:
                xi /= np.maximum(xi.sum(axis=(0, 1), keepdims=True), tiny)

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

        if epoch > 1 and LLs[epoch - 1] == LL_epoch:
            if verbose:
                print("Log-likelihoods have plateaued--terminating training")
            break

        if val_set is not None:
            val_LL_epoch = 0.0
            for val_obs in val_set:
                val_obs_list = list(val_obs)
                val_LL_epoch += forward(hmm=hmm, obs=val_obs_list, scaling=True)[0]
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
