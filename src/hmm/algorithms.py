"""HMM algorithms: forward, backward, viterbi, baum_welch."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from einops import rearrange

from hmm.continuous import gaussian_pdf

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
    hmm: AnyHMM,
    obs_seqs: list[Sequence[int] | Sequence[npt.NDArray]],
    epochs: int = 20,
    val_set: list[Sequence[int] | Sequence[npt.NDArray]] | None = None,
    update_pi: bool = True,
    update_a: bool = True,
    update_b: bool = True,
    scaling: bool = True,
    graph: bool = False,
    fname: str = "ll.eps",
    verbose: bool = False,
    reg_covar: float | None = None,
) -> HMM:
    """EM algorithm to update Pi, A, and B for the HMM.

    Implements the Baum-Welch algorithm from Rabiner (1989).

    For GaussianHMM with update_b=True, uses continuous emission
    reestimation formulas (Rabiner Section VIII, Eq. 53-54).

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
        reg_covar: covariance regularization constant (default: from hmm.reg_covar or 1e-6)

    Returns:
        Trained HMM model
    """
    # Use reg_covar from hmm if not provided
    if reg_covar is None:
        reg_covar = getattr(hmm, "reg_covar", 1e-6)
    # Check if this is a GaussianHMM (doesn't have M attribute)
    is_gaussian = not hasattr(hmm, "M")

    LLs: list[float] = []
    val_LLs: list[float] = []
    best_A = copy.deepcopy(hmm.A)
    best_Pi = copy.deepcopy(hmm.Pi)
    best_epoch = -1
    best_val_LL = float("-inf") if val_set is not None else None

    # Check if this is a MixtureGaussianHMM (has n_mixtures attribute)
    is_mixture = hasattr(hmm, "n_mixtures") and hmm.n_mixtures > 1
    is_continuous = is_gaussian  # Both GaussianHMM and MixtureGaussianHMM are continuous

    # For GaussianHMM: store best means and covariances for validation
    if is_continuous:
        best_means = copy.deepcopy(hmm.means)
        best_covars = copy.deepcopy(hmm.covars)
        if is_mixture:
            best_weights = copy.deepcopy(hmm.weights)
    else:
        best_B = copy.deepcopy(hmm.B)

    for epoch in range(epochs):
        LL_epoch = 0.0
        expect_si_all = np.zeros(hmm.N, dtype=float)
        expect_si_all_TM1 = np.zeros(hmm.N, dtype=float)
        expect_si_sj_all = np.zeros([hmm.N, hmm.N], dtype=float)
        expect_si_sj_all_TM1 = np.zeros([hmm.N, hmm.N], dtype=float)
        expect_si_t0_all = np.zeros(hmm.N, dtype=float)

        # For GaussianHMM: accumulators for means and covariances
        if is_continuous:
            if is_mixture:
                expect_mix_sum = np.zeros([hmm.N, hmm.n_mixtures], dtype=float)
                expect_obs_sum = np.zeros([hmm.N, hmm.n_mixtures, hmm.n_features], dtype=float)
                expect_obs_cov = np.zeros(
                    [hmm.N, hmm.n_mixtures, hmm.n_features, hmm.n_features], dtype=float
                )
            else:
                expect_obs_sum = np.zeros([hmm.N, hmm.n_features], dtype=float)
                expect_obs_cov = np.zeros([hmm.N, hmm.n_features, hmm.n_features], dtype=float)
        else:
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

            w_k = 1.0

            obs_symbols = obs[:]

            gamma_raw = alpha * beta
            gamma = gamma_raw / gamma_raw.sum(0)

            expect_si_t0_all += w_k * gamma[:, 0]
            expect_si_all += w_k * gamma.sum(1)
            expect_si_all_TM1 += w_k * gamma[:, : T - 1].sum(1)

            xi = np.zeros([hmm.N, hmm.N, T - 1], dtype=float)
            for t in range(T - 1):
                for i in range(hmm.N):
                    xi[i, :, t] = (
                        alpha[i, t]
                        * hmm.A[i, :]
                        * hmm.get_emission_probs(obs[t + 1])
                        * beta[:, t + 1]
                    )

                if not scaling:
                    xi[:, :, t] = xi[:, :, t] / xi[:, :, t].sum()

            expect_si_sj_all += w_k * xi.sum(2)
            expect_si_sj_all_TM1 += w_k * xi[:, :, : T - 1].sum(2)

            if update_b:
                if is_continuous:
                    # For continuous HMM: accumulate weighted observations
                    # gamma_jk_t = gamma[j,t] * c_jk * N(O_t | mu_jk, sigma_jk) / b_j(O_t)
                    for t in range(T):
                        obs_t = np.asarray(obs[t])
                        b_j = hmm.get_emission_probs(obs_t)

                        if is_mixture:
                            # Mixture Gaussian case
                            for j in range(hmm.N):
                                for k in range(hmm.n_mixtures):
                                    c_jk = hmm.weights[j, k]
                                    mu_jk = hmm.means[j, k]
                                    sigma_jk = hmm.covars[j, k]

                                    # Gaussian PDF for this mixture component
                                    pdf = gaussian_pdf(obs_t, mu_jk, sigma_jk, reg_covar)
                                    if b_j[j] > 0:
                                        gamma_jk = gamma[j, t] * c_jk * pdf / b_j[j]
                                    else:
                                        gamma_jk = 0.0

                                    expect_mix_sum[j, k] += gamma_jk
                                    expect_obs_sum[j, k] += gamma_jk * obs_t
                                    diff = np.asarray(obs_t) - np.asarray(mu_jk)
                                    diff_outer = np.outer(diff, diff)
                                    expect_obs_cov[j, k] += gamma_jk * diff_outer
                        else:
                            # Single Gaussian case
                            for j in range(hmm.N):
                                expect_obs_sum[j] += gamma[j, t] * obs_t
                                diff = obs_t - hmm.means[j]
                                expect_obs_cov[j] += gamma[j, t] * np.outer(diff, diff)
                else:
                    # Discrete B matrix update
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
            if is_continuous:
                if is_mixture:
                    # Update mixture weights (Rabiner Eq. 52)
                    for j in range(hmm.N):
                        if expect_si_all[j] > 0:
                            hmm.weights[j, :] = expect_mix_sum[j, :] / expect_si_all[j]

                    # Update means (Rabiner Eq. 53)
                    for j in range(hmm.N):
                        for k in range(hmm.n_mixtures):
                            if expect_mix_sum[j, k] > 0:
                                hmm.means[j, k] = expect_obs_sum[j, k] / expect_mix_sum[j, k]

                    # Update covariances (Rabiner Eq. 54) with regularization
                    for j in range(hmm.N):
                        for k in range(hmm.n_mixtures):
                            if expect_mix_sum[j, k] > 0:
                                hmm.covars[j, k] = (expect_obs_cov[j, k] / expect_mix_sum[j, k]) + (
                                    reg_covar * np.eye(hmm.n_features)
                                )
                else:
                    # Single Gaussian case - Update means (Rabiner Eq. 53)
                    for j in range(hmm.N):
                        if expect_si_all[j] > 0:
                            hmm.means[j] = expect_obs_sum[j] / expect_si_all[j]

                    # Update covariances (Rabiner Eq. 54) with regularization
                    for j in range(hmm.N):
                        if expect_si_all[j] > 0:
                            hmm.covars[j] = (expect_obs_cov[j] / expect_si_all[j]) + (
                                reg_covar * np.eye(hmm.n_features)
                            )
            else:
                # Discrete B matrix update
                for i in range(hmm.N):
                    if expect_si_all[i] > 0:
                        expect_si_vk_all[i, :] = expect_si_vk_all[i, :] / expect_si_all[i]

                hmm.B = expect_si_vk_all

                for i in hmm.F:
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

            if best_val_LL is None or val_LL_epoch > best_val_LL:
                best_A = copy.deepcopy(hmm.A)
                best_Pi = copy.deepcopy(hmm.Pi)
                if is_continuous:
                    best_means = copy.deepcopy(hmm.means)
                    best_covars = copy.deepcopy(hmm.covars)
                    if is_mixture:
                        best_weights = copy.deepcopy(hmm.weights)
                else:
                    best_B = copy.deepcopy(hmm.B)
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
        hmm.A = best_A
        hmm.Pi = best_Pi
        if hasattr(hmm, "B"):
            hmm.B = best_B
        elif hasattr(hmm, "weights"):
            hmm.means = best_means
            hmm.covars = best_covars
            hmm.weights = best_weights
        else:
            hmm.means = best_means
            hmm.covars = best_covars

    return hmm
