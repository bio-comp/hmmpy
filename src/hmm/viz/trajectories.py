"""Trajectory visualization functions for HMMs."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_state_probabilities(
    hmm: Any,
    obs: Sequence[int],
    alpha: NDArray,
    beta: NDArray | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 4),
    **kwargs: Any,
) -> Figure:
    """Plot forward state probabilities over time.

    Args:
        hmm: HMM model
        obs: Observation sequence
        alpha: Forward variables (N x T)
        beta: Backward variables (optional)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to plot

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    T = len(obs)
    t = range(T)

    colors = ["steelblue", "coral", "mediumseagreen", "orchid"][: hmm.N]

    for state in range(hmm.N):
        ax.plot(
            t, alpha[state, :], "-o", color=colors[state], label=f"P(state={state}|O)", **kwargs
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("State Probability")
    ax.set_title("Forward State Probabilities Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(t)
    ax.set_xticklabels([str(obs[i]) for i in t])

    return fig


def plot_viterbi_path(
    hmm: Any,
    obs: Sequence[int],
    states: list[int],
    alpha: NDArray,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
    **kwargs: Any,
) -> Figure:
    """Plot Viterbi decoded path with state probabilities.

    Args:
        hmm: HMM model
        obs: Observation sequence
        states: Decoded state sequence from Viterbi
        alpha: Forward variables (N x T)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to plot

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    T = len(obs)
    t = range(T)

    colors = ["steelblue", "coral", "mediumseagreen", "orchid"][: hmm.N]

    for state in range(hmm.N):
        ax.fill_between(t, 0, alpha[state, :], alpha=0.2, color=colors[state])
        ax.plot(t, alpha[state, :], "-o", color=colors[state], label=f"State {state}", **kwargs)

    for i, s in enumerate(states):
        ax.axvline(i, color=colors[s], linestyle="--", alpha=0.5)
        ax.plot(i, alpha[s, i], "ko", markersize=10)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("State Probability")
    ax.set_title("Viterbi Path Decoding")
    ax.legend()
    ax.set_xticks(t)
    ax.set_xticklabels([f"{obs[i]}\n(S{states[i]})" for i in t])

    return fig


def plot_baum_welch_convergence(
    log_likelihoods: list[float],
    val_likelihoods: list[float] | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 4),
    **kwargs: Any,
) -> Figure:
    """Plot Baum-Welch training convergence.

    Args:
        log_likelihoods: List of log-likelihoods per epoch
        val_likelihoods: Optional validation log-likelihoods
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to plot

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    epochs = range(len(log_likelihoods))
    ax.plot(epochs, log_likelihoods, "-o", color="steelblue", label="Training", **kwargs)

    if val_likelihoods is not None:
        ax.plot(
            epochs,
            val_likelihoods,
            "-s",
            color="coral",
            label="Validation",
            **kwargs,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Baum-Welch Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
