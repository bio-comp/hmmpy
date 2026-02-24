"""Matrix visualization functions for HMMs."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_transition_matrix(
    hmm: Any,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = "Blues",
    **kwargs: Any,
) -> Any:
    """Plot transition matrix as heatmap.

    Args:
        hmm: HMM model with A attribute (transition matrix)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        cmap: Colormap name
        **kwargs: Additional arguments passed to imshow

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    im = ax.imshow(hmm.A, cmap=cmap, vmin=0, vmax=1, **kwargs)
    ax.set_xticks(range(hmm.N))
    ax.set_yticks(range(hmm.N))
    ax.set_xticklabels([f"State {i}" for i in range(hmm.N)])
    ax.set_yticklabels([f"State {i}" for i in range(hmm.N)])
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("Transition Matrix A\n(P(next state | current state))")

    for i in range(hmm.N):
        for j in range(hmm.N):
            ax.text(j, i, f"{hmm.A[i, j]:.2f}", ha="center", va="center", fontsize=11)

    plt.colorbar(im, ax=ax)
    return fig  # type: ignore[return-value]


def plot_emission_matrix(
    hmm: Any,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 5),
    cmap: str = "Greens",
    **kwargs: Any,
) -> Any:
    """Plot emission matrix as heatmap.

    Args:
        hmm: HMM model with B attribute (emission matrix)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        cmap: Colormap name
        **kwargs: Additional arguments passed to imshow

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    im = ax.imshow(hmm.B, cmap=cmap, vmin=0, vmax=1, **kwargs)
    ax.set_xticks(range(hmm.M))
    ax.set_yticks(range(hmm.N))
    ax.set_xticklabels([f"Sym {i}" for i in range(hmm.M)])
    ax.set_yticklabels([f"State {i}" for i in range(hmm.N)])
    ax.set_xlabel("Observation Symbol")
    ax.set_ylabel("State")
    ax.set_title("Emission Matrix B\n(P(observation | state))")

    for i in range(hmm.N):
        for j in range(hmm.M):
            ax.text(j, i, f"{hmm.B[i, j]:.2f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax)
    return fig  # type: ignore[return-value]


def plot_initial_distribution(
    hmm: Any,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 4),
    colors: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot initial state distribution as bar chart.

    Args:
        hmm: HMM model with Pi attribute (initial distribution)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        colors: List of colors for bars
        **kwargs: Additional arguments passed to bar

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # type: ignore[assignment]

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(hmm.N)]  # type: ignore[misc]

    ax.bar(range(hmm.N), hmm.Pi, color=colors, **kwargs)
    ax.set_xticks(range(hmm.N))
    ax.set_xticklabels([f"State {i}" for i in range(hmm.N)])
    ax.set_ylabel("Probability")
    ax.set_title("Initial State Distribution Pi")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return fig  # type: ignore[return-value]


def plot_hmm_matrices(
    hmm: Any,
    figsize: tuple[int, int] = (14, 4),
    **kwargs: Any,
) -> Any:
    """Plot all HMM matrices in a single figure.

    Args:
        hmm: HMM model
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to subplots

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, **kwargs)
    plot_transition_matrix(hmm, ax=axes[0])
    plot_emission_matrix(hmm, ax=axes[1])
    plot_initial_distribution(hmm, ax=axes[2])
    plt.tight_layout()
    return fig  # type: ignore[return-value]
