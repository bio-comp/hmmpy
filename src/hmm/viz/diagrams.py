"""State diagram visualization using networkx."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_state_diagram(
    hmm: Any,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
    threshold: float = 0.01,
    **kwargs: Any,
) -> Figure:
    """Plot HMM as a state diagram with transitions using networkx.

    Args:
        hmm: HMM model with A attribute (transition matrix)
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        threshold: Minimum transition probability to show edge
        **kwargs: Additional arguments

    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx required for state diagrams: pip install networkx") from e

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    G = nx.DiGraph()

    for i in range(hmm.N):
        G.add_node(i, label=f"S{i}")

    for i in range(hmm.N):
        for j in range(hmm.N):
            if hmm.A[i, j] > threshold:
                G.add_edge(i, j, weight=hmm.A[i, j])

    pos = nx.circular_layout(G)

    node_colors = plt.cm.Blues(np.linspace(0.3, 0.8, hmm.N))
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=1500, node_color=node_colors, edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight="bold")

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        alpha=0.6,
        edge_color="gray",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=10)

    ax.set_title("HMM State Diagram")
    ax.axis("off")

    return fig


def plot_gaussian_ellipses(
    means: np.ndarray,
    covariances: np.ndarray,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
    n_std: float = 2.0,
    **kwargs: Any,
) -> Figure:
    """Plot covariance ellipses for Gaussian distributions.

    Args:
        means: Array of shape (n_states, n_dims) containing means
        covariances: Array of shape (n_states, n_dims, n_dims) containing covariances
        ax: Optional matplotlib axes
        figsize: Figure size (width, height)
        n_std: Number of standard deviations for ellipse size
        **kwargs: Additional arguments

    Returns:
        matplotlib Figure
    """
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_states = means.shape[0]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_states)]

    for i in range(n_states):
        mean = means[i]
        cov = covariances[i]

        if mean.ndim == 2:
            # 2D data: draw ellipses
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                facecolor=colors[i],
                alpha=0.3,
                edgecolor=colors[i],
                linewidth=2,
            )
            ax.add_patch(ellipse)
            ax.plot(mean[0], mean[1], "o", color=colors[i], markersize=10)
        else:
            # 1D data: plot full Gaussian PDF
            std = float(np.sqrt(np.squeeze(cov)))
            x = np.linspace(mean[0] - 4 * std, mean[0] + 4 * std, 200)
            pdf = np.exp(-0.5 * ((x - mean[0]) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            ax.plot(x, pdf, color=colors[i], linewidth=2, label=f"State {i}")
            ax.fill_between(x, pdf, alpha=0.2, color=colors[i])

    # Set labels based on dimensionality
    if mean.ndim == 2:
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    else:
        ax.set_xlabel("Observation Value")
        ax.set_ylabel("Probability Density")

    ax.set_title("Gaussian Emissions")
    ax.grid(True, alpha=0.3)

    # Only set equal aspect for 2D
    if mean.ndim == 2:
        ax.set_aspect("equal")

    ax.legend()
    return fig
