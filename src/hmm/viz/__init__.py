"""HMM Visualization Module

Functions for visualizing Hidden Markov Model parameters, state trajectories,
and state diagrams.
"""

from hmm.viz.matrices import (
    plot_transition_matrix,
    plot_emission_matrix,
    plot_initial_distribution,
    plot_hmm_matrices,
)
from hmm.viz.trajectories import (
    plot_state_probabilities,
    plot_viterbi_path,
    plot_baum_welch_convergence,
)
from hmm.viz.diagrams import plot_state_diagram, plot_gaussian_ellipses

__all__ = [
    "plot_transition_matrix",
    "plot_emission_matrix",
    "plot_initial_distribution",
    "plot_hmm_matrices",
    "plot_state_probabilities",
    "plot_viterbi_path",
    "plot_baum_welch_convergence",
    "plot_state_diagram",
    "plot_gaussian_ellipses",
]
