"""HMM Protocol defining the interface for all HMM implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy.typing as npt


class HMMProtocol(Protocol):
    """Protocol defining the required interface for any HMM implementation."""

    N: int
    Pi: npt.NDArray
    A: npt.NDArray

    def get_emission_probs(self, obs_t: npt.NDArray) -> npt.NDArray:
        """Returns emission probabilities for all states given a single observation."""
        ...

    def m_step(
        self,
        obs_seqs: list[Sequence[npt.NDArray]],
        gammas: list[npt.NDArray],
        xis: list[npt.NDArray],
        update_pi: bool = True,
        update_a: bool = True,
        update_b: bool = True,
    ) -> None:
        """
        Performs the Maximization step to update model parameters in-place.

        Args:
            obs_seqs: The original list of observation sequences.
            gammas: Expected state occupancies. List of arrays of shape (N, T).
            xis: Expected transitions. List of arrays of shape (N, N, T-1).
            update_pi: Whether to update initial state probabilities.
            update_a: Whether to update transition probabilities.
            update_b: Whether to update emission parameters (B, means, covars, etc.).
        """
        ...
