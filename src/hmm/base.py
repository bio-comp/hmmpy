"""Protocol definitions shared by trainable HMM implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy.typing as npt

Observation: TypeAlias = int | npt.NDArray
ObservationSequence: TypeAlias = Sequence[Observation]


class HMMProtocol(Protocol):
    """Protocol defining the interface required by the algorithm module."""

    N: int
    Pi: npt.NDArray
    A: npt.NDArray

    def get_emission_probs(self, obs_t: Observation) -> npt.NDArray:
        """Return emission probabilities for all states for a single observation."""
        ...

    def get_all_emission_probs(self, obs_seq: ObservationSequence) -> npt.NDArray:
        """Return emission probabilities for all states across an observation sequence."""
        ...

    def m_step(
        self,
        obs_seqs: list[ObservationSequence],
        gammas: list[npt.NDArray],
        xis: list[npt.NDArray],
        update_pi: bool = True,
        update_a: bool = True,
        update_b: bool = True,
    ) -> None:
        """Update model parameters in-place from expected sufficient statistics."""
        ...
