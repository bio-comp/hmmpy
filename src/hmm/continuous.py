"""Continuous Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt
from numpy import random as rand


class GaussianHMM:
    """
    Continuous Hidden Markov Model with single multivariate Gaussian emissions per state.
    Follows Rabiner (1989) Section VIII.

    Attributes:
        N: Number of hidden states
        n_features: Dimensionality of the continuous observation vectors
        A: Transition probability matrix (N x N)
        Pi: Initial state distribution (N,)
        means: Mean vectors for each state (N x n_features)
        covars: Covariance matrices for each state (N x n_features x n_features)
        Labels: State labels
    """

    def __init__(
        self,
        n_states: int = 1,
        n_features: int = 1,
        A: npt.NDArray | None = None,
        Pi: npt.NDArray | None = None,
        means: npt.NDArray | None = None,
        covars: npt.NDArray | None = None,
        Labels: list[int] | None = None,
        reg_covar: float = 1e-6,
    ) -> None:
        """Initialize a Continuous HMM with Gaussian emissions.

        Args:
            n_states: Number of hidden states
            n_features: Dimensionality of observation vectors
            A: Transition probability matrix (N x N)
            Pi: Initial state distribution (N,)
            means: Mean vectors (N x n_features)
            covars: Covariance matrices (N x n_features x n_features)
            Labels: State labels
            reg_covar: Regularization constant added to covariance matrices to prevent singularity
        """
        self.N = n_states
        self.n_features = n_features
        self.reg_covar = reg_covar

        # Initialize Transition Matrix A
        if A is not None:
            self.A = np.array(A, dtype=float)
            assert np.shape(self.A) == (self.N, self.N)
        else:
            raw_A = rand.uniform(size=self.N * self.N).reshape((self.N, self.N))
            self.A = (raw_A.T / raw_A.T.sum(0)).T
            if n_states == 1:
                self.A = self.A.reshape((1, 1))

        # Initialize Initial Distribution Pi
        if Pi is not None:
            self.Pi = np.array(Pi, dtype=float)
            assert len(self.Pi) == self.N
        else:
            self.Pi = np.array(1.0 / self.N).repeat(self.N)

        # Initialize Continuous Emission Parameters (means and covariances)
        if means is not None:
            self.means = np.array(means, dtype=float)
            assert self.means.shape == (self.N, self.n_features)
        else:
            self.means = rand.randn(self.N, self.n_features)

        if covars is not None:
            self.covars = np.array(covars, dtype=float)
            assert self.covars.shape == (self.N, self.n_features, self.n_features)
        else:
            self.covars = np.array([np.eye(self.n_features) for _ in range(self.N)], dtype=float)

        if Labels is not None:
            self.Labels = list(Labels)
        else:
            self.Labels = list(range(self.N))

    def emission_prob(self, state: int, obs: int | npt.NDArray) -> float:
        """Calculate b_j(O) = N(O, mu_j, U_j).

        Returns the probability density of observation vector 'obs' given 'state'.

        Args:
            state: State index (0 to N-1)
            obs: Observation vector (n_features,)

        Returns:
            Probability density at obs
        """
        mu = self.means[state]
        cov = self.covars[state]
        d = self.n_features
        diff = np.asarray(obs) - np.asarray(mu)

        # Add regularization to prevent singular covariance
        reg_cov = cov + (self.reg_covar * np.eye(d))

        if d == 1:
            diff_scalar = float(np.squeeze(diff))
            var = float(np.squeeze(reg_cov))
            return float(np.exp(-0.5 * (diff_scalar**2) / var) / np.sqrt(2 * np.pi * var))
        else:
            cov_inv = np.linalg.inv(reg_cov)
            det_cov = np.linalg.det(reg_cov)
            exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
            return float(np.exp(exponent) / np.sqrt(((2 * np.pi) ** d) * det_cov))

    def get_emission_probs(self, obs_t: int | npt.NDArray) -> npt.NDArray:
        """Returns emission probabilities for all states given observation obs_t.

        Args:
            obs_t: Observation vector (n_features,)

        Returns:
            Array of shape (N,) with probability densities for each state
        """
        probs = np.zeros(self.N)
        for i in range(self.N):
            probs[i] = self.emission_prob(i, obs_t)
        return probs

    def __repr__(self) -> str:
        retn = ""
        retn += f"num hiddens: {self.N}\n"
        retn += f"features (dimensions): {self.n_features}\n"
        retn += f"\nA:\n {self.A}\n"
        retn += f"Pi:\n {self.Pi}\n"
        retn += f"Means:\n {self.means}\n"
        return retn


class MixtureGaussianHMM:
    """
    Continuous Hidden Markov Model with mixture of Gaussian emissions per state.
    Follows Rabiner (1989) Section VIII, Equations 49-54.

    Attributes:
        N: Number of hidden states
        n_features: Dimensionality of the continuous observation vectors
        n_mixtures: Number of Gaussian mixtures per state
        A: Transition probability matrix (N x N)
        Pi: Initial state distribution (N,)
        weights: Mixture weights for each state (N x n_mixtures)
        means: Mean vectors for each state and mixture (N x n_mixtures x n_features)
        covars: Covariance matrices (N x n_mixtures x n_features x n_features)
        Labels: State labels
    """

    def __init__(
        self,
        n_states: int = 1,
        n_features: int = 1,
        n_mixtures: int = 1,
        A: npt.NDArray | None = None,
        Pi: npt.NDArray | None = None,
        weights: npt.NDArray | None = None,
        means: npt.NDArray | None = None,
        covars: npt.NDArray | None = None,
        Labels: list[int] | None = None,
        reg_covar: float = 1e-6,
    ) -> None:
        """Initialize a Mixture Gaussian HMM.

        Args:
            n_states: Number of hidden states
            n_features: Dimensionality of observation vectors
            n_mixtures: Number of Gaussian mixtures per state
            A: Transition probability matrix (N x N)
            Pi: Initial state distribution (N,)
            weights: Mixture weights (N x n_mixtures)
            means: Mean vectors (N x n_mixtures x n_features)
            covars: Covariance matrices (N x n_mixtures x n_features x n_features)
            Labels: State labels
            reg_covar: Regularization constant added to covariance matrices to prevent singularity
        """
        self.N = n_states
        self.n_features = n_features
        self.n_mixtures = n_mixtures
        self.reg_covar = reg_covar

        if A is not None:
            self.A = np.array(A, dtype=float)
            assert np.shape(self.A) == (self.N, self.N)
        else:
            raw_A = rand.uniform(size=self.N * self.N).reshape((self.N, self.N))
            self.A = (raw_A.T / raw_A.T.sum(0)).T
            if n_states == 1:
                self.A = self.A.reshape((1, 1))

        if Pi is not None:
            self.Pi = np.array(Pi, dtype=float)
            assert len(self.Pi) == self.N
        else:
            self.Pi = np.array(1.0 / self.N).repeat(self.N)

        if weights is not None:
            self.weights = np.array(weights, dtype=float)
            assert self.weights.shape == (self.N, self.n_mixtures)
        else:
            raw_weights = rand.uniform(size=self.N * self.n_mixtures).reshape(
                (self.N, self.n_mixtures)
            )
            self.weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

        if means is not None:
            self.means = np.array(means, dtype=float)
            assert self.means.shape == (self.N, self.n_mixtures, self.n_features)
        else:
            self.means = rand.randn(self.N, self.n_mixtures, self.n_features)

        if covars is not None:
            self.covars = np.array(covars, dtype=float)
            assert self.covars.shape == (self.N, self.n_mixtures, self.n_features, self.n_features)
        else:
            self.covars = np.array(
                [[np.eye(self.n_features) for _ in range(self.n_mixtures)] for _ in range(self.N)],
                dtype=float,
            )

        if Labels is not None:
            self.Labels = list(Labels)
        else:
            self.Labels = list(range(self.N))

    def _gaussian_pdf(self, mean: npt.NDArray, cov: npt.NDArray, obs: npt.NDArray) -> float:
        """Calculate Gaussian PDF for multivariate case.

        Args:
            mean: Mean vector (n_features,)
            cov: Covariance matrix (n_features x n_features)
            obs: Observation vector (n_features,)

        Returns:
            Probability density
        """
        d = self.n_features
        diff = np.asarray(obs) - np.asarray(mean)

        # Add regularization to prevent singular covariance
        reg_cov = cov + (self.reg_covar * np.eye(d))

        if d == 1:
            diff_scalar = float(np.squeeze(diff))
            var = float(np.squeeze(reg_cov))
            return float(np.exp(-0.5 * (diff_scalar**2) / var) / np.sqrt(2 * np.pi * var))
        else:
            cov_inv = np.linalg.inv(reg_cov)
            det_cov = np.linalg.det(reg_cov)
            exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
            return float(np.exp(exponent) / np.sqrt(((2 * np.pi) ** d) * det_cov))

    def emission_prob(self, state: int, obs: int | npt.NDArray) -> float:
        """Calculate b_j(O) = sum_k c_jk * N(O, mu_jk, U_jk).

        Returns the probability density of observation vector 'obs' given 'state'
        using mixture of Gaussians.

        Args:
            state: State index (0 to N-1)
            obs: Observation vector (n_features,)

        Returns:
            Probability density at obs
        """
        density = 0.0
        for k in range(self.n_mixtures):
            c_jk = self.weights[state, k]
            mu_jk = self.means[state, k]
            cov_jk = self.covars[state, k]
            density += c_jk * self._gaussian_pdf(mu_jk, cov_jk, obs)
        return density

    def get_emission_probs(self, obs_t: int | npt.NDArray) -> npt.NDArray:
        """Returns emission probabilities for all states given observation obs_t.

        Args:
            obs_t: Observation vector (n_features,)

        Returns:
            Array of shape (N,) with probability densities for each state
        """
        probs = np.zeros(self.N)
        for i in range(self.N):
            probs[i] = self.emission_prob(i, obs_t)
        return probs

    def __repr__(self) -> str:
        retn = ""
        retn += f"num hiddens: {self.N}\n"
        retn += f"features (dimensions): {self.n_features}\n"
        retn += f"num mixtures: {self.n_mixtures}\n"
        retn += f"\nA:\n {self.A}\n"
        retn += f"Pi:\n {self.Pi}\n"
        retn += f"Weights:\n {self.weights}\n"
        retn += f"Means:\n {self.means}\n"
        return retn
