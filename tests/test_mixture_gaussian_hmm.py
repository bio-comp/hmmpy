"""Tests for MixtureGaussianHMM class."""

import numpy as np
import pytest

from hmm.continuous import MixtureGaussianHMM


class TestMixtureGaussianHMMClass:
    """Tests for MixtureGaussianHMM class."""

    def test_create_mixture_gaussian_hmm(self) -> None:
        """Test creating MixtureGaussianHMM with provided parameters."""
        n_states = 2
        n_features = 1
        n_mixtures = 3

        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        weights = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
        means = np.array([[[0.0], [5.0], [10.0]], [[15.0], [20.0], [25.0]]])
        covars = np.array([[[[1.0]], [[2.0]], [[3.0]]], [[[1.0]], [[2.0]], [[3.0]]]])

        hmm = MixtureGaussianHMM(
            n_states=n_states,
            n_features=n_features,
            n_mixtures=n_mixtures,
            A=A,
            Pi=Pi,
            weights=weights,
            means=means,
            covars=covars,
        )

        assert hmm.N == n_states
        assert hmm.n_features == n_features
        assert hmm.n_mixtures == n_mixtures
        assert np.allclose(hmm.A, A)
        assert np.allclose(hmm.Pi, Pi)
        assert np.allclose(hmm.weights, weights)
        assert np.allclose(hmm.means, means)
        assert np.allclose(hmm.covars, covars)

    def test_create_mixture_gaussian_hmm_random_init(self) -> None:
        """Test creating MixtureGaussianHMM with random initialization."""
        hmm = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=2)

        assert hmm.N == 2
        assert hmm.n_features == 1
        assert hmm.n_mixtures == 2
        assert hmm.A.shape == (2, 2)
        assert hmm.Pi.shape == (2,)
        assert hmm.weights.shape == (2, 2)
        assert hmm.means.shape == (2, 2, 1)
        assert hmm.covars.shape == (2, 2, 1, 1)

    def test_weights_sum_to_one(self) -> None:
        """Test that mixture weights sum to 1 for each state."""
        hmm = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=3)

        for i in range(hmm.N):
            assert np.isclose(hmm.weights[i].sum(), 1.0)


class TestMixtureGaussianEmission:
    """Tests for mixture emission probability calculations."""

    def test_emission_prob_single_mixture(self) -> None:
        """Test emission probability calculation with single mixture per state."""
        n_states = 1
        n_features = 1
        n_mixtures = 1

        means = np.array([[[0.0]]])
        covars = np.array([[[[1.0]]]])
        weights = np.array([[1.0]])

        hmm = MixtureGaussianHMM(
            n_states=n_states,
            n_features=n_features,
            n_mixtures=n_mixtures,
            means=means,
            covars=covars,
            weights=weights,
        )

        prob = hmm.emission_prob(0, np.array([0.0]))
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert np.isclose(prob, expected, rtol=1e-5)

    def test_emission_prob_multiple_mixtures(self) -> None:
        """Test emission probability with multiple mixtures."""
        n_states = 1
        n_features = 1
        n_mixtures = 2

        means = np.array([[[0.0], [5.0]]])
        covars = np.array([[[[1.0]], [[0.5]]]])
        weights = np.array([[0.7, 0.3]])

        hmm = MixtureGaussianHMM(
            n_states=n_states,
            n_features=n_features,
            n_mixtures=n_mixtures,
            means=means,
            covars=covars,
            weights=weights,
        )

        prob_at_zero = hmm.emission_prob(0, np.array([0.0]))
        prob_at_five = hmm.emission_prob(0, np.array([5.0]))

        assert prob_at_zero > 0
        assert prob_at_five > 0

    def test_get_emission_probs_shape(self) -> None:
        """Test get_emission_probs returns correct shape."""
        hmm = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=3)

        probs = hmm.get_emission_probs(np.array([0.0]))

        assert probs.shape == (2,)

    def test_get_all_emission_probs_uses_vectorized_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_all_emission_probs should not delegate per timestep to get_emission_probs."""
        hmm = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=2)

        def _fail_get_emission_probs(_obs_t: int | np.ndarray) -> np.ndarray:
            raise AssertionError("Vectorized path should not call get_emission_probs")

        monkeypatch.setattr(hmm, "get_emission_probs", _fail_get_emission_probs)

        obs = np.array([[0.0], [0.5], [1.0]])
        probs = hmm.get_all_emission_probs(obs)

        assert probs.shape == (2, 3)
        assert np.all(probs > 0.0)


class TestMixtureGaussianHMMTraining:
    """Tests for MixtureGaussianHMM training."""

    def test_baum_welch_training(self) -> None:
        """Test Baum-Welch training updates mixture parameters."""
        np.random.seed(42)
        true_means = np.array([[[-2.0], [2.0]], [[-2.0], [2.0]]])
        true_covars = np.array([[[[0.5]]], [[[0.5]]]])
        true_weights = np.array([[0.5, 0.5], [0.5, 0.5]])

        hmm = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=2)
        hmm.means = true_means.copy()
        hmm.covars = true_covars.copy()
        hmm.weights = true_weights.copy()

        obs = []
        for _ in range(30):
            state = np.random.choice(2, p=hmm.Pi)
            for _ in range(10):
                obs.append(
                    hmm.means[state, 0, 0] + np.sqrt(hmm.covars[state, 0, 0, 0]) * np.random.randn()
                )
        obs = np.array(obs).reshape(-1, 1)

        hmm2 = MixtureGaussianHMM(n_states=2, n_features=1, n_mixtures=2)
        initial_means = hmm2.means.copy()
        initial_weights = hmm2.weights.copy()

        from hmm.algorithms import baum_welch

        trained = baum_welch(hmm2, [obs], epochs=10, update_a=True, update_b=True, verbose=False)

        means_changed = not np.allclose(trained.means, initial_means)
        weights_changed = not np.allclose(trained.weights, initial_weights)

        assert means_changed or weights_changed
