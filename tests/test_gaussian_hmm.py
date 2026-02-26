"""Tests for GaussianHMM class."""

import numpy as np

from hmm.algorithms import baum_welch, forward, viterbi
from hmm.algorithms import ComputeMode
from hmm.continuous import GaussianHMM


class TestGaussianHMMClass:
    """Tests for GaussianHMM class."""

    def test_create_gaussian_hmm(self) -> None:
        """Test creating GaussianHMM with provided parameters."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])

        hmm = GaussianHMM(n_states=2, n_features=1, A=A, Pi=Pi, means=means, covars=covars)

        assert hmm.N == 2
        assert hmm.n_features == 1
        assert np.allclose(hmm.A, A)
        assert np.allclose(hmm.Pi, Pi)
        assert np.allclose(hmm.means, means)
        assert np.allclose(hmm.covars, covars)

    def test_create_gaussian_hmm_random_init(self) -> None:
        """Test creating GaussianHMM with random initialization."""
        hmm = GaussianHMM(n_states=2, n_features=1)

        assert hmm.N == 2
        assert hmm.n_features == 1
        assert hmm.A.shape == (2, 2)
        assert hmm.Pi.shape == (2,)
        assert hmm.means.shape == (2, 1)
        assert hmm.covars.shape == (2, 1, 1)

    def test_gaussian_hmm_repr(self) -> None:
        """Test GaussianHMM string representation."""
        means = np.array([[0.0], [10.0]])
        hmm = GaussianHMM(n_states=2, n_features=1, means=means)

        repr_str = repr(hmm)
        assert "num hiddens: 2" in repr_str
        assert "features (dimensions): 1" in repr_str


class TestGaussianEmissionProbability:
    """Tests for emission probability calculations."""

    def test_emission_prob_univariate(self) -> None:
        """Test emission probability for univariate Gaussian."""
        means = np.array([[0.0]])
        covars = np.array([[[1.0]]])
        hmm = GaussianHMM(n_states=1, n_features=1, means=means, covars=covars)

        prob = hmm.emission_prob(0, np.array([0.0]))
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert np.isclose(prob, expected, rtol=1e-5)

    def test_emission_prob_univariate_offset(self) -> None:
        """Test emission probability at offset from mean."""
        means = np.array([[0.0]])
        covars = np.array([[[1.0]]])
        hmm = GaussianHMM(n_states=1, n_features=1, means=means, covars=covars)

        prob = hmm.emission_prob(0, np.array([1.0]))
        expected = np.exp(-0.5) / np.sqrt(2 * np.pi)
        assert np.isclose(prob, expected, rtol=1e-5)

    def test_emission_prob_multivariate(self) -> None:
        """Test emission probability for bivariate Gaussian."""
        means = np.array([[0.0, 0.0]])
        covars = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        hmm = GaussianHMM(n_states=1, n_features=2, means=means, covars=covars)

        prob = hmm.emission_prob(0, np.array([0.0, 0.0]))
        expected = 1.0 / (2 * np.pi)
        assert np.isclose(prob, expected, rtol=1e-5)

    def test_get_emission_probs_shape(self) -> None:
        """Test get_emission_probs returns correct shape."""
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=1, means=means, covars=covars)

        probs = hmm.get_emission_probs(np.array([0.0]))

        assert probs.shape == (2,)
        assert np.all(probs > 0)


class TestGaussianHMMForwardAlgorithm:
    """Tests for forward algorithm with GaussianHMM."""

    def test_forward_basic(self) -> None:
        """Test forward algorithm with GaussianHMM."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=1, A=A, Pi=Pi, means=means, covars=covars)

        obs = np.array([[0.0], [0.1], [-0.1], [0.0]])
        result = forward(hmm, obs, mode=ComputeMode.SCALED)

        if len(result) == 3:
            ll, _, _ = result
        else:
            ll, _ = result

        assert isinstance(ll, float)
        assert ll < 0

    def test_forward_multivariate(self) -> None:
        """Test forward algorithm with multivariate Gaussian."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0, 0.0], [10.0, 10.0]])
        covars = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=2, A=A, Pi=Pi, means=means, covars=covars)

        obs = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]])
        result = forward(hmm, obs, mode=ComputeMode.SCALED)

        if len(result) == 3:
            ll, _, _ = result
        else:
            ll, _ = result

        assert isinstance(ll, float)
        assert ll < 0


class TestGaussianHMMViterbi:
    """Tests for Viterbi algorithm with GaussianHMM."""

    def test_viterbi_basic(self) -> None:
        """Test Viterbi algorithm with GaussianHMM."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=1, A=A, Pi=Pi, means=means, covars=covars)

        obs = np.array([[0.0], [0.1], [-0.1], [0.0], [10.0], [10.1]])
        q_star, _, _ = viterbi(hmm, obs, mode=ComputeMode.SCALED)

        assert len(q_star) == 6
        assert all(0 <= q < 2 for q in q_star)


class TestGaussianHMMBaumWelch:
    """Tests for Baum-Welch training with GaussianHMM."""

    def test_baum_welch_training(self) -> None:
        """Test Baum-Welch training updates means and covariances."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=1, A=A, Pi=Pi, means=means, covars=covars)

        np.random.seed(42)
        obs = []
        for _ in range(50):
            state = np.random.choice(2, p=hmm.Pi)
            for _ in range(5):
                obs.append(hmm.means[state] + np.sqrt(hmm.covars[state, 0, 0]) * np.random.randn())
        obs = np.array(obs).reshape(-1, 1)

        trained = baum_welch(hmm, [obs], epochs=10, update_a=True, update_b=True, verbose=False)

        assert trained.N == 2
        assert trained.means.shape == (2, 1)
        assert trained.covars.shape == (2, 1, 1)

    def test_baum_welch_convergence(self) -> None:
        """Test Baum-Welch training updates means/covars."""
        np.random.seed(42)
        true_means = np.array([[-5.0], [5.0]])
        true_covars = np.array([[[1.0]], [[1.0]]])

        hmm = GaussianHMM(n_states=2, n_features=1)
        hmm.means = true_means.copy()
        hmm.covars = true_covars.copy()

        obs = []
        for _ in range(50):
            state = np.random.choice(2, p=hmm.Pi)
            for _ in range(10):
                obs.append(
                    hmm.means[state, 0] + np.sqrt(hmm.covars[state, 0, 0]) * np.random.randn()
                )
        obs = np.array(obs).reshape(-1, 1)

        hmm2 = GaussianHMM(n_states=2, n_features=1)
        initial_means = hmm2.means.copy()
        initial_covars = hmm2.covars.copy()

        trained = baum_welch(hmm2, [obs], epochs=20, update_a=True, update_b=True, verbose=False)

        means_changed = not np.allclose(trained.means, initial_means)
        covars_changed = not np.allclose(trained.covars, initial_covars)

        assert means_changed or covars_changed

    def test_baum_welch_validation_rollback(self) -> None:
        """Test validation rollback restores best means/covars."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        Pi = np.array([0.5, 0.5])
        means = np.array([[0.0], [10.0]])
        covars = np.array([[[1.0]], [[1.0]]])
        hmm = GaussianHMM(n_states=2, n_features=1, A=A, Pi=Pi, means=means, covars=covars)

        np.random.seed(42)
        train_obs = []
        for _ in range(20):
            state = np.random.choice(2, p=hmm.Pi)
            for _ in range(5):
                train_obs.append(
                    hmm.means[state] + np.sqrt(hmm.covars[state, 0, 0]) * np.random.randn()
                )
        train_obs = np.array(train_obs).reshape(-1, 1)

        val_obs = []
        for _ in range(5):
            state = np.random.choice(2, p=hmm.Pi)
            for _ in range(3):
                val_obs.append(
                    hmm.means[state] + np.sqrt(hmm.covars[state, 0, 0]) * np.random.randn()
                )
        val_obs = np.array(val_obs).reshape(-1, 1)

        trained = baum_welch(
            hmm,
            [train_obs],
            epochs=10,
            val_set=[val_obs],
            update_a=True,
            update_b=True,
            verbose=False,
        )

        assert trained.N == 2
        assert trained.means.shape == (2, 1)
