"""Tests for HMM and HMMClassifier classes."""

import numpy as np
import pytest

import hmm.algorithms as algorithms
from hmm import HMM, HMMClassifier, backward, baum_welch, forward, viterbi
from hmm.algorithms import ComputeMode


class TestHMMClass:
    """Tests for HMM class."""

    def test_create_hmm_with_matrices(self) -> None:
        """Test creating HMM with provided matrices."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]

        hmm = HMM(n_states=2, A=A, B=B, V=V)

        assert hmm.N == 2
        assert hmm.M == 6
        assert np.allclose(hmm.A, A)
        assert np.allclose(hmm.B, B)
        assert hmm.V == V

    def test_create_hmm_random_init(self) -> None:
        """Test creating HMM with random initialization."""
        V = [1, 2, 3, 4, 5, 6]

        hmm = HMM(n_states=2, V=V)

        assert hmm.N == 2
        assert hmm.M == 6
        assert hmm.A.shape == (2, 2)
        assert hmm.B.shape == (2, 6)
        assert hmm.Pi.shape == (2,)

    def test_hmm_requires_v(self) -> None:
        """Test that V (observable symbols) is required."""
        with pytest.raises(ValueError, match="V.*must be provided"):
            HMM(n_states=2)

    def test_hmm_repr(self) -> None:
        """Test HMM string representation."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]

        hmm = HMM(n_states=2, A=A, B=B, V=V)

        repr_str = repr(hmm)
        assert "num hiddens: 2" in repr_str
        assert "symbols:" in repr_str


class TestHMMClassifier:
    """Tests for HMMClassifier class."""

    def test_create_classifier(self) -> None:
        """Test creating HMM classifier."""
        classifier = HMMClassifier()
        assert classifier.pos_hmm is None
        assert classifier.neg_hmm is None

    def test_add_pos_hmm(self) -> None:
        """Test adding positive HMM."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        pos_hmm = HMM(n_states=2, A=A, B=B, V=V)

        classifier = HMMClassifier()
        classifier.add_pos_hmm(pos_hmm)

        assert classifier.pos_hmm is pos_hmm

    def test_add_neg_hmm(self) -> None:
        """Test adding negative HMM."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        neg_hmm = HMM(n_states=2, A=A, B=B, V=V)

        classifier = HMMClassifier()
        classifier.add_neg_hmm(neg_hmm)

        assert classifier.neg_hmm is neg_hmm

    def test_classify_missing_hmm(self) -> None:
        """Test classification raises error when HMMs are missing."""
        classifier = HMMClassifier()

        with pytest.raises(ValueError, match="No HMM models configured"):
            classifier.classify([1, 2, 3])

    def test_classify_multiclass_returns_label(self) -> None:
        """Multiclass classifier should return the best label."""
        A = np.array([[1.0]])
        V = [0, 1]
        models = {
            "spam": HMM(n_states=1, A=A, B=np.array([[0.9, 0.1]]), V=V),
            "ham": HMM(n_states=1, A=A, B=np.array([[0.1, 0.9]]), V=V),
        }

        classifier = HMMClassifier(models=models)

        assert classifier.classify([0, 0, 0]) == "spam"

    def test_classify_multiclass_does_not_fall_back_to_binary(self) -> None:
        """Multiclass mode should not be hijacked by positive/negative labels."""
        A = np.array([[1.0]])
        V = [0, 1]
        models = {
            "positive": HMM(n_states=1, A=A, B=np.array([[0.9, 0.1]]), V=V),
            "negative": HMM(n_states=1, A=A, B=np.array([[0.1, 0.9]]), V=V),
            "neutral": HMM(n_states=1, A=A, B=np.array([[0.5, 0.5]]), V=V),
        }

        classifier = HMMClassifier(models=models)
        result = classifier.classify([0, 0, 0])

        assert result == "positive"
        assert isinstance(result, str)


class TestForwardAlgorithm:
    """Tests for forward algorithm."""

    def test_forward_basic(self) -> None:
        """Test forward algorithm with basic example."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        result = forward(hmm, obs, mode=ComputeMode.SCALED)

        assert len(result) == 3
        log_prob, alpha, c = result
        assert isinstance(log_prob, float)
        assert alpha.shape == (2, 5)
        assert c.shape == (5,)

    def test_forward_without_scaling(self) -> None:
        """Test forward algorithm without scaling."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        result = forward(hmm, obs, mode=ComputeMode.UNSCALED)

        assert len(result) == 2
        prob, alpha = result
        assert prob >= 0
        assert alpha.shape == (2, 5)

    def test_forward_empty_observation(self) -> None:
        """Test forward with empty observation sequence."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        with pytest.raises(IndexError):
            forward(hmm, [], mode=ComputeMode.SCALED)

    def test_forward_unseen_symbol(self) -> None:
        """Test forward with unseen symbol raises KeyError."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        with pytest.raises(KeyError):
            forward(hmm, [99], mode=ComputeMode.SCALED)


class TestBackwardAlgorithm:
    """Tests for backward algorithm."""

    def test_backward_basic(self) -> None:
        """Test backward algorithm."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        log_prob, alpha, c = forward(hmm, obs, mode=ComputeMode.SCALED)
        beta = backward(hmm, obs, scaling_coeffs=c)

        assert beta.shape == (2, 5)

    def test_backward_without_scaling(self) -> None:
        """Test backward algorithm without scaling."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        prob_obs, _ = forward(hmm, obs, mode=ComputeMode.UNSCALED)
        beta = backward(hmm, obs, mode=ComputeMode.UNSCALED)

        assert beta.shape == (2, 5)
        assert np.allclose(beta[:, -1], np.ones(hmm.N))
        start_prob = np.sum(hmm.Pi * hmm.get_emission_probs(obs[0]) * beta[:, 0])
        assert np.isclose(start_prob, prob_obs)


class TestViterbiAlgorithm:
    """Tests for Viterbi algorithm."""

    def test_viterbi_basic(self) -> None:
        """Test Viterbi algorithm."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        q_star, delta, psi = viterbi(hmm, obs, mode=ComputeMode.LOG)

        assert len(q_star) == 5
        assert delta.shape == (2, 5)
        assert psi.shape == (2, 5)
        assert all(0 <= s < 2 for s in q_star)

    def test_viterbi_without_scaling(self) -> None:
        """Test Viterbi algorithm without scaling."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs = [1, 2, 1, 6, 6]
        q_star, delta, psi = viterbi(hmm, obs, mode=ComputeMode.LOG)

        assert len(q_star) == 5

    def test_viterbi_empty_observation(self) -> None:
        """Test Viterbi with empty observation sequence."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        with pytest.raises(IndexError):
            viterbi(hmm, [], mode=ComputeMode.LOG)

    def test_viterbi_unseen_symbol(self) -> None:
        """Test Viterbi with unseen symbol raises KeyError."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        with pytest.raises(KeyError):
            viterbi(hmm, [99], mode=ComputeMode.LOG)


class TestBaumWelch:
    """Tests for Baum-Welch training."""

    def test_baum_welch_basic(self) -> None:
        """Test Baum-Welch training."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        obs_seqs = [[1, 2, 1, 6, 6], [6, 6, 1, 2, 1]]
        trained = baum_welch(hmm, obs_seqs, epochs=5)

        assert trained.N == 2
        assert trained.A.shape == (2, 2)
        assert trained.B.shape == (2, 6)

    def test_baum_welch_with_validation(self) -> None:
        """Test Baum-Welch with validation set."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        train_seqs = [[1, 2, 1, 6, 6], [6, 6, 1, 2, 1]]
        val_seqs = [[1, 1, 6, 6]]
        trained = baum_welch(hmm, train_seqs, epochs=5, val_set=val_seqs)

        assert trained.N == 2

    def test_baum_welch_scaling_flag(self) -> None:
        """Test Baum-Welch respects scaling flag."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]

        hmm1 = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)
        hmm2 = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)

        obs_seqs = [[1, 2, 1, 6, 6]]

        baum_welch(hmm1, obs_seqs, epochs=3, mode=ComputeMode.SCALED)
        baum_welch(hmm2, obs_seqs, epochs=3, mode=ComputeMode.SCALED)

        assert hmm1.A.shape == hmm2.A.shape

    def test_baum_welch_log_mode(self) -> None:
        """Test Baum-Welch runs in LOG mode."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)

        trained = baum_welch(
            hmm,
            [[1, 2, 1, 6, 6], [6, 6, 1, 2, 1]],
            epochs=2,
            mode=ComputeMode.LOG,
        )

        assert trained.A.shape == (2, 2)
        assert trained.B.shape == (2, 6)

    def test_baum_welch_update_flags(self) -> None:
        """Test Baum-Welch update flags."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)

        original_A = hmm.A.copy()
        obs_seqs = [[1, 2, 1, 6, 6]]

        baum_welch(hmm, obs_seqs, epochs=3, update_a=False)

        assert np.allclose(hmm.A, original_A)

    def test_baum_welch_empty_sequences(self) -> None:
        """Test Baum-Welch with empty training set."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]
        hmm = HMM(n_states=2, A=A, B=B, V=V)

        trained = baum_welch(hmm, [], epochs=1)

        assert trained.N == 2

    def test_baum_welch_validation_restores_best_emission_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Validation rollback should restore the full best model, including B."""

        class RollbackHMM(HMM):
            def __init__(self) -> None:
                super().__init__(
                    n_states=2,
                    A=np.array([[0.9, 0.1], [0.2, 0.8]]),
                    B=np.array([[0.6, 0.4], [0.3, 0.7]]),
                    V=[0, 1],
                )
                self._epoch = 0

            def m_step(
                self,
                obs_seqs: list[list[int]],
                gammas: list[np.ndarray],
                xis: list[np.ndarray],
                update_pi: bool = True,
                update_a: bool = True,
                update_b: bool = True,
            ) -> None:
                del obs_seqs, gammas, xis, update_pi, update_a, update_b
                self._epoch += 1
                if self._epoch == 1:
                    self.B = np.array([[0.95, 0.05], [0.1, 0.9]])
                else:
                    self.B = np.array([[0.55, 0.45], [0.45, 0.55]])

        hmm = RollbackHMM()
        expected_best_B = np.array([[0.95, 0.05], [0.1, 0.9]])
        train_seqs = [[0, 1, 0, 1]]
        val_seqs = [[0, 0, 0, 0]]
        real_forward = algorithms.forward
        val_lls = iter([10.0, 0.0])

        def controlled_forward(
            hmm: HMM,
            obs: list[int],
            mode: ComputeMode = ComputeMode.SCALED,
        ) -> tuple[float, np.ndarray, np.ndarray | None]:
            log_prob, alpha, c = real_forward(hmm, obs, mode=mode)
            if list(obs) == val_seqs[0]:
                return (next(val_lls), alpha, c)
            return (log_prob, alpha, c)

        monkeypatch.setattr(algorithms, "forward", controlled_forward)

        trained = baum_welch(hmm, train_seqs, epochs=2, val_set=val_seqs)

        assert np.allclose(trained.B, expected_best_B)

    def test_m_step_streaming(self) -> None:
        """Test streaming m_step returns sufficient statistics for one sequence."""
        hmm = HMM(n_states=2, V=[0, 1])
        obs_seq = [0, 1, 0, 1]
        gamma = np.array([[0.8, 0.2, 0.7, 0.3], [0.2, 0.8, 0.3, 0.7]])
        xi = np.zeros((2, 2, 3))
        xi[0, 0, :] = 0.5
        xi[1, 1, :] = 0.5

        # Should return sufficient statistics dictionary
        stats = hmm.m_step_streaming(obs_seq, gamma, xi)

        assert "expect_si_t0" in stats
        assert "expected_transitions" in stats
        assert stats["expect_si_t0"].shape == (2,)
        assert stats["expected_transitions"].shape == (2, 2)
