"""Tests for HMM and HMMClassifier classes."""

import numpy as np
import pytest

from hmm import HMM, HMMClassifier


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

        with pytest.raises(ValueError, match="pos/neg hmm.*missing"):
            classifier.classify([1, 2, 3])

    def test_classify_not_implemented(self) -> None:
        """Test classification raises NotImplementedError when algorithms missing."""
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        B = np.array([[1.0 / 6] * 6, [1.0 / 10] * 5 + [1.0 / 2]])
        V = [1, 2, 3, 4, 5, 6]

        pos_hmm = HMM(n_states=2, A=A, B=B, V=V)
        neg_hmm = HMM(n_states=2, A=A, B=B, V=V)

        classifier = HMMClassifier(pos_hmm=pos_hmm, neg_hmm=neg_hmm)

        with pytest.raises(NotImplementedError, match="forward algorithm"):
            classifier.classify([1, 2, 3])
