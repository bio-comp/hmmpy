"""Property-based tests for HMM using Hypothesis."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hmm import HMM, backward, baum_welch, forward, viterbi
from hmm.algorithms import ComputeMode


def valid_transition_matrix(n_states: int) -> np.ndarray:
    """Generate a valid row-stochastic transition matrix."""
    A = np.random.rand(n_states, n_states)
    A = A / A.sum(axis=1, keepdims=True)
    return A


def valid_emission_matrix(n_states: int, n_symbols: int) -> np.ndarray:
    """Generate a valid row-stochastic emission matrix."""
    B = np.random.rand(n_states, n_symbols)
    B = B / B.sum(axis=1, keepdims=True)
    return B


@given(n_states=st.integers(min_value=1, max_value=5))
def test_hmm_probabilities_sum_to_one(n_states: int) -> None:
    """HMM probabilities (A, B, Pi) should each sum to 1."""
    V = list(range(5))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, len(V))
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    assert np.allclose(hmm.A.sum(axis=1), 1.0), "Transition matrix rows should sum to 1"
    assert np.allclose(hmm.B.sum(axis=1), 1.0), "Emission matrix rows should sum to 1"
    assert np.isclose(hmm.Pi.sum(), 1.0), "Initial state distribution should sum to 1"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_forward_probability_non_negative(n_states: int, n_symbols: int) -> None:
    """Forward algorithm should return non-negative probabilities."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 5

    result = forward(hmm, obs, mode=ComputeMode.SCALED)
    log_prob = result[0]
    assert np.isfinite(log_prob), "Log probability should be finite"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_forward_backward_consistency(n_states: int, n_symbols: int) -> None:
    """Forward and backward algorithms should be consistent."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 3
    assume(len(obs) >= 2)

    log_prob, alpha, c = forward(hmm, obs, mode=ComputeMode.SCALED)
    beta = backward(hmm, obs, scaling_coeffs=c)

    for t in range(len(obs)):
        gamma_sum = np.sum(alpha[:, t] * beta[:, t])
        assert np.isclose(gamma_sum, c[t], atol=1e-6), (
            f"Forward-backward consistency failed at t={t}"
        )


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_viterbi_path_valid(n_states: int, n_symbols: int) -> None:
    """Viterbi path should be valid state indices."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 3

    path, _, _ = viterbi(hmm, obs, mode=ComputeMode.LOG)
    assert len(path) == len(obs), "Path length should match observation length"
    assert all(0 <= s < hmm.N for s in path), "All states in path should be valid"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_viterbi_prob_at_least_forward(n_states: int, n_symbols: int) -> None:
    """Viterbi probability should be <= Forward probability."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 3

    forward_prob, _, _ = forward(hmm, obs, mode=ComputeMode.SCALED)
    _, delta, _ = viterbi(hmm, obs, mode=ComputeMode.LOG)
    viterbi_log_prob = np.max(delta[:, -1])

    assert viterbi_log_prob <= forward_prob + 1e-3, "Viterbi prob should be <= Forward prob"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=10)
def test_baum_welch_increases_log_likelihood(n_states: int, n_symbols: int) -> None:
    """Baum-Welch should increase log-likelihood after training."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)

    obs_seqs = [V * 3, V * 2]

    initial_hmm = HMM(n_states=n_states, A=A.copy(), B=B.copy(), V=V)
    initial_ll = sum(forward(initial_hmm, obs, mode=ComputeMode.SCALED)[0] for obs in obs_seqs)

    trained_hmm = baum_welch(
        HMM(n_states=n_states, A=A.copy(), B=B.copy(), V=V),
        obs_seqs=obs_seqs,
        epochs=5,
        mode=ComputeMode.SCALED,
    )

    final_ll = sum(forward(trained_hmm, obs, mode=ComputeMode.SCALED)[0] for obs in obs_seqs)

    assert final_ll >= initial_ll - 1e-3, "Log-likelihood should not decrease after training"


@given(n_states=st.sampled_from([1, 2, 3]), n_symbols=st.integers(min_value=2, max_value=5))
def test_single_state_hmm(n_states: int, n_symbols: int) -> None:
    """Single state HMM should work correctly."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = [V[0]] * 5

    log_prob, alpha = forward(hmm, obs, mode=ComputeMode.UNSCALED)
    assert np.isfinite(log_prob), "Forward should work"

    path, delta, _ = viterbi(hmm, obs, mode=ComputeMode.LOG)
    assert len(path) == len(obs), "Viterbi path length should match"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_forward_scaling_works(n_states: int, n_symbols: int) -> None:
    """Forward with scaling should give consistent results."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 3

    log_prob_scaled, alpha_scaled, c = forward(hmm, obs, mode=ComputeMode.SCALED)
    prob_unscaled, alpha_unscaled = forward(hmm, obs, mode=ComputeMode.UNSCALED)

    log_prob_unscaled = np.log(prob_unscaled + 1e-300)

    assert np.isclose(log_prob_scaled, log_prob_unscaled, rtol=1e-3), (
        "Scaled and unscaled should match"
    )


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_backward_scaling_c(n_states: int, n_symbols: int) -> None:
    """Backward algorithm with scaling coefficients should be consistent."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)
    obs = V * 3

    log_prob, alpha, c = forward(hmm, obs, mode=ComputeMode.SCALED)
    beta = backward(hmm, obs, scaling_coeffs=c)

    for t in range(len(obs)):
        gamma_sum = np.sum(alpha[:, t] * beta[:, t])
        assert np.isclose(gamma_sum, c[t], atol=1e-5), f"Alpha * Beta should equal c[{t}]"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=10)
def test_baum_welch_preserves_hmm_structure(n_states: int, n_symbols: int) -> None:
    """Baum-Welch should preserve valid HMM structure (row-stochastic matrices)."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)

    obs_seqs = [V * 3, V * 2]

    trained = baum_welch(
        HMM(n_states=n_states, A=A.copy(), B=B.copy(), V=V),
        obs_seqs=obs_seqs,
        epochs=3,
        mode=ComputeMode.SCALED,
    )

    assert np.allclose(trained.A.sum(axis=1), 1.0, atol=1e-4), "A should remain row-stochastic"
    assert np.allclose(trained.B.sum(axis=1), 1.0, atol=1e-4), "B should remain row-stochastic"
    assert np.isclose(trained.Pi.sum(), 1.0, atol=1e-4), "Pi should remain valid distribution"


@given(
    n_states=st.integers(min_value=1, max_value=3), n_symbols=st.integers(min_value=2, max_value=5)
)
def test_empty_observation_fails_gracefully(n_states: int, n_symbols: int) -> None:
    """Empty observation sequence should be handled gracefully."""
    V = list(range(n_symbols))
    A = valid_transition_matrix(n_states)
    B = valid_emission_matrix(n_states, n_symbols)
    hmm = HMM(n_states=n_states, A=A, B=B, V=V)

    with pytest.raises((IndexError, ValueError)):
        forward(hmm, [])


def test_hmmclassifier_classify() -> None:
    """HMMClassifier classify method should work."""
    from hmm import HMMClassifier

    A = np.array([[0.7, 0.3], [0.3, 0.7]])
    B = np.array([[0.9, 0.1], [0.1, 0.9]])
    V = [0, 1]

    pos_hmm = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)
    neg_hmm = HMM(n_states=2, A=A.copy(), B=B.copy(), V=V)

    classifier = HMMClassifier(pos_hmm=pos_hmm, neg_hmm=neg_hmm)
    obs = [0, 1, 0, 1]

    result = classifier.classify(obs)
    assert isinstance(result, float)


def test_hmm_with_labels() -> None:
    """HMM with custom labels should work."""
    V = [0, 1]
    A = valid_transition_matrix(2)
    B = valid_emission_matrix(2, len(V))

    hmm = HMM(n_states=2, A=A, B=B, V=V, Labels=["state_a", "state_b"])
    assert hmm.Labels == ["state_a", "state_b"]


def test_hmm_with_fixed_emission() -> None:
    """HMM with fixed emission probabilities should work."""
    V = [0, 1]
    A = valid_transition_matrix(2)
    F = {0: np.array([1.0, 0.0])}

    hmm = HMM(n_states=2, A=A, V=V, F=F)
    assert hmm.F[0].tolist() == [1.0, 0.0]
