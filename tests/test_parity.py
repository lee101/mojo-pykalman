"""Numerical and behavioral parity with pykalman 0.11."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pykalman import KalmanFilter as PythonKalmanFilter

from mojo_pykalman import KalmanFilter
from mojo_pykalman._lib import addr, lib


@pytest.fixture
def model_data():
    rng = np.random.default_rng(42)
    timesteps, n_state, n_obs = 120, 4, 3
    transition = np.array(
        [
            [0.92, 0.08, 0.00, 0.00],
            [0.00, 0.87, 0.10, 0.00],
            [0.00, 0.00, 0.81, 0.12],
            [0.03, 0.00, 0.00, 0.76],
        ]
    )
    observation = rng.normal(size=(n_obs, n_state))
    transition_covariance = np.diag([0.06, 0.08, 0.05, 0.07])
    observation_covariance = np.diag([0.12, 0.18, 0.14])
    transition_offset = np.array([0.03, -0.02, 0.01, 0.04])
    observation_offset = np.array([0.2, -0.1, 0.05])
    initial_mean = np.array([0.5, -0.2, 0.1, 0.3])
    initial_covariance = np.diag([0.8, 0.7, 0.9, 0.6])
    parameters = dict(
        transition_matrices=transition,
        observation_matrices=observation,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        transition_offsets=transition_offset,
        observation_offsets=observation_offset,
        initial_state_mean=initial_mean,
        initial_state_covariance=initial_covariance,
    )
    observations = PythonKalmanFilter(**parameters).sample(
        timesteps, random_state=7
    )[1]
    return observations, parameters


def pair(parameters):
    return PythonKalmanFilter(**parameters), KalmanFilter(**parameters)


def test_constructor_signature_matches_upstream():
    assert inspect.signature(KalmanFilter) == inspect.signature(PythonKalmanFilter)


def test_default_filter_matches_upstream():
    observations = np.array([0.2, 0.4, -0.1, 0.3])
    expected = PythonKalmanFilter().filter(observations)
    actual = KalmanFilter().filter(observations)
    assert np.allclose(actual[0], expected[0], atol=1e-14)
    assert np.allclose(actual[1], expected[1], atol=1e-14)


def test_filter_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    expected_mean, expected_covariance = expected.filter(observations)
    actual_mean, actual_covariance = actual.filter(observations)
    assert np.allclose(actual_mean, expected_mean, atol=2e-13)
    assert np.allclose(actual_covariance, expected_covariance, atol=2e-13)


def test_smooth_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    expected_mean, expected_covariance = expected.smooth(observations)
    actual_mean, actual_covariance = actual.smooth(observations)
    assert np.allclose(actual_mean, expected_mean, atol=3e-13)
    assert np.allclose(actual_covariance, expected_covariance, atol=3e-13)


def test_loglikelihood_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    assert actual.loglikelihood(observations) == pytest.approx(
        expected.loglikelihood(observations), abs=2e-8
    )


def test_simd_scalar_tail_matches_upstream():
    rng = np.random.default_rng(9)
    n_state, n_obs, timesteps = 5, 3, 37
    parameters = dict(
        transition_matrices=np.eye(n_state) * 0.8
        + rng.normal(scale=0.02, size=(n_state, n_state)),
        observation_matrices=rng.normal(size=(n_obs, n_state)),
        transition_covariance=np.eye(n_state) * 0.1,
        observation_covariance=np.eye(n_obs) * 0.2,
        transition_offsets=rng.normal(scale=0.01, size=n_state),
        observation_offsets=rng.normal(scale=0.01, size=n_obs),
        initial_state_mean=np.zeros(n_state),
        initial_state_covariance=np.eye(n_state),
    )
    observations = rng.normal(size=(timesteps, n_obs))
    expected, actual = pair(parameters)
    for method in ("filter", "smooth"):
        expected_result = getattr(expected, method)(observations)
        actual_result = getattr(actual, method)(observations)
        assert np.allclose(actual_result[0], expected_result[0], atol=3e-13)
        assert np.allclose(actual_result[1], expected_result[1], atol=3e-13)
    assert actual.loglikelihood(observations) == pytest.approx(
        expected.loglikelihood(observations), abs=2e-8
    )


@pytest.mark.parametrize("n_state,n_obs", [(1, 1), (3, 2), (4, 3), (6, 5), (7, 3)])
def test_simd_boundaries_and_tails(n_state, n_obs):
    rng = np.random.default_rng(100 + n_state)
    parameters = dict(
        transition_matrices=np.eye(n_state) * 0.75,
        observation_matrices=rng.normal(size=(n_obs, n_state)),
        transition_covariance=np.eye(n_state) * 0.1,
        observation_covariance=np.eye(n_obs) * 0.2,
    )
    observations = rng.normal(size=(11, n_obs))
    expected, actual = pair(parameters)
    expected_result = expected.smooth(observations)
    actual_result = actual.smooth(observations)
    assert np.allclose(actual_result[0], expected_result[0], atol=3e-13)
    assert np.allclose(actual_result[1], expected_result[1], atol=3e-13)


def test_masked_rows_match_upstream(model_data):
    observations, parameters = model_data
    masked = np.ma.array(observations, mask=False)
    masked.mask[8, 1] = True
    masked.mask[40:48] = True
    masked.mask[-1, 0] = True
    expected, actual = pair(parameters)
    for method in ("filter", "smooth"):
        expected_result = getattr(expected, method)(masked)
        actual_result = getattr(actual, method)(masked)
        assert np.allclose(actual_result[0], expected_result[0], atol=3e-13)
        assert np.allclose(actual_result[1], expected_result[1], atol=3e-13)
    assert actual.loglikelihood(masked) == pytest.approx(
        expected.loglikelihood(masked), abs=2e-8
    )


def test_time_varying_parameters_match_upstream(model_data):
    observations, parameters = model_data
    timesteps = len(observations)
    varying = dict(parameters)
    varying["transition_matrices"] = np.stack(
        [
            parameters["transition_matrices"]
            + np.eye(4) * (0.003 * np.sin(t / 9.0))
            for t in range(timesteps - 1)
        ]
    )
    varying["transition_covariance"] = np.stack(
        [
            parameters["transition_covariance"] * (1.0 + 0.1 * np.cos(t / 7.0))
            for t in range(timesteps - 1)
        ]
    )
    varying["transition_offsets"] = np.stack(
        [parameters["transition_offsets"] + 0.01 * np.sin(t / 11.0) for t in range(timesteps - 1)]
    )
    varying["observation_matrices"] = np.stack(
        [parameters["observation_matrices"] + 0.005 * np.cos(t / 8.0) for t in range(timesteps)]
    )
    varying["observation_covariance"] = np.stack(
        [
            parameters["observation_covariance"] * (1.0 + 0.08 * np.sin(t / 10.0))
            for t in range(timesteps)
        ]
    )
    varying["observation_offsets"] = np.stack(
        [parameters["observation_offsets"] + 0.01 * np.cos(t / 13.0) for t in range(timesteps)]
    )
    expected, actual = pair(varying)
    for method in ("filter", "smooth"):
        expected_result = getattr(expected, method)(observations)
        actual_result = getattr(actual, method)(observations)
        assert np.allclose(actual_result[0], expected_result[0], atol=4e-13)
        assert np.allclose(actual_result[1], expected_result[1], atol=4e-13)


def test_filter_update_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    mean, covariance = expected.filter(observations[:20])
    expected_update = expected.filter_update(mean[-1], covariance[-1], observations[20])
    actual_update = actual.filter_update(mean[-1], covariance[-1], observations[20])
    assert np.allclose(actual_update[0], expected_update[0], atol=2e-13)
    assert np.allclose(actual_update[1], expected_update[1], atol=2e-13)


def test_filter_update_missing_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    mean, covariance = expected.filter(observations[:20])
    expected_update = expected.filter_update(mean[-1], covariance[-1], None)
    actual_update = actual.filter_update(mean[-1], covariance[-1], None)
    assert np.allclose(actual_update[0], expected_update[0], atol=2e-13)
    assert np.allclose(actual_update[1], expected_update[1], atol=2e-13)


def test_filter_update_overrides_match_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    mean, covariance = expected.filter(observations[:12])
    overrides = dict(
        transition_matrix=parameters["transition_matrices"] * 0.98,
        transition_offset=parameters["transition_offsets"] + 0.02,
        transition_covariance=parameters["transition_covariance"] * 1.1,
        observation_matrix=parameters["observation_matrices"] * 1.02,
        observation_offset=parameters["observation_offsets"] - 0.01,
        observation_covariance=parameters["observation_covariance"] * 0.9,
    )
    expected_update = expected.filter_update(
        mean[-1], covariance[-1], observations[12], **overrides
    )
    actual_update = actual.filter_update(
        mean[-1], covariance[-1], observations[12], **overrides
    )
    assert np.allclose(actual_update[0], expected_update[0], atol=2e-13)
    assert np.allclose(actual_update[1], expected_update[1], atol=2e-13)


def test_sample_matches_upstream(model_data):
    _, parameters = model_data
    expected, actual = pair(parameters)
    expected_states, expected_observations = expected.sample(30, random_state=123)
    actual_states, actual_observations = actual.sample(30, random_state=123)
    assert np.array_equal(actual_states, expected_states)
    assert np.array_equal(actual_observations, expected_observations)


def test_sample_time_varying_matches_upstream(model_data):
    _, parameters = model_data
    steps = 8
    varying = dict(parameters)
    for name in ("transition_matrices", "transition_covariance", "transition_offsets"):
        varying[name] = np.repeat(parameters[name][None], steps - 1, axis=0)
    for name in ("observation_matrices", "observation_covariance", "observation_offsets"):
        varying[name] = np.repeat(parameters[name][None], steps, axis=0)
    expected, actual = pair(varying)
    expected_result = expected.sample(steps, random_state=321)
    actual_result = actual.sample(steps, random_state=321)
    assert np.array_equal(actual_result[0], expected_result[0])
    assert np.array_equal(actual_result[1], expected_result[1])


def test_default_em_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    assert actual.em(observations, n_iter=3) is actual
    expected.em(observations, n_iter=3)
    for name in (
        "transition_covariance",
        "observation_covariance",
        "initial_state_mean",
        "initial_state_covariance",
    ):
        assert np.allclose(getattr(actual, name), getattr(expected, name), atol=2e-12)


def test_all_parameter_em_matches_upstream(model_data):
    observations, parameters = model_data
    expected, actual = pair(parameters)
    expected.em(observations, n_iter=2, em_vars="all")
    actual.em(observations, n_iter=2, em_vars="all")
    for name in (
        "transition_matrices",
        "observation_matrices",
        "transition_offsets",
        "observation_offsets",
        "transition_covariance",
        "observation_covariance",
        "initial_state_mean",
        "initial_state_covariance",
    ):
        assert np.allclose(getattr(actual, name), getattr(expected, name), atol=3e-12)


def test_selective_em_leaves_other_parameters_unchanged(model_data):
    observations, parameters = model_data
    selected = ["transition_matrices", "observation_matrices"]
    expected, actual = pair(parameters)
    expected.em(observations, n_iter=2, em_vars=selected)
    actual.em(observations, n_iter=2, em_vars=selected)
    assert np.allclose(actual.transition_matrices, expected.transition_matrices, atol=2e-12)
    assert np.allclose(actual.observation_matrices, expected.observation_matrices, atol=2e-12)
    assert np.array_equal(actual.transition_covariance, parameters["transition_covariance"])
    assert np.array_equal(actual.observation_covariance, parameters["observation_covariance"])


def test_masked_em_matches_upstream(model_data):
    observations, parameters = model_data
    observations = np.ma.array(observations, mask=False)
    observations.mask[::9, 0] = True
    expected, actual = pair(parameters)
    variables = ["observation_matrices", "observation_covariance", "observation_offsets"]
    expected.em(observations, n_iter=2, em_vars=variables)
    actual.em(observations, n_iter=2, em_vars=variables)
    for name in variables:
        assert np.allclose(getattr(actual, name), getattr(expected, name), atol=3e-12)


def test_em_does_not_decrease_likelihood(model_data):
    observations, parameters = model_data
    model = KalmanFilter(**parameters)
    before = model.loglikelihood(observations)
    model.em(observations, n_iter=2)
    assert model.loglikelihood(observations) >= before - 1e-10


def test_singular_innovation_uses_compatible_fallback():
    parameters = dict(
        transition_matrices=np.eye(2),
        observation_matrices=np.array([[1.0, 0.0], [1.0, 0.0]]),
        transition_covariance=np.zeros((2, 2)),
        observation_covariance=np.zeros((2, 2)),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2),
    )
    observations = np.array([[1.0, 1.0], [1.5, 1.5], [0.5, 0.5]])
    expected, actual = pair(parameters)
    expected_result = expected.filter(observations)
    actual_result = actual.filter(observations)
    assert np.allclose(actual_result[0], expected_result[0], atol=1e-12)
    assert np.allclose(actual_result[1], expected_result[1], atol=1e-12)
    assert actual.loglikelihood(observations) == pytest.approx(
        expected.loglikelihood(observations), abs=1e-6
    )


def test_singular_predicted_covariance_uses_compatible_smoother_fallback():
    parameters = dict(
        transition_matrices=np.eye(2),
        observation_matrices=np.eye(2),
        transition_covariance=np.zeros((2, 2)),
        observation_covariance=np.eye(2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.zeros((2, 2)),
    )
    observations = np.zeros((4, 2))
    expected, actual = pair(parameters)
    expected_result = expected.smooth(observations)
    actual_result = actual.smooth(observations)
    assert np.allclose(actual_result[0], expected_result[0], atol=1e-12)
    assert np.allclose(actual_result[1], expected_result[1], atol=1e-12)


def test_inconsistent_dimensions_raise():
    with pytest.raises(ValueError, match="not consistent"):
        KalmanFilter(
            transition_matrices=np.eye(3),
            observation_matrices=np.ones((2, 4)),
        )


def test_ffi_boundary_rejects_malformed_buffers_before_call():
    model = KalmanFilter(n_dim_state=2, n_dim_obs=1)
    with pytest.raises(ValueError, match="filtered_state_mean"):
        model.filter_update(np.zeros(1), np.eye(2), [0.0])
    with pytest.raises(ValueError, match="transition_matrix"):
        model.filter_update(
            np.zeros(2), np.eye(2), [0.0], transition_matrix=np.eye(3)
        )
    with pytest.raises(ValueError, match="at least one observation"):
        model.em(np.empty((0, 1)), n_iter=1)
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.zeros((3, 3))[:, ::2])
    assert lib().mpk_smooth_pair(0, 0, 0, 1, 1) == 0


def test_float64_boundary_rejects_silent_narrowing():
    with pytest.raises(TypeError, match="complex"):
        KalmanFilter(transition_matrices=np.eye(2, dtype=np.complex128)).filter(
            np.zeros((2, 2))
        )
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(TypeError, match="narrowed"):
            KalmanFilter(
                transition_matrices=np.eye(2, dtype=np.longdouble)
            ).filter(np.zeros((2, 2)))
