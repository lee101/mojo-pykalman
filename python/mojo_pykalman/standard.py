"""A pykalman-compatible linear Gaussian Kalman filter backed by Mojo."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, i64, lib

EM_VARIABLES = {
    "transition_matrices",
    "observation_matrices",
    "transition_offsets",
    "observation_offsets",
    "transition_covariance",
    "observation_covariance",
    "initial_state_mean",
    "initial_state_covariance",
}


def _determine_dimensionality(candidates, default):
    dimensions = []
    for value, index in candidates:
        if value is None:
            continue
        shape = np.asanyarray(value).shape
        try:
            dimensions.append(int(shape[index]))
        except IndexError as error:
            raise ValueError("parameters must have the documented dimensions") from error
    if default is not None:
        dimensions.append(int(default))
    if not dimensions:
        return 1
    if any(value <= 0 for value in dimensions):
        raise ValueError("state and observation dimensions must be positive")
    if any(value != dimensions[0] for value in dimensions):
        raise ValueError(
            "The shape of all parameters is not consistent. Please re-check their values."
        )
    return dimensions[0]


def _matrix(value, default, rows, columns, name):
    array = f64(default if value is None else value)
    if array.ndim not in (2, 3) or array.shape[-2:] != (rows, columns):
        raise ValueError(
            f"{name} must have shape ({rows}, {columns}) or (time, {rows}, {columns})"
        )
    return array


def _vector(value, default, width, name):
    array = f64(default if value is None else value)
    if array.ndim not in (1, 2) or array.shape[-1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (time, {width})")
    return array


def _at(array, t):
    return array[t] if array.ndim > 2 else array


def _at_vector(array, t):
    return array[t] if array.ndim > 1 else array


def _varying(array, base_ndim):
    return int(array.ndim == base_ndim + 1)


def _call(name, *arguments):
    if getattr(lib(), name)(*arguments) != 1:
        raise RuntimeError(f"{name} rejected invalid FFI arguments")


def _check_length(array, needed, base_ndim, name):
    if array.ndim == base_ndim + 1 and array.shape[0] < needed:
        raise ValueError(f"{name} has {array.shape[0]} time steps; {needed} are required")


def _parse_observations(value, n_obs):
    observations = np.ma.atleast_2d(value)
    if observations.shape[0] == 1 and observations.shape[1] > 1:
        observations = observations.T
    if observations.shape[1] != n_obs:
        raise ValueError(
            f"observations have dimension {observations.shape[1]}, expected {n_obs}"
        )
    data = f64(np.ma.getdata(observations))
    mask = np.ma.getmaskarray(observations)
    missing = i64(np.any(mask, axis=1))
    return observations, data, missing


@dataclass
class _Inference:
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    loglikelihoods: np.ndarray


@dataclass
class _Smoothing:
    means: np.ndarray
    covariances: np.ndarray
    gains: np.ndarray


def _filter_numpy(data, missing, parameters):
    A, b, Q, C, d, R, mu0, P0 = parameters
    timesteps, n_obs = data.shape
    n_state = mu0.shape[0]
    predicted_means = np.zeros((timesteps, n_state))
    predicted_covariances = np.zeros((timesteps, n_state, n_state))
    gains = np.zeros((timesteps, n_state, n_obs))
    filtered_means = np.zeros((timesteps, n_state))
    filtered_covariances = np.zeros((timesteps, n_state, n_state))
    loglikelihoods = np.zeros(timesteps)
    for t in range(timesteps):
        if t == 0:
            predicted_means[t] = mu0
            predicted_covariances[t] = P0
        else:
            At = _at(A, t - 1)
            predicted_means[t] = At @ filtered_means[t - 1] + _at_vector(b, t - 1)
            predicted_covariances[t] = (
                At @ filtered_covariances[t - 1] @ At.T + _at(Q, t - 1)
            )
        if missing[t]:
            filtered_means[t] = predicted_means[t]
            filtered_covariances[t] = predicted_covariances[t]
            continue
        Ct = _at(C, t)
        residual = data[t] - Ct @ predicted_means[t] - _at_vector(d, t)
        innovation = Ct @ predicted_covariances[t] @ Ct.T + _at(R, t)
        inverse = np.linalg.pinv(innovation)
        gains[t] = predicted_covariances[t] @ Ct.T @ inverse
        filtered_means[t] = predicted_means[t] + gains[t] @ residual
        filtered_covariances[t] = (
            predicted_covariances[t] - gains[t] @ Ct @ predicted_covariances[t]
        )
        try:
            likelihood_cholesky = np.linalg.cholesky(innovation)
        except np.linalg.LinAlgError:
            likelihood_cholesky = np.linalg.cholesky(
                innovation + 1.0e-7 * np.eye(n_obs)
            )
        likelihood_residual = np.linalg.solve(likelihood_cholesky, residual)
        loglikelihoods[t] = -0.5 * (
            n_obs * np.log(2.0 * np.pi)
            + 2.0 * np.log(np.diag(likelihood_cholesky)).sum()
            + likelihood_residual @ likelihood_residual
        )
    return _Inference(
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        loglikelihoods,
    )


def _run_filter(data, missing, parameters, *, store_predictions=False):
    A, b, Q, C, d, R, mu0, P0 = parameters
    timesteps, n_obs = data.shape
    n_state = mu0.shape[0]
    for value, needed, ndim, name in (
        (A, max(timesteps - 1, 0), 2, "transition_matrices"),
        (b, max(timesteps - 1, 0), 1, "transition_offsets"),
        (Q, max(timesteps - 1, 0), 2, "transition_covariance"),
        (C, timesteps, 2, "observation_matrices"),
        (d, timesteps, 1, "observation_offsets"),
        (R, timesteps, 2, "observation_covariance"),
    ):
        _check_length(value, needed, ndim, name)
    prediction_steps = timesteps if store_predictions else 1
    predicted_means = np.empty((prediction_steps, n_state))
    predicted_covariances = np.empty((prediction_steps, n_state, n_state))
    filtered_means = np.empty((timesteps, n_state))
    filtered_covariances = np.empty((timesteps, n_state, n_state))
    loglikelihoods = np.empty(timesteps)
    width = max(n_state, n_obs)
    scratch = np.empty(4 * width * width)
    status = lib().mpk_filter(
        addr(data),
        addr(missing),
        addr(A),
        addr(C),
        addr(Q),
        addr(R),
        addr(b),
        addr(d),
        addr(mu0),
        addr(P0),
        addr(predicted_means),
        addr(predicted_covariances),
        addr(filtered_means),
        addr(filtered_covariances),
        addr(loglikelihoods),
        addr(scratch),
        timesteps,
        n_state,
        n_obs,
        _varying(A, 2),
        _varying(C, 2),
        _varying(Q, 2),
        _varying(R, 2),
        _varying(b, 1),
        _varying(d, 1),
        int(store_predictions),
    )
    if not status:
        return _filter_numpy(data, missing, parameters)
    return _Inference(
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        loglikelihoods,
    )


def _smooth_numpy(A, inference):
    filtered_means = inference.filtered_means
    filtered_covariances = inference.filtered_covariances
    timesteps, n_state = filtered_means.shape
    means = filtered_means.copy()
    covariances = filtered_covariances.copy()
    gains = np.zeros((max(timesteps - 1, 0), n_state, n_state))
    for t in reversed(range(timesteps - 1)):
        gains[t] = (
            filtered_covariances[t]
            @ _at(A, t).T
            @ np.linalg.pinv(inference.predicted_covariances[t + 1])
        )
        means[t] += gains[t] @ (means[t + 1] - inference.predicted_means[t + 1])
        covariances[t] += (
            gains[t]
            @ (covariances[t + 1] - inference.predicted_covariances[t + 1])
            @ gains[t].T
        )
    return _Smoothing(means, covariances, gains)


def _run_smooth(A, inference):
    timesteps, n_state = inference.filtered_means.shape
    means = np.empty_like(inference.filtered_means)
    covariances = np.empty_like(inference.filtered_covariances)
    gains_storage = np.empty((max(timesteps - 1, 1), n_state, n_state))
    scratch = np.empty(3 * n_state * n_state)
    status = lib().mpk_smooth(
        addr(A),
        addr(inference.filtered_means),
        addr(inference.filtered_covariances),
        addr(inference.predicted_means),
        addr(inference.predicted_covariances),
        addr(means),
        addr(covariances),
        addr(gains_storage),
        addr(scratch),
        timesteps,
        n_state,
        _varying(A, 2),
    )
    if not status:
        return _smooth_numpy(A, inference)
    return _Smoothing(means, covariances, gains_storage[: timesteps - 1])


def _smooth_pair(smoothing):
    timesteps, n_state = smoothing.means.shape
    pairwise = np.empty((timesteps, n_state, n_state))
    gains = smoothing.gains
    if not gains.size:
        gains = np.empty((1, n_state, n_state))
    _call(
        "mpk_smooth_pair",
        addr(smoothing.covariances),
        addr(gains),
        addr(pairwise),
        timesteps,
        n_state,
    )
    return pairwise


def _check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState")


class KalmanFilter:
    """Linear-Gaussian Kalman filter, RTS smoother, and EM estimator."""

    def __init__(
        self,
        transition_matrices=None,
        observation_matrices=None,
        transition_covariance=None,
        observation_covariance=None,
        transition_offsets=None,
        observation_offsets=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        random_state=None,
        em_vars=None,
        n_dim_state=None,
        n_dim_obs=None,
    ):
        n_dim_state = _determine_dimensionality(
            [
                (transition_matrices, -2),
                (transition_offsets, -1),
                (transition_covariance, -2),
                (initial_state_mean, -1),
                (initial_state_covariance, -2),
                (observation_matrices, -1),
            ],
            n_dim_state,
        )
        n_dim_obs = _determine_dimensionality(
            [
                (observation_matrices, -2),
                (observation_offsets, -1),
                (observation_covariance, -2),
            ],
            n_dim_obs,
        )
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.em_vars = em_vars
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self._em_vars = (
            [
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance",
            ]
            if em_vars is None
            else em_vars
        )

    def _initialize_parameters(self):
        n, m = self.n_dim_state, self.n_dim_obs
        A = _matrix(self.transition_matrices, np.eye(n), n, n, "transition_matrices")
        b = _vector(self.transition_offsets, np.zeros(n), n, "transition_offsets")
        Q = _matrix(
            self.transition_covariance, np.eye(n), n, n, "transition_covariance"
        )
        C = _matrix(
            self.observation_matrices, np.eye(m, n), m, n, "observation_matrices"
        )
        d = _vector(self.observation_offsets, np.zeros(m), m, "observation_offsets")
        R = _matrix(
            self.observation_covariance, np.eye(m), m, m, "observation_covariance"
        )
        mu0 = _vector(self.initial_state_mean, np.zeros(n), n, "initial_state_mean")
        P0 = _matrix(
            self.initial_state_covariance,
            np.eye(n),
            n,
            n,
            "initial_state_covariance",
        )
        if mu0.ndim != 1 or P0.ndim != 2:
            raise ValueError("initial state parameters cannot be time-varying")
        return A, b, Q, C, d, R, mu0, P0

    def _observations_and_parameters(self, X):
        parameters = self._initialize_parameters()
        _, data, missing = _parse_observations(X, self.n_dim_obs)
        if not len(data):
            raise ValueError("at least one observation is required")
        return data, missing, parameters

    def filter(self, X):
        data, missing, parameters = self._observations_and_parameters(X)
        result = _run_filter(data, missing, parameters)
        return result.filtered_means, result.filtered_covariances

    def smooth(self, X):
        data, missing, parameters = self._observations_and_parameters(X)
        inference = _run_filter(data, missing, parameters, store_predictions=True)
        smoothing = _run_smooth(parameters[0], inference)
        return smoothing.means, smoothing.covariances

    def loglikelihood(self, X):
        data, missing, parameters = self._observations_and_parameters(X)
        return float(_run_filter(data, missing, parameters).loglikelihoods.sum())

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_matrix=None,
        transition_offset=None,
        transition_covariance=None,
        observation_matrix=None,
        observation_offset=None,
        observation_covariance=None,
    ):
        A, b, Q, C, d, R, _, _ = self._initialize_parameters()

        def choose(value, default, ndim, name):
            selected = default if value is None else f64(value)
            if selected.ndim != ndim:
                raise ValueError(
                    f"{name} is not constant for all time. You must specify it manually."
                )
            return selected

        A = choose(transition_matrix, A, 2, "transition_matrix")
        b = choose(transition_offset, b, 1, "transition_offset")
        Q = choose(transition_covariance, Q, 2, "transition_covariance")
        C = choose(observation_matrix, C, 2, "observation_matrix")
        d = choose(observation_offset, d, 1, "observation_offset")
        R = choose(observation_covariance, R, 2, "observation_covariance")
        expected_shapes = (
            (A, (self.n_dim_state, self.n_dim_state), "transition_matrix"),
            (b, (self.n_dim_state,), "transition_offset"),
            (Q, (self.n_dim_state, self.n_dim_state), "transition_covariance"),
            (C, (self.n_dim_obs, self.n_dim_state), "observation_matrix"),
            (d, (self.n_dim_obs,), "observation_offset"),
            (R, (self.n_dim_obs, self.n_dim_obs), "observation_covariance"),
        )
        for value, shape, name in expected_shapes:
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        first = np.zeros(self.n_dim_obs)
        state_mean = _vector(
            filtered_state_mean, None, self.n_dim_state, "filtered_state_mean"
        )
        state_covariance = _matrix(
            filtered_state_covariance,
            None,
            self.n_dim_state,
            self.n_dim_state,
            "filtered_state_covariance",
        )
        if state_mean.ndim != 1 or state_covariance.ndim != 2:
            raise ValueError("filtered state parameters cannot be time-varying")
        if observation is not None:
            observation_array = np.ma.asarray(observation)
            if observation_array.shape != (self.n_dim_obs,):
                raise ValueError(
                    f"observation must have shape ({self.n_dim_obs},)"
                )
            second = f64(np.ma.getdata(observation_array))
            missing_observation = np.any(np.ma.getmaskarray(observation_array))
            data = f64(np.vstack([first, second]))
        else:
            missing_observation = True
            data = f64(np.vstack([first, np.zeros(self.n_dim_obs)]))
        missing = i64([1, int(missing_observation)])
        parameters = (
            A,
            b,
            Q,
            C,
            d,
            R,
            state_mean,
            state_covariance,
        )
        result = _run_filter(data, missing, parameters)
        return result.filtered_means[1], result.filtered_covariances[1]

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        A, b, Q, C, d, R, mu0, P0 = self._initialize_parameters()
        if not isinstance(n_timesteps, (int, np.integer)) or n_timesteps < 0:
            raise ValueError("n_timesteps must be a non-negative integer")
        for value, needed, ndim, name in (
            (A, max(n_timesteps - 1, 0), 2, "transition_matrices"),
            (b, max(n_timesteps - 1, 0), 1, "transition_offsets"),
            (Q, max(n_timesteps - 1, 0), 2, "transition_covariance"),
            (C, n_timesteps, 2, "observation_matrices"),
            (d, n_timesteps, 1, "observation_offsets"),
            (R, n_timesteps, 2, "observation_covariance"),
        ):
            _check_length(value, needed, ndim, name)
        rng = _check_random_state(self.random_state if random_state is None else random_state)
        n, m = self.n_dim_state, self.n_dim_obs
        states = np.zeros((n_timesteps, n))
        observations = np.zeros((n_timesteps, m))
        if initial_state is None:
            initial_state = rng.multivariate_normal(mu0, P0)
        for t in range(n_timesteps):
            if t == 0:
                states[t] = initial_state
            else:
                states[t] = (
                    _at(A, t - 1) @ states[t - 1]
                    + _at_vector(b, t - 1)
                    + rng.multivariate_normal(np.zeros(n), _at(Q, t - 1))
                )
            observations[t] = (
                _at(C, t) @ states[t]
                + _at_vector(d, t)
                + rng.multivariate_normal(np.zeros(m), _at(R, t))
            )
        return states, np.ma.array(observations)

    def em(self, X, y=None, n_iter=10, em_vars=None):
        del y
        data, missing, initialized = self._observations_and_parameters(X)
        parameters = list(initialized)
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
            raise ValueError("n_iter must be a non-negative integer")
        selected = self._em_vars if em_vars is None else em_vars
        selected = EM_VARIABLES if selected == "all" else set(selected)
        unknown = set(selected) - EM_VARIABLES
        if unknown:
            raise ValueError(f"unknown EM variables: {sorted(unknown)}")
        names = [
            "transition_matrices",
            "transition_offsets",
            "transition_covariance",
            "observation_matrices",
            "observation_offsets",
            "observation_covariance",
            "initial_state_mean",
            "initial_state_covariance",
        ]
        for name, value in zip(names, parameters):
            base = 1 if "offsets" in name or name == "initial_state_mean" else 2
            if name in selected and value.ndim != base:
                warnings.warn(
                    f"{name} has {value.ndim} dimensions now; after fitting, "
                    f"it will have dimension {base}",
                    stacklevel=2,
                )

        timesteps, n_obs = data.shape
        n_state = self.n_dim_state
        for _ in range(n_iter):
            A, b, Q, C, d, R, mu0, P0 = parameters
            inference = _run_filter(
                data, missing, tuple(parameters), store_predictions=True
            )
            smoothing = _run_smooth(A, inference)
            pairwise = _smooth_pair(smoothing)

            new_C = C
            if "observation_matrices" in selected:
                cross = np.empty((n_obs, n_state))
                second = np.empty((n_state, n_state))
                _call(
                    "mpk_em_observation_matrix_stats",
                    addr(data),
                    addr(missing),
                    addr(d),
                    addr(smoothing.means),
                    addr(smoothing.covariances),
                    addr(cross),
                    addr(second),
                    timesteps,
                    n_state,
                    n_obs,
                    _varying(d, 1),
                )
                new_C = f64(cross @ np.linalg.pinv(second))

            new_R = R
            if "observation_covariance" in selected:
                new_R = np.empty((n_obs, n_obs))
                _call(
                    "mpk_em_observation_covariance",
                    addr(data),
                    addr(missing),
                    addr(new_C),
                    addr(d),
                    addr(smoothing.means),
                    addr(smoothing.covariances),
                    addr(new_R),
                    timesteps,
                    n_state,
                    n_obs,
                    _varying(new_C, 2),
                    _varying(d, 1),
                )

            new_A = A
            if "transition_matrices" in selected:
                cross = np.empty((n_state, n_state))
                second = np.empty((n_state, n_state))
                _call(
                    "mpk_em_transition_matrix_stats",
                    addr(b),
                    addr(smoothing.means),
                    addr(smoothing.covariances),
                    addr(pairwise),
                    addr(cross),
                    addr(second),
                    timesteps,
                    n_state,
                    _varying(b, 1),
                )
                new_A = f64(cross @ np.linalg.pinv(second))

            new_Q = Q
            if "transition_covariance" in selected:
                new_Q = np.empty((n_state, n_state))
                _call(
                    "mpk_em_transition_covariance",
                    addr(new_A),
                    addr(b),
                    addr(smoothing.means),
                    addr(smoothing.covariances),
                    addr(pairwise),
                    addr(new_Q),
                    timesteps,
                    n_state,
                    _varying(new_A, 2),
                    _varying(b, 1),
                )

            new_mu0 = smoothing.means[0].copy() if "initial_state_mean" in selected else mu0
            new_P0 = P0
            if "initial_state_covariance" in selected:
                x0 = smoothing.means[0]
                x0x0 = smoothing.covariances[0] + np.outer(x0, x0)
                new_P0 = (
                    x0x0
                    - np.outer(new_mu0, x0)
                    - np.outer(x0, new_mu0)
                    + np.outer(new_mu0, new_mu0)
                )

            new_b = b
            if "transition_offsets" in selected:
                new_b = np.empty(n_state)
                _call(
                    "mpk_em_transition_offset",
                    addr(new_A),
                    addr(smoothing.means),
                    addr(new_b),
                    timesteps,
                    n_state,
                    _varying(new_A, 2),
                )

            new_d = d
            if "observation_offsets" in selected:
                new_d = np.empty(n_obs)
                _call(
                    "mpk_em_observation_offset",
                    addr(data),
                    addr(missing),
                    addr(new_C),
                    addr(smoothing.means),
                    addr(new_d),
                    timesteps,
                    n_state,
                    n_obs,
                    _varying(new_C, 2),
                )
            parameters = [new_A, new_b, new_Q, new_C, new_d, new_R, new_mu0, new_P0]

        (
            self.transition_matrices,
            self.transition_offsets,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_offsets,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = parameters
        return self
