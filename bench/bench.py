"""Benchmark Mojo kernels against pykalman on identical models and data."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from pykalman import KalmanFilter as PythonKalmanFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_pykalman import KalmanFilter  # noqa: E402


def machine() -> str:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in stream
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; NumPy {np.__version__}"


def model(timesteps: int, n_state: int, n_obs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(scale=0.04, size=(n_state, n_state))
    transition = np.eye(n_state) * 0.82 + raw
    radius = max(abs(np.linalg.eigvals(transition)))
    transition *= min(0.94 / radius, 1.0)
    observation = rng.normal(size=(n_obs, n_state))
    parameters = dict(
        transition_matrices=transition,
        observation_matrices=observation,
        transition_covariance=np.eye(n_state) * 0.08,
        observation_covariance=np.eye(n_obs) * 0.15,
        transition_offsets=rng.normal(scale=0.02, size=n_state),
        observation_offsets=rng.normal(scale=0.05, size=n_obs),
        initial_state_mean=np.zeros(n_state),
        initial_state_covariance=np.eye(n_state),
    )
    observations = np.ascontiguousarray(rng.normal(size=(timesteps, n_obs)))
    return observations, parameters


def time_best(function, repeat=2):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cases():
    observations, parameters = model(20_000, 4, 2)
    mojo = KalmanFilter(**parameters)
    python = PythonKalmanFilter(**parameters)
    yield (
        "filter, T=20k, state=4, obs=2",
        lambda: mojo.filter(observations),
        lambda: python.filter(observations),
        2,
    )
    yield (
        "loglikelihood, T=20k, state=4, obs=2",
        lambda: mojo.loglikelihood(observations),
        lambda: python.loglikelihood(observations),
        2,
    )

    observations, parameters = model(8_000, 8, 4, seed=1)
    mojo = KalmanFilter(**parameters)
    python = PythonKalmanFilter(**parameters)
    yield (
        "smooth, T=8k, state=8, obs=4",
        lambda: mojo.smooth(observations),
        lambda: python.smooth(observations),
        2,
    )

    observations, parameters = model(2_000, 6, 3, seed=2)
    yield (
        "EM all, 1 iter, T=2k, state=6, obs=3",
        lambda: KalmanFilter(**parameters).em(observations, n_iter=1, em_vars="all"),
        lambda: PythonKalmanFilter(**parameters).em(
            observations, n_iter=1, em_vars="all"
        ),
        2,
    )

    if os.environ.get("MPK_BENCH_PROFILE") != "1":
        return

    for timesteps, n_state, n_obs in (
        (2_000, 16, 8),
        (500, 32, 16),
        (120, 64, 32),
    ):
        observations, parameters = model(
            timesteps, n_state, n_obs, seed=n_state
        )
        mojo = KalmanFilter(**parameters)
        python = PythonKalmanFilter(**parameters)
        yield (
            f"profile filter, T={timesteps}, state={n_state}, obs={n_obs}",
            lambda mojo=mojo, observations=observations: mojo.filter(observations),
            lambda python=python, observations=observations: python.filter(observations),
            2,
        )
        yield (
            f"profile smooth, T={timesteps}, state={n_state}, obs={n_obs}",
            lambda mojo=mojo, observations=observations: mojo.smooth(observations),
            lambda python=python, observations=observations: python.smooth(observations),
            2,
        )


def main():
    print(f"Machine: {machine()}")
    print()
    print("| Case | mojo-pykalman | pykalman | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo, python, repeat in cases():
        mojo()
        python()
        mojo_time = time_best(mojo, repeat)
        python_time = time_best(python, repeat)
        ratio = python_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{python_time * 1e3:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
