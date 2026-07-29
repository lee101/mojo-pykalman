# mojo-pykalman

`mojo-pykalman` is a standalone Mojo port of the compute-heavy linear-Gaussian
path in [`pykalman`](https://github.com/pykalman/pykalman). It keeps an
upstream-compatible `KalmanFilter` constructor and method signatures while
moving whole-sequence recurrences and EM sufficient-statistic accumulation
into one compiled shared library.

The Python import is `mojo_pykalman`, allowing it to coexist with the real
`pykalman` package for parity testing. For the covered API, changing

```python
from pykalman import KalmanFilter
```

to

```python
from mojo_pykalman import KalmanFilter
```

is the only required source change.

## Coverage

Covered:

- `KalmanFilter` with the same constructor signature as `pykalman` 0.11.2
- batch `filter` and Rauch-Tung-Striebel `smooth`
- `filter_update` for streaming observations
- `loglikelihood`
- `em`, including the default, selective, and `"all"` parameter sets
- constant and time-varying transition/observation matrices, covariances, and
  offsets
- upstream-compatible whole-row handling of masked observations
- `sample`, including `RandomState`-compatible deterministic output
- a pseudoinverse fallback for singular innovations and predicted
  covariances

Not covered:

- `UnscentedKalmanFilter` and `AdditiveUnscentedKalmanFilter`
- the Cholesky square-root filters in `pykalman.sqrt`
- arbitrary nonlinear transition or observation functions
- float32 kernels; the compatibility API uses float64

The fast path requires positive-definite innovation and predicted covariance
matrices. Singular models remain correct, but transparently fall back to a
NumPy pseudoinverse recurrence for that call.

## Install

The repository pins the verified Mojo nightly and includes upstream
`pykalman` for parity tests:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-pykalman.so`. Set `MOJO_PYKALMAN_LIB` to use
an already-built library at another location.

## Usage

```python
import numpy as np
from mojo_pykalman import KalmanFilter

observations = np.array([[0.9], [2.1], [2.8], [4.2]])
model = KalmanFilter(
    transition_matrices=np.array([[1.0]]),
    observation_matrices=np.array([[1.0]]),
    transition_covariance=np.array([[0.1]]),
    observation_covariance=np.array([[0.2]]),
)

filtered_means, filtered_covariances = model.filter(observations)
smoothed_means, smoothed_covariances = model.smooth(observations)
model.em(observations, n_iter=5)
print(smoothed_means[:, 0])
print(model.loglikelihood(observations))
```

Run it from the Pixi environment so `python/` is on `PYTHONPATH`.

## Benchmarks

Measured with `pixi run bench`; values are the best of two locked repetitions.
A speedup greater than 1 means this port was faster. These are real wall-clock
measurements, including Python wrapper and output allocation time.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic;
NumPy 2.5.1.

| Case | mojo-pykalman | pykalman | Speedup |
|---|---:|---:|---:|
| filter, T=20k, state=4, obs=2 | 13.35 ms | 5776.72 ms | 432.71x |
| loglikelihood, T=20k, state=4, obs=2 | 13.04 ms | 13589.72 ms | 1042.00x |
| smooth, T=8k, state=8, obs=4 | 52.58 ms | 3332.95 ms | 63.39x |
| EM all, 1 iter, T=2k, state=6, obs=3 | 12.98 ms | 1333.28 ms | 102.73x |

`pykalman` performs a Python loop and calls SciPy's pseudoinverse once per
timestep. The Mojo path performs the same dense recurrences in a single FFI
call, which is why long sequences benefit even though both implementations
operate on small matrices.

There is no GPU path. Although individual dense products become
compute-intensive at large state sizes, every timestep depends on the previous
one and the per-step matrices in the measured range are too small to sustain
GPU occupancy. Transfers and repeated launches would dominate. CPU threading
is likewise not used for these fine-grained, timestep-dependent operations.

## How it works

`src/kalman.mojo` is one compilation unit containing filtering, smoothing,
pairwise smoothed covariance, and all EM accumulation kernels. It uses
row-major, C-contiguous float64 buffers. Python allocates and owns inputs,
outputs, and scratch memory; raw buffer addresses cross the C ABI as 64-bit
integers and Mojo reconstructs mutable pointers inside each exported wrapper.
There is no ownership transfer or allocation inside the shared library.

The filter and smoother use SIMD dense products with scalar remainder loops and
direct Cholesky solves for symmetric positive-definite systems. Smoother scratch
uses three state-square buffers, and the filter gain reuses scratch instead of
allocating a sequence-sized output that no public method consumes. The EM
E-step stays in Mojo, as do the M-step's time-indexed accumulations; NumPy
computes the two parameter pseudoinverses after those statistics have been
reduced to small state-sized matrices. The update order matches upstream
exactly.
