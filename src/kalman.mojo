"""Dense linear-Gaussian filtering, RTS smoothing, and EM statistics."""

from std.math import log, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()


def matrix_at(x: Ptr, t: Int, rows: Int, cols: Int, varying: Int) -> Ptr:
    return x + t * rows * cols if varying != 0 else x


def vector_at(x: Ptr, t: Int, width: Int, varying: Int) -> Ptr:
    return x + t * width if varying != 0 else x


def copy(src: Ptr, dst: Ptr, size: Int):
    var i = 0
    while i + W <= size:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < size:
        dst[i] = src[i]
        i += 1


def fill(dst: Ptr, value: Float64, size: Int):
    var vector = SIMD[DType.float64, W](value)
    var i = 0
    while i + W <= size:
        dst.store(i, vector)
        i += W
    while i < size:
        dst[i] = value
        i += 1


def dot(lhs: Ptr, rhs: Ptr, size: Int) -> Float64:
    var vector = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= size:
        vector += lhs.load[width=W](i) * rhs.load[width=W](i)
        i += W
    var result = vector.reduce_add()
    while i < size:
        result += lhs[i] * rhs[i]
        i += 1
    return result


def row_times_matrix(
    lhs: Ptr, rhs: Ptr, dst: Ptr, inner: Int, columns: Int
):
    var j = 0
    while j + W <= columns:
        var vector = SIMD[DType.float64, W](0.0)
        for k in range(inner):
            vector += lhs[k] * rhs.load[width=W](k * columns + j)
        dst.store(j, vector)
        j += W
    while j < columns:
        var result = 0.0
        for k in range(inner):
            result += lhs[k] * rhs[k * columns + j]
        dst[j] = result
        j += 1


def row_times_matrix_add(
    lhs: Ptr, rhs: Ptr, base: Ptr, dst: Ptr, inner: Int, columns: Int
):
    var j = 0
    while j + W <= columns:
        var vector = base.load[width=W](j)
        for k in range(inner):
            vector += lhs[k] * rhs.load[width=W](k * columns + j)
        dst.store(j, vector)
        j += W
    while j < columns:
        var result = base[j]
        for k in range(inner):
            result += lhs[k] * rhs[k * columns + j]
        dst[j] = result
        j += 1


def matrix_multiply(
    lhs: Ptr,
    rhs: Ptr,
    dst: Ptr,
    rows: Int,
    inner: Int,
    columns: Int,
):
    var i = 0
    while i + 4 <= rows:
        var j = 0
        while j + W <= columns:
            var v0 = SIMD[DType.float64, W](0.0)
            var v1 = SIMD[DType.float64, W](0.0)
            var v2 = SIMD[DType.float64, W](0.0)
            var v3 = SIMD[DType.float64, W](0.0)
            for k in range(inner):
                var rhs_vector = rhs.load[width=W](k * columns + j)
                v0 += lhs[i * inner + k] * rhs_vector
                v1 += lhs[(i + 1) * inner + k] * rhs_vector
                v2 += lhs[(i + 2) * inner + k] * rhs_vector
                v3 += lhs[(i + 3) * inner + k] * rhs_vector
            dst.store(i * columns + j, v0)
            dst.store((i + 1) * columns + j, v1)
            dst.store((i + 2) * columns + j, v2)
            dst.store((i + 3) * columns + j, v3)
            j += W
        while j < columns:
            for row in range(i, i + 4):
                var result = 0.0
                for k in range(inner):
                    result += lhs[row * inner + k] * rhs[k * columns + j]
                dst[row * columns + j] = result
            j += 1
        i += 4
    while i < rows:
        row_times_matrix(
            lhs + i * inner,
            rhs,
            dst + i * columns,
            inner,
            columns,
        )
        i += 1


def matrix_multiply_add(
    lhs: Ptr,
    rhs: Ptr,
    base: Ptr,
    dst: Ptr,
    rows: Int,
    inner: Int,
    columns: Int,
):
    var i = 0
    while i + 4 <= rows:
        var j = 0
        while j + W <= columns:
            var v0 = base.load[width=W](i * columns + j)
            var v1 = base.load[width=W]((i + 1) * columns + j)
            var v2 = base.load[width=W]((i + 2) * columns + j)
            var v3 = base.load[width=W]((i + 3) * columns + j)
            for k in range(inner):
                var rhs_vector = rhs.load[width=W](k * columns + j)
                v0 += lhs[i * inner + k] * rhs_vector
                v1 += lhs[(i + 1) * inner + k] * rhs_vector
                v2 += lhs[(i + 2) * inner + k] * rhs_vector
                v3 += lhs[(i + 3) * inner + k] * rhs_vector
            dst.store(i * columns + j, v0)
            dst.store((i + 1) * columns + j, v1)
            dst.store((i + 2) * columns + j, v2)
            dst.store((i + 3) * columns + j, v3)
            j += W
        while j < columns:
            for row in range(i, i + 4):
                var result = base[row * columns + j]
                for k in range(inner):
                    result += lhs[row * inner + k] * rhs[k * columns + j]
                dst[row * columns + j] = result
            j += 1
        i += 4
    while i < rows:
        row_times_matrix_add(
            lhs + i * inner,
            rhs,
            base + i * columns,
            dst + i * columns,
            inner,
            columns,
        )
        i += 1


def matrix_multiply_transpose_add(
    lhs: Ptr,
    rhs: Ptr,
    base: Ptr,
    dst: Ptr,
    rows: Int,
    columns: Int,
    inner: Int,
    scale: Float64,
):
    for i in range(rows):
        var j = 0
        while j + 4 <= columns:
            var v0 = SIMD[DType.float64, W](0.0)
            var v1 = SIMD[DType.float64, W](0.0)
            var v2 = SIMD[DType.float64, W](0.0)
            var v3 = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= inner:
                var lhs_vector = lhs.load[width=W](i * inner + k)
                v0 += lhs_vector * rhs.load[width=W](j * inner + k)
                v1 += lhs_vector * rhs.load[width=W]((j + 1) * inner + k)
                v2 += lhs_vector * rhs.load[width=W]((j + 2) * inner + k)
                v3 += lhs_vector * rhs.load[width=W]((j + 3) * inner + k)
                k += W
            var r0 = v0.reduce_add()
            var r1 = v1.reduce_add()
            var r2 = v2.reduce_add()
            var r3 = v3.reduce_add()
            while k < inner:
                var lhs_value = lhs[i * inner + k]
                r0 += lhs_value * rhs[j * inner + k]
                r1 += lhs_value * rhs[(j + 1) * inner + k]
                r2 += lhs_value * rhs[(j + 2) * inner + k]
                r3 += lhs_value * rhs[(j + 3) * inner + k]
                k += 1
            var offset = i * columns + j
            dst[offset] = base[offset] + scale * r0
            dst[offset + 1] = base[offset + 1] + scale * r1
            dst[offset + 2] = base[offset + 2] + scale * r2
            dst[offset + 3] = base[offset + 3] + scale * r3
            j += 4
        while j < columns:
            dst[i * columns + j] = (
                base[i * columns + j]
                + scale
                * dot(lhs + i * inner, rhs + j * inner, inner)
            )
            j += 1


def matrix_multiply_transpose(
    lhs: Ptr,
    rhs: Ptr,
    dst: Ptr,
    rows: Int,
    columns: Int,
    inner: Int,
):
    for i in range(rows):
        var j = 0
        while j + 4 <= columns:
            var v0 = SIMD[DType.float64, W](0.0)
            var v1 = SIMD[DType.float64, W](0.0)
            var v2 = SIMD[DType.float64, W](0.0)
            var v3 = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= inner:
                var lhs_vector = lhs.load[width=W](i * inner + k)
                v0 += lhs_vector * rhs.load[width=W](j * inner + k)
                v1 += lhs_vector * rhs.load[width=W]((j + 1) * inner + k)
                v2 += lhs_vector * rhs.load[width=W]((j + 2) * inner + k)
                v3 += lhs_vector * rhs.load[width=W]((j + 3) * inner + k)
                k += W
            var r0 = v0.reduce_add()
            var r1 = v1.reduce_add()
            var r2 = v2.reduce_add()
            var r3 = v3.reduce_add()
            while k < inner:
                var lhs_value = lhs[i * inner + k]
                r0 += lhs_value * rhs[j * inner + k]
                r1 += lhs_value * rhs[(j + 1) * inner + k]
                r2 += lhs_value * rhs[(j + 2) * inner + k]
                r3 += lhs_value * rhs[(j + 3) * inner + k]
                k += 1
            var offset = i * columns + j
            dst[offset] = r0
            dst[offset + 1] = r1
            dst[offset + 2] = r2
            dst[offset + 3] = r3
            j += 4
        while j < columns:
            dst[i * columns + j] = dot(
                lhs + i * inner, rhs + j * inner, inner
            )
            j += 1


def factor_spd(src: Ptr, chol: Ptr, d: Int) -> Bool:
    for i in range(d):
        for j in range(i + 1):
            var acc = 0.5 * (src[i * d + j] + src[j * d + i])
            acc -= dot(chol + i * d, chol + j * d, j)
            if i == j:
                if acc <= 1e-15:
                    return False
                chol[i * d + i] = sqrt(acc)
            else:
                chol[i * d + j] = acc / chol[j * d + j]
    for i in range(d):
        for j in range(i + 1, d):
            chol[i * d + j] = chol[j * d + i]
    return True


def solve_spd_rows(rhs: Ptr, dst: Ptr, chol: Ptr, rows: Int, d: Int):
    var row = 0
    while row + 4 <= rows:
        var s0 = dst + row * d
        var s1 = dst + (row + 1) * d
        var s2 = dst + (row + 2) * d
        var s3 = dst + (row + 3) * d
        var b0 = rhs + row * d
        var b1 = rhs + (row + 1) * d
        var b2 = rhs + (row + 2) * d
        var b3 = rhs + (row + 3) * d
        for i in range(d):
            var v0 = SIMD[DType.float64, W](0.0)
            var v1 = SIMD[DType.float64, W](0.0)
            var v2 = SIMD[DType.float64, W](0.0)
            var v3 = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= i:
                var factor = chol.load[width=W](i * d + k)
                v0 += factor * s0.load[width=W](k)
                v1 += factor * s1.load[width=W](k)
                v2 += factor * s2.load[width=W](k)
                v3 += factor * s3.load[width=W](k)
                k += W
            var a0 = v0.reduce_add()
            var a1 = v1.reduce_add()
            var a2 = v2.reduce_add()
            var a3 = v3.reduce_add()
            while k < i:
                var factor = chol[i * d + k]
                a0 += factor * s0[k]
                a1 += factor * s1[k]
                a2 += factor * s2[k]
                a3 += factor * s3[k]
                k += 1
            var diagonal = chol[i * d + i]
            s0[i] = (b0[i] - a0) / diagonal
            s1[i] = (b1[i] - a1) / diagonal
            s2[i] = (b2[i] - a2) / diagonal
            s3[i] = (b3[i] - a3) / diagonal
        for ri in range(d):
            var i = d - 1 - ri
            var v0 = SIMD[DType.float64, W](0.0)
            var v1 = SIMD[DType.float64, W](0.0)
            var v2 = SIMD[DType.float64, W](0.0)
            var v3 = SIMD[DType.float64, W](0.0)
            var k = i + 1
            while k + W <= d:
                var factor = chol.load[width=W](i * d + k)
                v0 += factor * s0.load[width=W](k)
                v1 += factor * s1.load[width=W](k)
                v2 += factor * s2.load[width=W](k)
                v3 += factor * s3.load[width=W](k)
                k += W
            var a0 = v0.reduce_add()
            var a1 = v1.reduce_add()
            var a2 = v2.reduce_add()
            var a3 = v3.reduce_add()
            while k < d:
                var factor = chol[i * d + k]
                a0 += factor * s0[k]
                a1 += factor * s1[k]
                a2 += factor * s2[k]
                a3 += factor * s3[k]
                k += 1
            var diagonal = chol[i * d + i]
            s0[i] = (s0[i] - a0) / diagonal
            s1[i] = (s1[i] - a1) / diagonal
            s2[i] = (s2[i] - a2) / diagonal
            s3[i] = (s3[i] - a3) / diagonal
        row += 4
    while row < rows:
        var solution = dst + row * d
        var values = rhs + row * d
        for i in range(d):
            var acc = values[i] - dot(chol + i * d, solution, i)
            solution[i] = acc / chol[i * d + i]
        for ri in range(d):
            var i = d - 1 - ri
            var acc = solution[i] - dot(
                chol + i * d + i + 1, solution + i + 1, d - i - 1
            )
            solution[i] = acc / chol[i * d + i]
        row += 1


def filter_impl(
    observations: Ptr,
    missing: IPtr,
    transitions: Ptr,
    observation_matrices: Ptr,
    transition_covariances: Ptr,
    observation_covariances: Ptr,
    transition_offsets: Ptr,
    observation_offsets: Ptr,
    initial_mean: Ptr,
    initial_covariance: Ptr,
    predicted_means: Ptr,
    predicted_covariances: Ptr,
    filtered_means: Ptr,
    filtered_covariances: Ptr,
    loglikelihoods: Ptr,
    scratch: Ptr,
    timesteps: Int,
    n_state: Int,
    n_obs: Int,
    transitions_vary: Int,
    observations_vary: Int,
    transition_covariances_vary: Int,
    observation_covariances_vary: Int,
    transition_offsets_vary: Int,
    observation_offsets_vary: Int,
    store_predictions: Int,
) -> Bool:
    var width = n_state if n_state > n_obs else n_obs
    var block = width * width
    var innovation = scratch
    var innovation_inverse = scratch + block
    var chol = scratch + 2 * block
    var temp = scratch + 3 * block

    for t in range(timesteps):
        var prediction_t = t if store_predictions != 0 else 0
        var predicted_mean = predicted_means + prediction_t * n_state
        var predicted_covariance = (
            predicted_covariances + prediction_t * n_state * n_state
        )
        if t == 0:
            copy(initial_mean, predicted_mean, n_state)
            copy(initial_covariance, predicted_covariance, n_state * n_state)
        else:
            var transition = matrix_at(
                transitions, t - 1, n_state, n_state, transitions_vary
            )
            var transition_covariance = matrix_at(
                transition_covariances,
                t - 1,
                n_state,
                n_state,
                transition_covariances_vary,
            )
            var transition_offset = vector_at(
                transition_offsets, t - 1, n_state, transition_offsets_vary
            )
            var previous_mean = filtered_means + (t - 1) * n_state
            var previous_covariance = (
                filtered_covariances + (t - 1) * n_state * n_state
            )
            for i in range(n_state):
                predicted_mean[i] = transition_offset[i] + dot(
                    transition + i * n_state, previous_mean, n_state
                )
            matrix_multiply(
                transition,
                previous_covariance,
                temp,
                n_state,
                n_state,
                n_state,
            )
            matrix_multiply_transpose_add(
                temp,
                transition,
                transition_covariance,
                predicted_covariance,
                n_state,
                n_state,
                n_state,
                1.0,
            )

        var filtered_mean = filtered_means + t * n_state
        var filtered_covariance = filtered_covariances + t * n_state * n_state
        if missing[t] != 0:
            copy(predicted_mean, filtered_mean, n_state)
            copy(predicted_covariance, filtered_covariance, n_state * n_state)
            loglikelihoods[t] = 0.0
            continue

        var observation_matrix = matrix_at(
            observation_matrices, t, n_obs, n_state, observations_vary
        )
        var observation_covariance = matrix_at(
            observation_covariances,
            t,
            n_obs,
            n_obs,
            observation_covariances_vary,
        )
        var observation_offset = vector_at(
            observation_offsets, t, n_obs, observation_offsets_vary
        )

        # temp holds P C.T, which also gives C P by transposition.
        matrix_multiply_transpose(
            predicted_covariance,
            observation_matrix,
            temp,
            n_state,
            n_obs,
            n_state,
        )
        matrix_multiply_add(
            observation_matrix,
            temp,
            observation_covariance,
            innovation,
            n_obs,
            n_state,
            n_obs,
        )
        if not factor_spd(innovation, chol, n_obs):
            return False

        var gain = innovation
        solve_spd_rows(temp, gain, chol, n_state, n_obs)

        var logdet = 0.0
        for i in range(n_obs):
            logdet += 2.0 * log(chol[i * n_obs + i])
            var predicted_observation = observation_offset[i] + dot(
                observation_matrix + i * n_state, predicted_mean, n_state
            )
            innovation_inverse[i] = (
                observations[t * n_obs + i] - predicted_observation
            )

        for i in range(n_state):
            filtered_mean[i] = predicted_mean[i] + dot(
                gain + i * n_obs, innovation_inverse, n_obs
            )
        matrix_multiply_transpose_add(
            gain,
            temp,
            predicted_covariance,
            filtered_covariance,
            n_state,
            n_state,
            n_obs,
            -1.0,
        )
        copy(innovation_inverse, temp, n_obs)
        solve_spd_rows(innovation_inverse, innovation_inverse, chol, 1, n_obs)
        var mahalanobis = dot(temp, innovation_inverse, n_obs)
        loglikelihoods[t] = -0.5 * (
            Float64(n_obs) * 1.8378770664093453 + logdet + mahalanobis
        )
    return True


def smooth_impl(
    transitions: Ptr,
    filtered_means: Ptr,
    filtered_covariances: Ptr,
    predicted_means: Ptr,
    predicted_covariances: Ptr,
    smoothed_means: Ptr,
    smoothed_covariances: Ptr,
    smoothing_gains: Ptr,
    scratch: Ptr,
    timesteps: Int,
    n_state: Int,
    transitions_vary: Int,
) -> Bool:
    var block = n_state * n_state
    var chol = scratch
    var temp = scratch + block
    var product = scratch + 2 * block
    copy(
        filtered_means + (timesteps - 1) * n_state,
        smoothed_means + (timesteps - 1) * n_state,
        n_state,
    )
    copy(
        filtered_covariances + (timesteps - 1) * block,
        smoothed_covariances + (timesteps - 1) * block,
        block,
    )
    for reverse_t in range(timesteps - 1):
        var t = timesteps - 2 - reverse_t
        var transition = matrix_at(
            transitions, t, n_state, n_state, transitions_vary
        )
        var filtered_covariance = filtered_covariances + t * block
        var predicted_covariance = predicted_covariances + (t + 1) * block
        if not factor_spd(predicted_covariance, chol, n_state):
            return False
        # temp = P_filtered A.T
        matrix_multiply_transpose(
            filtered_covariance,
            transition,
            temp,
            n_state,
            n_state,
            n_state,
        )
        var gain = smoothing_gains + t * block
        solve_spd_rows(temp, gain, chol, n_state, n_state)

        var next_smoothed_mean = smoothed_means + (t + 1) * n_state
        var next_predicted_mean = predicted_means + (t + 1) * n_state
        var smoothed_mean = smoothed_means + t * n_state
        var filtered_mean = filtered_means + t * n_state
        for i in range(n_state):
            chol[i] = next_smoothed_mean[i] - next_predicted_mean[i]
        for i in range(n_state):
            smoothed_mean[i] = filtered_mean[i] + dot(
                gain + i * n_state, chol, n_state
            )

        var next_smoothed_covariance = smoothed_covariances + (t + 1) * block
        for i in range(block):
            temp[i] = next_smoothed_covariance[i] - predicted_covariance[i]
        matrix_multiply(
            gain,
            temp,
            product,
            n_state,
            n_state,
            n_state,
        )
        var smoothed_covariance = smoothed_covariances + t * block
        matrix_multiply_transpose_add(
            product,
            gain,
            filtered_covariance,
            smoothed_covariance,
            n_state,
            n_state,
            n_state,
            1.0,
        )
    return True


@export("mpk_filter")
def mpk_filter(
    observations: Int,
    missing: Int,
    transitions: Int,
    observation_matrices: Int,
    transition_covariances: Int,
    observation_covariances: Int,
    transition_offsets: Int,
    observation_offsets: Int,
    initial_mean: Int,
    initial_covariance: Int,
    predicted_means: Int,
    predicted_covariances: Int,
    filtered_means: Int,
    filtered_covariances: Int,
    loglikelihoods: Int,
    scratch: Int,
    timesteps: Int,
    n_state: Int,
    n_obs: Int,
    transitions_vary: Int,
    observations_vary: Int,
    transition_covariances_vary: Int,
    observation_covariances_vary: Int,
    transition_offsets_vary: Int,
    observation_offsets_vary: Int,
    store_predictions: Int,
) abi("C") -> Int:
    if (
        observations == 0
        or missing == 0
        or transitions == 0
        or observation_matrices == 0
        or transition_covariances == 0
        or observation_covariances == 0
        or transition_offsets == 0
        or observation_offsets == 0
        or initial_mean == 0
        or initial_covariance == 0
        or predicted_means == 0
        or predicted_covariances == 0
        or filtered_means == 0
        or filtered_covariances == 0
        or loglikelihoods == 0
        or scratch == 0
        or timesteps <= 0
        or n_state <= 0
        or n_obs <= 0
        or transitions_vary < 0
        or transitions_vary > 1
        or observations_vary < 0
        or observations_vary > 1
        or transition_covariances_vary < 0
        or transition_covariances_vary > 1
        or observation_covariances_vary < 0
        or observation_covariances_vary > 1
        or transition_offsets_vary < 0
        or transition_offsets_vary > 1
        or observation_offsets_vary < 0
        or observation_offsets_vary > 1
        or store_predictions < 0
        or store_predictions > 1
    ):
        return 0
    return 1 if filter_impl(
        Ptr(unsafe_from_address=observations),
        IPtr(unsafe_from_address=missing),
        Ptr(unsafe_from_address=transitions),
        Ptr(unsafe_from_address=observation_matrices),
        Ptr(unsafe_from_address=transition_covariances),
        Ptr(unsafe_from_address=observation_covariances),
        Ptr(unsafe_from_address=transition_offsets),
        Ptr(unsafe_from_address=observation_offsets),
        Ptr(unsafe_from_address=initial_mean),
        Ptr(unsafe_from_address=initial_covariance),
        Ptr(unsafe_from_address=predicted_means),
        Ptr(unsafe_from_address=predicted_covariances),
        Ptr(unsafe_from_address=filtered_means),
        Ptr(unsafe_from_address=filtered_covariances),
        Ptr(unsafe_from_address=loglikelihoods),
        Ptr(unsafe_from_address=scratch),
        timesteps,
        n_state,
        n_obs,
        transitions_vary,
        observations_vary,
        transition_covariances_vary,
        observation_covariances_vary,
        transition_offsets_vary,
        observation_offsets_vary,
        store_predictions,
    ) else 0


@export("mpk_smooth")
def mpk_smooth(
    transitions: Int,
    filtered_means: Int,
    filtered_covariances: Int,
    predicted_means: Int,
    predicted_covariances: Int,
    smoothed_means: Int,
    smoothed_covariances: Int,
    smoothing_gains: Int,
    scratch: Int,
    timesteps: Int,
    n_state: Int,
    transitions_vary: Int,
) abi("C") -> Int:
    if (
        transitions == 0
        or filtered_means == 0
        or filtered_covariances == 0
        or predicted_means == 0
        or predicted_covariances == 0
        or smoothed_means == 0
        or smoothed_covariances == 0
        or smoothing_gains == 0
        or scratch == 0
        or timesteps <= 0
        or n_state <= 0
        or transitions_vary < 0
        or transitions_vary > 1
    ):
        return 0
    return 1 if smooth_impl(
        Ptr(unsafe_from_address=transitions),
        Ptr(unsafe_from_address=filtered_means),
        Ptr(unsafe_from_address=filtered_covariances),
        Ptr(unsafe_from_address=predicted_means),
        Ptr(unsafe_from_address=predicted_covariances),
        Ptr(unsafe_from_address=smoothed_means),
        Ptr(unsafe_from_address=smoothed_covariances),
        Ptr(unsafe_from_address=smoothing_gains),
        Ptr(unsafe_from_address=scratch),
        timesteps,
        n_state,
        transitions_vary,
    ) else 0


@export("mpk_smooth_pair")
def mpk_smooth_pair(
    smoothed_covariances: Int,
    smoothing_gains: Int,
    pairwise_covariances: Int,
    timesteps: Int,
    n_state: Int,
) abi("C") -> Int:
    if (
        smoothed_covariances == 0
        or smoothing_gains == 0
        or pairwise_covariances == 0
        or timesteps <= 0
        or n_state <= 0
    ):
        return 0
    var covariances = Ptr(unsafe_from_address=smoothed_covariances)
    var gains = Ptr(unsafe_from_address=smoothing_gains)
    var pairwise = Ptr(unsafe_from_address=pairwise_covariances)
    var block = n_state * n_state
    fill(pairwise, 0.0, block)
    for t in range(1, timesteps):
        var covariance = covariances + t * block
        var gain = gains + (t - 1) * block
        var pair = pairwise + t * block
        for i in range(n_state):
            for j in range(n_state):
                var acc = 0.0
                for k in range(n_state):
                    acc += covariance[i * n_state + k] * gain[j * n_state + k]
                pair[i * n_state + j] = acc
    return 1


@export("mpk_em_observation_matrix_stats")
def mpk_em_observation_matrix_stats(
    observations: Int,
    missing: Int,
    observation_offsets: Int,
    smoothed_means: Int,
    smoothed_covariances: Int,
    cross: Int,
    second: Int,
    timesteps: Int,
    n_state: Int,
    n_obs: Int,
    offsets_vary: Int,
) abi("C") -> Int:
    if (
        observations == 0
        or missing == 0
        or observation_offsets == 0
        or smoothed_means == 0
        or smoothed_covariances == 0
        or cross == 0
        or second == 0
        or timesteps <= 0
        or n_state <= 0
        or n_obs <= 0
        or offsets_vary < 0
        or offsets_vary > 1
    ):
        return 0
    var obs = Ptr(unsafe_from_address=observations)
    var mask = IPtr(unsafe_from_address=missing)
    var offsets = Ptr(unsafe_from_address=observation_offsets)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var covariances = Ptr(unsafe_from_address=smoothed_covariances)
    var lhs = Ptr(unsafe_from_address=cross)
    var rhs = Ptr(unsafe_from_address=second)
    fill(lhs, 0.0, n_obs * n_state)
    fill(rhs, 0.0, n_state * n_state)
    for t in range(timesteps):
        if mask[t] != 0:
            continue
        var offset = vector_at(offsets, t, n_obs, offsets_vary)
        var mean = means + t * n_state
        var covariance = covariances + t * n_state * n_state
        for i in range(n_obs):
            var centered = obs[t * n_obs + i] - offset[i]
            for j in range(n_state):
                lhs[i * n_state + j] += centered * mean[j]
        for i in range(n_state):
            for j in range(n_state):
                rhs[i * n_state + j] += (
                    covariance[i * n_state + j] + mean[i] * mean[j]
                )
    return 1


@export("mpk_em_transition_matrix_stats")
def mpk_em_transition_matrix_stats(
    transition_offsets: Int,
    smoothed_means: Int,
    smoothed_covariances: Int,
    pairwise_covariances: Int,
    cross: Int,
    second: Int,
    timesteps: Int,
    n_state: Int,
    offsets_vary: Int,
) abi("C") -> Int:
    if (
        transition_offsets == 0
        or smoothed_means == 0
        or smoothed_covariances == 0
        or pairwise_covariances == 0
        or cross == 0
        or second == 0
        or timesteps <= 0
        or n_state <= 0
        or offsets_vary < 0
        or offsets_vary > 1
    ):
        return 0
    var offsets = Ptr(unsafe_from_address=transition_offsets)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var covariances = Ptr(unsafe_from_address=smoothed_covariances)
    var pairwise = Ptr(unsafe_from_address=pairwise_covariances)
    var lhs = Ptr(unsafe_from_address=cross)
    var rhs = Ptr(unsafe_from_address=second)
    fill(lhs, 0.0, n_state * n_state)
    fill(rhs, 0.0, n_state * n_state)
    var block = n_state * n_state
    for t in range(1, timesteps):
        var offset = vector_at(offsets, t - 1, n_state, offsets_vary)
        var current_mean = means + t * n_state
        var previous_mean = means + (t - 1) * n_state
        var previous_covariance = covariances + (t - 1) * block
        var pair = pairwise + t * block
        for i in range(n_state):
            for j in range(n_state):
                lhs[i * n_state + j] += (
                    pair[i * n_state + j]
                    + current_mean[i] * previous_mean[j]
                    - offset[i] * previous_mean[j]
                )
                rhs[i * n_state + j] += (
                    previous_covariance[i * n_state + j]
                    + previous_mean[i] * previous_mean[j]
                )
    return 1


@export("mpk_em_observation_covariance")
def mpk_em_observation_covariance(
    observations: Int,
    missing: Int,
    observation_matrices: Int,
    observation_offsets: Int,
    smoothed_means: Int,
    smoothed_covariances: Int,
    result: Int,
    timesteps: Int,
    n_state: Int,
    n_obs: Int,
    matrices_vary: Int,
    offsets_vary: Int,
) abi("C") -> Int:
    if (
        observations == 0
        or missing == 0
        or observation_matrices == 0
        or observation_offsets == 0
        or smoothed_means == 0
        or smoothed_covariances == 0
        or result == 0
        or timesteps <= 0
        or n_state <= 0
        or n_obs <= 0
        or matrices_vary < 0
        or matrices_vary > 1
        or offsets_vary < 0
        or offsets_vary > 1
    ):
        return 0
    var obs = Ptr(unsafe_from_address=observations)
    var mask = IPtr(unsafe_from_address=missing)
    var matrices = Ptr(unsafe_from_address=observation_matrices)
    var offsets = Ptr(unsafe_from_address=observation_offsets)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var covariances = Ptr(unsafe_from_address=smoothed_covariances)
    var dst = Ptr(unsafe_from_address=result)
    fill(dst, 0.0, n_obs * n_obs)
    var count = 0
    for t in range(timesteps):
        if mask[t] != 0:
            continue
        count += 1
        var matrix = matrix_at(matrices, t, n_obs, n_state, matrices_vary)
        var offset = vector_at(offsets, t, n_obs, offsets_vary)
        var mean = means + t * n_state
        var covariance = covariances + t * n_state * n_state
        for i in range(n_obs):
            var err_i = obs[t * n_obs + i] - offset[i]
            for k in range(n_state):
                err_i -= matrix[i * n_state + k] * mean[k]
            for j in range(n_obs):
                var err_j = obs[t * n_obs + j] - offset[j]
                for k in range(n_state):
                    err_j -= matrix[j * n_state + k] * mean[k]
                var acc = err_i * err_j
                for a in range(n_state):
                    for b in range(n_state):
                        acc += (
                            matrix[i * n_state + a]
                            * covariance[a * n_state + b]
                            * matrix[j * n_state + b]
                        )
                dst[i * n_obs + j] += acc
    if count > 0:
        for i in range(n_obs * n_obs):
            dst[i] /= Float64(count)
    return 1


@export("mpk_em_transition_covariance")
def mpk_em_transition_covariance(
    transitions: Int,
    transition_offsets: Int,
    smoothed_means: Int,
    smoothed_covariances: Int,
    pairwise_covariances: Int,
    result: Int,
    timesteps: Int,
    n_state: Int,
    matrices_vary: Int,
    offsets_vary: Int,
) abi("C") -> Int:
    if (
        transitions == 0
        or transition_offsets == 0
        or smoothed_means == 0
        or smoothed_covariances == 0
        or pairwise_covariances == 0
        or result == 0
        or timesteps <= 0
        or n_state <= 0
        or matrices_vary < 0
        or matrices_vary > 1
        or offsets_vary < 0
        or offsets_vary > 1
    ):
        return 0
    var matrices = Ptr(unsafe_from_address=transitions)
    var offsets = Ptr(unsafe_from_address=transition_offsets)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var covariances = Ptr(unsafe_from_address=smoothed_covariances)
    var pairwise = Ptr(unsafe_from_address=pairwise_covariances)
    var dst = Ptr(unsafe_from_address=result)
    var block = n_state * n_state
    fill(dst, 0.0, block)
    for t in range(timesteps - 1):
        var matrix = matrix_at(matrices, t, n_state, n_state, matrices_vary)
        var offset = vector_at(offsets, t, n_state, offsets_vary)
        var mean = means + t * n_state
        var next_mean = means + (t + 1) * n_state
        var covariance = covariances + t * block
        var next_covariance = covariances + (t + 1) * block
        var pair = pairwise + (t + 1) * block
        for i in range(n_state):
            var err_i = next_mean[i] - offset[i]
            for k in range(n_state):
                err_i -= matrix[i * n_state + k] * mean[k]
            for j in range(n_state):
                var err_j = next_mean[j] - offset[j]
                for k in range(n_state):
                    err_j -= matrix[j * n_state + k] * mean[k]
                var acc = err_i * err_j + next_covariance[i * n_state + j]
                for a in range(n_state):
                    for b in range(n_state):
                        acc += (
                            matrix[i * n_state + a]
                            * covariance[a * n_state + b]
                            * matrix[j * n_state + b]
                        )
                for k in range(n_state):
                    acc -= (
                        pair[i * n_state + k] * matrix[j * n_state + k]
                        + matrix[i * n_state + k] * pair[j * n_state + k]
                    )
                dst[i * n_state + j] += acc
    if timesteps > 1:
        for i in range(block):
            dst[i] /= Float64(timesteps - 1)
    return 1


@export("mpk_em_transition_offset")
def mpk_em_transition_offset(
    transitions: Int,
    smoothed_means: Int,
    result: Int,
    timesteps: Int,
    n_state: Int,
    matrices_vary: Int,
) abi("C") -> Int:
    if (
        transitions == 0
        or smoothed_means == 0
        or result == 0
        or timesteps <= 0
        or n_state <= 0
        or matrices_vary < 0
        or matrices_vary > 1
    ):
        return 0
    var matrices = Ptr(unsafe_from_address=transitions)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var dst = Ptr(unsafe_from_address=result)
    fill(dst, 0.0, n_state)
    for t in range(1, timesteps):
        var matrix = matrix_at(matrices, t - 1, n_state, n_state, matrices_vary)
        for i in range(n_state):
            var value = means[t * n_state + i]
            for j in range(n_state):
                value -= matrix[i * n_state + j] * means[(t - 1) * n_state + j]
            dst[i] += value
    if timesteps > 1:
        for i in range(n_state):
            dst[i] /= Float64(timesteps - 1)
    return 1


@export("mpk_em_observation_offset")
def mpk_em_observation_offset(
    observations: Int,
    missing: Int,
    observation_matrices: Int,
    smoothed_means: Int,
    result: Int,
    timesteps: Int,
    n_state: Int,
    n_obs: Int,
    matrices_vary: Int,
) abi("C") -> Int:
    if (
        observations == 0
        or missing == 0
        or observation_matrices == 0
        or smoothed_means == 0
        or result == 0
        or timesteps <= 0
        or n_state <= 0
        or n_obs <= 0
        or matrices_vary < 0
        or matrices_vary > 1
    ):
        return 0
    var obs = Ptr(unsafe_from_address=observations)
    var mask = IPtr(unsafe_from_address=missing)
    var matrices = Ptr(unsafe_from_address=observation_matrices)
    var means = Ptr(unsafe_from_address=smoothed_means)
    var dst = Ptr(unsafe_from_address=result)
    fill(dst, 0.0, n_obs)
    var count = 0
    for t in range(timesteps):
        if mask[t] != 0:
            continue
        count += 1
        var matrix = matrix_at(matrices, t, n_obs, n_state, matrices_vary)
        for i in range(n_obs):
            var value = obs[t * n_obs + i]
            for j in range(n_state):
                value -= matrix[i * n_state + j] * means[t * n_state + j]
            dst[i] += value
    if count > 0:
        for i in range(n_obs):
            dst[i] /= Float64(count)
    return 1
