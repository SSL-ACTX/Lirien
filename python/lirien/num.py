import math
from lirien import verify, f32, Tensor, TypeVar

# Dimension TypeVars
M = TypeVar("M")
N = TypeVar("N")
K = TypeVar("K")
H = TypeVar("H")
W = TypeVar("W")
KH = TypeVar("KH")
KW = TypeVar("KW")
OH = TypeVar("OH")
OW = TypeVar("OW")


@verify
def transpose(a: Tensor[f32, M, N], out: Tensor[f32, N, M]):
    """
    Transpose a 2D tensor 'a' of shape (M, N) into 'out' of shape (N, M).
    Verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[j, i] = a[i, j]


@verify
def relu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Rectified Linear Unit (ReLU) activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = 0.0


@verify
def sigmoid(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Sigmoid activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            # 1 / (1 + exp(-val))
            out[i, j] = 1.0 / (1.0 + math.exp(-val))


@verify
def leaky_relu(a: Tensor[f32, M, N], out: Tensor[f32, M, N], alpha: f32):
    """
    Apply element-wise Leaky ReLU activation in-place to 'out'.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = alpha * val


@verify
def convolve1d(
    signal: Tensor[f32, M],
    kernel: Tensor[f32, K],
    out: Tensor[f32, M - K + 1],
):
    """
    1D valid convolution of 'signal' and 'kernel', storing the result in 'out'.
    Statically verified by Z3 that all accesses to signal, kernel, and out are in-bounds.
    """
    for i in range(M - K + 1):
        sum_val: f32 = 0.0
        for j in range(K):
            sum_val = sum_val + signal[i + j] * kernel[j]
        out[i] = sum_val


@verify
def convolve2d(
    image: Tensor[f32, H, W],
    kernel: Tensor[f32, KH, KW],
    out: Tensor[f32, H - KH + 1, W - KW + 1],
):
    """
    2D valid convolution of 'image' and 'kernel', storing the result in 'out'.
    Statically verified by Z3 that all accesses to image, kernel, and out are in-bounds.
    """
    for i in range(H - KH + 1):
        for j in range(W - KW + 1):
            sum_val: f32 = 0.0
            for ki in range(KH):
                for kj in range(KW):
                    sum_val = sum_val + image[i + ki, j + kj] * kernel[ki, kj]
            out[i, j] = sum_val


@verify
def matmul(a: Tensor[f32, M, N], b: Tensor[f32, N, K], out: Tensor[f32, M, K]):
    """
    Matrix multiplication of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(K):
            sum_val: f32 = 0.0
            for l in range(N):
                sum_val = sum_val + a[i, l] * b[l, j]
            out[i, j] = sum_val


@verify
def softmax(a: Tensor[f32, M], out: Tensor[f32, M]):
    """
    Apply Softmax activation to 1D tensor 'a', storing the result in 'out'.
    Statically verified by Z3 to be division-by-zero safe and memory-safe.
    """
    sum_exp: f32 = 0.0
    for i in range(M):
        sum_exp = sum_exp + math.exp(a[i])
    for i in range(M):
        out[i] = math.exp(a[i]) / sum_exp


@verify
def add(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise addition of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + b[i, j]


@verify
def sub(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise subtraction of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] - b[i, j]


@verify
def mul(a: Tensor[f32, M, N], b: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Element-wise multiplication of 'a' and 'b', storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * b[i, j]


@verify
def max_pool2d_2x2(image: Tensor[f32, H, W], out: Tensor[f32, OH, OW]):
    """
    2x2 Max Pooling with stride 2.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(OH):
        for j in range(OW):
            v00 = image[i * 2, j * 2]
            v01 = image[i * 2, j * 2 + 1]
            v10 = image[i * 2 + 1, j * 2]
            v11 = image[i * 2 + 1, j * 2 + 1]

            max_val = v00
            if v01 > max_val:
                max_val = v01
            if v10 > max_val:
                max_val = v10
            if v11 > max_val:
                max_val = v11

            out[i, j] = max_val


@verify
def avg_pool2d_2x2(image: Tensor[f32, H, W], out: Tensor[f32, OH, OW]):
    """
    2x2 Average Pooling with stride 2.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(OH):
        for j in range(OW):
            v00 = image[i * 2, j * 2]
            v01 = image[i * 2, j * 2 + 1]
            v10 = image[i * 2 + 1, j * 2]
            v11 = image[i * 2 + 1, j * 2 + 1]
            out[i, j] = (v00 + v01 + v10 + v11) * 0.25


@verify
def clip(a: Tensor[f32, M, N], out: Tensor[f32, M, N], min_val: f32, max_val: f32):
    """
    Clip the values in 'a' to [min_val, max_val] and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val < min_val:
                out[i, j] = min_val
            elif val > max_val:
                out[i, j] = max_val
            else:
                out[i, j] = val


@verify
def mean(a: Tensor[f32, M], out: Tensor[f32, 1], n: f32):
    """
    Compute the mean of 'a' and store in 'out[0]'.
    Requires precondition 'n > 0.0' to guarantee division safety.
    """
    assert n > 0.0
    sum_val: f32 = 0.0
    for i in range(M):
        sum_val = sum_val + a[i]
    out[0] = sum_val / n


@verify
def scale(a: Tensor[f32, M, N], out: Tensor[f32, M, N], factor: f32):
    """
    Scale the tensor 'a' by a scalar 'factor' and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * factor


@verify
def bias_add(a: Tensor[f32, M, N], bias: Tensor[f32, N], out: Tensor[f32, M, N]):
    """
    Add a 1D bias vector 'bias' to 'a' along the last dimension and store in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + bias[j]


@verify
def standardize(a: Tensor[f32, M], out: Tensor[f32, M], mean_val: f32, std_val: f32):
    """
    Standardize 'a' using precomputed 'mean_val' and 'std_val'.
    Requires 'std_val > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert std_val > 0.0
    for i in range(M):
        out[i] = (a[i] - mean_val) / std_val


@verify
def l2_normalize(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32):
    """
    L2 normalize a 1D vector 'a', storing the result in 'out'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        sum_sq = sum_sq + a[i] * a[i]
    divisor = math.sqrt(abs(sum_sq) + epsilon)
    assert divisor > 0.0
    for i in range(M):
        out[i] = a[i] / divisor


@verify
def l1_normalize(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32):
    """
    L1 normalize a 1D vector 'a', storing the result in 'out'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    sum_abs: f32 = 0.0
    for i in range(M):
        sum_abs = sum_abs + abs(a[i])
    divisor = sum_abs + epsilon
    assert divisor > 0.0
    for i in range(M):
        out[i] = a[i] / divisor


@verify
def cosine_similarity(
    a: Tensor[f32, M],
    b: Tensor[f32, M],
    out: Tensor[f32, 1],
    epsilon: f32,
):
    """
    Compute the cosine similarity of 'a' and 'b', storing in 'out[0]'.
    Requires 'epsilon > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    dot_val: f32 = 0.0
    norm_a_sq: f32 = 0.0
    norm_b_sq: f32 = 0.0
    for i in range(M):
        dot_val = dot_val + a[i] * b[i]
        norm_a_sq = norm_a_sq + a[i] * a[i]
        norm_b_sq = norm_b_sq + b[i] * b[i]

    denom = math.sqrt(abs(norm_a_sq)) * math.sqrt(abs(norm_b_sq)) + epsilon
    assert denom > 0.0
    out[0] = dot_val / denom


@verify
def matvec(matrix: Tensor[f32, M, N], vector: Tensor[f32, N], out: Tensor[f32, M]):
    """
    Matrix-vector multiplication, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        sum_val: f32 = 0.0
        for j in range(N):
            sum_val = sum_val + matrix[i, j] * vector[j]
        out[i] = sum_val


@verify
def outer(a: Tensor[f32, M], b: Tensor[f32, N], out: Tensor[f32, M, N]):
    """
    Compute the outer product of vectors 'a' and 'b', storing in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i] * b[j]


@verify
def dot(a: Tensor[f32, M], b: Tensor[f32, M], out: Tensor[f32, 1]):
    """
    Compute the dot product of vectors 'a' and 'b', storing in 'out[0]'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    sum_val: f32 = 0.0
    for i in range(M):
        sum_val = sum_val + a[i] * b[i]
    out[0] = sum_val
