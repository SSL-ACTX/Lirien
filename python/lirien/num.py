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
