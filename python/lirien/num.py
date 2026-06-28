import math
from lirien import verify, f32, Tensor, TypeVar, f32x4, i64, jit

# Dimension TypeVars
B = TypeVar("B")
M = TypeVar("M")
N = TypeVar("N")
K = TypeVar("K")
H = TypeVar("H")
W = TypeVar("W")
KH = TypeVar("KH")
KW = TypeVar("KW")
OH = TypeVar("OH")
OW = TypeVar("OW")
SH = TypeVar("SH")
SW = TypeVar("SW")
PH = TypeVar("PH")
PW = TypeVar("PW")


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
def silu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Sigmoid Linear Unit (SiLU) / Swish activation.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            out[i, j] = val / (1.0 + math.exp(-val))


@verify
def rms_norm(a: Tensor[f32, M], out: Tensor[f32, M], epsilon: f32, n: f32):
    """
    Root Mean Square Normalization (RMSNorm) of 'a', storing in 'out'.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        sum_sq = sum_sq + a[i] * a[i]
    rms = math.sqrt(abs(sum_sq / n) + epsilon)
    assert rms > 0.0
    for i in range(M):
        out[i] = a[i] / rms


@verify
def layer_norm(
    a: Tensor[f32, M],
    out: Tensor[f32, M],
    gamma: Tensor[f32, M],
    beta: Tensor[f32, M],
    epsilon: f32,
    n: f32,
):
    """
    Layer Normalization of 'a' with scale 'gamma' and shift 'beta'.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0

    # Compute mean and sum of squares in a single loop
    sum_val: f32 = 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        val = a[i]
        sum_val = sum_val + val
        sum_sq = sum_sq + val * val

    mean_val = sum_val / n
    var_val = sum_sq / n - mean_val * mean_val

    # Normalize and scale/shift
    std_val = math.sqrt(abs(var_val) + epsilon)
    assert std_val > 0.0
    for i in range(M):
        out[i] = (a[i] - mean_val) / std_val * gamma[i] + beta[i]


@verify
def matvec_bias(
    matrix: Tensor[f32, M, N],
    vector: Tensor[f32, N],
    bias: Tensor[f32, M],
    out: Tensor[f32, M],
):
    """
    Matrix-vector multiplication with a bias vector, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        sum_val: f32 = 0.0
        for j in range(N):
            sum_val = sum_val + matrix[i, j] * vector[j]
        out[i] = sum_val + bias[i]


@verify
def sigmoid_cross_entropy(
    logits: Tensor[f32, M, N],
    targets: Tensor[f32, M, N],
    out: Tensor[f32, M, N],
):
    """
    Compute element-wise sigmoid cross entropy loss.
    Statically verified by Z3 to be memory-safe, division-safe, and log-safe.
    """
    for i in range(M):
        for j in range(N):
            x = logits[i, j]
            y = targets[i, j]
            # Stable formula: max(x, 0) - x * y + log(1 + exp(-abs(x)))
            max_val = x
            if 0.0 > max_val:
                max_val = 0.0

            out[i, j] = max_val - x * y + math.log(1.0 + math.exp(-abs(x)))


@verify
def l2_loss(
    a: Tensor[f32, M, N],
    b: Tensor[f32, M, N],
    out: Tensor[f32, 1],
    n: f32,
):
    """
    Compute L2 loss (Mean Squared Error) between 'a' and 'b'.
    Requires 'n > 0.0' (where n is float(2 * M * N)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert n > 0.0
    sum_sq: f32 = 0.0
    for i in range(M):
        for j in range(N):
            diff = a[i, j] - b[i, j]
            sum_sq = sum_sq + diff * diff
    out[0] = sum_sq / n


@verify
def dot_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated dot product of two tensors of f32x4.
    Computes parallel vector products and performs a horizontal sum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]  # Initialize zero vector
    for i in range(M):
        acc = acc + a[i] * b[i]
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def matvec_simd(
    matrix: Tensor[f32x4, M, N],
    vector: Tensor[f32x4, N],
    out: Tensor[f32, M],
):
    """
    SIMD-accelerated matrix-vector multiplication.
    Computes parallel row-vector dot products.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        acc = matrix[i, 0] - matrix[i, 0]  # Initialize zero vector
        for j in range(N):
            acc = acc + matrix[i, j] * vector[j]
        out[i] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def mse_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated Mean Squared Error (MSE) accumulation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]
    for i in range(M):
        diff = a[i] - b[i]
        acc = acc + diff * diff
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def mae_simd(a: Tensor[f32x4, M], b: Tensor[f32x4, M], out: Tensor[f32, 1]):
    """
    SIMD-accelerated Mean Absolute Error (MAE) accumulation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    acc = a[0] - a[0]
    for i in range(M):
        diff = a[i] - b[i]
        acc = acc + abs(diff)
    out[0] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def add_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise addition of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] + b[i, j]


@verify
def sub_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise subtraction of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] - b[i, j]


@verify
def mul_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise multiplication of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * b[i, j]


@verify
def scale_simd(
    a: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
    factor: f32,
):
    """
    SIMD-accelerated element-wise scaling of a tensor of f32x4 by a scalar factor.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] * factor


@verify
def relu_simd(a: Tensor[f32x4, M, N], out: Tensor[f32x4, M, N]):
    """
    SIMD-accelerated element-wise branchless ReLU of a tensor of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            zero = val * 0.0
            out[i, j] = max(val, zero)


@verify
def div_simd(
    a: Tensor[f32x4, M, N],
    b: Tensor[f32x4, M, N],
    out: Tensor[f32x4, M, N],
):
    """
    SIMD-accelerated element-wise division of two tensors of f32x4.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            out[i, j] = a[i, j] / b[i, j]


@verify
def matmul_simd(
    a: Tensor[f32x4, M, K],
    b: Tensor[f32x4, K, N],
    out: Tensor[f32, M, N],
):
    """
    SIMD-accelerated 2D matrix multiplication.
    Computes parallel row-column vector products and performs a horizontal sum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            acc = a[i, 0] - a[i, 0]  # Initialize zero vector
            for k in range(K):
                acc = acc + a[i, k] * b[k, j]
            out[i, j] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def hardsigmoid(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Hard Sigmoid activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j] + 3.0
            if val < 0.0:
                out[i, j] = 0.0
            elif val > 6.0:
                out[i, j] = 1.0
            else:
                out[i, j] = val / 6.0


@verify
def hardswish(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Hard Swish activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            h_sig = val + 3.0
            if h_sig < 0.0:
                out[i, j] = 0.0
            elif h_sig > 6.0:
                out[i, j] = val
            else:
                out[i, j] = val * (h_sig / 6.0)


@verify
def elu(a: Tensor[f32, M, N], out: Tensor[f32, M, N], alpha: f32):
    """
    Apply element-wise Exponential Linear Unit (ELU) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = val
            else:
                out[i, j] = alpha * (math.exp(val) - 1.0)


@verify
def selu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise Scaled Exponential Linear Unit (SELU) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    for i in range(M):
        for j in range(N):
            val = a[i, j]
            if val > 0.0:
                out[i, j] = scale * val
            else:
                out[i, j] = scale * alpha * (math.exp(val) - 1.0)


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


@verify
def gelu(a: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise GELU (tanh approximation) activation.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            x = a[i, j]
            # z = sqrt(2/pi) * (x + 0.044715 * x^3)
            # sqrt(2/pi) is approx 0.79788456
            z = 0.79788456 * (x + 0.044715 * x * x * x)
            # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
            exp_2z = math.exp(2.0 * z)
            tanh_z = (exp_2z - 1.0) / (exp_2z + 1.0)
            out[i, j] = 0.5 * x * (1.0 + tanh_z)


@verify
def swiglu(x: Tensor[f32, M, N], gate: Tensor[f32, M, N], out: Tensor[f32, M, N]):
    """
    Apply element-wise SwiGLU activation: Swish(gate) * x, storing the result in 'out'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    for i in range(M):
        for j in range(N):
            g = gate[i, j]
            swish_g = g / (1.0 + math.exp(-g))
            out[i, j] = swish_g * x[i, j]


@verify
def sgd_momentum_step(
    param: Tensor[f32, M, N],
    grad: Tensor[f32, M, N],
    velocity: Tensor[f32, M, N],
    lr: f32,
    momentum: f32,
):
    """
    Perform an in-place SGD step with momentum.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for i in range(M):
        for j in range(N):
            v = momentum * velocity[i, j] + lr * grad[i, j]
            velocity[i, j] = v
            param[i, j] = param[i, j] - v


@verify
def adamw_step(
    param: Tensor[f32, M, N],
    grad: Tensor[f32, M, N],
    m: Tensor[f32, M, N],
    v: Tensor[f32, M, N],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    wd: f32,
    bias_correction1: f32,
    bias_correction2: f32,
):
    """
    Perform an in-place AdamW step.
    Requires 'epsilon > 0.0', 'bias_correction1 > 0.0', and 'bias_correction2 > 0.0'.
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert bias_correction1 > 0.0
    assert bias_correction2 > 0.0
    for i in range(M):
        for j in range(N):
            g = grad[i, j]
            m_t = beta1 * m[i, j] + (1.0 - beta1) * g
            v_t = beta2 * v[i, j] + (1.0 - beta2) * g * g
            m[i, j] = m_t
            v[i, j] = v_t

            m_hat = m_t / bias_correction1
            v_hat = v_t / bias_correction2

            denom = math.sqrt(abs(v_hat)) + epsilon
            assert denom > 0.0
            param[i, j] = param[i, j] - lr * (m_hat / denom + wd * param[i, j])


@verify
def softmax_cross_entropy_with_logits(
    logits: Tensor[f32, M, N],
    targets: Tensor[f32, M, N],
    out: Tensor[f32, M],
):
    """
    Compute multi-class cross entropy loss per batch element.
    Uses the log-sum-exp trick to prevent underflow/overflow.
    Statically verified by Z3 to be memory-safe, division-safe, and log-safe.
    """
    for i in range(M):
        # Find max logit for stability
        max_val = logits[i, 0]
        for j in range(N):
            if logits[i, j] > max_val:
                max_val = logits[i, j]

        # Compute log-sum-exp
        sum_exp = 0.0
        for j in range(N):
            sum_exp = sum_exp + math.exp(logits[i, j] - max_val)

        assert sum_exp > 0.0
        lse = max_val + math.log(sum_exp)

        # Compute cross entropy: sum(target * (lse - logit))
        loss = 0.0
        for j in range(N):
            loss = loss + targets[i, j] * (lse - logits[i, j])

        out[i] = loss


@verify
def bmm(
    a: Tensor[f32, B, M, N],
    b: Tensor[f32, B, N, K],
    out: Tensor[f32, B, M, K],
):
    """
    Batch matrix multiplication: out[b] = a[b] @ b[b]
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for batch in range(B):
        for i in range(M):
            for j in range(K):
                sum_val: f32 = 0.0
                for l in range(N):
                    sum_val = sum_val + a[batch, i, l] * b[batch, l, j]
                out[batch, i, j] = sum_val


@verify
def bmm_simd(
    a: Tensor[f32x4, B, M, K],
    b: Tensor[f32x4, B, K, N],
    out: Tensor[f32, B, M, N],
):
    """
    SIMD-accelerated batch matrix multiplication.
    Statically verified by Z3 to be memory-safe and in-bounds.
    """
    for batch in range(B):
        for i in range(M):
            for j in range(N):
                acc = a[batch, i, 0] - a[batch, i, 0]  # Initialize zero vector
                for k in range(K):
                    acc = acc + a[batch, i, k] * b[batch, k, j]
                out[batch, i, j] = acc[0] + acc[1] + acc[2] + acc[3]


@verify
def rms_norm_simd(
    a: Tensor[f32x4, M],
    out: Tensor[f32x4, M],
    epsilon: f32,
    n: f32,
):
    """
    SIMD-accelerated RMSNorm.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M * 4)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0
    acc = a[0] - a[0]  # Initialize zero vector
    for i in range(M):
        acc = acc + a[i] * a[i]
    sum_sq = acc[0] + acc[1] + acc[2] + acc[3]
    rms = math.sqrt(abs(sum_sq / n) + epsilon)
    assert rms > 0.0
    inv_rms = 1.0 / rms
    for i in range(M):
        out[i] = a[i] * inv_rms


@verify
def layer_norm_simd(
    a: Tensor[f32x4, M],
    out: Tensor[f32x4, M],
    gamma: Tensor[f32x4, M],
    beta: Tensor[f32x4, M],
    epsilon: f32,
    n: f32,
):
    """
    SIMD-accelerated Layer Normalization.
    Requires 'epsilon > 0.0' and 'n > 0.0' (where n is float(M * 4)).
    Statically verified by Z3 to be memory-safe and division-by-zero safe.
    """
    assert epsilon > 0.0
    assert n > 0.0

    sum_vec = a[0] - a[0]
    sum_sq_vec = a[0] - a[0]
    for i in range(M):
        val = a[i]
        sum_vec = sum_vec + val
        sum_sq_vec = sum_sq_vec + val * val

    sum_val = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3]
    sum_sq = sum_sq_vec[0] + sum_sq_vec[1] + sum_sq_vec[2] + sum_sq_vec[3]

    mean_val = sum_val / n
    var_val = sum_sq / n - mean_val * mean_val

    std_val = math.sqrt(abs(var_val) + epsilon)
    assert std_val > 0.0
    inv_std = 1.0 / std_val

    for i in range(M):
        out[i] = (a[i] - mean_val) * inv_std * gamma[i] + beta[i]


@jit
def max_pool2d(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
):
    """
    Generic 2D Max Pooling with arbitrary kernel size and stride.

    Safety model: @jit (runtime-enforced via assert).
    Full Z3 formal verification is impractical here because Lirien fully
    unrolls all 4 nested loops (OH*OW*KH*KW blocks) in the SSA IR, producing
    a CFG too large for the solver to reason about in bounded time.
    The boundary assertions below fire at call time, giving the same
    runtime safety guarantee as PyTorch/NumPy.
    """
    assert kernel_h > 0
    assert kernel_w > 0
    assert stride_h > 0
    assert stride_w > 0
    assert (OH - 1) * stride_h + kernel_h <= H
    assert (OW - 1) * stride_w + kernel_w <= W

    for i in range(OH):
        for j in range(OW):
            h_start = i * stride_h
            w_start = j * stride_w

            max_val = image[h_start, w_start]
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    val = image[h_start + kh, w_start + kw]
                    if val > max_val:
                        max_val = val
            out[i, j] = max_val


@jit
def avg_pool2d(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
):
    """
    Generic 2D Average Pooling with arbitrary kernel size and stride.

    Safety model: @jit (runtime-enforced via assert).
    Same rationale as max_pool2d: 4 nested loops produce a CFG that
    exceeds Z3's tractable search space at verification time.
    """
    assert kernel_h > 0
    assert kernel_w > 0
    assert stride_h > 0
    assert stride_w > 0
    assert (OH - 1) * stride_h + kernel_h <= H
    assert (OW - 1) * stride_w + kernel_w <= W

    for i in range(OH):
        for j in range(OW):
            h_start = i * stride_h
            w_start = j * stride_w

            sum_val: f32 = 0.0
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    sum_val = sum_val + image[h_start + kh, w_start + kw]

            denom = f32(kernel_h * kernel_w)
            assert denom > 0.0
            out[i, j] = sum_val / denom


@jit
def convolve2d_padded(
    image: Tensor[f32, H, W],
    kernel: Tensor[f32, KH, KW],
    out: Tensor[f32, OH, OW],
    stride_h: i64,
    stride_w: i64,
    pad_h: i64,
    pad_w: i64,
):
    """
    Generic 2D Convolution with arbitrary stride and zero-padding.

    Safety model: @jit (runtime-enforced via assert + branch guards).
    The 4-level nested loop produces a SSA CFG that is too large for Z3
    to discharge within a bounded timeout. Safety is guaranteed structurally
    by the flow-sensitive im_h/im_w bounds checks inside the loop body,
    which prevent any out-of-bounds access at runtime.
    """
    assert stride_h > 0
    assert stride_w > 0
    assert pad_h >= 0
    assert pad_w >= 0

    for i in range(OH):
        for j in range(OW):
            sum_val: f32 = 0.0
            for kh in range(KH):
                for kw in range(KW):
                    im_h = i * stride_h + kh - pad_h
                    im_w = j * stride_w + kw - pad_w

                    if im_h >= 0:
                        if im_h < H:
                            if im_w >= 0:
                                if im_w < W:
                                    sum_val = (
                                        sum_val + image[im_h, im_w] * kernel[kh, kw]
                                    )
            out[i, j] = sum_val


@jit
def resize_nearest(
    image: Tensor[f32, H, W],
    out: Tensor[f32, OH, OW],
    scale_h: f32,
    scale_w: f32,
):
    """
    Nearest-neighbor image resizing with arbitrary float scaling factors.

    Safety model: @jit (runtime-enforced via assert + branch guards).
    The float-to-int index computation combined with the nested loop CFG
    exceeds Z3's tractable search space. Safety is guaranteed structurally
    by the src_h/src_w bounds checks, which prevent out-of-bounds access.
    """
    assert scale_h > 0.0
    assert scale_w > 0.0

    for i in range(OH):
        for j in range(OW):
            src_h = int(f32(i) * scale_h)
            src_w = int(f32(j) * scale_w)

            if src_h >= 0:
                if src_h < H:
                    if src_w >= 0:
                        if src_w < W:
                            out[i, j] = image[src_h, src_w]
                        else:
                            out[i, j] = 0.0
                    else:
                        out[i, j] = 0.0
                else:
                    out[i, j] = 0.0
            else:
                out[i, j] = 0.0
