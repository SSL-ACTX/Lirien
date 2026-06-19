import time
from lirien import verify, i64, f64


# Lirien implementation
@verify
def mandelbrot_pixel(c_re: f64, c_im: f64, max_iter: i64) -> i64:
    z_re = 0.0
    z_im = 0.0
    for i in range(max_iter):
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = 2.0 * z_re * z_im + c_im
        z_re = new_re
        z_im = new_im
        if z_re * z_re + z_im * z_im > 4.0:
            return i
    return max_iter


# Pure Python implementation for baseline
def mandelbrot_pixel_py(c_re, c_im, max_iter):
    z_re = 0.0
    z_im = 0.0
    for i in range(max_iter):
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = 2.0 * z_re * z_im + c_im
        z_re = new_re
        z_im = new_im
        if z_re * z_re + z_im * z_im > 4.0:
            return i
    return max_iter


def run_bench(width=200, height=200, max_iter=1000):
    print(f"Benchmarking Mandelbrot ({width}x{height}, max_iter={max_iter})...")

    # Generate coordinates manually
    x_vals = [(-2.0 + (2.5 * i / width)) for i in range(width)]
    y_vals = [(-1.25 + (2.5 * i / height)) for i in range(height)]

    # 1. Pure Python
    start = time.perf_counter()
    for y in y_vals:
        for x in x_vals:
            mandelbrot_pixel_py(x, y, max_iter)
    end = time.perf_counter()
    py_time = end - start
    print(f"Pure Python: {py_time:.4f}s")

    # 2. Lirien
    # Warmup
    mandelbrot_pixel(0.0, 0.0, 10)

    start = time.perf_counter()
    for y in y_vals:
        for x in x_vals:
            mandelbrot_pixel(x, y, max_iter)
    end = time.perf_counter()
    lirien_time = end - start
    print(f"Lirien JIT:    {lirien_time:.4f}s")

    if lirien_time > 0:
        print(f"\nSpeedup: {py_time / lirien_time:.1f}x")
    else:
        print("\nLirien execution was too fast to measure accurately.")


if __name__ == "__main__":
    run_bench()
