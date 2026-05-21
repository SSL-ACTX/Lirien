"""
L.I.L.A. (Liquid Logic and Affine Ownership) Showcase
===================================================

This example demonstrates how Lila uses formal verification (Z3)
to prove mathematical safety at compile-time and lowers to 
bare-metal machine code (Cranelift) for extreme performance.

Features demonstrated:
1. Liquid Types (Refinement Types) with complex predicates
2. Memory-Mapped Structs with geometric invariants
3. Verified Numerical Stability (Safe Math)
4. Verified Buffer Interop (Zero Bounds-Checking)
"""

import math
import array
from lila import verify, i64, f64, struct, Refined, Buffer

# ---------------------------------------------------------
# 1. Liquid Types & Complex Logical Predicates
# ---------------------------------------------------------
# A type that is even, positive, and less than 1000.
# Z3 ensures any variable of this type satisfies all three conditions.
SpecificInt = Refined[i64, lambda x: (x > 0) and (x < 1000) and (x % 2 == 0)]

@verify
def safe_divide_even(n: SpecificInt) -> i64:
    # Z3 proves n > 0, so this division is mathematically safe.
    # No runtime check for ZeroDivisionError is needed!
    return 1000 // n

# ---------------------------------------------------------
# 2. Memory-Mapped Structs & Geometric Invariants
# ---------------------------------------------------------
@struct
class Rectangle:
    width: i64
    height: i64

# A valid rectangle must have positive dimensions.
ValidRect = Refined[Rectangle, lambda r: (r.width > 0) and (r.height > 0)]

@verify
def compute_area(r: ValidRect) -> i64:
    # Z3 tracks invariants even across struct fields.
    return r.width * r.height

# ---------------------------------------------------------
# 3. Verified Numerical Stability (Safe Math)
# ---------------------------------------------------------
# A refined float guaranteed to be non-negative.
NonNegativeFloat = Refined[f64, lambda x: x >= 0.0]

@verify
def verified_sqrt(x: NonNegativeFloat) -> f64:
    # math.sqrt(x) is only defined for x >= 0.
    # Lila proves this condition holds at compile-time.
    return math.sqrt(x)

# ---------------------------------------------------------
# 4. Verified Buffer Interop (Zero-Cost Safety)
# ---------------------------------------------------------
# Buffer[T] provides direct memory access to arrays/memoryviews.
# Lila statically proves that the index `i` is always in bounds.
@verify
def scale_buffer(data: Buffer[f64], factor: f64) -> None:
    n = len(data)
    for i in range(n):
        # Index 'i' is proven to be in [0, n)
        data[i] = data[i] * factor

def run_showcase():
    print("=== L.I.L.A. Feature Showcase ===")
    
    # Showcase 1: Complex Refinements
    print("\n1. Liquid Types (Nested Logic):")
    res = safe_divide_even(50)
    print(f"   1000 // 50 = {res} (Proven safe by Z3)")

    # Showcase 2: Struct Invariants
    print("\n2. Structs (Geometric Invariants):")
    r = Rectangle(10, 20)
    area = compute_area(r)
    print(f"   Rectangle(10, 20) area: {area} (Proven positive fields)")

    # Showcase 3: Numerical Stability
    print("\n3. Numerical Stability (Verified Sqrt):")
    val = 16.0
    res = verified_sqrt(val)
    print(f"   sqrt({val}) = {res} (Proven non-negative domain)")

    # Showcase 4: Buffer performance
    print("\n4. Verified Buffer Interop (Zero Bounds-Checking):")
    arr = array.array("d", [1.0, 2.0, 3.0, 4.0])
    print(f"   Original array: {list(arr)}")
    scale_buffer(arr, 2.5)
    print(f"   Modified array: {list(arr)}")

if __name__ == "__main__":
    run_showcase()
