# Lila

> [!WARNING]
> Lila is a proof-of-concept exploration into formal verification and affine types for Python. It is **not** a production-ready tool, is highly unstable, and should not be used in critical systems.

**A Formally Verified, Affine-Typed JIT Compiler for Python via Z3 and Cranelift**

Lila is a research compiler infrastructure designed to explore Liquid Types (refinement types) and Affine Types (ownership semantics) within a Python-like DSL. By extracting Python AST and lowering it to a verified SSA form, Lila bypasses the CPython interpreter and GIL for high-performance, proven execution.

---

## Motivation
Python is beloved for its developer experience, but its type-hinting system is purely cosmetic—metadata that the interpreter ignores at runtime. This "trust-based" model forces a choice:
1.  **Safety:** Accept the heavy overhead of runtime checks and the Global Interpreter Lock (GIL).
2.  **Performance:** Drop into C/C++/Rust, losing Python's productivity and introducing manual memory management risks.

Lila exists to break this dichotomy. It takes Python's "hints" and turns them into **mathematically enforced laws**. By using formal verification to prove code safety at compile-time, we can bypass the interpreter entirely, executing Python at bare-metal speeds without sacrificing the sound logic that prevents crashes, data races, and memory corruption.

---

## Research Focus
- **Formal Verification:** Using the **Z3 SMT Solver** to prove logical invariants and memory safety at compile-time.
- **Fractional Permissions:** A flow-sensitive, symbolic weight partitioning system that proves memory safety and data-race freedom using SMT arithmetic.
- **JIT Lowering:** Mapping verified SSA directly to **Cranelift IR** for bare-metal performance.

---

## The Lila Pipeline: From AST to Bare Metal
The transformation from dynamic Python to verified machine code follows a strict five-stage pipeline:

1.  **Extraction:** The Python AST is captured and transformed into a versioned **Static Single Assignment (SSA)** Intermediate Representation.
2.  **Optimization:** The Rust middle-end applies multiple passes, including **Constant Folding**, **Dead Code Elimination (DCE)**, and **Type Propagation** to refine the SSA graph.
3.  **Formal Verification:** 
    *   **Logic Engine:** Every path condition and operation is mapped to SMT-LIB logic and verified by Z3.
    *   **Permission Verifier:** A flow-sensitive system using **Fractional Permissions** ($Exclusive=1.0, Shared \in (0, 1)$) to prove absence of aliasing violations and use-after-moves.
4.  **Backend Lowering:** The verified SSA is lowered into **Cranelift IR**, which handles target-specific register allocation and machine code generation.
5.  **Hot-Swapping:** Using PyO3, the original Python function object is intercepted and redirected to a native C-ABI trampoline pointing to the JIT-compiled memory.

---

## System Capabilities and Examples

### 1. Liquid Types: Provable Logical Invariants
Lila uses refinement types to prove that operations are mathematically safe before they ever execute.
```python
from lila import verify, i64, Refined

# Define a refinement: x must be strictly greater than 0
Positive = Refined[i64, lambda x: x > 0]

@verify
def divide_verified(n: i64, d: Positive) -> i64:
    # Z3 proves d > 0. Runtime ZeroDivisionError is mathematically impossible.
    return n // d
```

### 2. Fractional Permissions: Compile-Time Memory Safety
Lila uses a symbolic weight partitioning system (inspired by Separation Logic) to prevent data races and use-after-move errors.
```python
from lila import verify, i64, Held, Hand, Peek, struct

@verify
def consume(x: Held[i64]) -> i64:
    return x

@verify
def illegal_use(x: Held[i64]) -> i64:
    val = consume(x)  # x is moved here (1.0 permission consumed)
    return x + 1      # COMPILE ERROR: Use-after-move (0.0 permission remaining)

@struct
class Data: val: i64

@verify
def illegal_alias(d: Hand[Data]) -> i64:
    r1 = Peek(d)       # Shared permission created
    # d.val = 10      # ERROR: Cannot mutate 'd' while shared permissions (Peek) are active
    return r1.val

@verify
def safe_borrow(d: Hand[Data]) -> i64:
    # We can temporarily borrow 'd' immutably
    with Peek(d) as p:
        read_val = p.val
    
    # The 'with' block acts as a lexical borrow checker. 
    # Exiting the block returns the fractional permission to 'd'.
    # Now it is safe to mutate 'd' again because we hold 1.0 permission.
    d.val = 20 
    return d.val + read_val
```

### 3. Memory-Mapped Structs
Define zero-overhead, C-compatible structures that exist outside the CPython heap.
```python
from lila import struct, f64, i32

@struct
class Point:
    x: f64
    y: f64
    id: i32

    def move_by(self, dx: f64, dy: f64):
        self.x += dx
        self.y += dy
```

### 4. Tagged Unions: Formally Verified Enums
Lila supports type-safe tagged unions (Enums) with Z3-proven variant access.
```python
from lila import verify, i64, struct, enum

@struct
class SomePayload:
    val: i64

@enum
class Option:
    Some: SomePayload
    NoneVariant: None

@verify
def create_some(x: i64) -> i64:
    s = SomePayload(x)
    opt = Option.Some(s)
    if opt.is_Some():
        # Z3 proves this extraction is safe because of the branch above
        return opt.as_Some().val
    return -1
```

### 5. Recursive Functions: Inductive Reasoning
Lila can formally prove properties of recursive functions using inductive hypotheses and precise interval analysis. It ensures that recursive calls satisfy the function's own refinements and that return values satisfy refinements across inductive steps.
```python
from lila import verify, i64, Refined

# Result is always >= 1
StrictPositive = Refined[i64, lambda x: x >= 1]
# Input bounded to prevent signed overflow during induction
SmallPos = Refined[i64, lambda x: (0 <= x) & (x <= 20)]

@verify
def factorial(n: SmallPos) -> StrictPositive:
    if n <= 1:
        return 1
    # Lila proves inductively: n * fac(n-1) >= 2 * 1 >= 1
    return n * factorial(n - 1)

@verify
def fib(n: SmallPos) -> i64:
    if n <= 0: return 0
    if n == 1: return 1
    # Verified with nested recursive calls
    return fib(n - 1) + fib(n - 2)
```

### 6. Higher-Order Functions: First-Class Verified Logic
Lila supports closures and lambdas with full formal verification of their capture environments. Captured variables are automatically detected, bundled into heap-allocated environments, and verified for safe access.
```python
from lila import verify, i64, Closure

@verify
def make_adder(x: i64) -> Closure[[i64], i64]:
    # Lila performs capture analysis and heap-allocates the environment
    return lambda y: x + y

add5 = make_adder(5)
result = add5(10) # result = 15
```

### 7. Verified Buffer and NumPy Interop
Seamlessly operate on high-performance memory buffers (like NumPy arrays) with Z3-proven bounds checking.
```python
from lila import verify, Buffer, f64
import numpy as np

@verify
def scale_vector(vec: Buffer[f64], factor: f64) -> None:
    # Lila proves 'i' is always within [0, len(vec))
    for i in range(len(vec)):
        vec[i] *= factor
```

### 8. Source-Level Diagnostics
Lila provides visual highlights for verification failures, mapping IR-level logic errors back to your original Python source code for immediate debugging.
```text
[Lila Warning] Lila Verification Failed for 'divide_unsafe': Potential division by zero at v2
  --> source.py:3:12
   |
 3 |    return n // d
   |                ^--- Logic error detected here
```

### 9. Centralized Granular Tracing
Lila features a highly controllable diagnostics system. You can toggle granular debug levels for specific sub-systems (Liveness, Z3 Translation, SSA Optimization) directly from Python to isolate issues in the compiler pipeline.

```python
from lila import configure_tracing, LIVENESS, VERIFY, SSA

# Only see detailed liveness logs and SSA optimizations, keep everything else at info
configure_tracing({
    LIVENESS: "debug", 
    SSA: "debug",
    VERIFY: "info"
})

@verify
def debug_target(x: Held[i64]):
    # Detailed state transitions will now appear in your console
    ...
```

### 10. GIL-less Parallelism
Since Lila code operates on raw memory and avoids `PyObject` manipulation, it can execute across multiple threads without ever acquiring the Global Interpreter Lock.

---

## Technical Specifications

| Feature | Support / Technology |
| :--- | :--- |
| **Core Architecture** | Multi-crate Rust Workspace (`lila-core`, `lila-ir`, `lila-verify`, `lila-backend`, `lila-bridge`) |
| **Numeric Types** | `i8` through `u64`, `f32`, `f64` |
| **Functional Types**| `FnPointer`, `Closure`, `Callable` |
| **Complex Types** | Structs, Tagged Unions (Enums), Tuples |
| **Concurrency** | GIL-less multi-threading (`parallel_for`) |
| **Memory Model** | Fractional & Structural Permissions (Data-race free, Affine semantics) |
| **Logic Solver** | Z3 SMT Solver v4.12+ (BV & Float Theories) |
| **JIT Backend** | Cranelift 0.100+ |
| **AOT Caching** | Binary IR Serialization (`bincode`, `seahash`) |
| **Interoperability** | PyO3, ctypes, NumPy, Buffer Protocol |
| **Diagnostics** | Source-level visual highlights, Centralized Granular Tracing |
| **Optimization Passes**| SSA-DCE (Self-Protecting), Constant Folding, Type Propagation |

---

## Getting Started

### Prerequisites
- **Rust Toolchain** (latest stable)
- **Python 3.8+**
- **Z3 Solver** (shared library v4.12+)

### Installation and Testing
```bash
# Build Lila in release mode
maturin develop --release

# Run verification test suite
cargo test
python -m unittest tests/python/*.py
```

---

## Limitations (The "Strict" in Strict Python)
To maintain mathematical soundness, Lila imposes a "Closed World Assumption":
*   No dynamic attribute access (getattr, setattr).
*   No eval() or exec().
*   Functions must have explicit type annotations.
*   Recursive calls must have proven termination (WIP).

---

## Future Research and Technical Roadmap

### 1. Transition to Aeneas and F*
While Z3 provides high-performance SMT solving, moving towards **Aeneas** and the **F\*** programming language would allow for **Absolute Formal Proof**. This transition would enable the project to move from "highly likely safe" to "mathematically certain" by extracting verified models directly from the Rust implementation.

### 2. Automated Loop Invariant Synthesis
Researching **Abstract Interpretation** or **Inductive Logic Programming** to automatically derive loop invariants for Z3 or F*, significantly expanding the range of provable programs.

### 3. IEEE 754 Floating-Point Proofs
Extending the bit-accurate model to prove stability and bound precision loss in complex numerical algorithms.

---

<div align="center">

Built with 🦀 & 🐍 by [Seuriin](https://github.com/SSL-ACTX)

</div>


</div>
