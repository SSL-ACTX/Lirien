# Lila

> [!WARNING]
> Lila is a proof-of-concept exploration into formal verification for Python. It is **not** a production-ready tool, is highly unstable, and should not be used in critical systems.

**A Formally Verified JIT Compiler for Python via Z3 and Cranelift**

Lila is a research compiler infrastructure designed to explore Liquid Types (refinement types) within a Python-like DSL. By extracting Python AST and lowering it to a verified SSA form, Lila bypasses the CPython interpreter and GIL for high-performance, proven execution.

---

## Motivation
Python is beloved for its developer experience, but its type-hinting system is purely cosmetic—metadata that the interpreter ignores at runtime. This "trust-based" model forces a choice:
1.  **Safety:** Accept the heavy overhead of runtime checks and the Global Interpreter Lock (GIL).
2.  **Performance:** Drop into C/C++/Rust, losing Python's productivity and introducing manual memory management risks.

Lila exists to break this dichotomy. It takes Python's "hints" and turns them into **mathematically enforced laws**. By using formal verification to prove code safety at compile-time, we can bypass the interpreter entirely, executing Python at bare-metal speeds without sacrificing the sound logic that prevents crashes and data corruption.

---

## Research Focus
- **Formal Verification:** Using the **Z3 SMT Solver** to prove logical invariants and memory safety at compile-time.
- **Refinement Types:** Attaching logical predicates to data to prove complex invariants across all reachable paths.
- **JIT Lowering:** Mapping verified SSA directly to **Cranelift IR** for bare-metal performance.

---

## The Lila Pipeline: From AST to Bare Metal
The transformation from dynamic Python to verified machine code follows a strict five-stage pipeline:

1.  **Extraction:** The Python AST is captured and transformed into a versioned **Static Single Assignment (SSA)** Intermediate Representation.
2.  **Optimization:** The Rust middle-end applies multiple passes, including **Constant Folding**, **Dead Code Elimination (DCE)**, and **Type Propagation** to refine the SSA graph.
3.  **Formal Verification:** Every path condition, mathematical operation, and memory access is mapped to SMT-LIB logic and rigorously proven by the **Z3 Solver**. Refinement types are checked across all reachable paths.
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

### 2. Memory-Mapped Structs
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

### 3. Tagged Unions: Formally Verified Enums
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

### 4. Recursive Functions: Inductive Reasoning
Lila can formally prove properties of recursive functions using inductive hypotheses and precise interval analysis.
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
```

### 5. Higher-Order Functions: First-Class Verified Logic
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

### 6. Verified Buffer and NumPy Interop
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

### 7. Source-Level Diagnostics
Lila provides visual highlights for verification failures, mapping IR-level logic errors back to your original Python source code for immediate debugging.

### 8. GIL-less Parallelism
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
python -m unittest discover tests/python
```

---

## Limitations (The "Strict" in Strict Python)
To maintain mathematical soundness, Lila imposes a "Closed World Assumption":
*   No dynamic attribute access (getattr, setattr).
*   No eval() or exec().
*   Functions must have explicit type annotations.

---

<div align="center">

Built with 🦀 & 🐍 by [Seuriin](https://github.com/SSL-ACTX)

</div>
