# RFC: Project Lila 
**Title:** A Formally Verified, Affine-Typed JIT Compiler for Python via Z3 and Cranelift  
**Domain:** Compiler Design, Formal Verification, Just-In-Time Compilation  
**Status:** Draft / Research Proposal  

## 1. Abstract
Project Lila is a stealth compiler disguised as a Python type-hinting library. It introduces Liquid Types (refinement types) and Affine Types (ownership semantics) to Python. By extracting the Python AST at module load time, Lila transforms the code into a Static Single Assignment (SSA) intermediate representation (IR). This IR is mathematically proven by the Z3 SMT Solver for logical correctness (bounds checking, invariants) and borrow-checked by a Rust middle-end. Upon successful verification, the SSA is lowered into Cranelift IR, JIT-compiled into bare-metal machine code, and hot-swapped with the original Python function via a PyO3 C-ABI trampoline—entirely bypassing the CPython Interpreter, `PyObject` overhead, and the Global Interpreter Lock (GIL).

## 2. Motivation
Python's dynamic nature incurs massive runtime overhead and defers critical logical errors (e.g., out-of-bounds, type mismatches, data races) to runtime. Current solutions (e.g., Numba, PyPy, Cython) focus on type inference and execution speed but do not guarantee mathematical correctness or strict memory ownership. Lila aims to achieve **zero-cost abstractions with compile-time formal verification**, pushing Python into the domain of high-performance, memory-safe systems programming.

## 3. Architecture Overview
The pipeline consists of four distinct phases:
1.  **Frontend (Python):** DSL for constraints and ownership.
2.  **Middle-end (Rust):** AST Extraction $\rightarrow$ CFG Generation $\rightarrow$ SSA Transformation.
3.  **Verification (Z3 & Rust):** Theorem proving and Affine analysis.
4.  **Backend (Cranelift):** IR Lowering $\rightarrow$ JIT Compilation $\rightarrow$ C-ABI hot-swapping.

---

## 4. Subsystem Specifications

### 4.1. The Frontend (Python DSL)
The system leverages Python's `typing.Annotated` and metaclasses to define constraints that are valid Python syntax but act as compiler directives.

```python
from lila import verify, Mut, Ref, Owned
from lila.types import Refined, u32

# Z3 Refinement: An integer strictly between 0 and 100
Percentile = Refined[u32, lambda x: (x >= 0) & (x <= 100)]

@verify
def apply_boost(base: Mut[Percentile], boost: Ref[u32]) -> None:
    # Z3 will flag a compile error if it cannot prove base + boost <= 100.
    # The developer MUST write a guard to pass compilation.
    if base.val + boost.val <= 100:
        base.val += boost.val
    else:
        base.val = 100
```

### 4.2. SSA Transformation Pipeline
Python variables are highly mutable. Z3, however, requires immutable logical propositions, and Cranelift requires explicit register definitions. To bridge this, the Rust core converts the dynamic AST into **Static Single Assignment (SSA)** form.

**Pipeline Steps:**
1.  **AST Extraction:** Rust (via `PyO3` / `RustPython-AST`) parses the targeted function.
2.  **Control Flow Graph (CFG):** The AST is mapped into basic blocks.
3.  **Dominance Frontiers & $\phi$-nodes:** Rust calculates dominance frontiers to insert $\Phi$ (Phi) nodes where control flow merges.
4.  **Versioning:** Variables are versioned so every assignment creates a new variable.

*Example Transformation:*
```python
# Original Python
x = 10
if condition:
    x = 20
return x
```
*Generated SSA Representation (Internal Rust struct):*
```text
b0:
  v_x_0 = 10
  branch condition ? b1 : b2
b1:
  v_x_1 = 20
  jump b3
b2:
  jump b3
b3:
  v_x_2 = phi(v_x_0 from b2, v_x_1 from b1)
  return v_x_2
```

### 4.3. Formal Verification (Z3 SMT Solver)
Once in SSA form, the IR is mapped directly into SMT-LIB format and fed to the Z3 solver. 
*   **Branch Path Analysis:** Every basic block translates to a logical constraint. The $\phi$-nodes are translated into Z3 `If-Then-Else` (ITE) expressions.
*   **Constraint Satisfaction:** Z3 attempts to find a model where a runtime fault (like an array index out of bounds) is `True`. If a satisfying model is found, the code is structurally unsafe, and compilation is aborted, returning the exact path that leads to the error.
*   *Benefit:* If Z3 proves the bounds, the Cranelift backend will omit standard runtime bounds-checking instructions, yielding maximum performance.

### 4.4. Affine Type System (Borrow Checker)
Parallel to Z3 verification, Rust analyzes the SSA for memory ownership violations based on the Python type hints (`Mut`, `Ref`, `Owned`).
*   **Move Semantics:** If `v_arg_0` (typed as `Owned`) is passed to another function, the SSA graph is marked such that any subsequent read of `v_arg_0` or its derivative versions raises a "Use-After-Move" compile error.
*   **Aliasing Rules:** Enforces that a variable can have multiple `Ref`s XOR a single `Mut` at any given basic block.

### 4.5. The Cranelift Backend
Once the SSA IR passes both Z3 and the Borrow Checker, it is lowered into **Cranelift IR (CLIF)**.
1.  **Direct Mapping:** Because the IR is already in SSA form, mapping to Cranelift’s `Value` types is virtually 1-to-1.
2.  **JIT Memory Allocation:** Cranelift compiles the CLIF to machine code in executable memory (e.g., using `mmap` with `PROT_EXEC`).
3.  **PyO3 Trampoline:** A C-callable function pointer is generated. Using PyO3, the original Python function object's `__call__` method is forcefully overwritten to invoke our native function pointer. 

---

## 5. Memory Model and ABI
To achieve C-level speeds, Lila code cannot interact with `PyObject`. 
1.  **Primitives:** Native mapping of `int`, `float`, and `bool` to hardware registers (`i64`, `f64`, `b1`).
2.  **Complex Data:** Arrays and Structs are implemented as custom Rust memory buffers exposed to Python via the standard `buffer` protocol (similar to NumPy). 
3.  **C-ABI:** The JIT function signature uses standard System V AMD64 ABI.

## 6. Constraints and Limitations
*   **Strict Subset:** Lila will only support a rigid, static subset of Python within `@verify` blocks. No dynamic typing, no `eval()`, no monkey-patching, no heterogeneous lists.
*   **Compilation Latency:** While Cranelift is fast, Z3 theorem proving is an NP-Complete problem. Heavily complex mathematical functions may experience a slight delay upon module initialization.
*   **Closed World Assumption:** JIT-compiled functions cannot dynamically call unverified standard Python functions (as that breaks the mathematical proofs).

---

## 7. Implementation Roadmap

### Phase 1: The Scaffolding (Weeks 1-2)
*   Define the Python `lila` DSL (`Mut`, `Ref`, `Refined`).
*   Implement the PyO3 hook to intercept the AST instead of executing the function.

### Phase 2: The SSA Engine (Weeks 3-5)
*   Build the Rust-based Python AST to Control Flow Graph (CFG) converter.
*   Implement the Dominator Tree algorithm to insert $\phi$-nodes.
*   Yield the finalized Rust SSA structures.

### Phase 3: The Verification Core (Weeks 6-8)
*   Integrate `z3-rs` (Rust bindings for Z3).
*   Write the translation layer: SSA $\rightarrow$ SMT constraints.
*   Implement the Borrow Checker pass over the SSA.

### Phase 4: Cranelift Execution (Weeks 9-12)
*   Integrate `cranelift-jit`.
*   Write the lowering pass: SSA $\rightarrow$ Cranelift IR.
*   Implement memory trampolines to execute the JIT code and pass results back to the CPython interpreter.

---
**Prepared By:** Systems Architecture Team  
**Next Steps:** Approve RFC and commence Phase 1 repository initialization.
