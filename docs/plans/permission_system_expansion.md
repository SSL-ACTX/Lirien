# Expansion of Lila's Fractional Permission System

## Background & Motivation
Lila's current fractional permission system successfully proves the absence of aliasing violations and use-after-moves by dynamically partitioning permissions ($Exclusive=1.0, Shared \in (0, 1)$) at the root variable level using Z3. 

To elevate Lila to a fully-featured, safe systems programming environment, we will expand this system across three major pillars: **Structural Permissions**, **Data-Race Freedom**, and **Developer Ergonomics (NLL)**.

## Scope & Impact
This expansion will impact the Python DSL, the SSA Intermediate Representation (IR), and the Z3 SMT translation layer. It will enable users to write complex, parallelized Python code that is mathematically guaranteed to be free of data races and pointer aliasing bugs.

## Proposed Solution: The Three Pillars

### Pillar 1: Structural (Field-Level) Permissions
**Objective:** Allow disjoint borrowing of composite data types (e.g., mutably borrowing `struct.field_a` while separately borrowing `struct.field_b`).
*   **IR Representation:** Evolve the `value_roots` map from `Value -> RootValue` to `Value -> (RootValue, AccessPath)`.
*   **Z3 Encoding:** Implement a hierarchical permission tree in Z3. The sum of permissions for any child node cannot exceed the permission held by its parent, and disjoint sibling paths are tracked independently.
*   **Impact:** Solves "coarse-grained borrowing" issues common in early Rust, allowing complex state mutations without false-positive alias rejections.

### Pillar 2: Concurrency & Static Data-Race Freedom
**Objective:** Provide a mechanism for safe, multi-threaded execution.
*   **Python DSL:** Introduce a `lila.parallel_for` intrinsic.
*   **Verification Strategy:** Leverage the fractional math natively. If a loop body mutates a captured variable, it requires a permission of `1.0`. For a `parallel_for`, Z3 will attempt to partition the available permission across $N$ concurrent loop iterations. Since $N * 1.0 > 1.0$, Z3 will correctly reject concurrent mutation of shared state, mathematically proving data-race freedom. Shared reads require only a fraction $\epsilon$, where $N * \epsilon \le 1.0$, which is provably safe.
*   **Backend:** Lower `parallel_for` to Cranelift threads/tasks (potentially using a lightweight thread pool).

### Pillar 3: Ergonomics & Non-Lexical Lifetimes (NLL)
**Objective:** Minimize the need for explicit `Mut` and `Ref` type annotations, making Lila feel like idiomatic Python.
*   **NLL Inference:** Combine existing SSA liveness analysis with Z3 constraint solving to infer the lifetimes of borrows. If a mutable borrow is no longer "live" (no future uses in the CFG), its permission fraction is automatically returned to the parent, enabling subsequent borrows without explicit release statements.
*   **Context Managers:** Introduce Python `with` block support (e.g., `with borrow_mut(obj) as b:`) for cases where developers want to explicitly scope a borrow, lowering directly to `Acquire` and `Release` IR instructions.

## Phased Implementation Plan

1.  **Phase 1: Path-Aware IR**
    *   Update `src/ssa/ir.rs` and the CFG builder to track access paths for `StructOffset` and `ArrayLoad`.
    *   Refactor `src/verification/permissions/mod.rs` to assert hierarchical Z3 constraints.
2.  **Phase 2: Ergonomics & `with` Blocks**
    *   Add AST parsing for `with` blocks and lower them to explicit borrow regions.
    *   Refine the liveness-based permission release logic.
3.  **Phase 3: The Parallel Frontier**
    *   Add `parallel_for` to the Python DSL and SSA IR.
    *   Implement the concurrency Z3 constraints (multiplying required permissions by $N$).
    *   Implement the Cranelift backend multithreading runtime.

## Alternatives Considered
*   **Separation Logic natively in Z3:** Instead of fractions, use arrays to model the heap and use formal Separation Logic. *Discarded:* Z3 handles arithmetic fractions (Real numbers) much faster than complex quantified array logic. Fractional permissions give us ~90% of Separation Logic's power at a fraction of the compilation time.
*   **Global Interpreter Lock (GIL):** *Discarded:* Lila is designed to bypass the GIL. We must prove thread-safety statically.

## Verification & Rollback
*   Each phase will be accompanied by exhaustive `unittest` coverage in `tests/python/memory/`.
*   A failure in Phase 3 (Parallelism) will not impact the serial execution safety guaranteed by Phase 1 and 2.
