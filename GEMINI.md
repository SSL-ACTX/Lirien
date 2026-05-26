# Project L.I.L.A.: Development Rules & Architecture

This document outlines the core architectural constraints, design patterns, and development workflows for Project Lila. These rules must be strictly adhered to during all future development to maintain mathematical soundness, type safety, and the "Python x Rust" developer experience.

## 1. Architectural Constraints

### 1.1 Formal Verification First
*   **Permission Verifier:** A flow-sensitive system using **Fractional Permissions** ($Exclusive=1.0, Shared \in (0, 1)$) to prove absence of aliasing violations and use-after-moves. Safety must be mathematically provable in Z3 using `Context::thread_local()`.
*   **Symbolic Partitioning:** Shared weights are generated as symbolic constants; Z3 proves safety by confirming a valid partitioning exists where $\sum \text{weights} \le 1.0$.

### 1.2 Strict SSA (Static Single Assignment)
The Intermediate Representation (IR) enforces strict SSA form.
*   **Values are immutable.** Once a `Value` is assigned via `get_def`, it can never be reassigned.
*   Control flow joins must be handled explicitly via `Phi` nodes.
*   Do not mutate variables in the IR; instead, create a new `Value` and update the block's `variable_defs` mapping.

### 1.3 Absolute Memory Layouts (No Pointer Aliasing)
Lila structs are compiled to flat, C-compatible memory layouts.
*   **Use Byte Offsets:** Nested structures are **inline**. The IR must calculate the absolute byte offset from the root object (`StructOffset` or `StructLoad`) rather than chaining pointer dereferences.
*   **Never clobber pointers:** A `StructSet` modifies the value at an offset; it does *not* overwrite the root object's pointer address.

## 2. Python DSL Guidelines

### 2.1 Zero-Boilerplate Experience
The Python-side DSL must feel like native Python, hiding all low-level C-ABI details.
*   **No explicit `ctypes` in user code:** Users should never have to manually call `ctypes.pointer` or `ctypes.addressof`.
*   **Native Types:** Always use Lila-native types (`i64`, `u8`, `f32`, `Mut`, `Ref`) in annotations. Do not expose `ctypes.c_int64` to the user.
*   **Automatic Unwrapping:** The `@verify` decorator must automatically resolve Lila objects to their underlying memory buffers before calling the JIT function.

### 2.2 Struct Generation
*   The `@struct` decorator dynamically generates a `ctypes.Structure` class behind the scenes.
*   Nested structs are inlined by placing the child's `__lila_ctypes__` directly into the parent's `_fields_` list.

## 3. Rust Codebase Rules

### 3.1 Verification Workflow
*   **Rust First:** Always run `cargo check` and `cargo test` **before** running `maturin develop`. Never attempt to build the Python module if the Rust core is in an inconsistent or failing state.

### 3.2 Centralized Diagnostics
*   Do not use `println!` or `eprintln!`.
*   Use the `tracing` crate for all logging (e.g., `info!(target: "lila::jit", ...)`).
*   All IR instructions must carry a `SourceLocation` to map errors back to the Python source line.

### 3.3 Adding New Instructions
When adding a new capability to the DSL:
1.  Update `Type` and `InstructionKind` in `src/ssa/ir.rs`.
2.  Update the AST `visitor.rs` to generate the new IR.
3.  Update the `CFGBuilder` layout engine if the size/alignment changes.
4.  Update the Z3 formal model in `src/verification/z3/` (arithmetic, memory, or control_flow).
5.  Update the Cranelift lowering in `src/backend/cranelift/lower/`.
6.  Update `src/ssa/optimization/dce.rs` and `type_propagation.rs` to ensure the instruction is handled.
7.  Write a Python integration test verifying the full pipeline.

### 3.4 Z3 0.20 API Usage
The project uses `z3-rs` v0.20.0, which has been configured/modified for ergonomics using `thread_local` contexts.
*   **No Explicit Context:** Most AST constructor methods (e.g., `BV::from_i64`, `Int::from_i64`, `Bool::from_bool`, `BV::new_const`) do **not** take a `&Context` argument. They use `Context::thread_local()` internally.
*   **Comparison Methods:** Use standard `eq`, `lt`, `gt`, etc.
*   **Verification Entry:** The Z3 context is initialized in `src/verification/mod.rs` via `Context::thread_local()`.
