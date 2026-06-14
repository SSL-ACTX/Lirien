# Project L.I.L.A.: Development Rules & Architecture

This document outlines the core architectural constraints, design patterns, and development workflows for Project Lila. These rules must be strictly adhered to during all future development to maintain mathematical soundness, type safety, and the "Python x Rust" developer experience.

## 1. Architectural Constraints

### 1.1 Formal Verification First
*   **Logic Verification:** A path-aware, flow-sensitive system using Z3 to prove the absence of logical errors, out-of-bounds accesses, and refinement type violations. Safety must be mathematically provable in Z3 using `Context::thread_local()`.
*   **Refinement Types:** Lila uses refinement types (liquid types) to attach logical predicates to data. Z3 verifies that all assignments and function calls satisfy these predicates across all reachable paths.

### 1.2 Strict SSA (Static Single Assignment)
The Intermediate Representation (IR) enforces strict SSA form.
*   **Values are immutable.** Once a `Value` is assigned via `get_def`, it can never be reassigned.
*   Control flow joins must be handled explicitly via `Phi` nodes.
*   Do not mutate variables in the IR; instead, create a new `Value` and update the block's `variable_defs` mapping.

### 1.3 Absolute Memory Layouts (No Pointer Aliasing)
Lila structs are compiled to flat, C-compatible memory layouts.
*   **Use Byte Offsets:** Nested structures are **inline**. The IR must calculate the absolute byte offset from the root object (`StructOffset` or `StructLoad`) rather than chaining pointer dereferences.
*   **Never clobber pointers:** A `StructSet` modifies the value at an offset; it does *not* overwrite the root object's pointer address.

### 1.4 Zero-Cost Static Dispatch (`Protocol`)
*   **Specialization over Virtualization:** Lila uses `typing.Protocol` for static dispatch. The monomorphization engine must clone and specialize functions for every unique struct type passed to a Protocol parameter.
*   **Static Call Mapping:** The IR builder must resolve Protocol method calls to direct `Call` instructions using the mangled name `ClassName_methodName`.

### 1.5 Null-Pointer Optimization (`Box[T] | None`)
*   **Zero-Overhead Optionals:** `Optional[Box[T]]` and `Box[T] | None` must be represented as raw 64-bit pointers where `None` is `0x0`.
*   **Mandatory Verification:** The Z3 verifier MUST prove non-nullity before any `PointerLoad` or `PointerStore` instruction. The IR builder must automatically insert these checks for `.val` or field access.

## 2. Python DSL Guidelines

### 2.1 Zero-Boilerplate Experience
The Python-side DSL must feel like native Python, hiding all low-level C-ABI details.
*   **No explicit `ctypes` in user code:** Users should never have to manually call `ctypes.pointer` or `ctypes.addressof`.
*   **Native Types:** Always use Lila-native types (`i64`, `u8`, `f32`, `bool`) in annotations. Do not expose `ctypes.c_int64` to the user.
*   **Automatic Unwrapping:** The `@verify` decorator must automatically resolve Lila objects to their underlying memory buffers before calling the JIT function.

### 2.2 Struct Generation
*   The `@struct` decorator dynamically generates a `ctypes.Structure` class behind the scenes.
*   Nested structs are inlined by placing the child's `__lila_ctypes__` directly into the parent's `_fields_` list.

## 3. Rust Codebase Rules

### 3.1 Verification Workflow
*   **Rust First:** Always run `cargo check` **before** running `maturin develop`. Never attempt to build the Python module if the Rust core is in an inconsistent or failing state.
*   **Maturin Reflection:** AFTER changes to the Rust core are verified, you MUST run `maturin develop` to reflect these changes in the Python environment. It won't magically reflect. Gemini AI is dumb as hell so it has to be taught like a damn kid—never forget this step.
*   **Tool Usage:** Using Python scripts, `sed`, or `cat` to edit or update files is strictly banned. Use native tool functions like `replace` or `write_file` directly.

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

### 3.4 Z3 0.20 API Usage (NO CONTEXT BULLSHIT)
*   **NO EXPLICIT CONTEXT:** The project uses `z3-rs` v0.20.0, which is configured for ergonomics using `thread_local` contexts. You MUST NOT pass `&Context` to AST constructors (e.g., `BV::from_i64`, `Bool::new_const`). It uses `Context::thread_local()` internally.
*   **STOP TRYING TO ADD CTX:** If you see a compiler error, it is NOT because you missing a context argument. It is because you are using the wrong method or a deprecated one. Do not ever re-introduce `ctx` arguments to the logic layers. This rule is absolute and permanent.

### 3.5 Development Discipline
*   **ZERO WARNINGS:** The project maintains a strict zero-warning policy. ALL compiler warnings and Clippy lints MUST be resolved before committing. Use `cargo clippy -- -D warnings` to verify.
*   **FULL TEST SUITE:** Never assume a refactor is correct by running a single subfolder of tests. You MUST run the entire suite using `PYTHONPATH=./python python3 -m unittest discover tests/python`.
*   **NO SHORTCUTS:** Validation is not "it compiled". Validation is "all 100+ integration tests passed".


## 4. Testing Standards

### 4.1 Python Tests (unittest)
*   **Framework:** Use the standard Python `unittest` library. Do not use `pytest`.
*   **Execution:** Run tests using `PYTHONPATH=./python python -m unittest discover tests/python`.
*   **Structure:** Each test file must be runnable directly (e.g., `if __name__ == "__main__": unittest.main()`).

## 5. AI Agent Execution Mandates (STRICT & NON-NEGOTIABLE)

### 5.1 OBEY THE USER EXACTLY
*   **The User's Diagnosis is Absolute Law:** If the user tells you a bug is in the `git diff`, or points you to a specific file or cause, you drop your generic debugging bullshit and investigate exactly what they told you IMMEDIATELY. The user's diagnosis is the highest signal.
*   **Do NOT Stall or Loop:** Do NOT fall back on generic heuristics, infinite test running scripts, or over-analyzing irrelevant files to "buy time". If your automated approach fails or infinite-loops, and the user gives you a direct hint, immediately pivot and follow their explicit path.

### 5.2 READ BEFORE ACTING
*   **Consult This File FIRST:** You must read and abide by the rules in this `GEMINI.md` file before attempting to execute tasks. Do not assume standard generic workflows (like running raw `cargo build` or `maturin develop` without verifying the project's exact required sequence).
*   **Acknowledge Flaws:** If you fail to follow these rules, recognize that it is a critical flaw and fundamentally dangerous to the codebase. Do not bullshit the user or feign blindness.
