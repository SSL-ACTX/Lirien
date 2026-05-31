# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## Current Capabilities
- [x] Basic Arithmetic & Bitwise Ops
- [x] Floating-Point Precision Verification (IEEE 754 bit-accurate)
- [x] Simple Control Flow (`if`, `while`)
- [x] Memory Safety (Buffer/Array bounds checking)
- [x] Fractional Permissions (Aliasing & Move semantics)
- [x] Composite Types (`struct`, `enum`, `tuple`)
- [x] Refined Types (`Refined[T, lambda x: ...]`)
- [x] JIT Compilation (via Cranelift)

## Roadmap

### 1. Enhanced Control Flow (Completed)
- [x] **For-Loops and Range Iteration**: Support `for i in range(...)` by lowering to SSA while-loops with automatic invariant generation.
- [x] **Early Returns**: Support `return` from within loops and nested `if` blocks.
- [x] **Break/Continue**: Support loop control statements.

### 2. Usability Improvements (In Progress)
- [x] **Structural Pattern Matching**: Support Python `match` statements for Enums.
- [x] **Type Inference**: Reduce the need for explicit type annotations on local variables.
- [x] **Improved Diagnostics**: Better error messages for verification failures with source-level highlights.
- [x] **Refactoring**: Extract the monolithic `@verify` decorator logic in `compiler.py` into private module-level helper functions for better maintainability and testability.

### 3. Advanced Features
- [x] **Recursive Functions**: Support verification of recursive logic (using inductive reasoning and interval-assisted overflow proofs).
- [x] **Higher-Order Functions**: Basic support for lambdas and passing functions.
- [ ] **SIMD Auto-vectorization**: Leverage Cranelift's SIMD capabilities for verified buffer operations.
