# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## Roadmap & Future Directions

1. [ ] **Automated Loop Invariant Synthesis**: Researching Abstract Interpretation to automatically derive loop invariants.

## Completed Features
- [x] **Flat Value Types (`@value`)**: Support for stack-allocated, non-boxed structs and inline nested buffer allocations to reduce heap pressure.
- [x] **Zero-Cost ADTs**: Full support for variants with primitive, tuple, and None payloads.
- [x] **Optimized Match Dispatch**: Cranelift `switch` based jump tables for $O(1)$ variant dispatch.
- [x] **Z3-Backed Exhaustiveness Checking**: Formal proof that all ADT variants are handled in `match` blocks.
- [x] **Pattern Guards (`if` clauses)**: Integration of Liquid Type refinements into `match` cases.
- [x] **Recursive ADTs (Boxed Variants)**: Support for heap-allocated recursive data structures (e.g., Linked Lists).
- [x] **Nested Pattern Matching**: Recursive destructuring of nested ADTs and structs.
- [x] **Interval Analysis**: Optimized range tracking to skip redundant Z3 solver calls for simple proofs.
- [x] **Native SIMD Support**: High-performance vector types (`f32x4`, `i32x4`, etc.) with native CPU lowering.
- [x] **IEEE 754 Floating-Point Verification**: Formal proofs for float operations, including div-by-zero and domain checks.
- [x] **GIL-less Parallelism**: `parallel_for` on raw memory buffers.
- [x] **Liquid Types**: Base support for formal verification with Z3.
- [x] **Type-Level Arithmetic (`N + 1`)**: Symbolic shape constraints and arithmetic on TypeVars for rank-polymorphic generic code.
- [x] **Zero-Copy Array Slicing (`arr[start:end]`)**: First-class IR support for memory views, fully verified by Z3 to be safe and in-bounds.
- [x] **Pythonic Result ADTs**: Generic `Result[T, E]` type with `Ok()` and `Err()` constructors, leveraging optimized match dispatch.
- [x] **Cross-Function Type Inference**: Global registry for specialized function signatures, enabling zero-annotation composition.
- [x] **Literal Types & Loop Unrolling**: Hijacking `typing.Literal` for "Const Generics" and AST-level unrolling.
- [x] **Monomorphization (TypeVar)**: C++ style templates using Python's `TypeVar` for zero-overhead generic code.
- [x] **Centralized Granular Tracing**: Rust & Python tracing system.
