# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## ADT Expansion Roadmap

1. [ ] **Jump Table Optimization ($O(1)$ dispatch)**: Use Cranelift's `switch` instruction to emit a native assembly jump table for `match` statements, bypassing the current branch chain.
2. [ ] **Z3-Backed Exhaustiveness Checking**: Prove via Z3 that all possible ADT variants are handled in a `match` block.
3. [ ] **Nested Pattern Matching**: Support recursive destructuring of nested ADTs and structs within `match` cases.
4. [ ] **Pattern Guards (`if` clauses)**: Integrate Liquid Type refinements into `match` cases via `if` conditions.
5. [ ] **Recursive ADTs (Boxed Variants)**: Support heap-allocated recursive data structures like linked lists and trees.

## Completed Features
- [x] **Zero-Cost ADTs**: Support for variants with primitive, tuple, and None payloads.
- [x] **Basic Pattern Matching**: Destructuring support for single-level ADT payloads.
- [x] **GIL-less Parallelism**: `parallel_for` on raw memory buffers.
- [x] **Liquid Types**: Base support for formal verification with Z3.
- [x] **Centralized Granular Tracing**: Rust & Python tracing system.
