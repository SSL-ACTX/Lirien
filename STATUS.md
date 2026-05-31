# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## Ongoing Development & Plans

### 1. Advanced Verification Features
- [ ] **Liquid Type Inference (Predicate Abstraction)**: Implement a feedback loop where Z3-required invariants at $\phi$-nodes are automatically extracted and applied as inferred Liquid Types, reducing manual annotation requirements.
- [ ] **Enhanced Inductive Reasoning**: Expand the Inductive Hypothesis engine to handle more complex recursive patterns and multi-variable induction.

### 2. Performance Optimizations
- [ ] **SIMD Auto-vectorization**: Leverage Cranelift's SIMD capabilities for verified buffer operations, enabling high-performance parallel data processing.
- [ ] **Global IR Optimizations**: Implement cross-block optimizations (e.g., GVN, LICM) that are aware of refinement-based invariants.
