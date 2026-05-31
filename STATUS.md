# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## Ongoing Development & Plans

### 1. Advanced Verification Features
- [ ] **Structural (Field-Level) Permissions**: Enable disjoint borrowing of composite types by tracking hierarchical Z3 path constraints.
- [ ] **Liquid Type Inference (Predicate Abstraction)**: Implement a feedback loop where Z3-required invariants at $\phi$-nodes are automatically extracted and applied as inferred Liquid Types.

### 2. Concurrency & Performance
- [ ] **Static Data-Race Freedom**: Introduce `lila.parallel_for` and use fractional permissions to statically prove that concurrent tasks do not cause data races.
- [ ] **SIMD Auto-vectorization**: Leverage Cranelift's SIMD capabilities for verified buffer operations.

### 3. Developer Ergonomics
- [ ] **Non-Lexical Lifetimes (NLL)**: Use SMT-guided liveness to automatically infer borrow lifetimes and eliminate redundant `Mut`/`Ref` annotations.
- [ ] **Python Context Managers**: Support `with` blocks for explicit, safe scoped borrowing.
