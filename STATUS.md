# Lila Project Status & Roadmap

Lila is an experimental formal verification and JIT compiler for a safe subset of Python.

## Ongoing Development & Plans

### 1. Advanced Verification Features
- [ ] **Liquid Type Inference (Predicate Abstraction)**: Implement a feedback loop where Z3-required invariants at $\phi$-nodes are automatically extracted and applied as inferred Liquid Types.
- [ ] **Automated Loop Invariant Synthesis**: Researching Abstract Interpretation to automatically derive loop invariants for complex numerical algorithms.

### 2. Concurrency & Performance
- [x] **GIL-less Parallelism**: Support for high-performance multi-threading via `parallel_for` on raw memory buffers.
- [ ] **SIMD Auto-vectorization**: Leverage Cranelift's SIMD capabilities for verified buffer operations.

### 3. Developer Ergonomics
- [x] **Centralized Granular Tracing**: Implemented a controllable tracing system (Rust & Python) allowing component-specific debug levels.
- [ ] **Refinement Diagnostics**: Improved source-mapping for verification failures to provide precise location info in Python code.
