# Lirien — Project Status

Lirien is an experimental research compiler. It is under active development and is not production-ready. This document tracks the current implementation status and planned work.

---

## Active Development

| Area | Status |
| :--- | :--- |
| SSA IR & CFG builder | Stable |
| Z3 formal verification | Stable |
| Cranelift code generation | Stable |
| Python DSL & FFI layer | Stable |
| AOT IR caching | Stable |
| Flow-sensitive type narrowing | Stable |
| Optimization passes (DCE, constant folding, type propagation) | Stable |

---

## Implemented Features

### Type System
- Refinement types (`Refined[T, pred]` / `Annotated[T, pred]`) with Z3-discharged predicates across all reachable paths
- Symbolic refinement DSL (`V`) — point-free predicate expressions without explicit `lambda`
- Automatic postcondition inference via `Refined[T, ...]` using interval analysis
- Monomorphization via `TypeVar` — zero-overhead generics specialized per call site
- Const generics — integer `TypeVar` dimensions with symbolic arithmetic (`N + 1`)
- Variadic generics via `TypeVarTuple` and `Unpack` for rank-polymorphic functions
- `typing.Protocol` static dispatch — monomorphized for both `@struct` and `@adt` types
- `typing.overload` multiple dispatch — per-signature machine-code specialization
- `typing.Literal` loop unrolling — compile-time integer constants with exact Z3 induction values
- `TypedDict` zero-cost struct layout — string key access compiled to byte offsets
- Non-pointer value-type optionals (`T | None` / `Optional[T]`) using inline tagged layout (has_value tag + value)

### Memory & Data Structures
- `@struct` / `@value` — flat, C-ABI-compatible layouts with inlined nested structs
- `@adt` — tagged unions with O(1) Cranelift `switch`-based variant dispatch
- `Box[T]` — heap-allocated pointer type
- `Optional[Box[T]]` / `Box[T] | None` — null-pointer optimization (raw 64-bit pointer, `None` = `0x0`)
- Flow-sensitive smart casts — automatic type narrowing after `is None` / `is not None` guards for both pointer and value-type optionals
- `Tuple` / `NamedTuple` — recursively register-flattened; SRet convention for aggregates > 16 bytes
- `SizedArray[T, N]` — statically-sized arrays with Z3-verified index bounds
- `Buffer[T]` / `Buffer[...]` — Python buffer protocol interop with zero-copy slicing (including strided slicing `arr[start:end:step]`) and direct iteration
- `Tensor[T, *Shape]` — rank-polymorphic tensors with type-level shape tracking and verified 2D matrix multiplication (`@` operator)
- `Result[T, E]` — generic result type with `Ok` / `Err` variants
- `List[T]` — heap-allocated dynamic lists with bounds checking and length semantics modeled and verified by Z3

### Verification
- Arithmetic safety — division by zero, overflow
- Memory safety — null dereference, out-of-bounds access
- Refinement predicate checking across all CFG paths
- Match exhaustiveness — Z3 proves all `@adt` variants are handled
- Match guards (`case Pattern if condition:`) — guards encoded as SMT constraints
- Inductive reasoning for recursive functions
- Interval analysis — skips Z3 solver calls for trivially provable constraints
- Automated loop invariant synthesis — abstract interpretation to automatically derive loop invariants and entry-edge implication constraints
- Design by Contract — native `assert` statements promoted to preconditions, postconditions, and inductively verified loop invariants

### Code Generation
- Cranelift JIT backend — native machine code in executable memory
- C-ABI trampoline via PyO3 — verified functions replace the Python callable directly
- SIMD types — `f32x4`, `f64x2`, `i8x16`, `u8x16`, `i16x8`, `u16x8`, `i32x4`, `i64x2`
- GIL-free `parallel_for` on raw memory buffers
- IR-level kernel fusion — folds chains of element-wise tensor arithmetic operations to avoid intermediate allocations

### Developer Tooling
- `@jit` decorator — Cranelift compilation without Z3 verification
- `no_verification()` — thread-local context manager to disable Z3 for a block
- `tracing()` — nestable context manager for per-subsystem structured log output
- `configure_tracing()` — persistent global tracing configuration
- Source-mapped diagnostics — `SourceLocation` on every IR instruction
- AOT IR caching — `seahash`-keyed `.lir` binaries in `.lirien_cache/`, skip re-verification on hit
- Auto-derived `__repr__` and `__eq__` on `@struct`, `@value`, and `@adt` types

---

## Roadmap

- [x] Design by Contract (Native Assert Pattern)
- [ ] Custom Error Messages for Native Asserts
- [ ] Tuple/Aggregate Unpacking & Destructuring Assignment
- [ ] Intermediate Inline Assertions (Safety Proofs)
- [ ] Pattern Matching Destructuring

---

## Known Limitations

- **Closed-world assumption.** Dynamic attribute access (`getattr`, `setattr`, `__dict__`), `eval()`, and `exec()` are not supported. All parameters and return types require explicit annotations.
- **Compiler is not formally verified.** The Rust implementation (SSA builder, Z3 encoding, Cranelift lowering) has not been proven correct. See [SECURITY.md](SECURITY.md) for the full threat model.
