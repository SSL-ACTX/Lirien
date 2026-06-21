# Contributing to Lirien

Thanks for your interest. Lirien is an experimental research compiler — contributions are welcome, but the bar for correctness is high by design. Please read this document fully before opening a PR.

## Before You Start

- Check [STATUS.md](STATUS.md) and open issues to avoid duplicating work.
- For significant changes (new IR instructions, type system extensions, verifier logic), open an issue first to discuss the design. The architecture has non-obvious invariants that affect multiple layers simultaneously.

---

## Development Environment

### Prerequisites

| Tool | Version |
| :--- | :--- |
| Rust toolchain (stable) | 1.80 or later |
| Python | 3.10 or later |
| Z3 shared library | 4.12 or later |
| `maturin` | latest stable |

### Build Sequence

> [!IMPORTANT]
> Always follow this exact order. Never run `maturin develop` against a broken Rust workspace.

```bash
# 1. Verify the Rust workspace compiles cleanly.
cargo check

# 2. Run Clippy with warnings-as-errors.
cargo clippy -- -D warnings

# 3. Build and install the Python extension into the active environment.
maturin develop

# 4. Run the full test suite.
PYTHONPATH=./python python -m unittest discover tests/python
```

---

## Contribution Standards

### Zero Warnings

The project enforces a strict zero-warning policy. All Clippy lints must pass before a PR can be merged:

```bash
cargo clippy -- -D warnings
```

If your change introduces a lint that is genuinely a false positive, discuss it in the PR rather than suppressing it with `#[allow(...)]`.

### Full Test Suite

Never validate a change by running a single test file. The full suite must pass:

```bash
PYTHONPATH=./python python -m unittest discover tests/python
```

A change that breaks an unrelated test is a regression and will not be merged.

### Adding a New IR Instruction

Changes to the IR touch multiple layers. Follow this checklist in order:

1. `src/ssa/ir.rs` — update `Type` and `InstructionKind`
2. `src/builder/visitor/` — generate the new instruction from the AST
3. `src/builder/cfg.rs` — update the CFG layout engine if size/alignment changes
4. `src/verification/z3/` — extend the formal model (arithmetic, memory, or control flow)
5. `src/backend/cranelift/lower/` — lower the instruction to Cranelift IR
6. `src/ssa/optimization/dce.rs` and `type_propagation.rs` — handle the new instruction in optimization passes
7. `tests/python/` — write a Python integration test that exercises the full pipeline

### Z3 API Rules

The project uses `z3-rs` v0.20.0 with thread-local contexts. **Do not pass `&Context` to AST constructors.** The API uses `Context::thread_local()` internally. If you encounter a compiler error in the verification layer, it is not a missing context argument — it is a wrong method or a deprecated call.

### Diagnostics and Logging

- Do not use `println!` or `eprintln!` anywhere in the Rust codebase.
- Use the `tracing` crate: `info!(target: "lirien::subsystem", ...)`.
- All IR instructions must carry a `SourceLocation` to enable source-mapped error reporting.

### Python-Side Changes

When modifying the Python layer (decorators, FFI, type definitions):

- Include temporary `print` statements to trace data flow during debugging. Remove them before submitting.
- Do not use `sed`, `cat`, or shell scripts to patch files. Edit files directly.
- Do not expose `ctypes` internals to user-facing APIs. Users should never call `ctypes.pointer` or `ctypes.addressof` directly.

---

## Testing Standards

- Use the standard `unittest` library. Do not use `pytest`.
- Each test file must be independently runnable: `python test_foo.py`.
- Tests live under `tests/python/` organized by subsystem.
- Integration tests must exercise the full pipeline: Python DSL → SSA IR → Z3 → Cranelift → result.

---

## Pull Request Checklist

Before submitting:

- [ ] `cargo check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `maturin develop` succeeds
- [ ] `PYTHONPATH=./python python -m unittest discover tests/python` — full suite passes
- [ ] New behaviour is covered by at least one integration test
- [ ] No `println!` / `eprintln!` in Rust code
- [ ] No `ctx` arguments added to Z3 AST constructors

---

## License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0](LICENSE).
