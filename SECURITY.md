# Security Policy

## Scope of Formal Guarantees

Lirien is a *verifying compiler*. For programs it accepts, it uses Z3 to formally prove the absence of:

- Division by zero
- Out-of-bounds buffer accesses
- Null pointer dereferences (on `Optional[Box[T]]` / `Box[T] | None` types)
- Refinement type predicate violations across all reachable control-flow paths

These guarantees apply to the **input program**, not to the compiler implementation itself.

## What Lirien Does NOT Guarantee

> [!WARNING]
> Lirien is an experimental research compiler and is **not production-ready**.

- **The compiler is not formally verified.** The Rust codebase (SSA builder, Z3 encoding, Cranelift lowering) has not been proven correct against a formal specification. A bug in any of these layers could allow an unsafe program to pass verification.
- **Closed-world assumption.** Lirien only accepts a restricted subset of Python. Code that uses `getattr`, `setattr`, `eval()`, `exec()`, or unannotated functions is rejected. Mixing verified Lirien functions with arbitrary CPython code in ways that violate memory assumptions is outside the safety envelope.
- **Z3 soundness.** The formal guarantees are contingent on the correctness of Z3 itself and on the accuracy of Lirien's SMT encoding. Neither is unconditionally guaranteed.
- **Native code safety.** JIT-compiled functions run as native machine code. A bug in the Cranelift lowering pass could produce incorrect or unsafe machine instructions even for a formally verified IR.

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `main` branch | ✅ Active development |
| Tagged releases | ✅ Best-effort |
| Older branches | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately via email:

**seuriin@gmail.com**

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce (minimal Python snippet or Rust test case if applicable)
- Whether the issue affects the verifier (Z3 encoding), the code generator (Cranelift lowering), or the Python FFI layer

You can expect an acknowledgement within **72 hours**. Given the experimental nature of the project, fixes will be prioritised based on severity and exploitability.

## Threat Model

Lirien's threat model is **not** adversarial input from untrusted users. It is designed as a developer tool — the person writing `@verify` functions is assumed to be the same person operating the compiler. Security reports are primarily relevant to:

1. **Verifier unsoundness** — a program that should fail verification passes, leading to unsafe native code
2. **Memory safety bugs in the compiler itself** — crashes, use-after-free, or undefined behaviour in the Rust codebase
3. **FFI boundary violations** — incorrect ctypes marshalling that corrupts memory across the Python/native boundary
