# Roadmap: Unified Verification Pass (Liquid Types + SSA + Z3)

## 1. Abstract
This document outlines the transition of Project Lila's verification system from a multi-phase, static analysis pipeline into a unified constraint-solving graph. By merging Static Single Assignment (SSA) dataflow, Liquid Types (refinement inference), and Z3 SMT solving into a single pass, Lila will achieve more powerful type inference and robust formal verification.

## 2. Current Architecture vs. Proposed Architecture

**Current State:**
1.  Liveness & Interval Analysis (Dataflow facts computed statically).
2.  Fractional Permission tracking.
3.  Type propagation (resolving basic types like `i32`, `f64`).
4.  Z3 Translation (building models based on prior static passes).
*Limitation:* The separation means type inference cannot benefit from SMT-level logic, and SMT verification cannot easily guide type inference.

**Proposed State:**
*   **Unified Constraint Graph:** The SSA graph becomes a constraint graph. Every instruction emits both dataflow constraints (for execution) and logical constraints (for verification).
*   **Liquid Types:** Variables carry refinement types (e.g., `{v: i32 | v > 0}`). These predicates are propagated through the graph and merged at $\phi$-nodes using Z3 `If-Then-Else` (ITE) logic.
*   **Single-Pass SMT Solving:** Instead of separate interval and type analyses, the entire graph of refinements and permissions is encoded into Z3. Z3 solves the constraints to simultaneously prove safety and infer the strongest possible refinement types for intermediate variables.

## 3. Implementation Phases

### Phase 1: SSA Augmentation (Constraint Nodes)
*   Modify `src/ssa/ir.rs` to allow instructions to carry logical predicates alongside basic types.
*   Extend the `Type` enum to represent Refinement Types (`Type::Refined(Box<Type>, Constraint)`).
*   Ensure branch conditions (e.g., `if x < 10:`) inject constraints into the dominance frontiers of their respective blocks.

### Phase 2: Refinement Propagation
*   Update `src/ssa/optimization/type_propagation.rs`. When types are propagated, combine their refinement constraints.
*   For arithmetic operations, generate verification conditions (VCs). For example, `z = x + y` yields a type `{v | v == x + y}`.

### Phase 3: Unified Z3 Translation
*   Refactor `src/verification/z3/mod.rs` and `src/verification/mod.rs`.
*   Remove the dependency on the static `interval::analyze`. Rely directly on Z3's arithmetic solver to prove bounds.
*   Encode Fractional Permissions directly as SMT constraints (e.g., tracking the sum of permissions for a resource at each SSA node).

### Phase 4: Liquid Type Inference (Predicate Abstraction)
*   Implement a feedback loop: If Z3 requires a specific invariant at a $\phi$-node to prove memory safety (e.g., an array bound), extract that invariant and apply it as an inferred Liquid Type for the variable.
*   This reduces the need for developers to manually annotate every intermediate variable.

## 4. Benefits for Lila
*   **Developer Experience:** The "Python x Rust" DSL becomes magically smart. SMT-guided type inference means less boilerplate for the user.
*   **Mathematical Soundness:** Eliminating the gap between dataflow passes and SMT solving prevents edge cases where static analysis misses a dynamic invariant that Z3 could have proven.
*   **Performance:** A unified pass reduces redundant graph traversals.

## 5. Next Steps
1.  Begin extending the `Type` system in `src/ssa/ir.rs` to support `Refined` variants.
2.  Draft a prototype Z3 encoding for a simple SSA $\phi$-node merging two different refinement constraints.
