use crate::ssa::analysis::liveness::LivenessAnalysisResults;
use crate::ssa::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::{HashMap, HashSet};
use z3::ast::{Bool, Real};
use z3::{SatResult, Solver};

pub struct PermissionVerifier<'a> {
    func: &'a Function,
    // Maps each Value to the Root it originates from.
    value_roots: HashMap<Value, Value>,
    uid: usize,
}

impl<'a> PermissionVerifier<'a> {
    pub fn new(func: &'a Function) -> Self {
        let mut verifier = Self {
            func,
            value_roots: HashMap::new(),
            uid: 0,
        };
        verifier.analyze_roots();
        verifier
    }

    pub fn set_uid(&mut self, uid: usize) {
        self.uid = uid;
    }

    fn analyze_roots(&mut self) {
        // 1. Initial roots: all Held values and arguments
        for i in 0..self.func.arg_count {
            let v = Value(i);
            self.value_roots.insert(v, v);
        }
        for i in 0..self.func.value_count {
            let v = Value(i);
            if let Type::Held(_) = self.func.get_type(v) {
                self.value_roots.insert(v, v);
            }
        }

        // 2. Fixpoint propagation for derived values
        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.func.blocks {
                for inst in &block.instructions {
                    if let Some(def) = inst.get_def() {
                        let mut root = None;
                        match &inst.kind {
                            InstructionKind::Peek(_, src)
                            | InstructionKind::Hand(_, src)
                            | InstructionKind::StructOffset(_, src, _)
                            | InstructionKind::StructLoad(_, src, _)
                            | InstructionKind::ArrayLoad(_, src, _)
                            | InstructionKind::BufferLoad(_, src, _)
                            | InstructionKind::TupleExtract(_, src, _)
                            | InstructionKind::EnumExtract(_, src, _)
                            | InstructionKind::StructSet(_, src, _, _, _)
                            | InstructionKind::ArrayStore(_, src, _, _, _)
                            | InstructionKind::BufferStore(_, src, _, _, _) => {
                                root = self.value_roots.get(src).copied();
                            }
                            InstructionKind::Phi(_, mappings) => {
                                for v in mappings.values() {
                                    if let Some(r) = self.value_roots.get(v) {
                                        root = Some(*r);
                                        break;
                                    }
                                }
                            }
                            _ => {
                                let ty = self.func.get_type(def);
                                if ty.is_pointer_like() && !self.value_roots.contains_key(&def) {
                                    root = Some(def);
                                }
                            }
                        }

                        if let Some(r) = root {
                            if self.value_roots.insert(def, r) != Some(r) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn is_consuming(&self, inst: &InstructionKind, val: Value) -> bool {
        let ty = self.func.get_type(val);
        if !matches!(ty, Type::Held(_)) {
            return false;
        }

        match inst {
            InstructionKind::Call(_, _, args) => args.contains(&val),
            InstructionKind::Return(Some(v)) => *v == val,
            InstructionKind::StructSet(_, _, _, v, _) => *v == val,
            InstructionKind::ArrayStore(_, _, _, v, _) => *v == val,
            InstructionKind::BufferStore(_, _, _, v, _) => *v == val,
            InstructionKind::TupleCreate(_, elts) => elts.contains(&val),
            InstructionKind::EnumCreate(_, _, _, Some(v)) => *v == val,
            _ => false,
        }
    }

    fn requires_exclusive(&self, inst: &InstructionKind, val: Value) -> bool {
        match inst {
            InstructionKind::StructSet(_, obj, _, _, _) => *obj == val,
            InstructionKind::ArrayStore(_, arr, _, _, _) => *arr == val,
            InstructionKind::BufferStore(_, buf, _, _, _) => *buf == val,
            InstructionKind::Hand(_, src) => *src == val,
            _ => false,
        }
    }

    pub fn generate_assertions(
        &self,
        solver: &Solver,
        liveness: &LivenessAnalysisResults,
        value_perms: &HashMap<Value, Real>,
        block_conditions: &HashMap<BlockId, Bool>,
    ) -> Result<(), String> {
        let zero = Real::from_rational(0, 1);
        let one = Real::from_rational(1, 1);

        // 1. Constrain symbolic permission weights
        for (&v, p_var) in value_perms {
            let ty = self.func.get_type(v);
            match ty {
                Type::Held(_) | Type::Hand(_) => {
                    solver.assert(p_var.eq(&one));
                }
                _ if ty.is_pointer_like() => {
                    solver.assert(p_var.gt(&zero));
                    solver.assert(p_var.le(&one));
                }
                _ => {}
            }
        }

        let all_roots: HashSet<Value> = self.value_roots.values().cloned().collect();

        // 2. Perform flow-sensitive verification
        for block in &self.func.blocks {
            let path_cond = block_conditions
                .get(&block.id)
                .ok_or_else(|| format!("Missing path condition for block {:?}", block.id))?;

            for (idx, inst) in block.instructions.iter().enumerate() {
                let live_out = liveness
                    .inst_live_out
                    .get(&(block.id.0, idx))
                    .ok_or_else(|| {
                        format!("Missing liveness for block {:?} inst {}", block.id, idx)
                    })?;

                // Active values = values live after this instruction + the value defined by this instruction
                let mut active_values = live_out.clone();
                if let Some(def) = inst.get_def() {
                    active_values.insert(def);
                }

                // A. Check Fractional Permission Sum for each Root
                for &root in &all_roots {
                    let mut terms = Vec::new();
                    for &v in &active_values {
                        if self.value_roots.get(&v) == Some(&root) {
                            if let Some(p_var) = value_perms.get(&v) {
                                terms.push(p_var);
                            }
                        }
                    }

                    if !terms.is_empty() {
                        let sum = if terms.len() == 1 {
                            terms[0].clone()
                        } else {
                            Real::add(terms.as_slice())
                        };

                        // Assert that total permission on a root must not exceed 1.0
                        solver.assert(path_cond.implies(sum.le(&one)));

                        // Only check for consistency if we have multiple terms,
                        // as single terms are already constrained to be <= 1.0
                        if terms.len() > 1 && solver.check() == SatResult::Unsat {
                            return Err(format!(
                                "Memory safety violation: No valid fractional permission partitioning exists for root {:?} at instruction {} in block {:?}. (Possible aliasing or use-after-move)",
                                root, idx, block.id
                            ));
                        }
                    }
                }

                // B. Check for Linear Move Violations & Exclusive Access
                for &u in &inst.get_uses() {
                    let is_moving = self.is_consuming(&inst.kind, u);
                    let is_mutating = self.requires_exclusive(&inst.kind, u);

                    if is_moving || is_mutating {
                        if let Some(&root) = self.value_roots.get(&u) {
                            for &v in live_out {
                                if Some(v) == inst.get_def() {
                                    continue;
                                } // Skip the value we just defined (the new state)

                                if self.value_roots.get(&v) == Some(&root) {
                                    // Potential violation: trying to move/mutate while other references are live.
                                    // This includes 'u' itself if it's still live after this instruction.
                                    // Optimization: only call solver if it's not obviously safe
                                    solver.push();
                                    solver.assert(path_cond);
                                    let check_res = solver.check();
                                    solver.pop(1);

                                    if check_res == SatResult::Sat {
                                        return Err(format!(
                                            "Memory safety violation: Root {:?} is {} via value {:?}, but still referenced by live value {:?} at instruction {} in block {:?}.",
                                            root,
                                            if is_moving { "moved" } else { "mutated" },
                                            u,
                                            v,
                                            idx,
                                            block.id
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
