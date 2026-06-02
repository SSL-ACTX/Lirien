pub mod tracker;

use lila_ir::analysis::liveness::LivenessAnalysisResults;
use lila_ir::ir::{AccessPath, BlockId, Function, InstructionKind, PathElement, Type, Value};
use std::collections::{HashMap, HashSet};
use z3::ast::{Bool, Real};
use z3::{SatResult, Solver};

pub struct PermissionVerifier<'a> {
    pub func: &'a Function,
    // Maps each Value to the Root it originates from and its AccessPath.
    pub value_roots: HashMap<Value, (Value, AccessPath)>,
    pub parallel_blocks: HashSet<BlockId>,
    pub uid: usize,
}

impl<'a> PermissionVerifier<'a> {
    pub fn new(func: &'a Function) -> Self {
        let mut verifier = Self {
            func,
            value_roots: HashMap::new(),
            parallel_blocks: HashSet::new(),
            uid: 0,
        };
        verifier.analyze_roots();
        verifier.analyze_parallel_blocks();
        verifier
    }

    fn analyze_parallel_blocks(&mut self) {
        let mut parallel_roots = Vec::new();
        for block in &self.func.blocks {
            for inst in &block.instructions {
                if let InstructionKind::ParallelFor {
                    body_block,
                    exit_block,
                    ..
                } = &inst.kind
                {
                    parallel_roots.push((*body_block, *exit_block));
                }
            }
        }

        for (body, exit) in parallel_roots {
            let mut visited = HashSet::new();
            let mut stack = vec![body];
            while let Some(curr) = stack.pop() {
                if curr == exit || !visited.insert(curr) {
                    continue;
                }
                self.parallel_blocks.insert(curr);
                if let Some(block) = self.func.blocks.iter().find(|b| b.id == curr) {
                    for &succ in &block.successors {
                        stack.push(succ);
                    }
                }
            }
        }
    }

    pub fn set_uid(&mut self, uid: usize) {
        self.uid = uid;
    }

    fn get_field_index(&self, src: Value, offset: usize) -> usize {
        let ty = self.func.get_type(src);
        let mut inner_ty = ty;
        while let Type::Hand(t) | Type::Peek(t) | Type::Held(t) | Type::Refined(t, _) = inner_ty {
            inner_ty = t.as_ref().clone();
        }

        if let Type::Struct(name) = inner_ty {
            if let Some(fields) = self.func.struct_layouts.get(&name) {
                let mut curr_offset = 0;
                for (i, (_, f_ty)) in fields.iter().enumerate() {
                    let align = f_ty.align(&self.func.struct_layouts);
                    curr_offset = (curr_offset + align - 1) & !(align - 1);
                    if curr_offset == offset {
                        return i;
                    }
                    curr_offset += f_ty.size(&self.func.struct_layouts);
                }
            }
        }
        offset
    }

    fn analyze_roots(&mut self) {
        // 1. Initial roots: all Held values and arguments
        for i in 0..self.func.arg_count {
            let v = Value(i);
            self.value_roots.insert(v, (v, AccessPath::default()));
        }
        for i in 0..self.func.value_count {
            let v = Value(i);
            if let Type::Held(_) = self.func.get_type(v) {
                self.value_roots.insert(v, (v, AccessPath::default()));
            }
        }

        // 2. Fixpoint propagation for derived values
        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.func.blocks {
                for inst in &block.instructions {
                    if let Some(def) = inst.get_def() {
                        let mut root_info = None;
                        match &inst.kind {
                            InstructionKind::Peek(_, src)
                            | InstructionKind::Hand(_, src)
                            | InstructionKind::StructLoad(_, src, _)
                            | InstructionKind::BufferLoad(_, src, _) => {
                                root_info = self.value_roots.get(src).cloned();
                            }
                            InstructionKind::StructSet(_, src, _, _, _)
                            | InstructionKind::ArrayStore(_, src, _, _, _)
                            | InstructionKind::BufferStore(_, src, _, _, _) => {
                                root_info = self.value_roots.get(src).cloned();
                            }

                            InstructionKind::StructOffset(_, src, offset) => {
                                let field_idx = self.get_field_index(*src, *offset);
                                root_info = self
                                    .value_roots
                                    .get(src)
                                    .map(|(r, p)| (*r, p.extend(PathElement::Field(field_idx))));
                            }
                            InstructionKind::ArrayLoad(_, src, idx_val) => {
                                root_info = self
                                    .value_roots
                                    .get(src)
                                    .map(|(r, p)| (*r, p.extend(PathElement::Index(*idx_val))));
                            }
                            InstructionKind::TupleExtract(_, src, idx) => {
                                root_info = self
                                    .value_roots
                                    .get(src)
                                    .map(|(r, p)| (*r, p.extend(PathElement::Field(*idx))));
                            }
                            InstructionKind::EnumExtract(_, src, idx) => {
                                root_info = self
                                    .value_roots
                                    .get(src)
                                    .map(|(r, p)| (*r, p.extend(PathElement::Field(*idx))));
                            }
                            InstructionKind::Phi(_, mappings) => {
                                for v in mappings.values() {
                                    if let Some(ri) = self.value_roots.get(v) {
                                        root_info = Some(ri.clone());
                                        break;
                                    }
                                }
                            }
                            _ => {
                                let ty = self.func.get_type(def);
                                if ty.is_pointer_like() && !self.value_roots.contains_key(&def) {
                                    root_info = Some((def, AccessPath::default()));
                                }
                            }
                        }

                        if let Some(ri) = root_info {
                            if self.value_roots.get(&def) != Some(&ri) {
                                self.value_roots.insert(def, ri);
                                changed = true;
                            }
                        }
                    }
                }
            }
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

        // Symbolic loop count for parallel partitioning
        let loop_count = Real::new_const(format!("loop_count_{}", self.uid));
        solver.assert(loop_count.gt(&one));

        let all_roots: HashSet<Value> = self.value_roots.values().map(|(r, _)| *r).collect();

        // 1. Perform flow-sensitive verification
        for block in &self.func.blocks {
            let path_cond = block_conditions
                .get(&block.id)
                .ok_or_else(|| format!("Missing path condition for block {:?}", block.id))?;

            if !self.parallel_blocks.contains(&block.id) {
                continue;
            }

            for (idx, inst) in block.instructions.iter().enumerate() {
                let live_out_opt = liveness.inst_live_out.get(&(block.id.0, idx));
                tracing::debug!(target: "lila::verify", "Parallel Block {} Inst {}: {:?}, live_out lookup: {:?}", block.id.0, idx, inst.kind, live_out_opt);

                let live_out = live_out_opt.ok_or_else(|| {
                    format!("Missing liveness for block {:?} inst {}", block.id, idx)
                })?;

                // Active values = values live after this instruction + the value defined by this instruction
                let mut active_values = live_out.clone();
                if let Some(def) = inst.get_def() {
                    active_values.insert(def);
                }

                // Check Fractional Permission Sum for each Root
                for &root in &all_roots {
                    let root_values: Vec<Value> = active_values
                        .iter()
                        .filter(|v| self.value_roots.get(v).map(|(r, _)| *r) == Some(root))
                        .copied()
                        .collect();

                    // Find terminal values (those with no live descendants)
                    let terminal_values: Vec<Value> = root_values
                        .iter()
                        .filter(|&v| {
                            let (_, path) = self.value_roots.get(v).unwrap();
                            !root_values.iter().any(|&other| {
                                if other == *v {
                                    return false;
                                }
                                let (_, other_path) = self.value_roots.get(&other).unwrap();
                                path.is_prefix_of(other_path) && *path != *other_path
                            })
                        })
                        .copied()
                        .collect();

                    // Group terminal values by path and assert sum <= 1.0
                    let mut path_groups: HashMap<AccessPath, Vec<Value>> = HashMap::new();
                    for &v in &terminal_values {
                        let (_, path) = self.value_roots.get(&v).unwrap();
                        path_groups.entry(path.clone()).or_default().push(v);
                    }

                    for (path, group) in path_groups {
                        let mut terms = Vec::new();
                        for &v in &group {
                            if let Some(p_var) = value_perms.get(&v) {
                                // Constrain weights on-demand
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
                                terms.push(p_var);
                            }
                        }

                        if !terms.is_empty() {
                            let sum = if terms.len() == 1 {
                                terms[0].clone()
                            } else {
                                Real::add(terms.as_slice())
                            };

                            let parallel_sum = sum * loop_count.clone();
                            solver.assert(path_cond.implies(parallel_sum.le(&one)));

                            if solver.check() == SatResult::Unsat {
                                let error_msg = format!("Possible data-race: Parallel iterations conflict on shared state for root {:?} at path {}", root, path);

                                return Err(format!(
                                    "{} at instruction {} in block {:?}.",
                                    error_msg, idx, block.id
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
