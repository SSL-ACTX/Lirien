use crate::ssa::analysis::liveness::LivenessAnalysisResults;
use crate::ssa::ir::{AccessPath, BlockId, Function, InstructionKind, PathElement, Type, Value};
use std::collections::{HashMap, HashSet};
use z3::ast::{Bool, Real};
use z3::{SatResult, Solver};

pub struct PermissionVerifier<'a> {
    func: &'a Function,
    // Maps each Value to the Root it originates from and its AccessPath.
    value_roots: HashMap<Value, (Value, AccessPath)>,
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

    fn is_consuming(&self, inst: &InstructionKind, val: Value) -> bool {
        let ty = self.func.get_type(val);
        // Only pointer-like types can be "consumed" or moved.
        // Peek references are specifically designed to be reused.
        if !ty.is_pointer_like() || matches!(ty, Type::Peek(_)) {
            return false;
        }

        match inst {
            InstructionKind::Call(_, _, args) => args.contains(&val),
            InstructionKind::Return(Some(v)) => *v == val,
            InstructionKind::StructSet(_, obj, _, v, _) => *v == val || *obj == val,
            InstructionKind::ArrayStore(_, arr, _, v, _) => *v == val || *arr == val,
            InstructionKind::BufferStore(_, buf, _, v, _) => *v == val || *buf == val,
            InstructionKind::TupleCreate(_, elts) => elts.contains(&val),
            InstructionKind::EnumCreate(_, _, _, Some(v)) => *v == val,
            InstructionKind::Release(v) => *v == val,
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

    fn get_target_path(&self, inst: &InstructionKind, u: Value) -> Option<AccessPath> {
        let (_, path) = self.value_roots.get(&u)?;
        match inst {
            InstructionKind::StructSet(_, obj, offset, _, _) if *obj == u => {
                let field_idx = self.get_field_index(*obj, *offset);
                Some(path.extend(PathElement::Field(field_idx)))
            }
            InstructionKind::StructLoad(_, obj, offset) if *obj == u => {
                let field_idx = self.get_field_index(*obj, *offset);
                Some(path.extend(PathElement::Field(field_idx)))
            }
            InstructionKind::StructOffset(_, obj, offset) if *obj == u => {
                let field_idx = self.get_field_index(*obj, *offset);
                Some(path.extend(PathElement::Field(field_idx)))
            }
            InstructionKind::ArrayLoad(_, src, idx_val) if *src == u => {
                Some(path.extend(PathElement::Index(*idx_val)))
            }
            InstructionKind::ArrayStore(_, arr, idx_val, _, _) if *arr == u => {
                Some(path.extend(PathElement::Index(*idx_val)))
            }
            InstructionKind::BufferLoad(_, src, idx_val) if *src == u => {
                Some(path.extend(PathElement::Index(*idx_val)))
            }
            InstructionKind::BufferStore(_, buf, idx_val, _, _) if *buf == u => {
                Some(path.extend(PathElement::Index(*idx_val)))
            }
            InstructionKind::TupleExtract(_, src, idx) if *src == u => {
                Some(path.extend(PathElement::Field(*idx)))
            }
            InstructionKind::EnumExtract(_, src, idx) if *src == u => {
                Some(path.extend(PathElement::Field(*idx)))
            }
            _ => Some(path.clone()),
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

        let all_roots: HashSet<Value> = self.value_roots.values().map(|(r, _)| *r).collect();

        // 2. Perform flow-sensitive verification
        for block in &self.func.blocks {
            let path_cond = block_conditions
                .get(&block.id)
                .ok_or_else(|| format!("Missing path condition for block {:?}", block.id))?;

            for (idx, inst) in block.instructions.iter().enumerate() {
                let live_out_opt = liveness.inst_live_out.get(&(block.id.0, idx));
                tracing::debug!(target: "lila::verify", "Block {} Inst {}: {:?}, live_out lookup: {:?}", block.id.0, idx, inst.kind, live_out_opt);

                let live_out = live_out_opt.ok_or_else(|| {
                    format!("Missing liveness for block {:?} inst {}", block.id, idx)
                })?;

                // Active values = values live after this instruction + the value defined by this instruction
                let mut active_values = live_out.clone();
                if let Some(def) = inst.get_def() {
                    active_values.insert(def);
                }

                // A. Check Fractional Permission Sum for each Root
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
                                terms.push(p_var);
                            }
                        }

                        if !terms.is_empty() {
                            let sum = if terms.len() == 1 {
                                terms[0].clone()
                            } else {
                                Real::add(terms.as_slice())
                            };

                            solver.assert(path_cond.implies(sum.le(&one)));

                            if terms.len() > 1 && solver.check() == SatResult::Unsat {
                                return Err(format!(
                                    "Memory safety violation: No valid fractional permission partitioning exists for root {:?} at path {} at instruction {} in block {:?}.",
                                    root, path, idx, block.id
                                ));
                            }
                        }
                    }
                }

                // B. Check for Linear Move Violations & Exclusive Access
                for &u in &inst.get_uses() {
                    let is_moving = self.is_consuming(&inst.kind, u);
                    let is_mutating = self.requires_exclusive(&inst.kind, u);

                    if is_moving || is_mutating {
                        if let Some((root, _u_orig_path)) = self.value_roots.get(&u) {
                            let affected_path = self.get_target_path(&inst.kind, u).unwrap();
                            tracing::debug!(target: "lila::verify", "Checking conflict for {:?} at path {}, live_out: {:?}", u, affected_path, live_out);
                            for &v in live_out {
                                if Some(v) == inst.get_def() {
                                    continue;
                                }
                                if let Some((v_root, v_path)) = self.value_roots.get(&v) {
                                    tracing::debug!(target: "lila::verify", "  Comparing with {:?} at path {}", v, v_path);
                                    if v_root == root && affected_path.overlaps(v_path) {
                                        // Ignore if v is an ancestor of the value being used (u),
                                        // as u is a reborrow/sub-reference of v.
                                        let (u_root, u_path) = self.value_roots.get(&u).unwrap();
                                        if u_root == v_root
                                            && v_path.is_prefix_of(u_path)
                                            && v_path != u_path
                                        {
                                            continue;
                                        }

                                        // Specific path overlap + Exclusive access/Move = Violation
                                        solver.push();
                                        solver.assert(path_cond);
                                        let check_res = solver.check();
                                        solver.pop(1);

                                        if check_res == SatResult::Sat {
                                            return Err(format!(
                                                "Memory safety violation: Root {:?} at path {} is {} via value {:?}, but overlapping path {} is still referenced by live value {:?} at instruction {} in block {:?}.",
                                                root,
                                                affected_path,
                                                if is_moving { "moved" } else { "mutated" },
                                                u,
                                                v_path,
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
        }

        Ok(())
    }
}
