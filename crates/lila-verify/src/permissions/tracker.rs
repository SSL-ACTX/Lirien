use lila_ir::analysis::liveness::LivenessAnalysisResults;
use lila_ir::ir::{AccessPath, Function, InstructionKind, PathElement, Type, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathState {
    Moved,
    ExclusiveBorrowed(Value),
    SharedBorrowed(HashSet<Value>),
    Active, // Unborrowed, fully owned
}

pub struct OwnershipTracker<'a> {
    func: &'a Function,
    value_roots: &'a HashMap<Value, (Value, AccessPath)>,
    liveness: &'a LivenessAnalysisResults,
}

impl<'a> OwnershipTracker<'a> {
    pub fn new(
        func: &'a Function,
        value_roots: &'a HashMap<Value, (Value, AccessPath)>,
        liveness: &'a LivenessAnalysisResults,
    ) -> Self {
        Self {
            func,
            value_roots,
            liveness,
        }
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

    fn is_consuming(&self, inst: &InstructionKind, val: Value) -> bool {
        let ty = self.func.get_type(val);
        if !ty.is_pointer_like() || matches!(ty, Type::Peek(_) | Type::Buffer(_)) {
            return false;
        }

        match inst {
            InstructionKind::Call(_, _, args) => args.contains(&val),
            InstructionKind::Return(Some(v)) => *v == val,
            InstructionKind::StructSet(_, _, _, v, _) => *v == val, // v is consumed
            InstructionKind::ArrayStore(_, _, _, v, _) => *v == val,
            InstructionKind::BufferStore(_, _, _, v, _) => *v == val,
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

    pub fn verify(&self) -> Result<(), String> {
        for block in &self.func.blocks {
            for (idx, inst) in block.instructions.iter().enumerate() {
                let live_out = self
                    .liveness
                    .inst_live_out
                    .get(&(block.id.0, idx))
                    .ok_or_else(|| {
                        format!("Missing liveness for block {:?} inst {}", block.id, idx)
                    })?;

                for &u in &inst.get_uses() {
                    let is_moving = self.is_consuming(&inst.kind, u);
                    let is_mutating = self.requires_exclusive(&inst.kind, u);

                    if is_moving || is_mutating {
                        if let Some((root, _u_orig_path)) = self.value_roots.get(&u) {
                            let affected_path = self.get_target_path(&inst.kind, u).unwrap();

                            // Find all other live values that originate from the same root
                            for &v in live_out {
                                if Some(v) == inst.get_def() {
                                    continue; // Ignore the definition of the current instruction
                                }
                                if let Some((v_root, v_path)) = self.value_roots.get(&v) {
                                    if v_root == root && affected_path.overlaps(v_path) {
                                        // Ignore if v is an ancestor of the value being used (u),
                                        // as u is a reborrow/sub-reference of v.
                                        let (_, u_path) = self.value_roots.get(&u).unwrap();
                                        if v_path.is_prefix_of(u_path) && v_path != u_path {
                                            continue;
                                        }

                                        let action = if is_moving { "moved" } else { "mutated" };
                                        return Err(format!(
                                            "Memory safety violation: Root {:?} at path {} is {} via value {:?}, but overlapping path {} is still referenced by live value {:?} at instruction {} in block {:?}.",
                                            root, affected_path, action, u, v_path, v, idx, block.id
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
