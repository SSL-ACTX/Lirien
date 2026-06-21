use crate::ir::{Function, InstructionKind, Type};
use std::collections::HashMap;

pub fn propagate_types(func: &mut Function) {
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;
        let mut new_types = HashMap::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                match &inst.kind {
                    InstructionKind::Assign(d, s) => {
                        let d_ty = func.get_type(*d);
                        let s_ty = func.get_type(*s);
                        if d_ty == Type::Unknown && s_ty != Type::Unknown {
                            new_types.insert(*d, s_ty);
                        } else if s_ty == Type::Unknown && d_ty != Type::Unknown {
                            new_types.insert(*s, d_ty);
                        }
                    }
                    InstructionKind::Add(d, l, r)
                    | InstructionKind::Sub(d, l, r)
                    | InstructionKind::Mul(d, l, r)
                    | InstructionKind::SDiv(d, l, r)
                    | InstructionKind::UDiv(d, l, r)
                    | InstructionKind::SRem(d, l, r)
                    | InstructionKind::URem(d, l, r)
                    | InstructionKind::And(d, l, r)
                    | InstructionKind::Or(d, l, r)
                    | InstructionKind::Xor(d, l, r)
                    | InstructionKind::Shl(d, l, r)
                    | InstructionKind::LShr(d, l, r)
                    | InstructionKind::AShr(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);
                        
                        if current_ty != Type::Unknown {
                            if l_ty == Type::Unknown {
                                new_types.insert(*l, current_ty.base_type().clone());
                            }
                            if r_ty == Type::Unknown {
                                new_types.insert(*r, current_ty.base_type().clone());
                            }
                        }

                        if current_ty == Type::Unknown {
                            let l_base = l_ty.base_type();
                            let r_base = r_ty.base_type();
                            let base_ty = if l_base != &Type::Unknown {
                                l_base
                            } else {
                                r_base
                            };
                            if base_ty != &Type::Unknown {
                                let inner_ty = base_ty.clone();
                                let op_str = match &inst.kind {
                                    InstructionKind::Add(_, _, _) => "+",
                                    InstructionKind::Sub(_, _, _) => "-",
                                    InstructionKind::Mul(_, _, _) => "*",
                                    InstructionKind::SDiv(_, _, _)
                                    | InstructionKind::UDiv(_, _, _) => "/",
                                    InstructionKind::SRem(_, _, _)
                                    | InstructionKind::URem(_, _, _) => "%",
                                    InstructionKind::And(_, _, _) => "&",
                                    InstructionKind::Or(_, _, _) => "|",
                                    InstructionKind::Xor(_, _, _) => "^",
                                    _ => "",
                                };
                                if !op_str.is_empty() {
                                    let constraint = format!("(= {{v}} ({} {} {}))", op_str, l, r);
                                    new_types
                                        .insert(*d, Type::Refined(Box::new(inner_ty), constraint));
                                } else {
                                    new_types.insert(*d, inner_ty);
                                }
                            }
                        }
                    }
                    InstructionKind::Not(d, s)
                    | InstructionKind::Abs(d, s)
                    | InstructionKind::Neg(d, s) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let s_ty = func.get_type(*s);
                            if s_ty != Type::Unknown {
                                let inner_ty = match s_ty {
                                    Type::Refined(inner, _) => *inner,
                                    other => other,
                                };
                                let constraint = match &inst.kind {
                                    InstructionKind::Neg(_, _) => {
                                        format!("(= {{v}} (- 0 {}))", s)
                                    }
                                    InstructionKind::Abs(_, _) => {
                                        format!("(= {{v}} (ite (>= {} 0) {} (- 0 {})))", s, s, s)
                                    }
                                    _ => "".to_string(),
                                };
                                if !constraint.is_empty() {
                                    new_types.insert(*d, Type::Refined(Box::new(inner_ty), constraint));
                                } else {
                                    new_types.insert(*d, inner_ty);
                                }
                            }
                        }
                    }
                    InstructionKind::Min(d, l, r)
                    | InstructionKind::Max(d, l, r)
                    | InstructionKind::Avg(d, l, r) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let l_ty = func.get_type(*l);
                            let r_ty = func.get_type(*r);
                            let base_ty = if l_ty != Type::Unknown { l_ty } else { r_ty };
                            if base_ty != Type::Unknown {
                                let inner_ty = match base_ty {
                                    Type::Refined(inner, _) => *inner,
                                    other => other,
                                };
                                let constraint = match &inst.kind {
                                    InstructionKind::Min(_, _, _) => {
                                        format!("(= {{v}} (ite (<= {} {}) {} {}))", l, r, l, r)
                                    }
                                    InstructionKind::Max(_, _, _) => {
                                        format!("(= {{v}} (ite (>= {} {}) {} {}))", l, r, l, r)
                                    }
                                    InstructionKind::Avg(_, _, _) => {
                                        format!("(= {{v}} (/ (+ {} {} 1) 2))", l, r)
                                    }
                                    _ => "".to_string(),
                                };
                                if !constraint.is_empty() {
                                    new_types.insert(*d, Type::Refined(Box::new(inner_ty), constraint));
                                } else {
                                    new_types.insert(*d, inner_ty);
                                }
                            }
                        }
                    }
                    InstructionKind::Eq(d, l, r)
                    | InstructionKind::Ne(d, l, r)
                    | InstructionKind::SLt(d, l, r)
                    | InstructionKind::SLe(d, l, r)
                    | InstructionKind::SGt(d, l, r)
                    | InstructionKind::SGe(d, l, r)
                    | InstructionKind::ULt(d, l, r)
                    | InstructionKind::ULe(d, l, r)
                    | InstructionKind::UGt(d, l, r)
                    | InstructionKind::UGe(d, l, r)
                    | InstructionKind::FLt(d, l, r)
                    | InstructionKind::FLe(d, l, r)
                    | InstructionKind::FGt(d, l, r)
                    | InstructionKind::FGe(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);

                        if current_ty == Type::Unknown {
                            if l_ty != Type::Unknown {
                                new_types.insert(*d, l_ty);
                            } else if r_ty != Type::Unknown {
                                new_types.insert(*d, r_ty);
                            }
                        }
                    }
                    InstructionKind::FAdd(d, l, r)
                    | InstructionKind::FSub(d, l, r)
                    | InstructionKind::FMul(d, l, r)
                    | InstructionKind::FDiv(d, l, r) => {
                        let l_ty = func.get_type(*l);
                        let r_ty = func.get_type(*r);
                        let current_ty = func.get_type(*d);

                        if current_ty != Type::Unknown {
                            if l_ty == Type::Unknown {
                                new_types.insert(*l, current_ty.base_type().clone());
                            } else if current_ty.is_float32() && l_ty.is_float64() {
                                // Narrowing from f64 to f32
                                new_types.insert(*l, Type::F32);
                            }
                            if r_ty == Type::Unknown {
                                new_types.insert(*r, current_ty.base_type().clone());
                            } else if current_ty.is_float32() && r_ty.is_float64() {
                                // Narrowing from f64 to f32
                                new_types.insert(*r, Type::F32);
                            }
                        }

                        if current_ty == Type::Unknown {
                            let mut base_ty = Type::Unknown;
                            let l_base = l_ty.base_type();
                            let r_base = r_ty.base_type();

                            if l_base.is_float() || r_base.is_float() {
                                if l_base.is_float32() || r_base.is_float32() {
                                    base_ty = Type::F32;
                                } else if l_base.is_float64() || r_base.is_float64() {
                                    base_ty = Type::F64;
                                }
                            }

                            if base_ty != Type::Unknown {
                                let op_str = match &inst.kind {
                                    InstructionKind::FAdd(_, _, _) => "+",
                                    InstructionKind::FSub(_, _, _) => "-",
                                    InstructionKind::FMul(_, _, _) => "*",
                                    InstructionKind::FDiv(_, _, _) => "/",
                                    _ => "",
                                };
                                if !op_str.is_empty() {
                                    let constraint = format!("(= {{v}} ({} {} {}))", op_str, l, r);
                                    new_types
                                        .insert(*d, Type::Refined(Box::new(base_ty), constraint));
                                } else {
                                    new_types.insert(*d, base_ty);
                                }
                            }
                        }
                    }
                    InstructionKind::Phi(d, mappings) => {
                        let mut current_ty = func.get_type(*d);
                        if current_ty != Type::Unknown {
                            if let Type::Pointer(ref p_inner) = current_ty {
                                let mut needs_downgrade = false;
                                for val in mappings.values() {
                                    if let Type::NullablePointer(ref incoming_inner) = func.get_type(*val) {
                                        if incoming_inner == p_inner {
                                            needs_downgrade = true;
                                            break;
                                        }
                                    }
                                }
                                if needs_downgrade {
                                    let downgraded_ty = Type::NullablePointer(p_inner.clone());
                                    new_types.insert(*d, downgraded_ty.clone());
                                    current_ty = downgraded_ty;
                                }
                            }

                            for val in mappings.values() {
                                let v_ty = func.get_type(*val);
                                if v_ty == Type::Unknown {
                                    new_types.insert(*val, current_ty.clone());
                                } else if v_ty.is_float() && current_ty.is_float() && v_ty != current_ty {
                                    // Allow narrowing/widening of floats in phi
                                    new_types.insert(*val, current_ty.clone());
                                }
                            }
                        }

                        let mut base_ty = Type::Unknown;
                        let mut refined_sources = Vec::new();
                        let mut all_base_types_match = true;

                        for val in mappings.values() {
                            let ty = func.get_type(*val);
                            let (b_ty, is_refined) = match ty {
                                Type::Refined(inner, _) => (*inner, true),
                                other => (other, false),
                            };

                            if b_ty != Type::Unknown {
                                if base_ty == Type::Unknown {
                                    base_ty = b_ty.clone();
                                } else if base_ty != b_ty {
                                    if base_ty.is_float() && b_ty.is_float() {
                                        // Favor F32 if current_ty is F32, else F64
                                        if current_ty.is_float32() {
                                            base_ty = Type::F32;
                                        } else {
                                            base_ty = Type::F64;
                                        }
                                    } else if let (Type::Pointer(ref p1) | Type::NullablePointer(ref p1), Type::Pointer(ref p2) | Type::NullablePointer(ref p2)) = (&base_ty, &b_ty) {
                                        if p1 == p2 {
                                            if matches!(base_ty, Type::NullablePointer(_)) || matches!(b_ty, Type::NullablePointer(_)) {
                                                base_ty = Type::NullablePointer(p1.clone());
                                            }
                                        } else {
                                            all_base_types_match = false;
                                        }
                                    } else {
                                        all_base_types_match = false;
                                    }
                                }
                                if is_refined
                                    || b_ty.is_int()
                                    || b_ty.is_float()
                                    || b_ty == Type::Bool
                                {
                                    refined_sources.push(*val);
                                }
                            }
                        }

                        if base_ty != Type::Unknown && all_base_types_match {
                            let new_ty = if !refined_sources.is_empty() {
                                refined_sources.sort_by_key(|v| v.0);
                                let constraints: Vec<String> = refined_sources
                                    .iter()
                                    .map(|v| format!("(= {{v}} {})", v))
                                    .collect();
                                let combined = if constraints.len() == 1 {
                                    constraints[0].clone()
                                } else {
                                    format!("(| {})", constraints.join(" "))
                                };
                                Type::Refined(Box::new(base_ty), combined)
                            } else {
                                base_ty
                            };

                            if current_ty != new_ty {
                                new_types.insert(*d, new_ty);
                            }
                        }
                    }
                    InstructionKind::BufferLoad(d, buf, _idx) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            if let Type::Buffer(inner) = func.get_type(*buf).base_type() {
                                new_types.insert(*d, (**inner).clone());
                            }
                        }
                    }
                    InstructionKind::BufferStore(d, buf, _idx, _val, _ty) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*buf));
                        }
                    }
                    InstructionKind::IToF(d, _, ty)
                    | InstructionKind::FToI(d, _, ty)
                    | InstructionKind::FConv(d, _, ty) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, ty.clone());
                        }
                    }
                    InstructionKind::Return(Some(v)) => {
                        let ty = func.get_type(*v);
                        if func.return_type == Type::Unknown && ty != Type::Unknown {
                            func.return_type = ty.clone();
                            changed = true;
                        }
                        if ty == Type::Unknown && func.return_type != Type::Unknown {
                            new_types.insert(*v, func.return_type.clone());
                        }
                    }
                    InstructionKind::Alloc(d, t) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, Type::Pointer(Box::new(t.clone())));
                        }
                    }
                    InstructionKind::PointerLoad(d, p) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let p_ty = func.get_type(*p);
                            if let Type::Pointer(inner) = p_ty {
                                new_types.insert(*d, (*inner).clone());
                            }
                        }
                    }
                    InstructionKind::EnumIsVariant(d, _, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, Type::Bool);
                        }
                    }
                    InstructionKind::BufferLen(d, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, Type::I64);
                        }
                    }
                    InstructionKind::TensorLoad(d, tensor, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let tensor_ty = match func.get_type(*tensor) {
                                Type::Refined(inner, _) => *inner,
                                other => other,
                            };
                            if let Type::Tensor(inner, _) = tensor_ty {
                                new_types.insert(*d, *inner);
                            }
                        }
                    }
                    InstructionKind::TensorStore(d, tensor, _, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*tensor));
                        }
                    }
                    InstructionKind::TensorAdd(d, lhs, _)
                    | InstructionKind::TensorSub(d, lhs, _)
                    | InstructionKind::TensorMul(d, lhs, _)
                    | InstructionKind::TensorDiv(d, lhs, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*lhs));
                        }
                    }
                    InstructionKind::ArrayLoad(d, arr, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let arr_ty = func.get_type(*arr);
                            if let Type::Array(inner, _) = arr_ty {
                                new_types.insert(*d, *inner);
                            }
                        }
                    }
                    InstructionKind::ArraySlice(d, arr, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*arr));
                        }
                    }
                    InstructionKind::SIMDExtractLane(d, v, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let v_ty = func.get_type(*v);
                            match v_ty {
                                Type::F32X4 => {
                                    new_types.insert(*d, Type::F32);
                                }
                                Type::I32X4 => {
                                    new_types.insert(*d, Type::I32);
                                }
                                Type::F64X2 => {
                                    new_types.insert(*d, Type::F64);
                                }
                                Type::I64X2 => {
                                    new_types.insert(*d, Type::I64);
                                }
                                Type::I8X16 => {
                                    new_types.insert(*d, Type::I8);
                                }
                                Type::U8X16 => {
                                    new_types.insert(*d, Type::U8);
                                }
                                Type::I16X8 => {
                                    new_types.insert(*d, Type::I16);
                                }
                                Type::U16X8 => {
                                    new_types.insert(*d, Type::U16);
                                }
                                _ => {}
                            }
                        }
                    }
                    InstructionKind::SIMDInsertLane(d, v, _, _) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let v_ty = func.get_type(*v);
                            new_types.insert(*d, v_ty);
                        }
                    }
                    InstructionKind::StructCreate(d, _struct_name, args) => {
                        let current_ty = func.get_type(*d);
                        if let Type::Struct(name) | Type::NamedTuple(name) = current_ty {
                            if let Some(fields) = func.struct_layouts.get(&name) {
                                for (i, arg) in args.iter().enumerate() {
                                    if i < fields.len() {
                                        let arg_ty = func.get_type(*arg);
                                        if arg_ty == Type::Unknown {
                                            new_types.insert(*arg, fields[i].1.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    InstructionKind::StructLoad(d, obj, offset) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let obj_ty = func.get_type(*obj);
                            if let Type::Struct(name) | Type::NamedTuple(name) = obj_ty {
                                if let Some(fields) = func.struct_layouts.get(&name) {
                                    let mut curr_offset = 0;
                                    for (_, f_ty) in fields {
                                        let align = f_ty.align(&func.struct_layouts);
                                        curr_offset = (curr_offset + align - 1) & !(align - 1);
                                        if curr_offset == *offset {
                                            new_types.insert(*d, f_ty.clone());
                                            break;
                                        }
                                        curr_offset += f_ty.size(&func.struct_layouts);
                                    }
                                }
                            }
                        }
                    }
                    InstructionKind::StructSet(d, obj, offset, val, _ty) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            new_types.insert(*d, func.get_type(*obj));
                        }
                        let obj_ty = func.get_type(*obj);
                        if let Type::Struct(name) | Type::NamedTuple(name) = obj_ty {
                            if let Some(fields) = func.struct_layouts.get(&name) {
                                let mut curr_offset = 0;
                                for (_, f_ty) in fields {
                                    let align = f_ty.align(&func.struct_layouts);
                                    curr_offset = (curr_offset + align - 1) & !(align - 1);
                                    if curr_offset == *offset {
                                        if func.get_type(*val) == Type::Unknown {
                                            new_types.insert(*val, f_ty.clone());
                                        }
                                        break;
                                    }
                                    curr_offset += f_ty.size(&func.struct_layouts);
                                }
                            }
                        }
                    }
                    InstructionKind::EnumExtract(d, obj, tag_idx) => {
                        let current_ty = func.get_type(*d);
                        if current_ty == Type::Unknown {
                            let obj_ty = func.get_type(*obj);
                            if let Type::Enum(name) = obj_ty {
                                if let Some(variants) = func.enum_layouts.get(&name) {
                                    if let Some((_, ty)) = variants.get(*tag_idx) {
                                        new_types.insert(*d, ty.clone());
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if !new_types.is_empty() {
            for (val, ty) in new_types {
                func.set_type(val, ty);
                changed = true;
            }
        }
    }

    // Fix instruction kinds based on propagated types
    let mut instruction_updates = Vec::new();

    for block in &func.blocks {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            let new_kind = match &inst.kind {
                InstructionKind::Add(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if l_ty.is_float() || r_ty.is_float() {
                        Some(InstructionKind::FAdd(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::Sub(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if l_ty.is_float() || r_ty.is_float() {
                        Some(InstructionKind::FSub(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::Mul(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if l_ty.is_float() || r_ty.is_float() {
                        Some(InstructionKind::FMul(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::SDiv(d, l, r) | InstructionKind::UDiv(d, l, r) => {
                    let l_ty = func.get_type(*l);
                    let r_ty = func.get_type(*r);
                    if l_ty.is_float() || r_ty.is_float() {
                        Some(InstructionKind::FDiv(*d, *l, *r))
                    } else {
                        None
                    }
                }
                InstructionKind::ArrayLoad(d, arr, idx) => {
                    let ty = func.get_type(*arr);
                    if let Type::Buffer(_) = ty {
                        Some(InstructionKind::BufferLoad(*d, *arr, *idx))
                    } else {
                        None
                    }
                }
                InstructionKind::ArrayStore(d, arr, idx, val, _ty_hint) => {
                    let ty = func.get_type(*arr);
                    if let Type::Buffer(inner) = ty {
                        Some(InstructionKind::BufferStore(*d, *arr, *idx, *val, *inner))
                    } else {
                        None
                    }
                }
                InstructionKind::ConstInt(d, _) => {
                    if func.get_type(*d) == Type::Unknown {
                        func.value_types.insert(*d, Type::I64);
                    }
                    None
                }
                InstructionKind::ConstFloat(d, _) => {
                    let current_ty = func.get_type(*d);
                    if current_ty == Type::Unknown
                        || (current_ty.is_float64() && func.return_type.is_float32())
                    {
                        let ty = if func.return_type.is_float32() {
                            Type::F32
                        } else {
                            Type::F64
                        };
                        func.value_types.insert(*d, ty);
                    }
                    None
                }
                _ => None,
            };

            if let Some(k) = new_kind {
                instruction_updates.push((block.id, inst_idx, k));
            }
        }
    }

    for (block_id, inst_idx, new_kind) in instruction_updates {
        if let Some(block) = func.blocks.iter_mut().find(|b| b.id == block_id) {
            if inst_idx < block.instructions.len() {
                block.instructions[inst_idx].kind = new_kind;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, BlockId, Instruction, InstructionKind};

    #[test]
    fn test_type_propagation_phi() {
        let mut func = Function::new("test".to_string());

        let v0 = func.next_value();
        let v1 = func.next_value();
        let v2 = func.next_value();

        func.set_type(v0, Type::F64);

        let b0_id = BlockId(0);
        let b1_id = BlockId(1);

        let b0 = BasicBlock {
            id: b0_id,
            instructions: vec![
                Instruction::new(InstructionKind::ConstFloat(v0, 1.0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![],
            successors: vec![b1_id],
        };

        let mut phi_map = HashMap::new();
        phi_map.insert(b0_id, v0);
        phi_map.insert(b1_id, v2);

        let b1 = BasicBlock {
            id: b1_id,
            instructions: vec![
                Instruction::new(InstructionKind::Phi(v1, phi_map), None),
                Instruction::new(InstructionKind::Add(v2, v1, v0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![b0_id, b1_id],
            successors: vec![b1_id],
        };

        func.blocks.push(b0);
        func.blocks.push(b1);

        propagate_types(&mut func);

        assert_eq!(
            func.get_type(v1),
            Type::Refined(Box::new(Type::F64), "(| (= {v} v0) (= {v} v2))".to_string())
        );
        assert_eq!(
            func.get_type(v2),
            Type::Refined(Box::new(Type::F64), "(= {v} (+ v1 v0))".to_string())
        );

        let b1_ref = func.blocks.iter().find(|b| b.id == b1_id).unwrap();
        match &b1_ref.instructions[1].kind {
            InstructionKind::FAdd(d, l, r) => {
                assert_eq!(*d, v2);
                assert_eq!(*l, v1);
                assert_eq!(*r, v0);
            }
            k => panic!("Expected FAdd, got {:?}", k),
        }
    }

    #[test]
    fn test_type_propagation_buffer_phi() {
        let mut func = Function::new("test_buffer".to_string());

        let v0 = func.next_value(); // data: Buffer[f64]
        let v1 = func.next_value(); // factor: f64
        let v2 = func.next_value(); // len
        let v3 = func.next_value(); // 0
        let v4 = func.next_value(); // i (phi)
        let v6 = func.next_value(); // data (phi)
        let v7 = func.next_value(); // data[i]
        let v9 = func.next_value(); // data[i] * factor
        let v10 = func.next_value(); // data[i] = ...
        let v12 = func.next_value(); // i + 1

        func.set_type(v0, Type::Buffer(Box::new(Type::F64)));
        func.set_type(v1, Type::F64);

        let b0_id = BlockId(0);
        let b1_id = BlockId(1);
        let b2_id = BlockId(2);

        let b0 = BasicBlock {
            id: b0_id,
            instructions: vec![
                Instruction::new(InstructionKind::BufferLen(v2, v0), None),
                Instruction::new(InstructionKind::ConstInt(v3, 0), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![],
            successors: vec![b1_id],
        };

        let mut i_phi = HashMap::new();
        i_phi.insert(b0_id, v3);
        i_phi.insert(b2_id, v12);

        let mut data_phi = HashMap::new();
        data_phi.insert(b0_id, v0);
        data_phi.insert(b2_id, v10);

        let b1 = BasicBlock {
            id: b1_id,
            instructions: vec![
                Instruction::new(InstructionKind::Phi(v4, i_phi), None),
                Instruction::new(InstructionKind::Phi(v6, data_phi), None),
                Instruction::new(InstructionKind::Jump(b2_id), None), // Simplified
            ],
            predecessors: vec![b0_id, b2_id],
            successors: vec![b2_id],
        };

        let b2 = BasicBlock {
            id: b2_id,
            instructions: vec![
                Instruction::new(InstructionKind::BufferLoad(v7, v6, v4), None),
                Instruction::new(InstructionKind::FMul(v9, v7, v1), None),
                Instruction::new(
                    InstructionKind::BufferStore(v10, v6, v4, v9, Type::F64),
                    None,
                ),
                Instruction::new(InstructionKind::Add(v12, v4, v3), None),
                Instruction::new(InstructionKind::Jump(b1_id), None),
            ],
            predecessors: vec![b1_id],
            successors: vec![b1_id],
        };

        func.blocks.push(b0);
        func.blocks.push(b1);
        func.blocks.push(b2);

        propagate_types(&mut func);

        assert_eq!(func.get_type(v6), Type::Buffer(Box::new(Type::F64)));
        assert_eq!(func.get_type(v7), Type::F64);
        assert_eq!(func.get_type(v10), Type::Buffer(Box::new(Type::F64)));
    }
}
