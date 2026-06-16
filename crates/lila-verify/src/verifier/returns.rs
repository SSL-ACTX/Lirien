use crate::backend::SolverBackend;
use crate::verifier::TranslationContext;
use lila_ir::ir::{InstructionKind, Type};

pub fn verify_return_refinements<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) -> Result<(), String> {
    let ret_ty = t_ctx.func.return_type.clone();
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();
        for inst in &block.instructions {
            if let InstructionKind::Return(Some(ret_val)) = &inst.kind {
                let actual_ty = t_ctx.func.get_type(*ret_val);
                
                // Shape verification
                if let (Type::Tensor(_, src_dims), Type::Tensor(_, target_dims)) = (&actual_ty, &ret_ty) {
                    let has_ellipsis = target_dims.iter().any(|d| d == "...");
                    if !has_ellipsis && src_dims.len() != target_dims.len() {
                        let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                        return Err(format!("Tensor rank mismatch in return: expected {} dims, got {}{}", target_dims.len(), src_dims.len(), loc_info));
                    }
                    
                    if let Some(src_z3_dims) = t_ctx.z3_tensor_dims.get(ret_val).cloned() {
                        if has_ellipsis {
                            let ellipsis_pos = target_dims.iter().position(|d| d == "...").unwrap();
                            let num_fixed_before = ellipsis_pos;
                            let num_fixed_after = target_dims.len() - ellipsis_pos - 1;

                            if src_dims.len() < num_fixed_before + num_fixed_after {
                                let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                return Err(format!("Tensor rank too small for polymorphic target: expected at least {} dims, got {}{}", num_fixed_before + num_fixed_after, src_dims.len(), loc_info));
                            }

                            // Check fixed dims before ellipsis
                            for i in 0..num_fixed_before {
                                let target_dim_name = &target_dims[i];
                                let target_z3_dim = t_ctx.get_dim_var(target_dim_name);
                                t_ctx.backend.push();
                                t_ctx.backend.assert(&path_cond);
                                let eq = t_ctx.backend.int_eq(&src_z3_dims[i], &target_z3_dim);
                                let not_eq = t_ctx.backend.bool_not(&eq);
                                t_ctx.backend.assert(&not_eq);
                                if t_ctx.backend.check()? {
                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                    return Err(format!("Tensor shape mismatch in return (prefix): dimension '{}' (idx {}) does not match{}", target_dim_name, i, loc_info));
                                }
                                t_ctx.backend.pop(1);
                            }

                            // Check fixed dims after ellipsis
                            for i in 0..num_fixed_after {
                                let src_idx = src_dims.len() - num_fixed_after + i;
                                let target_idx = ellipsis_pos + 1 + i;
                                let target_dim_name = &target_dims[target_idx];
                                let target_z3_dim = t_ctx.get_dim_var(target_dim_name);
                                t_ctx.backend.push();
                                t_ctx.backend.assert(&path_cond);
                                let eq = t_ctx.backend.int_eq(&src_z3_dims[src_idx], &target_z3_dim);
                                let not_eq = t_ctx.backend.bool_not(&eq);
                                t_ctx.backend.assert(&not_eq);
                                if t_ctx.backend.check()? {
                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                    return Err(format!("Tensor shape mismatch in return (suffix): dimension '{}' (idx {}) does not match{}", target_dim_name, src_idx, loc_info));
                                }
                                t_ctx.backend.pop(1);
                            }
                        } else {
                            for (dim_idx, target_dim_name) in target_dims.iter().enumerate() {
                                let target_z3_dim = t_ctx.get_dim_var(target_dim_name);
                                t_ctx.backend.push();
                                t_ctx.backend.assert(&path_cond);
                                let eq = t_ctx.backend.int_eq(&src_z3_dims[dim_idx], &target_z3_dim);
                                let not_eq = t_ctx.backend.bool_not(&eq);
                                t_ctx.backend.assert(&not_eq);
                                
                                if t_ctx.backend.check()? {
                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                    return Err(format!("Tensor shape mismatch in return: dimension '{}' (idx {}) does not match{}", target_dim_name, dim_idx, loc_info));
                                }
                                t_ctx.backend.pop(1);
                            }
                        }
                    }
                }

                if let Some(ret_ref) = &t_ctx.func.ret_refinement {
                    if ret_ref == "..." {
                        continue;
                    }
                    let ty = t_ctx.func.get_type(*ret_val);
                    let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(ret_val) {
                        let bv_int = t_ctx.backend.bv_to_int(z3_bv, ty.is_signed());
                        crate::refinement::parse_refinement(ret_ref, &bv_int, Some(z3_bv))
                    } else if let Some(z3_int) = t_ctx.z3_ints.get(ret_val) {
                        crate::refinement::parse_refinement(ret_ref, z3_int, None)
                    } else if let Some(z3_float) = t_ctx.z3_floats.get(ret_val) {
                        crate::refinement::parse_float_refinement(ret_ref, z3_float)
                    } else {
                        continue;
                    };

                    if let Ok(expr) = res {
                        t_ctx.backend.push();
                        t_ctx.backend.assert(&path_cond);
                        let __tmp = t_ctx.backend.bool_not(&expr);
                        t_ctx.backend.assert(&__tmp);
                        if t_ctx.backend.check()? {
                            let loc_info = inst
                                .location
                                .map(|l| format!(" at {}", l))
                                .unwrap_or_default();
                            return Err(format!(
                                "Return refinement violation: value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                ret_val, ret_ref, loc_info
                            ));
                        }
                        t_ctx.backend.pop(1);
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn infer_return_refinement<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &TranslationContext<'_, B>,
) -> Result<Option<String>, String> {
    use lila_ir::analysis::interval::{Bound, Interval};
    let mut combined_interval: Option<Interval> = None;

    for block in &t_ctx.func.blocks {
        for inst in &block.instructions {
            if let InstructionKind::Return(Some(ret_val)) = &inst.kind {
                let interval = t_ctx.analysis.block_narrowing.get(&(*ret_val, block.id))
                    .or_else(|| t_ctx.analysis.intervals.get(ret_val));
                
                if let Some(interval) = interval {
                    combined_interval = match combined_interval {
                        Some(j) => Some(j.join(interval)),
                        None => Some(interval.clone()),
                    };
                }
            }
        }
    }

    if let Some(interval) = combined_interval {
        let mut parts = Vec::new();
        if let Bound::Finite(low) = interval.low {
            if t_ctx.func.return_type.is_float() {
                parts.push(format!("(>= {{v}} {})", low));
            } else {
                parts.push(format!("(>= {{v}} {})", low as i64));
            }
        }
        if let Bound::Finite(high) = interval.high {
            if t_ctx.func.return_type.is_float() {
                parts.push(format!("(<= {{v}} {})", high));
            } else {
                parts.push(format!("(<= {{v}} {})", high as i64));
            }
        }

        if parts.is_empty() {
            Ok(None)
        } else if parts.len() == 1 {
            Ok(Some(parts[0].clone()))
        } else {
            Ok(Some(format!("(and {})", parts.join(" "))))
        }
    } else {
        Ok(None)
    }
}
