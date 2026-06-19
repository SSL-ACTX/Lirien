use crate::backend::SolverBackend;
use crate::verifier::TranslationContext;
use lirien_ir::ir::{Instruction, InstructionKind, Type, Value};
use z3::ast::Bool;
use std::collections::HashMap;

pub fn translate<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
    inst: &Instruction,
    path_cond: &Bool,
) -> Result<(), String> {
    if let InstructionKind::Call(dest, target_name, _args) = &inst.kind {
        let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();

        let sig = if target_name == &t_ctx.func.name {
            // Recursive call: use current function's signature
            let mut arg_types = Vec::new();
            let mut arg_refinements = HashMap::new();
            for i in 0..t_ctx.func.arg_count {
                let v = Value(i);
                arg_types.push(t_ctx.func.get_type(v));
                if let Some(ref_str) = t_ctx.func.refinements.get(&v) {
                    arg_refinements.insert(i, ref_str.clone());
                }
            }
            Some(lirien_ir::registry::FunctionSignature {
                name: target_name.clone(),
                arg_types,
                arg_refinements,
                return_type: t_ctx.func.return_type.clone(),
                return_refinement: t_ctx.func.ret_refinement.clone(),
                pointer: 0,
            })
        } else {
            registry.get(target_name).cloned()
        };

        if let Some(sig) = sig {
            if let Some(ret_ref) = &sig.return_refinement {
                let ty = t_ctx.func.get_type(*dest);
                let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(dest) {
                    let bv_int = t_ctx.backend.bv_to_int(z3_bv, ty.is_signed());
                    crate::refinement::parse_refinement(ret_ref, &bv_int, Some(z3_bv))
                } else if let Some(z3_int) = t_ctx.z3_ints.get(dest) {
                    crate::refinement::parse_refinement(ret_ref, z3_int, None)
                } else if let Some(z3_float) = t_ctx.z3_floats.get(dest) {
                    crate::refinement::parse_float_refinement(ret_ref, z3_float)
                } else {
                    return Ok(());
                };

                if let Ok(expr) = res {
                    // Inductive Hypothesis: Assume the function holds for smaller inputs.
                    let __tmp = t_ctx.backend.bool_implies(path_cond, &expr);
                    t_ctx.backend.assert(&__tmp);
                }
            }
        }
    }
    Ok(())
}

pub fn verify_call_arguments<
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
    for block in &t_ctx.func.blocks {
        let path_cond = t_ctx.block_conditions.get(&block.id).unwrap().clone();
        for inst in &block.instructions {
            if let InstructionKind::Call(_, target_name, args) = &inst.kind {
                let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
                let sig = if target_name == &t_ctx.func.name {
                    // Recursive call
                    let mut arg_types = Vec::new();
                    let mut arg_refinements = HashMap::new();
                    for i in 0..t_ctx.func.arg_count {
                        let v = Value(i);
                        arg_types.push(t_ctx.func.get_type(v));
                        if let Some(ref_str) = t_ctx.func.refinements.get(&v) {
                            arg_refinements.insert(i, ref_str.clone());
                        }
                    }
                    Some(lirien_ir::registry::FunctionSignature {
                        name: target_name.clone(),
                        arg_types,
                        arg_refinements,
                        return_type: t_ctx.func.return_type.clone(),
                        return_refinement: t_ctx.func.ret_refinement.clone(),
                        pointer: 0,
                    })
                } else {
                    registry.get(target_name).cloned()
                };

                if let Some(sig) = sig {
                    let mut call_dim_map: HashMap<String, B::Int> = HashMap::new();
                    for (i, arg_val) in args.iter().enumerate() {
                        let arg_ty = t_ctx.func.get_type(*arg_val);
                        if i < sig.arg_types.len() {
                            let target_ty = &sig.arg_types[i];
                            if let (Type::Tensor(_, src_dims), Type::Tensor(_, target_dims)) = (&arg_ty, target_ty) {
                                let has_ellipsis = target_dims.iter().any(|d| d == "...");
                                if !has_ellipsis && src_dims.len() != target_dims.len() {
                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                    return Err(format!("Tensor rank mismatch in call to '{}': expected {} dims, got {}{}", target_name, target_dims.len(), src_dims.len(), loc_info));
                                }
                                if let Some(src_z3_dims) = t_ctx.z3_tensor_dims.get(arg_val) {
                                    if has_ellipsis {
                                        let ellipsis_pos = target_dims.iter().position(|d| d == "...").unwrap();
                                        let num_fixed_before = ellipsis_pos;
                                        let num_fixed_after = target_dims.len() - ellipsis_pos - 1;

                                        if src_dims.len() < num_fixed_before + num_fixed_after {
                                            let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                            return Err(format!("Tensor rank too small for polymorphic target in call to '{}': expected at least {} dims, got {}{}", target_name, num_fixed_before + num_fixed_after, src_dims.len(), loc_info));
                                        }

                                        // Check fixed dims before ellipsis
                                        for i in 0..num_fixed_before {
                                            let target_dim_name = &target_dims[i];
                                            let src_z3_dim = &src_z3_dims[i];
                                            if let Some(bound_z3_dim) = call_dim_map.get(target_dim_name) {
                                                t_ctx.backend.push();
                                                t_ctx.backend.assert(&path_cond);
                                                let eq = t_ctx.backend.int_eq(src_z3_dim, bound_z3_dim);
                                                let not_eq = t_ctx.backend.bool_not(&eq);
                                                t_ctx.backend.assert(&not_eq);
                                                if t_ctx.backend.check()? {
                                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                                    return Err(format!("Tensor shape mismatch in call to '{}' (prefix): dimension '{}' (idx {}) does not match previously bound value{}", target_name, target_dim_name, i, loc_info));
                                                }
                                                t_ctx.backend.pop(1);
                                            } else {
                                                call_dim_map.insert(target_dim_name.clone(), src_z3_dim.clone());
                                            }
                                        }

                                        // Check fixed dims after ellipsis
                                        for i in 0..num_fixed_after {
                                            let src_idx = src_dims.len() - num_fixed_after + i;
                                            let target_idx = ellipsis_pos + 1 + i;
                                            let target_dim_name = &target_dims[target_idx];
                                            let src_z3_dim = &src_z3_dims[src_idx];
                                            if let Some(bound_z3_dim) = call_dim_map.get(target_dim_name) {
                                                t_ctx.backend.push();
                                                t_ctx.backend.assert(&path_cond);
                                                let eq = t_ctx.backend.int_eq(src_z3_dim, bound_z3_dim);
                                                let not_eq = t_ctx.backend.bool_not(&eq);
                                                t_ctx.backend.assert(&not_eq);
                                                if t_ctx.backend.check()? {
                                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                                    return Err(format!("Tensor shape mismatch in call to '{}' (suffix): dimension '{}' (idx {}) does not match previously bound value{}", target_name, target_dim_name, src_idx, loc_info));
                                                }
                                                t_ctx.backend.pop(1);
                                            } else {
                                                call_dim_map.insert(target_dim_name.clone(), src_z3_dim.clone());
                                            }
                                        }
                                    } else {
                                        for (dim_idx, target_dim_name) in target_dims.iter().enumerate() {
                                            let src_z3_dim = &src_z3_dims[dim_idx];
                                            if let Some(bound_z3_dim) = call_dim_map.get(target_dim_name) {
                                                t_ctx.backend.push();
                                                t_ctx.backend.assert(&path_cond);
                                                let eq = t_ctx.backend.int_eq(src_z3_dim, bound_z3_dim);
                                                let not_eq = t_ctx.backend.bool_not(&eq);
                                                t_ctx.backend.assert(&not_eq);
                                                
                                                if t_ctx.backend.check()? {
                                                    let loc_info = inst.location.map(|l| format!(" at {}", l)).unwrap_or_default();
                                                    return Err(format!("Tensor shape mismatch in call to '{}': dimension '{}' (idx {}) does not match previously bound value{}", target_name, target_dim_name, dim_idx, loc_info));
                                                }
                                                t_ctx.backend.pop(1);
                                            } else {
                                                call_dim_map.insert(target_dim_name.clone(), src_z3_dim.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(ref_str) = sig.arg_refinements.get(&i) {
                            let arg_ty = t_ctx.func.get_type(*arg_val);
                            let res = if let Some(z3_bv) = t_ctx.z3_bvs.get(arg_val) {
                                let bv_int = t_ctx.backend.bv_to_int(z3_bv, arg_ty.is_signed());
                                crate::refinement::parse_refinement(
                                    ref_str,
                                    &bv_int,
                                    Some(z3_bv),
                                )
                            } else if let Some(z3_int) = t_ctx.z3_ints.get(arg_val) {
                                crate::refinement::parse_refinement(ref_str, z3_int, None)
                            } else if let Some(z3_float) = t_ctx.z3_floats.get(arg_val) {
                                crate::refinement::parse_float_refinement(ref_str, z3_float)
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
                                        "Argument refinement violation for function '{}' (arg {}): value of {:?} does not satisfy '{}' and may be violated on some reachable path{}.",
                                        target_name, i, arg_val, ref_str, loc_info
                                    ));
                                }
                                t_ctx.backend.pop(1);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
