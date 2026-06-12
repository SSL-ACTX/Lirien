use super::TranslationContext;
use lila_ir::ir::{Instruction, InstructionKind, Type};

pub fn translate<
    B: crate::backend::SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    ctx: &mut TranslationContext<'_, B>,
    inst: &Instruction,
    path_cond: &B::Bool,
) -> Result<(), String> {
    match &inst.kind {
        InstructionKind::ConstInt(dest, val) => {
            if let Some(z3_dest) = ctx.z3_bvs.get(dest) {
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let z3_val = ctx.backend.bv_from_i64(*val, bit_width);
                let __inner = ctx.backend.bv_eq(z3_dest, &z3_val);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::ConstFloat(dest, val) => {
            if let Some(z3_dest) = ctx.z3_floats.get(dest) {
                let ty = ctx.func.get_type(*dest);
                let z3_val = if ty.is_float32() {
                    ctx.backend.float_from_f32(*val as f32)
                } else {
                    ctx.backend.float_from_f64(*val)
                };
                let __inner = ctx.backend.float_eq(z3_dest, &z3_val);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Assign(dest, src) => {
            if let (Some(d), Some(s)) = (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(src)) {
                let __inner = ctx.backend.bv_eq(d, s);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let (Some(d), Some(s)) = (ctx.z3_floats.get(dest), ctx.z3_floats.get(src)) {
                let __inner = ctx.backend.float_eq(d, s);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            } else if let (Some(d), Some(s)) = (ctx.z3_arrays.get(dest), ctx.z3_arrays.get(src)) {
                let __inner = ctx.backend.array_eq(d, s);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Add(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_add(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FAdd(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let res = ctx.backend.float_add(z3_l, z3_r);
                let __inner = ctx.backend.float_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Sub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_sub(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FSub(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let res = ctx.backend.float_sub(z3_l, z3_r);
                let __inner = ctx.backend.float_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Mul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_mul(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FMul(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let res = ctx.backend.float_mul(z3_l, z3_r);
                let __inner = ctx.backend.float_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::MatMult(_dest, lhs, rhs) => {
            if let (Some(l_dims), Some(r_dims)) = (
                ctx.z3_tensor_dims.get(lhs),
                ctx.z3_tensor_dims.get(rhs),
            ) {
                if l_dims.len() == 2 && r_dims.len() == 2 {
                    let inner_dim_l = &l_dims[1];
                    let inner_dim_r = &r_dims[0];
                    
                    let eq = ctx.backend.int_eq(inner_dim_l, inner_dim_r);
                    let not_eq = ctx.backend.bool_not(&eq);
                    
                    ctx.backend.push();
                    ctx.backend.assert(path_cond);
                    ctx.backend.assert(&not_eq);
                    if ctx.backend.check()? {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Matrix multiplication dimension mismatch: inner dimensions must be equal{}",
                            loc_info
                        ));
                    }
                    ctx.backend.pop(1);

                    // Propagate dimensions: (M, N) @ (N, K) -> (M, K)
                    let res_dims = vec![l_dims[0].clone(), r_dims[1].clone()];
                    ctx.z3_tensor_dims.insert(*_dest, res_dims);
                }
            }
        }
        InstructionKind::TensorAdd(dest, lhs, rhs)
        | InstructionKind::TensorSub(dest, lhs, rhs)
        | InstructionKind::TensorMul(dest, lhs, rhs)
        | InstructionKind::TensorDiv(dest, lhs, rhs) => {
            if let (Some(l_dims), Some(r_dims)) = (
                ctx.z3_tensor_dims.get(lhs).cloned(),
                ctx.z3_tensor_dims.get(rhs).cloned(),
            ) {
                if l_dims.len() != r_dims.len() {
                    return Err(format!(
                        "Tensor rank mismatch in element-wise operation at v{}",
                        dest.0
                    ));
                }

                for i in 0..l_dims.len() {
                    ctx.backend.push();
                    ctx.backend.assert(path_cond);
                    let eq = ctx.backend.int_eq(&l_dims[i], &r_dims[i]);
                    let not_eq = ctx.backend.bool_not(&eq);
                    ctx.backend.assert(&not_eq);

                    if ctx.backend.check()? {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Tensor shape mismatch in element-wise operation (dimension {} mismatch){}",
                            i, loc_info
                        ));
                    }
                    ctx.backend.pop(1);
                }

                // Result has same shape
                ctx.z3_tensor_dims.insert(*dest, l_dims.clone());

                // Create a new Z3 array for the result
                let l_ty = ctx.func.get_type(*lhs);
                let (inner_ty, _) = match l_ty {
                    lila_ir::ir::Type::Tensor(inner, dims) => (inner, dims),
                    _ => unreachable!(),
                };

                let bit_width = inner_ty.int_bit_width().unwrap_or(64);
                let res_array = ctx.backend.array_const(
                    &format!("{}_v{}_tensor_res_{}", ctx.func.name, dest.0, ctx.uid),
                    inner_ty.is_float(),
                    bit_width,
                );
                ctx.z3_arrays.insert(*dest, res_array);
            }
        }
        InstructionKind::TensorScalarAdd(dest, tensor, _scalar)
        | InstructionKind::TensorScalarSub(dest, tensor, _scalar)
        | InstructionKind::TensorScalarMul(dest, tensor, _scalar)
        | InstructionKind::TensorScalarDiv(dest, tensor, _scalar) => {
            if let Some(dims) = ctx.z3_tensor_dims.get(tensor).cloned() {
                // Result has same shape as input tensor
                ctx.z3_tensor_dims.insert(*dest, dims);

                let t_ty = ctx.func.get_type(*tensor);
                let (inner_ty, _) = match t_ty {
                    lila_ir::ir::Type::Tensor(inner, _) => (inner, ()),
                    _ => unreachable!(),
                };

                let bit_width = inner_ty.int_bit_width().unwrap_or(64);
                let res_array = ctx.backend.array_const(
                    &format!("{}_v{}_tensor_scalar_res_{}", ctx.func.name, dest.0, ctx.uid),
                    inner_ty.is_float(),
                    bit_width,
                );
                ctx.z3_arrays.insert(*dest, res_array);
            }
        }
        InstructionKind::SDiv(dest, lhs, rhs) | InstructionKind::SRem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let bit_width = ctx.func.get_type(*rhs).int_bit_width().unwrap_or(64);
                let zero = ctx.backend.bv_from_i64(0, bit_width);
                let is_zero = ctx.backend.bv_eq(z3_r, &zero);

                // Optimization: Use interval analysis to skip Z3 check if divisor is non-zero
                let is_safe = if let Some(interval) = ctx.analysis.intervals.get(rhs) {
                    interval.is_strictly_positive() || interval.is_strictly_negative()
                } else {
                    false
                };

                if !is_safe {
                    ctx.backend.push();
                    ctx.backend.assert(path_cond);
                    ctx.backend.assert(&is_zero);
                    if ctx.backend.check()? {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Potential division by zero at v{}{}",
                            dest.0, loc_info
                        ));
                    }
                    ctx.backend.pop(1);
                }

                if let InstructionKind::SDiv(_, _, _) = &inst.kind {
                    let res = ctx.backend.bv_sdiv(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                } else {
                    let res = ctx.backend.bv_srem(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::FDiv(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let ty = ctx.func.get_type(*rhs);
                let zero = if ty.is_float32() {
                    ctx.backend.float_from_f32(0.0)
                } else {
                    ctx.backend.float_from_f64(0.0)
                };

                // Optimization: Use interval analysis to skip Z3 check if possible
                let is_safe = if let Some(interval) = ctx.analysis.intervals.get(rhs) {
                    interval.is_strictly_positive() || interval.is_strictly_negative()
                } else {
                    false
                };

                if !is_safe {
                    // Verify safety via Z3
                    ctx.backend.push();
                    ctx.backend.assert(path_cond);
                    let __tmp = ctx.backend.float_eq(z3_r, &zero);
                    ctx.backend.assert(&__tmp);
                    if ctx.backend.check()? {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Potential float division by zero at v{}{}",
                            dest.0, loc_info
                        ));
                    }
                    ctx.backend.pop(1);
                }

                let res = ctx.backend.float_div(z3_l, z3_r);
                let __inner = ctx.backend.float_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::UDiv(dest, lhs, rhs) | InstructionKind::URem(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                if let InstructionKind::UDiv(_, _, _) = &inst.kind {
                    let res = ctx.backend.bv_udiv(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                } else {
                    let res = ctx.backend.bv_urem(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::Eq(dest, lhs, rhs)
        | InstructionKind::Ne(dest, lhs, rhs)
        | InstructionKind::SLt(dest, lhs, rhs)
        | InstructionKind::SLe(dest, lhs, rhs)
        | InstructionKind::SGt(dest, lhs, rhs)
        | InstructionKind::SGe(dest, lhs, rhs)
        | InstructionKind::ULt(dest, lhs, rhs)
        | InstructionKind::ULe(dest, lhs, rhs)
        | InstructionKind::UGt(dest, lhs, rhs)
        | InstructionKind::UGe(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => ctx.backend.bv_eq(l, r),
                    InstructionKind::Ne(_, _, _) => {
                        let eq = ctx.backend.bv_eq(l, r);
                        ctx.backend.bool_not(&eq)
                    }
                    InstructionKind::SLt(_, _, _) => ctx.backend.bv_slt(l, r),
                    InstructionKind::SLe(_, _, _) => ctx.backend.bv_sle(l, r),
                    InstructionKind::SGt(_, _, _) => ctx.backend.bv_sgt(l, r),
                    InstructionKind::SGe(_, _, _) => ctx.backend.bv_sge(l, r),
                    InstructionKind::ULt(_, _, _) => ctx.backend.bv_ult(l, r),
                    InstructionKind::ULe(_, _, _) => ctx.backend.bv_ule(l, r),
                    InstructionKind::UGt(_, _, _) => ctx.backend.bv_ugt(l, r),
                    InstructionKind::UGe(_, _, _) => ctx.backend.bv_uge(l, r),
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = ctx.backend.bv_from_i64(1, bit_width);
                let zero = ctx.backend.bv_from_i64(0, bit_width);

                // Emulate ite manually since SolverBackend doesn't have `bool_ite`
                // (val == 1) <-> is_true
                // (val == 0) <-> !is_true
                let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                let __implies1 = ctx.backend.bool_implies(&is_true, &__is_true_eq_one);

                let __not_is_true = ctx.backend.bool_not(&is_true);
                let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                let __implies2 = ctx
                    .backend
                    .bool_implies(&__not_is_true, &__is_false_eq_zero);

                let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                ctx.backend.assert(&__tmp);
            } else if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::Eq(_, _, _) => ctx.backend.float_eq(l, r),
                    InstructionKind::Ne(_, _, _) => {
                        let eq = ctx.backend.float_eq(l, r);
                        ctx.backend.bool_not(&eq)
                    }
                    InstructionKind::SLt(_, _, _) | InstructionKind::FLt(_, _, _) => {
                        ctx.backend.float_lt(l, r)
                    }
                    InstructionKind::SLe(_, _, _) | InstructionKind::FLe(_, _, _) => {
                        ctx.backend.float_le(l, r)
                    }
                    InstructionKind::SGt(_, _, _) | InstructionKind::FGt(_, _, _) => {
                        ctx.backend.float_gt(l, r)
                    }
                    InstructionKind::SGe(_, _, _) | InstructionKind::FGe(_, _, _) => {
                        ctx.backend.float_ge(l, r)
                    }
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = ctx.backend.bv_from_i64(1, bit_width);
                let zero = ctx.backend.bv_from_i64(0, bit_width);

                let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                let __implies1 = ctx.backend.bool_implies(&is_true, &__is_true_eq_one);
                let __not_is_true = ctx.backend.bool_not(&is_true);
                let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                let __implies2 = ctx
                    .backend
                    .bool_implies(&__not_is_true, &__is_false_eq_zero);
                let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FLt(dest, lhs, rhs)
        | InstructionKind::FLe(dest, lhs, rhs)
        | InstructionKind::FGt(dest, lhs, rhs)
        | InstructionKind::FGe(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(l), Some(r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let is_true = match &inst.kind {
                    InstructionKind::FLt(_, _, _) => ctx.backend.float_lt(l, r),
                    InstructionKind::FLe(_, _, _) => ctx.backend.float_le(l, r),
                    InstructionKind::FGt(_, _, _) => ctx.backend.float_gt(l, r),
                    InstructionKind::FGe(_, _, _) => ctx.backend.float_ge(l, r),
                    _ => unreachable!(),
                };
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);
                let one = ctx.backend.bv_from_i64(1, bit_width);
                let zero = ctx.backend.bv_from_i64(0, bit_width);

                let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                let __implies1 = ctx.backend.bool_implies(&is_true, &__is_true_eq_one);
                let __not_is_true = ctx.backend.bool_not(&is_true);
                let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                let __implies2 = ctx
                    .backend
                    .bool_implies(&__not_is_true, &__is_false_eq_zero);
                let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::And(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let ty = ctx.func.get_type(*dest);
                if ty == Type::Bool {
                    let one = ctx.backend.bv_from_i64(1, 1);
                    let zero = ctx.backend.bv_from_i64(0, 1);
                    let l_eq = ctx.backend.bv_eq(z3_l, &one);
                    let r_eq = ctx.backend.bv_eq(z3_r, &one);
                    let both_true = ctx.backend.bool_and(&[&l_eq, &r_eq]);

                    let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                    let __implies1 = ctx.backend.bool_implies(&both_true, &__is_true_eq_one);
                    let __not_both = ctx.backend.bool_not(&both_true);
                    let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                    let __implies2 = ctx.backend.bool_implies(&__not_both, &__is_false_eq_zero);
                    let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                    ctx.backend.assert(&__tmp);
                } else {
                    let res = ctx.backend.bv_and(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::Or(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let ty = ctx.func.get_type(*dest);
                if ty == Type::Bool {
                    let one = ctx.backend.bv_from_i64(1, 1);
                    let zero = ctx.backend.bv_from_i64(0, 1);
                    let l_eq = ctx.backend.bv_eq(z3_l, &one);
                    let r_eq = ctx.backend.bv_eq(z3_r, &one);
                    let either_true = ctx.backend.bool_or(&[&l_eq, &r_eq]);

                    let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                    let __implies1 = ctx.backend.bool_implies(&either_true, &__is_true_eq_one);
                    let __not_both = ctx.backend.bool_not(&either_true);
                    let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                    let __implies2 = ctx.backend.bool_implies(&__not_both, &__is_false_eq_zero);
                    let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                    ctx.backend.assert(&__tmp);
                } else {
                    let res = ctx.backend.bv_or(z3_l, z3_r);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::Xor(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_xor(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::Not(dest, src) => {
            if let (Some(z3_dest), Some(z3_src)) = (ctx.z3_bvs.get(dest), ctx.z3_bvs.get(src)) {
                let ty = ctx.func.get_type(*dest);
                if ty == Type::Bool {
                    let one = ctx.backend.bv_from_i64(1, 1);
                    let zero = ctx.backend.bv_from_i64(0, 1);
                    let is_false = ctx.backend.bv_eq(z3_src, &zero);

                    let __is_true_eq_one = ctx.backend.bv_eq(z3_dest, &one);
                    let __implies1 = ctx.backend.bool_implies(&is_false, &__is_true_eq_one);
                    let __not_false = ctx.backend.bool_not(&is_false);
                    let __is_false_eq_zero = ctx.backend.bv_eq(z3_dest, &zero);
                    let __implies2 = ctx.backend.bool_implies(&__not_false, &__is_false_eq_zero);
                    let __both = ctx.backend.bool_and(&[&__implies1, &__implies2]);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__both);
                    ctx.backend.assert(&__tmp);
                } else {
                    let res = ctx.backend.bv_not(z3_src);
                    let __inner = ctx.backend.bv_eq(z3_dest, &res);
                    let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                    ctx.backend.assert(&__tmp);
                }
            }
        }
        InstructionKind::Shl(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_shl(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::LShr(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_lshr(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::AShr(dest, lhs, rhs) => {
            if let (Some(z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_bvs.get(dest),
                ctx.z3_bvs.get(lhs),
                ctx.z3_bvs.get(rhs),
            ) {
                let res = ctx.backend.bv_ashr(z3_l, z3_r);
                let __inner = ctx.backend.bv_eq(z3_dest, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FSqrt(dest, s_val)
        | InstructionKind::FSin(dest, s_val)
        | InstructionKind::FCos(dest, s_val) => {
            if let Some(_z3_dest) = ctx.z3_floats.get(dest) {
                match &inst.kind {
                    InstructionKind::FSin(_, _) | InstructionKind::FCos(_, _) => {}
                    InstructionKind::FSqrt(_, _) => {
                        if let Some(z3_src) = ctx.z3_floats.get(s_val) {
                            let ty = ctx.func.get_type(*s_val);
                            let zero = if ty.is_float32() {
                                ctx.backend.float_from_f32(0.0)
                            } else {
                                ctx.backend.float_from_f64(0.0)
                            };

                            // Optimization: Use interval analysis to skip Z3 check if possible
                            let is_safe = if let Some(interval) = ctx.analysis.intervals.get(s_val)
                            {
                                interval.is_strictly_positive()
                            } else {
                                false
                            };

                            if !is_safe {
                                // Verify safety via Z3
                                ctx.backend.push();
                                ctx.backend.assert(path_cond);
                                let __tmp = ctx.backend.float_lt(z3_src, &zero);
                                ctx.backend.assert(&__tmp);
                                if ctx.backend.check()? {
                                    let loc_info = inst
                                        .location
                                        .map(|l| format!(" at {}", l))
                                        .unwrap_or_default();
                                    return Err(format!(
                                        "Potential sqrt of negative number at v{}{}",
                                        dest.0, loc_info
                                    ));
                                }
                                ctx.backend.pop(1);
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        InstructionKind::FPow(dest, lhs, rhs) => {
            if let (Some(_z3_dest), Some(z3_l), Some(z3_r)) = (
                ctx.z3_floats.get(dest),
                ctx.z3_floats.get(lhs),
                ctx.z3_floats.get(rhs),
            ) {
                let ty = ctx.func.get_type(*lhs);
                let zero = if ty.is_float32() {
                    ctx.backend.float_from_f32(0.0)
                } else {
                    ctx.backend.float_from_f64(0.0)
                };

                // Optimization: Use interval analysis to skip Z3 check if base is strictly positive
                let is_safe = if let Some(interval) = ctx.analysis.intervals.get(lhs) {
                    interval.is_strictly_positive()
                } else {
                    false
                };

                if !is_safe {
                    // Verify safety via Z3
                    ctx.backend.push();
                    ctx.backend.assert(path_cond);

                    let is_base_zero = ctx.backend.float_eq(z3_l, &zero);
                    let is_exp_nonpositive = ctx.backend.float_le(z3_r, &zero);
                    let is_base_negative = ctx.backend.float_lt(z3_l, &zero);

                    let a1 = ctx.backend.bool_and(&[&is_base_zero, &is_exp_nonpositive]);
                    let domain_err = ctx.backend.bool_or(&[&a1, &is_base_negative]);

                    ctx.backend.assert(&domain_err);
                    if ctx.backend.check()? {
                        let loc_info = inst
                            .location
                            .map(|l| format!(" at {}", l))
                            .unwrap_or_default();
                        return Err(format!(
                            "Potential domain error in fpow at v{}{}",
                            dest.0, loc_info
                        ));
                    }
                    ctx.backend.pop(1);
                }
            }
        }

        InstructionKind::IToF(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_floats.get(dest), ctx.z3_bvs.get(src)) {
                let is_signed = !ctx.func.get_type(*src).is_unsigned();

                let is_f32 = ctx.func.get_type(*dest).is_float32();
                let res = ctx.backend.bv_to_float(s, is_signed, is_f32);
                let __inner = ctx.backend.float_eq(d, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FToI(dest, src, _) => {
            if let (Some(d), Some(s)) = (ctx.z3_bvs.get(dest), ctx.z3_floats.get(src)) {
                let is_signed = !ctx.func.get_type(*dest).is_unsigned();
                let bit_width = ctx.func.get_type(*dest).int_bit_width().unwrap_or(64);

                let res = ctx.backend.float_to_bv(s, is_signed, bit_width);
                let __inner = ctx.backend.bv_eq(d, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }
        InstructionKind::FConv(dest, src, target_ty) => {
            if let (Some(d), Some(s)) = (ctx.z3_floats.get(dest), ctx.z3_floats.get(src)) {
                let is_f32 = target_ty.is_float32();
                let res = ctx.backend.float_to_float(s, is_f32);
                let __inner = ctx.backend.float_eq(d, &res);
                let __tmp = ctx.backend.bool_implies(path_cond, &__inner);
                ctx.backend.assert(&__tmp);
            }
        }

        InstructionKind::SIMDSplat(..)
        | InstructionKind::SIMDExtractLane(..)
        | InstructionKind::SIMDInsertLane(..) => {}
        _ => {}
    }
    Ok(())
}
