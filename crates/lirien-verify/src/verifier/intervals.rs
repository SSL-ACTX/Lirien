use crate::backend::SolverBackend;
use crate::verifier::TranslationContext;
use lirien_ir::analysis::interval::Bound;

pub fn assert_derived_intervals<
    B: SolverBackend<
        Bool = z3::ast::Bool,
        Int = z3::ast::Int,
        Float = z3::ast::Float,
        BV = z3::ast::BV,
        Array = z3::ast::Array,
    >,
>(
    t_ctx: &mut TranslationContext<'_, B>,
) {
    for (val, interval) in &t_ctx.analysis.intervals {
        tracing::debug!(target: "lirien::verify", "Value v{} has interval {:?}", val.0, interval);
        if let Some(ty) = t_ctx.func.value_types.get(val) {
            if let Some(z3_bv) = t_ctx.z3_bvs.get(val) {
                if let Some(bit_width) = ty.int_bit_width() {
                    let is_signed = ty.is_signed();
                    if let Bound::Finite(low) = interval.low {
                        let low_bv = t_ctx.backend.bv_from_i64(low as i64, bit_width);
                        if is_signed {
                            let __tmp = t_ctx.backend.bv_sge(z3_bv, &low_bv);
                            t_ctx.backend.assert(&__tmp);
                        } else {
                            let __tmp = t_ctx.backend.bv_uge(z3_bv, &low_bv);
                            t_ctx.backend.assert(&__tmp);
                        }
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_bv = t_ctx.backend.bv_from_i64(high as i64, bit_width);
                        if is_signed {
                            let __tmp = t_ctx.backend.bv_sle(z3_bv, &high_bv);
                            t_ctx.backend.assert(&__tmp);
                        } else {
                            let __tmp = t_ctx.backend.bv_ule(z3_bv, &high_bv);
                            t_ctx.backend.assert(&__tmp);
                        }
                    }
                }
            } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                if let Bound::Finite(low) = interval.low {
                    let low_float = if ty.is_float32() {
                        t_ctx.backend.float_from_f32(low as f32)
                    } else {
                        t_ctx.backend.float_from_f64(low)
                    };
                    let __tmp = t_ctx.backend.float_ge(z3_float, &low_float);
                    t_ctx.backend.assert(&__tmp);
                }
                if let Bound::Finite(high) = interval.high {
                    let high_float = if ty.is_float32() {
                        t_ctx.backend.float_from_f32(high as f32)
                    } else {
                        t_ctx.backend.float_from_f64(high)
                    };
                    let __tmp = t_ctx.backend.float_le(z3_float, &high_float);
                    t_ctx.backend.assert(&__tmp);
                }
            }
        }
    }

    for ((val, b_id), interval) in &t_ctx.analysis.block_narrowing {
        if let Some(path_cond) = t_ctx.block_conditions.get(b_id) {
            if let Some(ty) = t_ctx.func.value_types.get(val) {
                if let Some(z3_bv) = t_ctx.z3_bvs.get(val) {
                    if let Some(bit_width) = ty.int_bit_width() {
                        let is_signed = ty.is_signed();
                        if let Bound::Finite(low) = interval.low {
                            let low_bv = t_ctx.backend.bv_from_i64(low as i64, bit_width);
                            if is_signed {
                                let __inner = t_ctx.backend.bv_sge(z3_bv, &low_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            } else {
                                let __inner = t_ctx.backend.bv_uge(z3_bv, &low_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            }
                        }
                        if let Bound::Finite(high) = interval.high {
                            let high_bv = t_ctx.backend.bv_from_i64(high as i64, bit_width);
                            if is_signed {
                                let __inner = t_ctx.backend.bv_sle(z3_bv, &high_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            } else {
                                let __inner = t_ctx.backend.bv_ule(z3_bv, &high_bv);
                                let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                                t_ctx.backend.assert(&__tmp);
                            }
                        }
                    }
                } else if let Some(z3_float) = t_ctx.z3_floats.get(val) {
                    if let Bound::Finite(low) = interval.low {
                        let low_float = if ty.is_float32() {
                            t_ctx.backend.float_from_f32(low as f32)
                        } else {
                            t_ctx.backend.float_from_f64(low)
                        };
                        let __inner = t_ctx.backend.float_ge(z3_float, &low_float);
                        let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                        t_ctx.backend.assert(&__tmp);
                    }
                    if let Bound::Finite(high) = interval.high {
                        let high_float = if ty.is_float32() {
                            t_ctx.backend.float_from_f32(high as f32)
                        } else {
                            t_ctx.backend.float_from_f64(high)
                        };
                        let __inner = t_ctx.backend.float_le(z3_float, &high_float);
                        let __tmp = t_ctx.backend.bool_implies(path_cond, &__inner);
                        t_ctx.backend.assert(&__tmp);
                    }
                }
            }
        }
    }
}
