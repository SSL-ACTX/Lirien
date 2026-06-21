use crate::backend::SolverBackend;
use z3::ast::{Array, Ast, Bool, Float, Int, BV};
use z3::{Context, Params, SatResult, Solver};
use z3_sys;

pub struct Z3Backend<'ctx> {
    ctx: &'ctx Context,
    solver: &'ctx Solver,
}

impl<'ctx> Z3Backend<'ctx> {
    pub fn new(ctx: &'ctx Context, solver: &'ctx Solver) -> Self {
        Self { ctx, solver }
    }

    fn unify_floats(&mut self, a: &Float, b: &Float) -> (Float, Float) {
        let sort_a = a.get_sort();
        let sort_b = b.get_sort();
        if sort_a == sort_b {
            return (a.clone(), b.clone());
        }

        unsafe {
            let context = self.ctx.get_z3_context();
            let ebits_a = z3_sys::Z3_fpa_get_ebits(context, sort_a.get_z3_sort());
            let ebits_b = z3_sys::Z3_fpa_get_ebits(context, sort_b.get_z3_sort());

            if ebits_a > ebits_b {
                // Promote b to sort of a
                let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(context)
                    .expect("Rounding mode failed");
                let promoted = z3_sys::Z3_mk_fpa_to_fp_float(
                    context,
                    rm,
                    b.get_z3_ast(),
                    sort_a.get_z3_sort(),
                );
                (a.clone(), Float::wrap(self.ctx, promoted.expect("Promotion failed")))
            } else {
                // Promote a to sort of b
                let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(context)
                    .expect("Rounding mode failed");
                let promoted = z3_sys::Z3_mk_fpa_to_fp_float(
                    context,
                    rm,
                    a.get_z3_ast(),
                    sort_b.get_z3_sort(),
                );
                (Float::wrap(self.ctx, promoted.expect("Promotion failed")), b.clone())
            }
        }
    }
}

impl<'ctx> SolverBackend for Z3Backend<'ctx> {
    type Bool = Bool;
    type Int = Int;
    type Float = Float;
    type BV = BV;
    type Array = Array;

    fn check(&mut self) -> Result<bool, String> {
        match self.solver.check() {
            SatResult::Sat => {
                if let Some(model) = self.solver.get_model() {
                    tracing::debug!(target: "lirien::verify", "[Z3 MODEL]:\n{:?}", model);
                } else {
                    tracing::debug!(target: "lirien::verify", "[Z3 MODEL]: None");
                }
                Ok(true)
            }
            SatResult::Unsat => Ok(false),
            SatResult::Unknown => {
                let reason = self.solver.get_reason_unknown().unwrap_or_else(|| "unknown".to_string());
                Err(format!("Z3 returned Unknown: {}", reason))
            }
        }
    }

    fn set_timeout(&mut self, timeout_ms: u32) {
        let mut params = Params::new();
        params.set_u32("timeout", timeout_ms);
        self.solver.set_params(&params);
    }

    fn assert(&mut self, cond: &Self::Bool) {
        self.solver.assert(cond);
    }

    fn assert_implies(&mut self, premise: &Self::Bool, conclusion: &Self::Bool) {
        self.solver.assert(premise.implies(conclusion));
    }

    fn push(&mut self) {
        self.solver.push();
    }

    fn pop(&mut self, num: u32) {
        self.solver.pop(num);
    }

    fn bool_const(&mut self, name: &str) -> Self::Bool {
        Bool::new_const(name)
    }

    fn bool_from_bool(&mut self, val: bool) -> Self::Bool {
        Bool::from_bool(val)
    }

    fn bool_not(&mut self, a: &Self::Bool) -> Self::Bool {
        a.not()
    }

    fn bool_and(&mut self, args: &[&Self::Bool]) -> Self::Bool {
        Bool::and(args)
    }

    fn bool_or(&mut self, args: &[&Self::Bool]) -> Self::Bool {
        Bool::or(args)
    }

    fn bool_implies(&mut self, a: &Self::Bool, b: &Self::Bool) -> Self::Bool {
        a.implies(b)
    }

    fn bool_eq(&mut self, a: &Self::Bool, b: &Self::Bool) -> Self::Bool {
        a.eq(b)
    }

    fn bool_ite(&mut self, cond: &Self::Bool, then: &Self::BV, orelse: &Self::BV) -> Self::BV {
        cond.ite(then, orelse)
    }

    fn float_ite(&mut self, cond: &Self::Bool, then: &Self::Float, orelse: &Self::Float) -> Self::Float {
        let (t, o) = self.unify_floats(then, orelse);
        unsafe {
            let ast = z3_sys::Z3_mk_ite(
                self.ctx.get_z3_context(),
                cond.get_z3_ast(),
                t.get_z3_ast(),
                o.get_z3_ast(),
            );
            Float::wrap(self.ctx, ast.expect("Z3_mk_ite failed"))
        }
    }

    fn int_const(&mut self, name: &str) -> Self::Int {
        Int::new_const(name)
    }

    fn int_from_i64(&mut self, val: i64) -> Self::Int {
        Int::from_i64(val)
    }

    fn int_add(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Int {
        Int::add(&[a, b])
    }

    fn int_sub(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Int {
        Int::sub(&[a, b])
    }

    fn int_eq(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool {
        a.eq(b)
    }

    fn int_ge(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool {
        a.ge(b)
    }

    fn int_lt(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool {
        a.lt(b)
    }

    fn int_to_bv(&mut self, a: &Self::Int, bit_width: u32) -> Self::BV {
        unsafe {
            BV::wrap(
                self.ctx,
                z3_sys::Z3_mk_int2bv(self.ctx.get_z3_context(), bit_width, a.get_z3_ast()).expect("Z3_mk_int2bv failed"),
            )
        }
    }

    fn bv_const(&mut self, name: &str, sz: u32) -> Self::BV {
        BV::new_const(name, sz)
    }

    fn bv_from_i64(&mut self, val: i64, sz: u32) -> Self::BV {
        BV::from_i64(val, sz)
    }

    fn bv_to_int(&mut self, a: &Self::BV, is_signed: bool) -> Self::Int {
        a.to_int(is_signed)
    }

    fn bv_eq(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.eq(b)
    }

    fn bv_add(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvadd(b)
    }

    fn bv_sub(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvsub(b)
    }

    fn bv_mul(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvmul(b)
    }

    fn bv_sdiv(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvsdiv(b)
    }

    fn bv_udiv(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvudiv(b)
    }

    fn bv_srem(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvsrem(b)
    }

    fn bv_urem(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvurem(b)
    }

    fn bv_and(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvand(b)
    }

    fn bv_or(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvor(b)
    }

    fn bv_xor(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvxor(b)
    }

    fn bv_not(&mut self, a: &Self::BV) -> Self::BV {
        a.bvnot()
    }

    fn bv_shl(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvshl(b)
    }

    fn bv_lshr(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvlshr(b)
    }

    fn bv_ashr(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV {
        a.bvashr(b)
    }

    fn bv_sext(&mut self, a: &Self::BV, sz: u32) -> Self::BV {
        a.sign_ext(sz)
    }

    fn bv_zext(&mut self, a: &Self::BV, sz: u32) -> Self::BV {
        a.zero_ext(sz)
    }

    fn bv_extract(&mut self, a: &Self::BV, high: u32, low: u32) -> Self::BV {
        a.extract(high, low)
    }

    fn bv_slt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvslt(b)
    }

    fn bv_sle(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvsle(b)
    }

    fn bv_sgt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvsgt(b)
    }

    fn bv_sge(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvsge(b)
    }

    fn bv_ult(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvult(b)
    }

    fn bv_ule(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvule(b)
    }

    fn bv_ugt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvugt(b)
    }

    fn bv_uge(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool {
        a.bvuge(b)
    }

    fn float_const(&mut self, name: &str, is_f32: bool) -> Self::Float {
        if is_f32 {
            Float::new_const(name, 8, 24)
        } else {
            Float::new_const(name, 11, 53)
        }
    }

    fn float_from_f32(&mut self, val: f32) -> Self::Float {
        Float::from_f32(val)
    }

    fn float_from_f64(&mut self, val: f64) -> Self::Float {
        Float::from_f64(val)
    }

    fn float_eq(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let ast = z3_sys::Z3_mk_eq(
                self.ctx.get_z3_context(),
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Bool::wrap(self.ctx, ast.expect("Z3_mk_eq failed"))
        }
    }

    fn float_add(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let ast = z3_sys::Z3_mk_fpa_add(
                self.ctx.get_z3_context(),
                rm,
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Float::wrap(self.ctx, ast.expect("Z3_mk_fpa_add failed"))
        }
    }

    fn float_sub(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let ast = z3_sys::Z3_mk_fpa_sub(
                self.ctx.get_z3_context(),
                rm,
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Float::wrap(self.ctx, ast.expect("Z3_mk_fpa_sub failed"))
        }
    }

    fn float_mul(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let ast = z3_sys::Z3_mk_fpa_mul(
                self.ctx.get_z3_context(),
                rm,
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Float::wrap(self.ctx, ast.expect("Z3_mk_fpa_mul failed"))
        }
    }

    fn float_div(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let ast = z3_sys::Z3_mk_fpa_div(
                self.ctx.get_z3_context(),
                rm,
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Float::wrap(self.ctx, ast.expect("Z3_mk_fpa_div failed"))
        }
    }

    fn float_lt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let ast = z3_sys::Z3_mk_fpa_lt(
                self.ctx.get_z3_context(),
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Bool::wrap(self.ctx, ast.expect("Z3_mk_fpa_lt failed"))
        }
    }

    fn float_le(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let ast = z3_sys::Z3_mk_fpa_leq(
                self.ctx.get_z3_context(),
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Bool::wrap(self.ctx, ast.expect("Z3_mk_fpa_leq failed"))
        }
    }

    fn float_gt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let ast = z3_sys::Z3_mk_fpa_gt(
                self.ctx.get_z3_context(),
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Bool::wrap(self.ctx, ast.expect("Z3_mk_fpa_gt failed"))
        }
    }

    fn float_ge(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        let (lhs, rhs) = self.unify_floats(a, b);
        unsafe {
            let ast = z3_sys::Z3_mk_fpa_geq(
                self.ctx.get_z3_context(),
                lhs.get_z3_ast(),
                rhs.get_z3_ast(),
            );
            Bool::wrap(self.ctx, ast.expect("Z3_mk_fpa_geq failed"))
        }
    }

    fn float_to_bv(&mut self, a: &Self::Float, is_signed: bool, bit_width: u32) -> Self::BV {
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let conv = if is_signed {
                z3_sys::Z3_mk_fpa_to_sbv(
                    self.ctx.get_z3_context(),
                    rm,
                    a.get_z3_ast(),
                    bit_width,
                )
            } else {
                z3_sys::Z3_mk_fpa_to_ubv(
                    self.ctx.get_z3_context(),
                    rm,
                    a.get_z3_ast(),
                    bit_width,
                )
            };
            BV::wrap(self.ctx, conv.expect("Z3_mk_fpa_to_bv failed"))
        }
    }

    fn float_to_float(&mut self, a: &Self::Float, is_f32: bool) -> Self::Float {
        let sort = if is_f32 {
            z3::Sort::float32()
        } else {
            z3::Sort::double()
        };
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let conv = z3_sys::Z3_mk_fpa_to_fp_float(
                self.ctx.get_z3_context(),
                rm,
                a.get_z3_ast(),
                sort.get_z3_sort(),
            );
            Float::wrap(self.ctx, conv.expect("Z3_mk_fpa_to_fp_float failed"))
        }
    }

    fn bv_to_float(&mut self, a: &Self::BV, is_signed: bool, is_f32: bool) -> Self::Float {
        let sort = if is_f32 {
            z3::Sort::float32()
        } else {
            z3::Sort::double()
        };
        unsafe {
            let rm = z3_sys::Z3_mk_fpa_round_nearest_ties_to_even(self.ctx.get_z3_context()).expect("Rounding mode failed");
            let conv = if is_signed {
                z3_sys::Z3_mk_fpa_to_fp_signed(
                    self.ctx.get_z3_context(),
                    rm,
                    a.get_z3_ast(),
                    sort.get_z3_sort(),
                )
            } else {
                z3_sys::Z3_mk_fpa_to_fp_unsigned(
                    self.ctx.get_z3_context(),
                    rm,
                    a.get_z3_ast(),
                    sort.get_z3_sort(),
                )
            };
            Float::wrap(self.ctx, conv.expect("Z3_mk_fpa_to_fp failed"))
        }
    }

    fn bv_bitcast_to_float(&mut self, a: &Self::BV, is_f32: bool) -> Self::Float {
        let sort = if is_f32 {
            z3::Sort::float32()
        } else {
            z3::Sort::double()
        };
        unsafe {
            let conv = z3_sys::Z3_mk_fpa_to_fp_bv(
                self.ctx.get_z3_context(),
                a.get_z3_ast(),
                sort.get_z3_sort(),
            );
            Float::wrap(self.ctx, conv.expect("Z3_mk_fpa_to_fp_bv failed"))
        }
    }

    fn float_bitcast_to_bv(&mut self, a: &Self::Float) -> Self::BV {
        unsafe {
            let conv = z3_sys::Z3_mk_fpa_to_ieee_bv(self.ctx.get_z3_context(), a.get_z3_ast());
            BV::wrap(self.ctx, conv.expect("Z3_mk_fpa_to_ieee_bv failed"))
        }
    }

    fn array_const(&mut self, name: &str, is_float: bool, bit_width: u32) -> Self::Array {
        let domain = z3::Sort::int();
        let range = if is_float {
            if bit_width == 32 {
                z3::Sort::float32()
            } else {
                z3::Sort::double()
            }
        } else {
            z3::Sort::bitvector(bit_width)
        };
        Array::new_const(name, &domain, &range)
    }

    fn array_select_bv(&mut self, a: &Self::Array, index: &Self::Int) -> Self::BV {
        a.select(index).as_bv().expect("array_select_bv failed")
    }

    fn array_select_float(&mut self, a: &Self::Array, index: &Self::Int, is_f32: bool) -> Self::Float {
        let res = a.select(index);
        let sort = res.get_sort();
        let kind = unsafe { z3_sys::Z3_get_sort_kind(self.ctx.get_z3_context(), sort.get_z3_sort()) };
        
        if kind == z3_sys::SortKind::FloatingPoint {
            res.as_float().expect("array_select_float: expected float")
        } else if kind == z3_sys::SortKind::Bv {
            let bv = res.as_bv().expect("array_select_float failed: expected bv");
            let bv_size = bv.get_size();
            let target_size = if is_f32 { 32 } else { 64 };
            
            if bv_size == target_size {
                self.bv_bitcast_to_float(&bv, is_f32)
            } else if bv_size > target_size {
                // Extract lower bits and bitcast
                let extracted = self.bv_extract(&bv, target_size - 1, 0);
                self.bv_bitcast_to_float(&extracted, is_f32)
            } else {
                // Zero-extend and bitcast (unusual case)
                let extended = self.bv_zext(&bv, target_size - bv_size);
                self.bv_bitcast_to_float(&extended, is_f32)
            }
        } else {
            panic!("array_select_float: unexpected sort kind {:?}", kind);
        }
    }

    fn array_select_int(&mut self, a: &Self::Array, index: &Self::Int) -> Self::Int {
        a.select(index).as_int().expect("array_select_int failed")
    }

    fn array_store_bv(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::BV,
    ) -> Self::Array {
        let array_sort = a.get_sort();
        let array_range_sort = array_sort.array_range().unwrap();
        let kind = unsafe { z3_sys::Z3_get_sort_kind(self.ctx.get_z3_context(), array_range_sort.get_z3_sort()) };
        
        if kind == z3_sys::SortKind::Bv {
            let bv_size = unsafe { z3_sys::Z3_get_bv_sort_size(self.ctx.get_z3_context(), array_range_sort.get_z3_sort()) };
            let val_size = val.get_size();
            if bv_size == val_size {
                a.store(index, val)
            } else if val_size > bv_size {
                let truncated = self.bv_extract(val, bv_size - 1, 0);
                a.store(index, &truncated)
            } else {
                let extended = self.bv_zext(val, bv_size - val_size);
                a.store(index, &extended)
            }
        } else {
             // Fallback
             a.store(index, val)
        }
    }

    fn array_store_float(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::Float,
        is_f32: bool,
    ) -> Self::Array {
        let array_sort = a.get_sort();
        let array_range_sort = array_sort.array_range().unwrap();
        let kind = unsafe { z3_sys::Z3_get_sort_kind(self.ctx.get_z3_context(), array_range_sort.get_z3_sort()) };
        
        if kind == z3_sys::SortKind::FloatingPoint {
            a.store(index, val)
        } else if kind == z3_sys::SortKind::Bv {
            let bv_size = unsafe { z3_sys::Z3_get_bv_sort_size(self.ctx.get_z3_context(), array_range_sort.get_z3_sort()) };
            let mut bv_val = self.float_bitcast_to_bv(val);
            let target_size = if is_f32 { 32 } else { 64 };
            
            // Adjust bv_val if float bit width doesn't match target bv width
            if bv_size != target_size {
                 if target_size > bv_size {
                     bv_val = self.bv_extract(&bv_val, bv_size - 1, 0);
                 } else {
                     bv_val = self.bv_zext(&bv_val, bv_size - target_size);
                 }
            }
            a.store(index, &bv_val)
        } else {
            a.store(index, val)
        }
    }

    fn array_store_int(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::Int,
    ) -> Self::Array {
        a.store(index, val)
    }

    fn array_eq(&mut self, a: &Self::Array, b: &Self::Array) -> Self::Bool {
        a.eq(b)
    }
}
