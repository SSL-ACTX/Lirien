use crate::backend::SolverBackend;
use z3::ast::{Array, Ast, Bool, Float, Int, RoundingMode, BV};
use z3::{Context, Params, SatResult, Solver};

pub struct Z3Backend<'ctx> {
    ctx: &'ctx Context,
    solver: &'ctx Solver,
}

impl<'ctx> Z3Backend<'ctx> {
    pub fn new(ctx: &'ctx Context, solver: &'ctx Solver) -> Self {
        Self { ctx, solver }
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
            SatResult::Sat => Ok(true),
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
        let refs = args.to_vec();
        Bool::and(&refs)
    }

    fn bool_or(&mut self, args: &[&Self::Bool]) -> Self::Bool {
        let refs = args.to_vec();
        Bool::or(&refs)
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
        cond.ite(then, orelse)
    }

    fn int_const(&mut self, name: &str) -> Self::Int {
        Int::new_const(name)
    }

    fn int_from_i64(&mut self, val: i64) -> Self::Int {
        Int::from_i64(val)
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
                z3_sys::Z3_mk_int2bv(self.ctx.get_z3_context(), bit_width, a.get_z3_ast()).unwrap(),
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
        let current_sz = a.get_size();
        if sz > current_sz {
            a.sign_ext(sz - current_sz)
        } else {
            a.clone()
        }
    }

    fn bv_zext(&mut self, a: &Self::BV, sz: u32) -> Self::BV {
        let current_sz = a.get_size();
        if sz > current_sz {
            a.zero_ext(sz - current_sz)
        } else {
            a.clone()
        }
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
        a.eq(b)
    }

    fn float_add(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        rm.add(a, b)
    }

    fn float_sub(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        rm.sub(a, b)
    }

    fn float_mul(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        rm.mul(a, b)
    }

    fn float_div(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        rm.div(a, b)
    }

    fn float_lt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        a.lt(b)
    }

    fn float_le(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        a.le(b)
    }

    fn float_gt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        a.gt(b)
    }

    fn float_ge(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool {
        a.ge(b)
    }

    fn float_to_bv(&mut self, a: &Self::Float, is_signed: bool, bit_width: u32) -> Self::BV {
        let rm = RoundingMode::round_nearest_ties_to_even();
        unsafe {
            let conv = if is_signed {
                z3_sys::Z3_mk_fpa_to_sbv(
                    self.ctx.get_z3_context(),
                    rm.get_z3_ast(),
                    a.get_z3_ast(),
                    bit_width,
                )
            } else {
                z3_sys::Z3_mk_fpa_to_ubv(
                    self.ctx.get_z3_context(),
                    rm.get_z3_ast(),
                    a.get_z3_ast(),
                    bit_width,
                )
            };
            BV::wrap(self.ctx, conv.unwrap())
        }
    }

    fn float_to_float(&mut self, a: &Self::Float, is_f32: bool) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        let sort = if is_f32 {
            z3::Sort::float32()
        } else {
            z3::Sort::double()
        };
        unsafe {
            let conv = z3_sys::Z3_mk_fpa_to_fp_float(
                self.ctx.get_z3_context(),
                rm.get_z3_ast(),
                a.get_z3_ast(),
                sort.get_z3_sort(),
            );
            Float::wrap(self.ctx, conv.unwrap())
        }
    }

    fn bv_to_float(&mut self, a: &Self::BV, is_signed: bool, is_f32: bool) -> Self::Float {
        let rm = RoundingMode::round_nearest_ties_to_even();
        let sort = if is_f32 {
            z3::Sort::float32()
        } else {
            z3::Sort::double()
        };
        unsafe {
            let conv = if is_signed {
                z3_sys::Z3_mk_fpa_to_fp_signed(
                    self.ctx.get_z3_context(),
                    rm.get_z3_ast(),
                    a.get_z3_ast(),
                    sort.get_z3_sort(),
                )
            } else {
                z3_sys::Z3_mk_fpa_to_fp_unsigned(
                    self.ctx.get_z3_context(),
                    rm.get_z3_ast(),
                    a.get_z3_ast(),
                    sort.get_z3_sort(),
                )
            };
            Float::wrap(self.ctx, conv.unwrap())
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
        a.select(index).as_bv().unwrap()
    }

    fn array_select_float(&mut self, a: &Self::Array, index: &Self::Int) -> Self::Float {
        a.select(index).as_float().unwrap()
    }

    fn array_select_int(&mut self, a: &Self::Array, index: &Self::Int) -> Self::Int {
        a.select(index).as_int().unwrap()
    }

    fn array_store_bv(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::BV,
    ) -> Self::Array {
        a.store(index, val)
    }

    fn array_store_float(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::Float,
    ) -> Self::Array {
        a.store(index, val)
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
