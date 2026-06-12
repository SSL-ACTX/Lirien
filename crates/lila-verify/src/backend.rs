pub trait SolverBackend {
    type Bool: Clone;
    type Int: Clone;
    type Float: Clone;
    type BV: Clone;
    type Array: Clone;

    fn check(&mut self) -> Result<bool, String>;
    fn set_timeout(&mut self, timeout_ms: u32);
    fn assert(&mut self, cond: &Self::Bool);
    fn assert_implies(&mut self, premise: &Self::Bool, conclusion: &Self::Bool);
    fn push(&mut self);
    fn pop(&mut self, num: u32);

    fn bool_const(&mut self, name: &str) -> Self::Bool;
    fn bool_from_bool(&mut self, val: bool) -> Self::Bool;
    fn bool_not(&mut self, a: &Self::Bool) -> Self::Bool;
    fn bool_and(&mut self, args: &[&Self::Bool]) -> Self::Bool;
    fn bool_or(&mut self, args: &[&Self::Bool]) -> Self::Bool;
    fn bool_implies(&mut self, a: &Self::Bool, b: &Self::Bool) -> Self::Bool;
    fn bool_eq(&mut self, a: &Self::Bool, b: &Self::Bool) -> Self::Bool;
    fn bool_ite(&mut self, cond: &Self::Bool, then: &Self::BV, orelse: &Self::BV) -> Self::BV;
    fn float_ite(&mut self, cond: &Self::Bool, then: &Self::Float, orelse: &Self::Float) -> Self::Float;

    fn int_const(&mut self, name: &str) -> Self::Int;
    fn int_from_i64(&mut self, val: i64) -> Self::Int;
    fn int_eq(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool;
    fn int_ge(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool;
    fn int_lt(&mut self, a: &Self::Int, b: &Self::Int) -> Self::Bool;
    fn int_to_bv(&mut self, a: &Self::Int, bit_width: u32) -> Self::BV;

    fn bv_const(&mut self, name: &str, sz: u32) -> Self::BV;
    fn bv_from_i64(&mut self, val: i64, sz: u32) -> Self::BV;
    fn bv_to_int(&mut self, a: &Self::BV, is_signed: bool) -> Self::Int;
    fn bv_eq(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_add(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_sub(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_mul(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_sdiv(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_udiv(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_srem(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_urem(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_and(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_or(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_xor(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_not(&mut self, a: &Self::BV) -> Self::BV;
    fn bv_shl(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_lshr(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_ashr(&mut self, a: &Self::BV, b: &Self::BV) -> Self::BV;
    fn bv_slt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_sle(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_sgt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_sge(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_ult(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_ule(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_ugt(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;
    fn bv_uge(&mut self, a: &Self::BV, b: &Self::BV) -> Self::Bool;

    fn float_const(&mut self, name: &str, is_f32: bool) -> Self::Float;
    fn float_from_f32(&mut self, val: f32) -> Self::Float;
    fn float_from_f64(&mut self, val: f64) -> Self::Float;
    fn float_eq(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool;
    fn float_add(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float;
    fn float_sub(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float;
    fn float_mul(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float;
    fn float_div(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Float;
    fn float_lt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool;
    fn float_le(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool;
    fn float_gt(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool;
    fn float_ge(&mut self, a: &Self::Float, b: &Self::Float) -> Self::Bool;
    fn float_to_bv(&mut self, a: &Self::Float, is_signed: bool, bit_width: u32) -> Self::BV;
    fn float_to_float(&mut self, a: &Self::Float, is_f32: bool) -> Self::Float;
    fn bv_to_float(&mut self, a: &Self::BV, is_signed: bool, is_f32: bool) -> Self::Float;

    fn array_const(&mut self, name: &str, is_float: bool, bit_width: u32) -> Self::Array;
    fn array_select_bv(&mut self, a: &Self::Array, index: &Self::Int) -> Self::BV;
    fn array_select_float(&mut self, a: &Self::Array, index: &Self::Int) -> Self::Float;
    fn array_select_int(&mut self, a: &Self::Array, index: &Self::Int) -> Self::Int;
    fn array_store_bv(&mut self, a: &Self::Array, index: &Self::Int, val: &Self::BV)
        -> Self::Array;
    fn array_store_float(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::Float,
    ) -> Self::Array;
    fn array_store_int(
        &mut self,
        a: &Self::Array,
        index: &Self::Int,
        val: &Self::Int,
    ) -> Self::Array;
    fn array_eq(&mut self, a: &Self::Array, b: &Self::Array) -> Self::Bool;
}
