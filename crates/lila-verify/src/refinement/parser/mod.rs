pub(crate) mod bitvectors;
pub(crate) mod booleans;
pub(crate) mod floats;
pub(crate) mod integers;
pub(crate) mod reals;

use crate::refinement::resolver::Resolver;
use z3::ast::{Array, Bool, Float, Int, Real, BV};

pub fn parse_bool_expr_with_resolver(sexpr: &str, resolver: &Resolver) -> Result<Bool, String> {
    booleans::parse_bool_expr(sexpr, None, None, None, None, None, Some(resolver))
}

pub fn parse_refinement(refinement: &str, v: &Int, v_bv: Option<&BV>) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    booleans::parse_bool_expr(&refinement, Some(v), None, None, None, v_bv, None)
}

pub fn parse_float_refinement(refinement: &str, v: &Float) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    booleans::parse_bool_expr(&refinement, None, None, Some(v), None, None, None)
}

pub fn parse_real_refinement(refinement: &str, v: &Real) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    booleans::parse_bool_expr(&refinement, None, Some(v), None, None, None, None)
}

pub fn parse_array_refinement(refinement: &str, v: &Array, is_float: bool) -> Result<Bool, String> {
    let refinement = refinement.replace("{v}", "VALUE_PLACEHOLDER");
    if is_float {
        booleans::parse_bool_expr(
            &refinement,
            None,
            None,
            Some(&Float::new_const_double("DUMMY")),
            Some(v),
            None,
            None,
        )
    } else {
        booleans::parse_bool_expr(
            &refinement,
            Some(&Int::new_const("DUMMY")),
            None,
            None,
            Some(v),
            Some(&BV::new_const("DUMMY", 64)),
            None,
        )
    }
}
