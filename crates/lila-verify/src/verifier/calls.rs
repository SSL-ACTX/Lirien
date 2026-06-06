use crate::verifier::TranslationContext;
use lila_ir::ir::{Instruction, Value};
use z3::ast::Bool;

pub fn translate<
    B: crate::backend::SolverBackend<
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
    if let lila_ir::ir::InstructionKind::Call(dest, target_name, _args) = &inst.kind {
        let registry = lila_ir::registry::GLOBAL_REGISTRY.lock().unwrap();

        let sig = if target_name == &t_ctx.func.name {
            // Recursive call: use current function's signature
            let mut arg_types = Vec::new();
            let mut arg_refinements = std::collections::HashMap::new();
            for i in 0..t_ctx.func.arg_count {
                let v = Value(i);
                arg_types.push(t_ctx.func.get_type(v));
                if let Some(ref_str) = t_ctx.func.refinements.get(&v) {
                    arg_refinements.insert(i, ref_str.clone());
                }
            }
            Some(lila_ir::registry::FunctionSignature {
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
                    crate::refinement_parser::parse_refinement(ret_ref, &bv_int, Some(z3_bv))
                } else if let Some(z3_int) = t_ctx.z3_ints.get(dest) {
                    crate::refinement_parser::parse_refinement(ret_ref, z3_int, None)
                } else if let Some(z3_float) = t_ctx.z3_floats.get(dest) {
                    crate::refinement_parser::parse_float_refinement(ret_ref, z3_float)
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
