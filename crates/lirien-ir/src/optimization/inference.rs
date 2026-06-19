use crate::ir::Function;

pub fn embed_intervals(_func: &mut Function) {
    // Architectural Choice: We skip embedding simple bounds as Liquid Types here because they
    // are handled more efficiently via direct SMT assertions in the verifier's 'Early Out' pass.
    // This avoids expensive string parsing in Z3 for trivial facts.
}
