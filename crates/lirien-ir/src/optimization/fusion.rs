use super::super::ir::{Function, InstructionKind, Value, FusedExpr, SourceLocation, Type};
use std::collections::HashMap;
use tracing::info;

pub fn fuse_tensor_kernels(func: &mut Function) {
    info!(target: "lirien::ssa::opt", "Starting fusion for '{}'", func.name);

    // 1. Compute use counts for all values in the function.
    let mut use_counts = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            for operand in inst.get_uses() {
                *use_counts.entry(operand).or_insert(0) += 1;
            }
        }
    }

    info!(target: "lirien::ssa::opt", "Use counts: {:?}", use_counts);

    // 2. Iterate through each block and fuse tensor operations.
    let value_types = &func.value_types;
    for block in &mut func.blocks {
        // Map from defined value in this block to its instruction kind and location.
        let mut def_map = HashMap::new();
        
        for inst in &mut block.instructions {
            let mut fused_opt = None;
            
            info!(target: "lirien::ssa::opt", "Processing instruction: {:?}", inst.kind);

            // Check if this instruction is a candidate to become or extend a fused tensor op.
            match &inst.kind {
                InstructionKind::TensorAdd(dest, lhs, rhs)
                    if should_fuse(*lhs, &def_map, &use_counts) || should_fuse(*rhs, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let left_expr = get_fused_expr(*lhs, &def_map, &use_counts, value_types, &mut inputs);
                        let right_expr = get_fused_expr(*rhs, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Add(Box::new(left_expr), Box::new(right_expr))));
                }
                InstructionKind::TensorSub(dest, lhs, rhs)
                    if should_fuse(*lhs, &def_map, &use_counts) || should_fuse(*rhs, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let left_expr = get_fused_expr(*lhs, &def_map, &use_counts, value_types, &mut inputs);
                        let right_expr = get_fused_expr(*rhs, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Sub(Box::new(left_expr), Box::new(right_expr))));
                }
                InstructionKind::TensorMul(dest, lhs, rhs)
                    if should_fuse(*lhs, &def_map, &use_counts) || should_fuse(*rhs, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let left_expr = get_fused_expr(*lhs, &def_map, &use_counts, value_types, &mut inputs);
                        let right_expr = get_fused_expr(*rhs, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Mul(Box::new(left_expr), Box::new(right_expr))));
                }
                InstructionKind::TensorDiv(dest, lhs, rhs)
                    if should_fuse(*lhs, &def_map, &use_counts) || should_fuse(*rhs, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let left_expr = get_fused_expr(*lhs, &def_map, &use_counts, value_types, &mut inputs);
                        let right_expr = get_fused_expr(*rhs, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Div(Box::new(left_expr), Box::new(right_expr))));
                }
                InstructionKind::TensorScalarAdd(dest, tensor, scalar)
                    if should_fuse(*tensor, &def_map, &use_counts) || should_fuse(*scalar, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let t_expr = get_fused_expr(*tensor, &def_map, &use_counts, value_types, &mut inputs);
                        let s_expr = get_fused_expr(*scalar, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Add(Box::new(t_expr), Box::new(s_expr))));
                }
                InstructionKind::TensorScalarSub(dest, tensor, scalar)
                    if should_fuse(*tensor, &def_map, &use_counts) || should_fuse(*scalar, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let t_expr = get_fused_expr(*tensor, &def_map, &use_counts, value_types, &mut inputs);
                        let s_expr = get_fused_expr(*scalar, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Sub(Box::new(t_expr), Box::new(s_expr))));
                }
                InstructionKind::TensorScalarMul(dest, tensor, scalar)
                    if should_fuse(*tensor, &def_map, &use_counts) || should_fuse(*scalar, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let t_expr = get_fused_expr(*tensor, &def_map, &use_counts, value_types, &mut inputs);
                        let s_expr = get_fused_expr(*scalar, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Mul(Box::new(t_expr), Box::new(s_expr))));
                }
                InstructionKind::TensorScalarDiv(dest, tensor, scalar)
                    if should_fuse(*tensor, &def_map, &use_counts) || should_fuse(*scalar, &def_map, &use_counts) => {
                        let mut inputs = Vec::new();
                        let t_expr = get_fused_expr(*tensor, &def_map, &use_counts, value_types, &mut inputs);
                        let s_expr = get_fused_expr(*scalar, &def_map, &use_counts, value_types, &mut inputs);
                        fused_opt = Some((*dest, inputs, FusedExpr::Div(Box::new(t_expr), Box::new(s_expr))));
                }
                _ => {}
            }

            if let Some((dest, inputs, expr)) = fused_opt {
                info!(target: "lirien::ssa::opt", "FUSED instruction generated for v{}!", dest.0);
                inst.kind = InstructionKind::TensorFused(dest, inputs, expr);
            }

            // Record the definition of this instruction.
            if let Some(def_val) = inst.get_def() {
                def_map.insert(def_val, (inst.kind.clone(), inst.location));
            }
        }
    }
}

fn should_fuse(
    val: Value,
    def_map: &HashMap<Value, (InstructionKind, Option<SourceLocation>)>,
    use_counts: &HashMap<Value, usize>,
) -> bool {
    if let Some((kind, _)) = def_map.get(&val) {
        let u_count = use_counts.get(&val).copied().unwrap_or(0);
        info!(target: "lirien::ssa::opt", "Checking should_fuse for v{} (u_count: {}), kind: {:?}", val.0, u_count, kind);
        if u_count == 1 {
            match kind {
                InstructionKind::TensorAdd(..)
                | InstructionKind::TensorSub(..)
                | InstructionKind::TensorMul(..)
                | InstructionKind::TensorDiv(..)
                | InstructionKind::TensorScalarAdd(..)
                | InstructionKind::TensorScalarSub(..)
                | InstructionKind::TensorScalarMul(..)
                | InstructionKind::TensorScalarDiv(..)
                | InstructionKind::TensorFused(..) => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

fn get_fused_expr(
    val: Value,
    def_map: &HashMap<Value, (InstructionKind, Option<SourceLocation>)>,
    use_counts: &HashMap<Value, usize>,
    value_types: &HashMap<Value, Type>,
    fused_inputs: &mut Vec<Value>,
) -> FusedExpr {
    if let Some((kind, _loc)) = def_map.get(&val) {
        if use_counts.get(&val).copied().unwrap_or(0) == 1 {
            match kind {
                InstructionKind::TensorAdd(_, l, r) => {
                    let left_expr = get_fused_expr(*l, def_map, use_counts, value_types, fused_inputs);
                    let right_expr = get_fused_expr(*r, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Add(Box::new(left_expr), Box::new(right_expr));
                }
                InstructionKind::TensorSub(_, l, r) => {
                    let left_expr = get_fused_expr(*l, def_map, use_counts, value_types, fused_inputs);
                    let right_expr = get_fused_expr(*r, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Sub(Box::new(left_expr), Box::new(right_expr));
                }
                InstructionKind::TensorMul(_, l, r) => {
                    let left_expr = get_fused_expr(*l, def_map, use_counts, value_types, fused_inputs);
                    let right_expr = get_fused_expr(*r, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Mul(Box::new(left_expr), Box::new(right_expr));
                }
                InstructionKind::TensorDiv(_, l, r) => {
                    let left_expr = get_fused_expr(*l, def_map, use_counts, value_types, fused_inputs);
                    let right_expr = get_fused_expr(*r, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Div(Box::new(left_expr), Box::new(right_expr));
                }
                InstructionKind::TensorScalarAdd(_, t, s) => {
                    let t_expr = get_fused_expr(*t, def_map, use_counts, value_types, fused_inputs);
                    let s_expr = get_fused_expr(*s, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Add(Box::new(t_expr), Box::new(s_expr));
                }
                InstructionKind::TensorScalarSub(_, t, s) => {
                    let t_expr = get_fused_expr(*t, def_map, use_counts, value_types, fused_inputs);
                    let s_expr = get_fused_expr(*s, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Sub(Box::new(t_expr), Box::new(s_expr));
                }
                InstructionKind::TensorScalarMul(_, t, s) => {
                    let t_expr = get_fused_expr(*t, def_map, use_counts, value_types, fused_inputs);
                    let s_expr = get_fused_expr(*s, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Mul(Box::new(t_expr), Box::new(s_expr));
                }
                InstructionKind::TensorScalarDiv(_, t, s) => {
                    let t_expr = get_fused_expr(*t, def_map, use_counts, value_types, fused_inputs);
                    let s_expr = get_fused_expr(*s, def_map, use_counts, value_types, fused_inputs);
                    return FusedExpr::Div(Box::new(t_expr), Box::new(s_expr));
                }
                InstructionKind::TensorFused(_, inputs, expr) => {
                    for &in_val in inputs {
                        if !fused_inputs.contains(&in_val) {
                            fused_inputs.push(in_val);
                        }
                    }
                    return expr.clone();
                }
                _ => {}
            }
        }
    }

    if !fused_inputs.contains(&val) {
        fused_inputs.push(val);
    }
    let is_tensor = value_types.get(&val).map(|t| t.is_tensor()).unwrap_or(false);
    if is_tensor {
        FusedExpr::Input(val)
    } else {
        FusedExpr::Scalar(val)
    }
}
