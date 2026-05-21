use crate::ssa::ir::{BlockId, Function, Value};
use std::collections::HashMap;
use z3::ast::{Array, Int, Real};
use z3::{Context, Solver};

pub mod arithmetic;
pub mod control_flow;
pub mod memory;

pub struct TranslationContext<'ctx> {
    pub ctx: &'ctx Context,
    pub solver: &'ctx Solver<'ctx>,
    pub func: &'ctx Function,
    pub z3_ints: HashMap<Value, Int<'ctx>>,
    pub z3_reals: HashMap<Value, Real<'ctx>>,
    pub z3_arrays: HashMap<Value, Array<'ctx>>,
    pub block_conditions: HashMap<BlockId, z3::ast::Bool<'ctx>>,
    pub edge_conditions: HashMap<(BlockId, BlockId), z3::ast::Bool<'ctx>>,
}

pub fn verify_with_context<'ctx>(
    ctx: &'ctx Context,
    solver: &Solver<'ctx>,
    func: &Function,
) -> Result<(), String> {
    let mut t_ctx = TranslationContext {
        ctx,
        solver,
        func,
        z3_ints: HashMap::new(),
        z3_reals: HashMap::new(),
        z3_arrays: HashMap::new(),
        block_conditions: HashMap::new(),
        edge_conditions: HashMap::new(),
    };

    // Initialize Z3 values for all SSA values
    memory::init_values(&mut t_ctx)?;

    t_ctx
        .block_conditions
        .insert(func.entry_block, z3::ast::Bool::from_bool(ctx, true));

    for block in &func.blocks {
        let path_cond = t_ctx
            .block_conditions
            .get(&block.id)
            .cloned()
            .unwrap_or_else(|| z3::ast::Bool::from_bool(ctx, false));

        for inst in &block.instructions {
            match &inst.kind {
                crate::ssa::ir::InstructionKind::Add(_, _, _)
                | crate::ssa::ir::InstructionKind::Sub(_, _, _)
                | crate::ssa::ir::InstructionKind::Mul(_, _, _)
                | crate::ssa::ir::InstructionKind::SDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::UDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::SRem(_, _, _)
                | crate::ssa::ir::InstructionKind::URem(_, _, _)
                | crate::ssa::ir::InstructionKind::FAdd(_, _, _)
                | crate::ssa::ir::InstructionKind::FSub(_, _, _)
                | crate::ssa::ir::InstructionKind::FMul(_, _, _)
                | crate::ssa::ir::InstructionKind::FDiv(_, _, _)
                | crate::ssa::ir::InstructionKind::FSqrt(_, _)
                | crate::ssa::ir::InstructionKind::FSin(_, _)
                | crate::ssa::ir::InstructionKind::FCos(_, _)
                | crate::ssa::ir::InstructionKind::FPow(_, _, _)
                | crate::ssa::ir::InstructionKind::ConstInt(_, _)
                | crate::ssa::ir::InstructionKind::ConstFloat(_, _)
                | crate::ssa::ir::InstructionKind::Eq(_, _, _)
                | crate::ssa::ir::InstructionKind::Ne(_, _, _)
                | crate::ssa::ir::InstructionKind::SLt(_, _, _)
                | crate::ssa::ir::InstructionKind::SLe(_, _, _)
                | crate::ssa::ir::InstructionKind::SGt(_, _, _)
                | crate::ssa::ir::InstructionKind::SGe(_, _, _)
                | crate::ssa::ir::InstructionKind::ULt(_, _, _)
                | crate::ssa::ir::InstructionKind::ULe(_, _, _)
                | crate::ssa::ir::InstructionKind::UGt(_, _, _)
                | crate::ssa::ir::InstructionKind::UGe(_, _, _)
                | crate::ssa::ir::InstructionKind::FLt(_, _, _)
                | crate::ssa::ir::InstructionKind::FLe(_, _, _)
                | crate::ssa::ir::InstructionKind::FGt(_, _, _)
                | crate::ssa::ir::InstructionKind::FGe(_, _, _)
                | crate::ssa::ir::InstructionKind::And(_, _, _)
                | crate::ssa::ir::InstructionKind::Or(_, _, _)
                | crate::ssa::ir::InstructionKind::IToF(_, _, _)
                | crate::ssa::ir::InstructionKind::FToI(_, _, _)
                | crate::ssa::ir::InstructionKind::Not(_, _) => {
                    arithmetic::translate(&mut t_ctx, inst, &path_cond)?;
                }

                crate::ssa::ir::InstructionKind::Jump(_)
                | crate::ssa::ir::InstructionKind::Branch(_, _, _)
                | crate::ssa::ir::InstructionKind::Phi(_, _) => {
                    control_flow::translate(&mut t_ctx, inst, &path_cond, block.id)?;
                }

                crate::ssa::ir::InstructionKind::ArrayLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::ArrayStore(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::BufferLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::BufferStore(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::BufferLen(_, _)
                | crate::ssa::ir::InstructionKind::StructLoad(_, _, _)
                | crate::ssa::ir::InstructionKind::StructOffset(_, _, _)
                | crate::ssa::ir::InstructionKind::StructSet(_, _, _, _, _)
                | crate::ssa::ir::InstructionKind::Borrow(_, _)
                | crate::ssa::ir::InstructionKind::MutBorrow(_, _) => {
                    memory::translate(&mut t_ctx, inst, &path_cond)?;
                }
                _ => {}
            }
        }
    }

    tracing::info!(target: "lila::verify::z3", "Proof successful for '{}'", func.name);
    Ok(())
}

pub fn update_block_condition<'ctx>(
    ctx: &'ctx Context,
    conditions: &mut HashMap<BlockId, z3::ast::Bool<'ctx>>,
    block: BlockId,
    cond: z3::ast::Bool<'ctx>,
) {
    let entry = conditions
        .entry(block)
        .or_insert_with(|| z3::ast::Bool::from_bool(ctx, false));
    *entry = z3::ast::Bool::or(ctx, &[entry, &cond]);
}
