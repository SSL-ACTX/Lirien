use super::{get_len, get_val, CodegenContext};
use crate::ssa::ir::{BlockId as SsaBlockId, InstructionKind, Type as SsaType};
use cranelift::prelude::*;
use cranelift_module::Module;

pub fn lower<M: Module>(
    ctx: &mut CodegenContext<M>,
    kind: &InstructionKind,
    current_ssa_block_id: SsaBlockId,
) -> Result<(), String> {
    match kind {
        InstructionKind::Jump(target) => {
            let dest_block = ctx.blocks[target];
            let mut args = Vec::new();

            for ssa_block_iter in &ctx.ssa_func.blocks {
                if ssa_block_iter.id == *target {
                    for inst_iter in &ssa_block_iter.instructions {
                        if let InstructionKind::Phi(dest_val, mappings) = &inst_iter.kind {
                            if let Some(src_val) = mappings.get(&current_ssa_block_id) {
                                args.push(get_val(&ctx.values, src_val));
                                if let SsaType::Buffer(_) = ctx.ssa_func.get_type(*dest_val) {
                                    args.push(get_len(&ctx.buffer_lengths, src_val));
                                }
                            }
                        }
                    }
                }
            }
            ctx.builder.ins().jump(dest_block, &args);
        }
        InstructionKind::Branch(cond, t, f) => {
            let c = get_val(&ctx.values, cond);
            let t_block = ctx.blocks[t];
            let f_block = ctx.blocks[f];

            let mut t_args = Vec::new();
            let mut f_args = Vec::new();

            for ssa_block_iter in &ctx.ssa_func.blocks {
                if ssa_block_iter.id == *t {
                    for inst_iter in &ssa_block_iter.instructions {
                        if let InstructionKind::Phi(dest_val, mappings) = &inst_iter.kind {
                            if let Some(src_val) = mappings.get(&current_ssa_block_id) {
                                t_args.push(get_val(&ctx.values, src_val));
                                if let SsaType::Buffer(_) = ctx.ssa_func.get_type(*dest_val) {
                                    t_args.push(get_len(&ctx.buffer_lengths, src_val));
                                }
                            }
                        }
                    }
                }
                if ssa_block_iter.id == *f {
                    for inst_iter in &ssa_block_iter.instructions {
                        if let InstructionKind::Phi(dest_val, mappings) = &inst_iter.kind {
                            if let Some(src_val) = mappings.get(&current_ssa_block_id) {
                                f_args.push(get_val(&ctx.values, src_val));
                                if let SsaType::Buffer(_) = ctx.ssa_func.get_type(*dest_val) {
                                    f_args.push(get_len(&ctx.buffer_lengths, src_val));
                                }
                            }
                        }
                    }
                }
            }

            let cond_b1 = ctx.builder.ins().icmp_imm(IntCC::NotEqual, c, 0);
            ctx.builder
                .ins()
                .brif(cond_b1, t_block, &t_args, f_block, &f_args);
        }
        InstructionKind::Return(val) => {
            if ctx.is_tuple_return {
                if let Some(v) = val {
                    let tuple_ptr = get_val(&ctx.values, v);
                    let dest_ptr = ctx.sret_ptr.expect("Missing SRet pointer");
                    let total_size = ctx.ssa_func.return_type.size(&ctx.ssa_func.struct_layouts);

                    let mut offset = 0;
                    while offset + 8 <= total_size {
                        let chunk = ctx.builder.ins().load(
                            types::I64,
                            MemFlags::new(),
                            tuple_ptr,
                            offset as i32,
                        );
                        ctx.builder
                            .ins()
                            .store(MemFlags::new(), chunk, dest_ptr, offset as i32);
                        offset += 8;
                    }
                    while offset < total_size {
                        let chunk = ctx.builder.ins().load(
                            types::I8,
                            MemFlags::new(),
                            tuple_ptr,
                            offset as i32,
                        );
                        ctx.builder
                            .ins()
                            .store(MemFlags::new(), chunk, dest_ptr, offset as i32);
                        offset += 1;
                    }
                }
                ctx.builder.ins().return_(&[]);
            } else {
                let cl_vals = match val {
                    Some(v) => vec![get_val(&ctx.values, v)],
                    None => vec![],
                };
                ctx.builder.ins().return_(&cl_vals);
            }
        }
        _ => return Err(format!("Not a control flow instruction: {:?}", kind)),
    }
    Ok(())
}
