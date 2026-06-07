use super::{get_len, get_val, CodegenContext};
use cranelift::prelude::*;
use cranelift_module::Module;
use lila_ir::ir::{BlockId as SsaBlockId, InstructionKind, Type as SsaType};
use std::collections::HashMap;

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
        InstructionKind::Match(selector, cases, default, _is_strict) => {
            let s_val = get_val(&ctx.values, selector);
            let default_ssa_id = default;

            let mut targets = cases.values().collect::<Vec<_>>();
            targets.push(default_ssa_id);

            let mut target_to_cl_block = HashMap::new();
            let mut trampolines = Vec::new();

            for &target_ssa_id in &targets {
                let mut phi_args = Vec::new();
                for ssa_block_iter in &ctx.ssa_func.blocks {
                    if ssa_block_iter.id == *target_ssa_id {
                        for inst_iter in &ssa_block_iter.instructions {
                            if let InstructionKind::Phi(dest_val, mappings) = &inst_iter.kind {
                                if let Some(src_val) = mappings.get(&current_ssa_block_id) {
                                    phi_args.push(get_val(&ctx.values, src_val));
                                    if let SsaType::Buffer(_) = ctx.ssa_func.get_type(*dest_val) {
                                        phi_args.push(get_len(&ctx.buffer_lengths, src_val));
                                    }
                                }
                            }
                        }
                    }
                }

                if phi_args.is_empty() {
                    target_to_cl_block.insert(target_ssa_id, ctx.blocks[target_ssa_id]);
                } else {
                    let trampoline = ctx.builder.create_block();
                    target_to_cl_block.insert(target_ssa_id, trampoline);
                    trampolines.push((trampoline, ctx.blocks[target_ssa_id], phi_args));
                }
            }

            let mut switch_obj = cranelift::frontend::Switch::new();
            for (tag, target_ssa_id) in cases {
                switch_obj.set_entry((*tag as u64).into(), target_to_cl_block[target_ssa_id]);
            }
            let cl_default_block = target_to_cl_block[default_ssa_id];
            switch_obj.emit(&mut ctx.builder, s_val, cl_default_block);

            // Now fill trampolines
            for (trampoline, target_cl_block, phi_args) in trampolines {
                ctx.builder.switch_to_block(trampoline);
                ctx.builder.ins().jump(target_cl_block, &phi_args);
            }
        }
        InstructionKind::Return(val) => {
            if ctx.ssa_func.return_type.is_simd() {
                if let Some(v) = val {
                    let vec_val = get_val(&ctx.values, v);
                    let dest_ptr = ctx.sret_ptr.expect("Missing SRet pointer for SIMD");
                    ctx.builder
                        .ins()
                        .store(MemFlags::new(), vec_val, dest_ptr, 0);
                }
                ctx.builder.ins().return_(&[]);
            } else if ctx.is_tuple_return {
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
