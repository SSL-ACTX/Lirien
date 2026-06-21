use cranelift::prelude::*;
use cranelift_module::Module;
use lirien_ir::ir::Function as SsaFunction;
use super::translate_type;

pub fn get_flattened_types(
    ssa_func: &SsaFunction,
    ty: &lirien_ir::ir::Type,
) -> Vec<types::Type> {
    match ty {
        lirien_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ssa_func.struct_layouts.get(name).unwrap();
            let mut res = Vec::new();
            for (_, f_ty) in fields {
                res.extend(get_flattened_types(ssa_func, f_ty));
            }
            res
        }
        lirien_ir::ir::Type::Tuple(ref types) => {
            let mut res = Vec::new();
            for t in types {
                res.extend(get_flattened_types(ssa_func, t));
            }
            res
        }
        lirien_ir::ir::Type::Buffer(_) => vec![types::I64, types::I64],
        lirien_ir::ir::Type::Tensor(_, ref dims) => {
            vec![types::I64; dims.len() + 1]
        }
        _ => vec![translate_type(ty)],
    }
}

pub fn get_field_info(
    ssa_func: &SsaFunction,
    ty: &lirien_ir::ir::Type,
    target_offset: i32,
    expected_count: usize,
    current_offset: &mut i32,
    val_idx: &mut usize,
) -> Option<usize> {
    let align = ty.align(&ssa_func.struct_layouts) as i32;
    *current_offset = (*current_offset + align - 1) & !(align - 1);

    if *current_offset == target_offset {
        let count = get_flattened_types(ssa_func, ty).len();
        if count == expected_count {
            return Some(*val_idx);
        }
    }

    match ty {
        lirien_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ssa_func.struct_layouts.get(name).unwrap();
            for (_, f_ty) in fields {
                if let Some(res) = get_field_info(ssa_func, f_ty, target_offset, expected_count, current_offset, val_idx) {
                    return Some(res);
                }
            }
        }
        lirien_ir::ir::Type::Tuple(ref types) => {
            for t in types {
                if let Some(res) = get_field_info(ssa_func, t, target_offset, expected_count, current_offset, val_idx) {
                    return Some(res);
                }
            }
        }
        lirien_ir::ir::Type::Buffer(_) => {
            *current_offset += 16;
            *val_idx += 2;
        }
        lirien_ir::ir::Type::Tensor(_, dims) => {
            *current_offset += 8 + 8 * dims.len() as i32;
            *val_idx += 1 + dims.len();
        }
        _ => {
            *current_offset += ty.size(&ssa_func.struct_layouts) as i32;
            *val_idx += 1;
        }
    }
    None
}

pub fn build_cranelift_signature(
    ssa_func: &SsaFunction,
    arg_types: &[lirien_ir::ir::Type],
    ret_ty: &lirien_ir::ir::Type,
    is_closure: bool,
    module: &impl Module,
) -> (Signature, bool, bool) {
    let mut sig = module.make_signature();
    let mut is_sret = false;
    let mut is_register_composite_ret = false;

    // 1. Handle Return Type
    if matches!(ret_ty, lirien_ir::ir::Type::NamedTuple(_) | lirien_ir::ir::Type::Tuple(_)) {
        let cl_types = get_flattened_types(ssa_func, ret_ty);
        if cl_types.len() <= 2 {
            for cl_ty in cl_types {
                sig.returns.push(AbiParam::new(cl_ty));
            }
            is_register_composite_ret = true;
        } else {
            sig.params.push(AbiParam::new(types::I64)); // SRet pointer
            is_sret = true;
        }
    } else if ret_ty.is_simd() || matches!(ret_ty, lirien_ir::ir::Type::Optional(_) | lirien_ir::ir::Type::Struct(_)) {
        sig.params.push(AbiParam::new(types::I64)); // SRet pointer
        is_sret = true;
    } else if *ret_ty != lirien_ir::ir::Type::Unknown {
        sig.returns.push(AbiParam::new(translate_type(ret_ty)));
    }

    // 2. Handle Arguments
    if is_closure {
        sig.params.push(AbiParam::new(types::I64)); // context pointer
    }

    for arg_ty in arg_types {
        match arg_ty {
            lirien_ir::ir::Type::NamedTuple(_) | lirien_ir::ir::Type::Tuple(_) => {
                let cl_types = get_flattened_types(ssa_func, arg_ty);
                for cl_ty in cl_types {
                    sig.params.push(AbiParam::new(cl_ty));
                }
            }
            lirien_ir::ir::Type::Buffer(_) => {
                sig.params.push(AbiParam::new(types::I64)); // Ptr
                sig.params.push(AbiParam::new(types::I64)); // Len
            }
            lirien_ir::ir::Type::Tensor(_, dims) => {
                sig.params.push(AbiParam::new(types::I64)); // Ptr
                for _ in 0..dims.len() {
                    sig.params.push(AbiParam::new(types::I64)); // Dim length
                }
            }
            _ if arg_ty.is_simd() => {
                sig.params.push(AbiParam::new(types::I64)); // Pass by pointer
            }
            _ => {
                sig.params.push(AbiParam::new(translate_type(arg_ty)));
            }
        }
    }

    (sig, is_sret, is_register_composite_ret)
}
