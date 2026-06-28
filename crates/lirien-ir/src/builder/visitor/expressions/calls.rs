use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, Type, Value};
use crate::{builder_error, push_inst};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(crate) fn visit_call(&mut self, s: ast::ExprCall) -> BuilderResult<Value> {
        let expr_offset = s.range.start().to_usize();

        // Check for List constructor (e.g. List[i64]() or list[i64]())
        if let ast::Expr::Subscript(sub) = &*s.func {
            if let ast::Expr::Name(n) = &*sub.value {
                if n.id.as_str() == "List" || n.id.as_str() == "list" {
                    let list_elem_ty = crate::builder::metadata::parse_type(
                        &sub.slice,
                        &self.type_aliases,
                        &self.named_tuple_names,
                        &self.typed_dict_names,
                        &self.enum_names,
                    )?;
                    let dest = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::ListCreate(dest, list_elem_ty.clone())
                    );
                    self.func.set_type(dest, Type::List(Box::new(list_elem_ty)));
                    return Ok(dest);
                }
            }
        }

        let (func_name, method_obj, is_indirect) = match &*s.func {
            ast::Expr::Name(n) => {
                let name = n.id.to_string();
                if self.variable_defs.contains_key(&name) {
                    (name, None, true)
                } else {
                    (name, None, false)
                }
            }
            ast::Expr::Attribute(attr) => {
                if let ast::Expr::Name(n) = &*attr.value {
                    if n.id.as_str() == "math" {
                        (format!("math.{}", attr.attr), None, false)
                    } else if self.func.enum_layouts.contains_key(n.id.as_str()) {
                        // Static enum variant constructor: EnumName.Variant(...)
                        (format!("{}_{}", n.id.as_str(), attr.attr), None, false)
                    } else {
                        let obj = self.visit_expr((*attr.value).clone())?;
                        let curr_ty = self.func.get_type(obj);

                        if let Type::Struct(struct_name) = curr_ty {
                            (format!("{}_{}", struct_name, attr.attr), Some(obj), false)
                        } else if let Type::Enum(enum_name) = curr_ty {
                            (format!("{}_{}", enum_name, attr.attr), Some(obj), false)
                        } else if let Type::Tensor(..) | Type::List(..) = curr_ty {
                            (attr.attr.to_string(), Some(obj), false)
                        } else {
                            return Err(builder_error!(
                                General,
                                "Cannot call method '{}' on non-struct/enum type {:?}",
                                attr.attr,
                                self.func.get_type(obj)
                            ));
                        }
                    }
                } else {
                    let obj = self.visit_expr((*attr.value).clone())?;
                    let curr_ty = self.func.get_type(obj);

                    if let Type::Struct(struct_name) = curr_ty {
                        (format!("{}_{}", struct_name, attr.attr), Some(obj), false)
                    } else if let Type::Enum(enum_name) = curr_ty {
                        (format!("{}_{}", enum_name, attr.attr), Some(obj), false)
                    } else if let Type::Tensor(..) | Type::List(..) = curr_ty {
                        (attr.attr.to_string(), Some(obj), false)
                    } else {
                        return Err(builder_error!(
                            General,
                            "Cannot call method '{}' on non-struct/enum type {:?}",
                            attr.attr,
                            self.func.get_type(obj)
                        ));
                    }
                }
            }
            _ => ("".to_string(), None, true),
        };

        if is_indirect {
            let fn_val = if func_name.is_empty() {
                self.visit_expr(*s.func.clone())?
            } else {
                self.read_variable(func_name, self.current_block)?
            };
            let fn_ty = self.func.get_type(fn_val);

            let (arg_types, ret_ty, static_target) = match fn_ty {
                Type::FnPointer(ref params, ref ret, ref target) => {
                    (params.clone(), (**ret).clone(), target.clone())
                }
                Type::Closure(_, ref params, ref ret, ref target) => {
                    (params.clone(), (**ret).clone(), target.clone())
                }
                _ => (Vec::new(), Type::Unknown, None),
            };

            let mut args = Vec::new();
            for (i, arg) in s.args.into_iter().enumerate() {
                let mut v = self.visit_expr(arg)?;
                if i < arg_types.len() {
                    let expected_ty = &arg_types[i];
                    if expected_ty.is_int() || expected_ty.is_float() || *expected_ty == Type::Bool
                    {
                        v = self.auto_load(v);
                    }
                } else {
                    v = self.auto_load(v);
                }
                args.push(v);
            }

            let dest = self.func.next_value();
            self.update_location(expr_offset);

            let exc_ptr = Value(0);

            if let Some(target) = static_target {
                // If it's a Closure, we must pass the context pointer as the first argument.
                if let Type::Closure(..) = fn_ty {
                    let mut call_args = vec![exc_ptr, fn_val];
                    call_args.extend(args);
                    push_inst!(self, InstructionKind::Call(dest, target, call_args));
                } else {
                    let mut call_args = vec![exc_ptr];
                    call_args.extend(args);
                    push_inst!(self, InstructionKind::Call(dest, target, call_args));
                }
            } else {
                let mut call_args = vec![exc_ptr];
                call_args.extend(args);
                push_inst!(self, InstructionKind::IndirectCall(dest, fn_val, call_args));
            }

            self.func.set_type(dest, ret_ty);
            self.check_and_propagate_exception()?;
            return Ok(dest);
        }

        // Check for Enum Creation
        let mut enum_info = None;
        for (name, variants) in &self.func.enum_layouts {
            let prefix = format!("{}_", name);
            if func_name.starts_with(&prefix) {
                let v_name = &func_name[prefix.len()..];
                if variants.iter().any(|(n, _)| n == v_name) {
                    enum_info = Some((name.clone(), v_name.to_string()));
                    break;
                }
            }
        }

        // Special case for direct Ok/Err calls if return type is Result-like
        if enum_info.is_none() && (func_name == "Ok" || func_name == "Err") && method_obj.is_none()
        {
            let enum_name = match self.func.return_type {
                Type::Enum(ref name) => Some(name.clone()),
                Type::Struct(ref name) if self.func.enum_layouts.contains_key(name) => {
                    Some(name.clone())
                }
                _ => None,
            };

            if let Some(name) = enum_name {
                if let Some(variants) = self.func.enum_layouts.get(&name) {
                    if variants.iter().any(|(v_name, _)| v_name == &func_name) {
                        enum_info = Some((name.clone(), func_name.clone()));
                    }
                }
            }
        }

        if let Some((enum_name, variant_name)) = enum_info {
            if method_obj.is_none() {
                let variants = self.func.enum_layouts.get(&enum_name).unwrap();
                let tag_idx = variants
                    .iter()
                    .position(|(name, _)| name == &variant_name)
                    .unwrap();

                let variant_ty = variants[tag_idx].1.clone();
                let payload = if s.args.is_empty() {
                    None
                } else if s.args.len() == 1 {
                    let mut v = self.visit_expr(s.args[0].clone())?;
                    if let Type::Pointer(inner) = &variant_ty {
                        // Automatic boxing
                        let ptr = self.func.next_value();
                        push_inst!(self, InstructionKind::Alloc(ptr, (**inner).clone(),));
                        self.func.set_type(ptr, variant_ty.clone());
                        push_inst!(self, InstructionKind::PointerStore(ptr, v));
                        v = ptr;
                    }
                    Some(v)
                } else {
                    // Multi-element payload: create a tuple
                    let mut elts = Vec::new();
                    let mut elt_types = Vec::new();
                    let target_elt_types = if let Type::Tuple(ref types) = variant_ty {
                        types.clone()
                    } else {
                        Vec::new()
                    };

                    for (i, arg) in s.args.iter().enumerate() {
                        let mut v = self.visit_expr(arg.clone())?;
                        if i < target_elt_types.len() {
                            let expected_ty = &target_elt_types[i];
                            if let Type::Pointer(inner) = expected_ty {
                                // Automatic boxing
                                let ptr = self.func.next_value();
                                push_inst!(self, InstructionKind::Alloc(ptr, (**inner).clone(),));
                                self.func.set_type(ptr, expected_ty.clone());
                                push_inst!(self, InstructionKind::PointerStore(ptr, v));
                                v = ptr;
                            }
                        }
                        elts.push(v);
                        elt_types.push(self.func.get_type(v));
                    }
                    let tuple_dest = self.func.next_value();
                    push_inst!(self, InstructionKind::TupleCreate(tuple_dest, elts));
                    self.func.set_type(tuple_dest, Type::Tuple(elt_types));
                    Some(tuple_dest)
                };

                let dest = self.func.next_value();
                push_inst!(
                    self,
                    InstructionKind::EnumCreate(dest, enum_name.to_string(), tag_idx, payload,)
                );
                self.func.set_type(dest, Type::Enum(enum_name.to_string()));
                return Ok(dest);
            }
        }

        // Check for Enum Methods (is_Variant, as_Variant)
        if let Some(obj) = method_obj {
            let obj_ty = self.func.get_type(obj);
            if let Type::Enum(enum_name) = obj_ty {
                let method = func_name.strip_prefix(&format!("{}_", enum_name)).unwrap();
                if method.starts_with("is_") {
                    let variant_name = method.strip_prefix("is_").unwrap();
                    let variants = self.func.enum_layouts.get(&enum_name).unwrap();
                    let tag_idx = variants
                        .iter()
                        .position(|(name, _)| name == variant_name)
                        .ok_or_else(|| {
                            builder_error!(
                                General,
                                "Unknown variant '{}' for enum '{}'",
                                variant_name,
                                enum_name
                            )
                        })?;

                    let dest = self.func.next_value();
                    push_inst!(self, InstructionKind::EnumIsVariant(dest, obj, tag_idx,));
                    self.func.set_type(dest, Type::Bool);
                    return Ok(dest);
                } else if method.starts_with("as_") {
                    let variant_name = method.strip_prefix("as_").unwrap();
                    let variants = self.func.enum_layouts.get(&enum_name).unwrap();
                    let tag_idx = variants
                        .iter()
                        .position(|(name, _)| name == variant_name)
                        .ok_or_else(|| {
                            builder_error!(
                                General,
                                "Unknown variant '{}' for enum '{}'",
                                variant_name,
                                enum_name
                            )
                        })?;

                    let payload_ty = variants[tag_idx].1.clone();
                    let dest = self.func.next_value();
                    push_inst!(self, InstructionKind::EnumExtract(dest, obj, tag_idx));
                    self.func.set_type(dest, payload_ty);
                    return Ok(dest);
                }
            } else if let Type::Tensor(inner, _) = obj_ty {
                let dest = self.func.next_value();
                let kind = match func_name.as_str() {
                    "sum" => InstructionKind::TensorSum(dest, obj),
                    "max" => InstructionKind::TensorMax(dest, obj),
                    "min" => InstructionKind::TensorMin(dest, obj),
                    _ => {
                        return Err(builder_error!(
                            General,
                            "Unknown Tensor method: {}",
                            func_name
                        ))
                    }
                };
                push_inst!(self, kind);
                self.func.set_type(dest, *inner);
                return Ok(dest);
            } else if let Type::List(inner) = obj_ty {
                if func_name.as_str() == "append" {
                    if s.args.len() != 1 {
                        return Err(builder_error!(General, "append() expects 1 argument"));
                    }
                    let mut val = self.visit_expr(s.args[0].clone())?;
                    val = self.auto_load(val);
                    let dest = self.func.next_value();
                    push_inst!(self, InstructionKind::ListAppend(dest, obj, val));
                    self.func.set_type(dest, Type::List(inner.clone()));

                    // Write back dest to the base list expression if it is a write target
                    if let ast::Expr::Attribute(attr) = &*s.func {
                        self.handle_assignment_target(&attr.value, dest)?;
                    }
                    return Ok(dest);
                } else {
                    return Err(builder_error!(
                        General,
                        "Unknown List method: {}",
                        func_name
                    ));
                }
            }
        }

        if self.func.struct_layouts.contains_key(&func_name) {
            let mut struct_args = Vec::new();
            for arg in s.args.clone() {
                struct_args.push(self.visit_expr(arg)?);
            }
            let dest = self.func.next_value();
            push_inst!(
                self,
                InstructionKind::StructCreate(dest, func_name.clone(), struct_args,)
            );
            if self.named_tuple_names.contains(&func_name) {
                self.func
                    .set_type(dest, Type::NamedTuple(func_name.clone()));
            } else {
                self.func.set_type(dest, Type::Struct(func_name.clone()));
            }
            return Ok(dest);
        }

        if func_name == "isinstance" {
            if s.args.len() != 2 {
                return Err(builder_error!(General, "isinstance() expects 2 arguments"));
            }
            let obj = self.visit_expr(s.args[0].clone())?;
            let obj_ty = self.func.get_type(obj);

            let target_ty_name = if let ast::Expr::Name(n) = &s.args[1] {
                n.id.to_string()
            } else {
                "unknown".to_string()
            };

            let matched = match target_ty_name.as_str() {
                "int" | "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" => {
                    obj_ty.is_int()
                }
                "float" | "f64" | "f32" => obj_ty.is_float(),
                "bool" => matches!(obj_ty, Type::Bool),
                _ => {
                    if let Type::Struct(name) = &obj_ty {
                        name == &target_ty_name
                    } else {
                        false
                    }
                }
            };

            let dest = self.func.next_value();
            push_inst!(
                self,
                InstructionKind::ConstInt(dest, if matched { 1 } else { 0 })
            );
            self.func.set_type(dest, Type::Bool);
            return Ok(dest);
        }

        if func_name == "f64" || func_name == "float" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "f64() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg_ty = self.func.get_type(arg);
            if matches!(arg_ty, Type::F64) {
                return Ok(arg);
            }
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::IToF(dest, arg, Type::F64));
            self.func.set_type(dest, Type::F64);
            return Ok(dest);
        } else if func_name == "f32" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "f32() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg_ty = self.func.get_type(arg);
            if matches!(arg_ty, Type::F32) {
                return Ok(arg);
            }
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::IToF(dest, arg, Type::F32));
            self.func.set_type(dest, Type::F32);
            return Ok(dest);
        } else if func_name == "i64" || func_name == "int" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "i64() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg_ty = self.func.get_type(arg);
            if matches!(arg_ty, Type::I64) {
                return Ok(arg);
            }
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FToI(dest, arg, Type::I64));
            self.func.set_type(dest, Type::I64);
            return Ok(dest);
        } else if func_name == "len" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "len() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let ty = self.func.get_type(arg);
            if let Type::Buffer(_) = ty {
                let dest = self.func.next_value();
                push_inst!(self, InstructionKind::BufferLen(dest, arg));
                self.func.set_type(dest, Type::I64);
                return Ok(dest);
            } else if let Type::List(_) = ty {
                let dest = self.func.next_value();
                push_inst!(self, InstructionKind::ListLen(dest, arg));
                self.func.set_type(dest, Type::I64);
                return Ok(dest);
            } else if let Type::Array(_, Some(size)) = ty {
                let dest = self.func.next_value();
                push_inst!(self, InstructionKind::ConstInt(dest, size as i64));
                self.func.set_type(dest, Type::I64);
                return Ok(dest);
            }
        } else if func_name == "parallel_for" {
            if s.args.len() != 2 {
                return Err(builder_error!(
                    General,
                    "parallel_for expects 2 arguments: range and body lambda"
                ));
            }

            // 1. Parse range
            let (start_v, stop_v, step_v) = if let ast::Expr::Call(range_call) = &s.args[0] {
                if let ast::Expr::Name(n) = &*range_call.func {
                    if n.id.as_str() == "range" {
                        let (start, end, step) = match range_call.args.len() {
                            1 => (None, self.visit_expr(range_call.args[0].clone())?, None),
                            2 => (
                                Some(self.visit_expr(range_call.args[0].clone())?),
                                self.visit_expr(range_call.args[1].clone())?,
                                None,
                            ),
                            3 => (
                                Some(self.visit_expr(range_call.args[0].clone())?),
                                self.visit_expr(range_call.args[1].clone())?,
                                Some(self.visit_expr(range_call.args[2].clone())?),
                            ),
                            _ => {
                                return Err(builder_error!(
                                    General,
                                    "Unsupported range() signature"
                                ))
                            }
                        };

                        let s_v = if let Some(v) = start {
                            v
                        } else {
                            let zero = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(zero, 0));
                            zero
                        };
                        let st_v = if let Some(v) = step {
                            v
                        } else {
                            let one = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(one, 1));
                            one
                        };
                        (s_v, end, st_v)
                    } else {
                        return Err(builder_error!(
                            General,
                            "parallel_for first argument must be range()"
                        ));
                    }
                } else {
                    return Err(builder_error!(
                        General,
                        "parallel_for first argument must be range()"
                    ));
                }
            } else {
                return Err(builder_error!(
                    General,
                    "parallel_for first argument must be range()"
                ));
            };

            // 2. Parse lambda
            if let ast::Expr::Lambda(lambda) = &s.args[1] {
                if lambda.args.args.len() != 1 {
                    return Err(builder_error!(
                        General,
                        "parallel_for lambda must take exactly 1 argument (the index)"
                    ));
                }
                let index_name = lambda.args.args[0].def.arg.to_string();

                let body_block = self.create_block();
                let exit_block = self.create_block();

                let index_var = self.func.next_value();
                self.func.set_type(index_var, Type::I64);

                // Capture analysis
                use crate::builder::capture_analysis::CaptureVisitor;
                use rustpython_ast::Visitor;
                let mut capture_visitor = CaptureVisitor::new(vec![index_name.clone()]);
                capture_visitor.visit_expr(*lambda.body.clone());

                let mut captures = Vec::new();
                for var_name in capture_visitor.captures {
                    if self.variable_defs.contains_key(&var_name) {
                        captures.push(self.read_variable(var_name, self.current_block)?);
                    }
                }

                let prev_block = self.current_block;
                self.link_blocks(prev_block, body_block);
                self.link_blocks(prev_block, exit_block);

                self.seal_block(body_block)?;
                self.start_block(body_block);
                self.write_variable(index_name, body_block, index_var);

                self.visit_expr(*lambda.body.clone())?;

                // Transition to exit block
                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(exit_block));
                    self.link_blocks(self.current_block, exit_block);
                }

                self.start_block(exit_block);
                self.seal_block(exit_block)?;

                // Add ParallelFor to the original block
                self.current_block = prev_block;
                self.update_location(expr_offset);
                push_inst!(
                    self,
                    InstructionKind::ParallelFor(
                        index_var, start_v, stop_v, step_v, body_block, exit_block, captures,
                    )
                );

                push_inst!(self, InstructionKind::Jump(exit_block));

                // Switch back to exit block for subsequent instructions
                self.current_block = exit_block;

                let dest = self.func.next_value();
                self.func.set_type(dest, Type::Unknown);
                return Ok(dest);
            } else {
                return Err(builder_error!(
                    General,
                    "parallel_for second argument must be a lambda"
                ));
            }
        } else if func_name == "math.sqrt" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "sqrt() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FSqrt(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.sin" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "sin() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FSin(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.cos" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "cos() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FCos(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.tan" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "tan() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FTan(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.asin" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "asin() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FAsin(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.acos" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "acos() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FAcos(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.atan" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "atan() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FAtan(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.exp" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "exp() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FExp(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.log" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "log() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FLog(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.log10" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "log10() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FLog10(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.floor" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "floor() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FFloor(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.ceil" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "ceil() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FCeil(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.trunc" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "trunc() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FTrunc(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "math.pow" {
            if s.args.len() != 2 {
                return Err(builder_error!(General, "pow() expects 2 arguments"));
            }
            let b = self.visit_expr(s.args[0].clone())?;
            let e = self.visit_expr(s.args[1].clone())?;
            let b = self.auto_load(b);
            let e = self.auto_load(e);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::FPow(dest, b, e));
            self.func.set_type(dest, self.func.get_type(b));
            return Ok(dest);
        } else if func_name == "abs" || func_name == "math.abs" {
            if s.args.len() != 1 {
                return Err(builder_error!(General, "abs() expects 1 argument"));
            }
            let arg = self.visit_expr(s.args[0].clone())?;
            let arg = self.auto_load(arg);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::Abs(dest, arg));
            self.func.set_type(dest, self.func.get_type(arg));
            return Ok(dest);
        } else if func_name == "min" || func_name == "math.min" {
            if s.args.len() != 2 {
                return Err(builder_error!(General, "min() expects 2 arguments"));
            }
            let l = self.visit_expr(s.args[0].clone())?;
            let r = self.visit_expr(s.args[1].clone())?;
            let l = self.auto_load(l);
            let r = self.auto_load(r);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::Min(dest, l, r));
            self.func.set_type(dest, self.func.get_type(l));
            return Ok(dest);
        } else if func_name == "max" || func_name == "math.max" {
            if s.args.len() != 2 {
                return Err(builder_error!(General, "max() expects 2 arguments"));
            }
            let l = self.visit_expr(s.args[0].clone())?;
            let r = self.visit_expr(s.args[1].clone())?;
            let l = self.auto_load(l);
            let r = self.auto_load(r);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::Max(dest, l, r));
            self.func.set_type(dest, self.func.get_type(l));
            return Ok(dest);
        } else if func_name == "avg" || func_name == "math.avg" {
            if s.args.len() != 2 {
                return Err(builder_error!(General, "avg() expects 2 arguments"));
            }
            let l = self.visit_expr(s.args[0].clone())?;
            let r = self.visit_expr(s.args[1].clone())?;
            let l = self.auto_load(l);
            let r = self.auto_load(r);
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::Avg(dest, l, r));
            self.func.set_type(dest, self.func.get_type(l));
            return Ok(dest);
        }

        // Look up return type and arg types in registry or self
        let mut ret_ty = Type::Unknown;
        let mut arg_types = Vec::new();
        if func_name == self.func.name {
            ret_ty = self.func.return_type.clone();
            for i in 0..self.func.arg_count {
                arg_types.push(self.func.get_type(Value(i)));
            }
        } else if let Ok(registry) = crate::registry::GLOBAL_REGISTRY.lock() {
            if let Some(sig) = registry.get(&func_name) {
                ret_ty = sig.return_type.clone();
                arg_types = sig.arg_types.clone();
            }
        }

        let mut args = Vec::new();
        let mut arg_idx = 0;
        if let Some(obj) = method_obj {
            args.push(obj);
            arg_idx += 1;
        }
        for arg in s.args {
            let mut v = self.visit_expr(arg)?;
            if arg_idx < arg_types.len() {
                let expected_ty = &arg_types[arg_idx];
                // Only auto-load if the function expects a primitive value
                if expected_ty.is_int() || expected_ty.is_float() || *expected_ty == Type::Bool {
                    v = self.auto_load(v);
                }
            } else {
                v = self.auto_load(v);
            }
            args.push(v);
            arg_idx += 1;
        }

        let dest = self.func.next_value();
        self.update_location(expr_offset);
        let exc_ptr = Value(0);
        let mut call_args = vec![exc_ptr];
        call_args.extend(args);
        push_inst!(
            self,
            InstructionKind::Call(dest, func_name.clone(), call_args)
        );
        self.func.set_type(dest, ret_ty);
        self.check_and_propagate_exception()?;
        Ok(dest)
    }

    pub(crate) fn visit_lambda(&mut self, s: ast::ExprLambda) -> BuilderResult<Value> {
        use crate::builder::capture_analysis::CaptureVisitor;
        use rustpython_ast::Visitor;

        let next_val = self.func.next_value().0;
        let lambda_name = format!("{}_lambda_{}", self.func.name, next_val);

        // 1. Capture Analysis
        let mut params = Vec::new();
        for arg in &s.args.args {
            params.push(arg.def.arg.to_string());
        }
        let mut capture_visitor = CaptureVisitor::new(params);
        capture_visitor.visit_expr(*s.body.clone());

        let mut captures = Vec::new();
        let mut capture_types = Vec::new();
        for var_name in capture_visitor.captures {
            if self.variable_defs.contains_key(&var_name) {
                let val = self.read_variable(var_name.clone(), self.current_block)?;
                let ty = self.func.get_type(val);
                captures.push((var_name, val));
                capture_types.push(ty);
            }
        }

        // 2. Build Lambda Function
        // The lambda will take (ctx_ptr, ...args)
        let mut lambda_builder = self.new_sub_builder(lambda_name.clone());

        // Define arguments in lambda
        // arg0 is always ctx_ptr
        lambda_builder.func.arg_count = 2 + s.args.args.len();
        lambda_builder.func.value_count = lambda_builder.func.arg_count;
        lambda_builder
            .func
            .value_types
            .insert(Value(0), Type::Pointer(Box::new(Type::I64))); // _exc_ptr
        lambda_builder
            .func
            .value_types
            .insert(Value(1), Type::Struct("ClosureEnv".to_string())); // ctx_ptr

        for (i, arg) in s.args.args.iter().enumerate() {
            let arg_ty = if let Some(ann) = &arg.def.annotation {
                crate::builder::metadata::parse_type(
                    ann,
                    &self.type_aliases,
                    &self.named_tuple_names,
                    &self.typed_dict_names,
                    &self.enum_names,
                )?
            } else {
                Type::Unknown
            };
            lambda_builder.func.value_types.insert(Value(i + 2), arg_ty);
            lambda_builder.write_variable(
                arg.def.arg.to_string(),
                lambda_builder.current_block,
                Value(i + 2),
            );
        }

        // If there are captures, load them from ctx_ptr
        if !captures.is_empty() {
            let mut offset = 8; // Offset 0 is fn_ptr
            for (name, ty) in captures.iter().zip(capture_types.iter()) {
                let align = ty.align(&self.func.struct_layouts);
                offset = (offset + align - 1) & !(align - 1);

                let dest = lambda_builder.func.next_value();
                push_inst!(
                    lambda_builder,
                    InstructionKind::StructLoad(dest, Value(1), offset,)
                );
                lambda_builder.func.set_type(dest, ty.clone());
                lambda_builder.write_variable(name.0.clone(), lambda_builder.current_block, dest);

                offset += ty.size(&self.func.struct_layouts);
            }
        }

        // Visit body
        let ret_val = lambda_builder.visit_expr(*s.body)?;
        push_inst!(lambda_builder, InstructionKind::Return(Some(ret_val)));
        lambda_builder.func.return_type = lambda_builder.func.get_type(ret_val);

        let ret_ty = lambda_builder.func.return_type.clone();
        if ret_ty != Type::Unknown && ret_ty != Type::Tuple(vec![]) {
            let block_ids: Vec<crate::ir::BlockId> = lambda_builder
                .func
                .blocks
                .iter()
                .filter(|b| {
                    b.instructions.last().map_or(false, |inst| {
                        matches!(inst.kind, InstructionKind::Return(None))
                    })
                })
                .map(|b| b.id)
                .collect();

            for bid in block_ids {
                let prev_block = lambda_builder.current_block;
                lambda_builder.current_block = bid;

                if let Some(block) = lambda_builder.func.blocks.iter_mut().find(|b| b.id == bid) {
                    block.instructions.pop();
                }

                let dummy = lambda_builder.dummy_value(&ret_ty)?;
                push_inst!(lambda_builder, InstructionKind::Return(Some(dummy)));

                lambda_builder.current_block = prev_block;
            }
        }

        // Optimization for lambda
        crate::optimization::optimize(&mut lambda_builder.func);

        // Store lambda for later compilation
        let lambda_func = lambda_builder.func;
        self.lambdas.push(lambda_func.clone());
        // Collect nested lambdas from the sub-builder
        self.lambdas.extend(lambda_builder.lambdas);

        // 3. Create Closure Instruction
        let dest = self.func.next_value();
        let capture_vals: Vec<Value> = captures.iter().map(|(_, v)| *v).collect();
        push_inst!(
            self,
            InstructionKind::Lambda(dest, lambda_name.clone(), capture_vals,)
        );

        let arg_types: Vec<Type> = (2..2 + s.args.args.len())
            .map(|i| lambda_func.get_type(Value(i)))
            .collect();
        self.func.set_type(
            dest,
            Type::Closure(
                lambda_name.clone(),
                arg_types,
                Box::new(lambda_func.return_type),
                Some(lambda_name),
            ),
        );

        Ok(dest)
    }
}
