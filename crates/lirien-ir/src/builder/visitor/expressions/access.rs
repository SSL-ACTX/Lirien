use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::{push_inst, builder_error};
use crate::ir::{InstructionKind, Type, Value};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(crate) fn visit_attribute(&mut self, s: ast::ExprAttribute, expr_offset: usize) -> BuilderResult<Value> {
        let mut obj = self.visit_expr(*s.value.clone())?;
        let mut curr_ty = self.func.get_type(obj);

        loop {
            // Handle .val or .value unwrap for Refined/Box types
            if s.attr.as_str() == "val" || s.attr.as_str() == "value"
            {
                if let Type::Pointer(inner) | Type::NullablePointer(inner) = &curr_ty {
                    if !matches!(**inner, Type::Struct(_)) {
                        let deref_val = self.func.next_value();
                        push_inst!(self, InstructionKind::PointerLoad(deref_val, obj));
                        self.func.set_type(deref_val, (**inner).clone());
                        return Ok(deref_val);
                    }
                    // If it's a pointer to a struct, we fall through to auto-deref in the match.
                } else if !matches!(curr_ty, Type::Struct(_)) {
                    // .val on a primitive is a no-op in IR (it's already the value)
                    return Ok(obj);
                }
            }

            match curr_ty {
                Type::Struct(ref struct_name) | Type::NamedTuple(ref struct_name) => {
                    let field_offset = self.get_field_offset(struct_name, s.attr.as_str()).ok_or_else(|| {
                        builder_error!(AttributeNotFound, struct_name.clone(), s.attr.to_string())
                    })?;

                    let fields = self.func.struct_layouts.get(struct_name).unwrap();
                    let field_ty = fields
                        .iter()
                        .find(|(f, _)| f == s.attr.as_str())
                        .unwrap()
                        .1
                        .clone();

                    if matches!(field_ty, Type::Unknown) {
                        return Err(builder_error!(
                            General,
                            "Field '{}' has unknown type in struct '{}'",
                            s.attr, struct_name
                        ));
                    }

                    let dest = self.func.next_value();
                    self.update_location(expr_offset);

                    if let Type::NamedTuple(_) = curr_ty {
                        push_inst!(self, InstructionKind::StructLoad(dest, obj, field_offset));
                    } else if field_ty.is_composite() {
                        push_inst!(self, InstructionKind::StructOffset(dest, obj, field_offset));
                    } else {
                        push_inst!(self, InstructionKind::StructLoad(dest, obj, field_offset));
                    }

                    self.func.set_type(dest, field_ty);
                    return Ok(dest);
                }
                Type::Pointer(inner) | Type::NullablePointer(inner) => {
                    // Auto-dereference for attribute access
                    let deref_val = self.func.next_value();
                    push_inst!(self, InstructionKind::PointerLoad(deref_val, obj));
                    self.func.set_type(deref_val, (*inner).clone());
                    obj = deref_val;
                    curr_ty = (*inner).clone();
                }
                _ => {
                    return Err(builder_error!(
                        General,
                        "Cannot resolve attribute '{}' on non-struct type {:?}",
                        s.attr, curr_ty
                    ));
                }
            }
        }
    }

    pub(crate) fn visit_subscript(&mut self, s: ast::ExprSubscript) -> BuilderResult<Value> {
        let arr = self.visit_expr(*s.value)?;
        let arr_ty = self.func.get_type(arr);
        let dest = self.func.next_value();

        match arr_ty {
            Type::Tensor(inner, dims) => {
                let mut indices = Vec::new();
                if let ast::Expr::Tuple(t) = &*s.slice {
                    for elt_expr in &t.elts {
                        let mut idx = self.visit_expr(elt_expr.clone())?;
                        idx = self.auto_load(idx);
                        indices.push(idx);
                    }
                } else {
                    let mut idx = self.visit_expr(*s.slice.clone())?;
                    idx = self.auto_load(idx);
                    indices.push(idx);
                }

                if indices.len() != dims.len() {
                    return Err(builder_error!(
                        General,
                        "Tensor indexing rank mismatch: expected {} indices, got {}",
                        dims.len(),
                        indices.len()
                    ));
                }

                push_inst!(self, InstructionKind::TensorLoad(dest, arr, indices));
                self.func.set_type(dest, *inner);
                Ok(dest)
            }
            _ => {
                match arr_ty {
                    Type::TypedDict(ref dict_name) => {
                        // Find the constant key (string)
                        let mut key_val = None;
                        if let ast::Expr::Constant(c) = &*s.slice {
                            if let ast::Constant::Str(s) = &c.value {
                                key_val = Some(s.to_string());
                            }
                        }

                        if let Some(key) = key_val {
                            let field_offset = self.get_field_offset(dict_name, &key).ok_or_else(|| {
                                builder_error!(General, "Key '{}' not found in TypedDict '{}'", key, dict_name)
                            })?;

                            let fields = self.func.struct_layouts.get(dict_name).unwrap();
                            let field_ty = fields
                                .iter()
                                .find(|(f, _)| f == &key)
                                .unwrap()
                                .1
                                .clone();

                            push_inst!(self, InstructionKind::StructLoad(dest, arr, field_offset));
                            self.func.set_type(dest, field_ty);
                            return Ok(dest);
                        } else {
                            return Err(builder_error!(General, "TypedDict key must be a constant string"));
                        }
                    }
                    _ => {
                        if let ast::Expr::Slice(slice_node) = &*s.slice {
                            return self.visit_array_slice(arr, slice_node, dest);
                        }
                        let mut idx = self.visit_expr(*s.slice)?;
                        idx = self.auto_load(idx);
                        match arr_ty {
                            Type::Buffer(inner) => {
                                push_inst!(self, InstructionKind::BufferLoad(dest, arr, idx));
                                self.func.set_type(dest, *inner);
                            }
                            Type::Array(inner, _) => {
                                push_inst!(self, InstructionKind::ArrayLoad(dest, arr, idx));
                                self.func.set_type(dest, *inner);
                            }
                            Type::Tuple(elt_types) => {
                                // Find the constant index
                                let mut idx_val = None;
                                for block in &self.func.blocks {
                                    for inst in &block.instructions {
                                        if let InstructionKind::ConstInt(v, val) = inst.kind {
                                            if v == idx {
                                                idx_val = Some(val as usize);
                                                break;
                                            }
                                        }
                                    }
                                }

                                if let Some(i) = idx_val {
                                    if i < elt_types.len() {
                                        let elt_ty = elt_types[i].clone();
                                        push_inst!(self, InstructionKind::TupleExtract(
                                            dest, arr, i,
                                        ));
                                        self.func.set_type(dest, elt_ty);
                                    } else {
                                        return Err(builder_error!(General, "Tuple index out of bounds: {}", i));
                                    }
                                } else {
                                    return Err(builder_error!(General, "Tuple index must be a constant"));
                                }
                            }
                            _ => {
                                push_inst!(self, InstructionKind::ArrayLoad(dest, arr, idx));
                            }
                        }
                    }
                }
                Ok(dest)
            }
        }
    }
}

impl CFGBuilder {
    pub(crate) fn visit_array_slice(
        &mut self,
        arr: Value,
        slice: &ast::ExprSlice,
        dest: Value,
    ) -> BuilderResult<Value> {
        let arr_ty = self.func.get_type(arr);
        let (inner_ty, total_size) = match arr_ty {
            Type::Array(inner, size) => (inner, size.map(|s| s as i64)),
            Type::Buffer(inner) => (inner, None),
            _ => {
                return Err(builder_error!(
                    General,
                    "Slicing only supported on SizedArray and Buffer"
                ))
            }
        };

        let lower = if let Some(l) = &slice.lower {
            let v = self.visit_expr(*l.clone())?;
            self.auto_load(v)
        } else {
            let v = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(v, 0));
            self.func.set_type(v, Type::I64);
            v
        };

        let upper = if let Some(u) = &slice.upper {
            let v = self.visit_expr(*u.clone())?;
            self.auto_load(v)
        } else if let Some(size) = total_size {
            let v = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(v, size));
            self.func.set_type(v, Type::I64);
            v
        } else {
            return Err(builder_error!(General, "Slice end index required for Buffers"));
        };

        let step = if let Some(s) = &slice.step {
            let step_val = self.visit_expr(*s.clone())?;
            self.auto_load(step_val)
        } else {
            let v = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(v, 1));
            self.func.set_type(v, Type::I64);
            v
        };

        let mut step_const = 1i64;
        if let Some(val) = self.get_constant_int(step) {
            if val == 0 {
                return Err(builder_error!(General, "Slice step cannot be zero"));
            }
            step_const = val;
        }

        push_inst!(self, InstructionKind::ArraySlice(dest, arr, lower, step));

        // Infer new size if both bounds are compile-time constants.
        let mut new_size = None;
        let mut lower_val = None;
        let mut upper_val = None;

        for block in &self.func.blocks {
            for inst in &block.instructions {
                if let InstructionKind::ConstInt(v, val) = inst.kind {
                    if v == lower {
                        lower_val = Some(val);
                    }
                    if v == upper {
                        upper_val = Some(val);
                    }
                }
            }
        }

        if let (Some(l), Some(u)) = (lower_val, upper_val) {
            new_size = if step_const > 0 && u > l {
                // Forward slice: ceil((u - l) / step)
                Some(((u - l + step_const - 1) / step_const) as usize)
            } else if step_const < 0 && l > u {
                // Reverse slice: ceil((l - u) / |step|)
                let abs_step = -step_const;
                Some(((l - u + abs_step - 1) / abs_step) as usize)
            } else {
                // Empty (step direction contradicts bounds ordering)
                Some(0)
            };
        }

        self.func.set_type(dest, Type::Array(inner_ty, new_size));
        Ok(dest)
    }
}
