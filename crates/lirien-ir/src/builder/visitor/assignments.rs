use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, Type, Value};
use crate::{builder_error, push_inst};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(super) fn handle_assignment_target(
        &mut self,
        target: &ast::Expr,
        value: Value,
    ) -> BuilderResult<()> {
        match target {
            ast::Expr::Name(name) => {
                self.write_variable(name.id.to_string(), self.current_block, value);
            }
            ast::Expr::Subscript(sub) => {
                let arr = self.visit_expr(*sub.value.clone())?;
                let arr_ty = self.func.get_type(arr);
                let dest_arr = self.func.next_value();

                match arr_ty {
                    Type::Tensor(inner, dims) => {
                        let mut indices = Vec::new();
                        if let ast::Expr::Tuple(t) = &*sub.slice {
                            for elt_expr in &t.elts {
                                let mut idx = self.visit_expr(elt_expr.clone())?;
                                idx = self.auto_load(idx);
                                indices.push(idx);
                            }
                        } else {
                            let mut idx = self.visit_expr(*sub.slice.clone())?;
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

                        push_inst!(
                            self,
                            InstructionKind::TensorStore(dest_arr, arr, indices, value,)
                        );
                        self.func.set_type(dest_arr, Type::Tensor(inner, dims));
                    }
                    Type::TypedDict(ref dict_name) => {
                        // Find the constant key (string)
                        let mut key_val = None;
                        if let ast::Expr::Constant(c) = &*sub.slice {
                            if let ast::Constant::Str(s) = &c.value {
                                key_val = Some(s.to_string());
                            }
                        }

                        if let Some(key) = key_val {
                            let field_offset =
                                self.get_field_offset(dict_name, &key).ok_or_else(|| {
                                    builder_error!(
                                        AttributeNotFound,
                                        dict_name.clone(),
                                        key.clone()
                                    )
                                })?;

                            let fields = self.func.struct_layouts.get(dict_name).unwrap();
                            let field_ty =
                                fields.iter().find(|(f, _)| f == &key).unwrap().1.clone();

                            push_inst!(
                                self,
                                InstructionKind::StructSet(
                                    dest_arr,
                                    arr,
                                    field_offset,
                                    value,
                                    field_ty,
                                )
                            );
                            self.func
                                .set_type(dest_arr, Type::TypedDict(dict_name.clone()));
                        } else {
                            return Err(builder_error!(
                                General,
                                "TypedDict key must be a constant string"
                            ));
                        }
                    }
                    _ => {
                        let mut idx = self.visit_expr(*sub.slice.clone())?;
                        idx = self.auto_load(idx);
                        match arr_ty {
                            Type::Buffer(inner) => {
                                push_inst!(
                                    self,
                                    InstructionKind::BufferStore(
                                        dest_arr,
                                        arr,
                                        idx,
                                        value,
                                        *inner.clone(),
                                    )
                                );
                                self.func.set_type(dest_arr, Type::Buffer(inner));
                            }
                            Type::List(inner) => {
                                push_inst!(
                                    self,
                                    InstructionKind::ListStore(dest_arr, arr, idx, value,)
                                );
                                self.func.set_type(dest_arr, Type::List(inner));
                            }
                            Type::Array(inner, size) => {
                                push_inst!(
                                    self,
                                    InstructionKind::ArrayStore(
                                        dest_arr,
                                        arr,
                                        idx,
                                        value,
                                        *inner.clone(),
                                    )
                                );
                                self.func.set_type(dest_arr, Type::Array(inner, size));
                            }
                            _ => {
                                push_inst!(
                                    self,
                                    InstructionKind::ArrayStore(
                                        dest_arr,
                                        arr,
                                        idx,
                                        value,
                                        Type::Unknown,
                                    )
                                );
                            }
                        }
                    }
                }

                if let ast::Expr::Name(name) = &*sub.value {
                    self.write_variable(name.id.to_string(), self.current_block, dest_arr);
                }
            }
            ast::Expr::Attribute(attr) => {
                // Check if the base of the attribute chain is a subscript
                // e.g., arr[i].x = val
                let mut current = &*attr.value;
                let mut is_subscript_base = false;
                while let ast::Expr::Attribute(inner_attr) = current {
                    current = &*inner_attr.value;
                }

                if let ast::Expr::Subscript(_) = current {
                    is_subscript_base = true;
                }

                if is_subscript_base {
                    // Evaluate the full attribute path as an expression to load the struct.
                    // Modify the struct locally, then recursively store it back.
                    let base_val = self.visit_expr(*attr.value.clone())?;
                    let base_ty = self.func.get_type(base_val);

                    let (offset, leaf_ty) = if let Type::Struct(struct_name) = &base_ty {
                        let field_offset = self
                            .get_field_offset(struct_name, attr.attr.as_str())
                            .ok_or_else(|| {
                            builder_error!(
                                AttributeNotFound,
                                struct_name.clone(),
                                attr.attr.to_string()
                            )
                        })?;
                        let fields = self.func.struct_layouts.get(struct_name).unwrap();
                        let ty = fields
                            .iter()
                            .find(|(f, _)| f == attr.attr.as_str())
                            .unwrap()
                            .1
                            .clone();
                        (field_offset, ty)
                    } else {
                        return Err(builder_error!(General, "Expected struct base"));
                    };

                    let dest_obj = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::StructSet(dest_obj, base_val, offset, value, leaf_ty,)
                    );
                    self.func.set_type(dest_obj, base_ty.clone());

                    // Now we need to store `dest_obj` back into whatever `attr.value` was.
                    // This means `handle_assignment_target` needs to be recursive!
                    self.handle_assignment_target(&attr.value, dest_obj)?;
                    return Ok(());
                }

                let (root_name, offset, leaf_ty) =
                    self.resolve_attribute_path(ast::Expr::Attribute(attr.clone()))?;
                let root_val = self.read_variable(root_name.clone(), self.current_block)?;
                let root_ty = self.func.get_type(root_val);

                let dest_obj = self.func.next_value();
                push_inst!(
                    self,
                    InstructionKind::StructSet(dest_obj, root_val, offset, value, leaf_ty,)
                );
                self.func.set_type(dest_obj, root_ty);

                self.write_variable(root_name, self.current_block, dest_obj);
            }
            ast::Expr::Tuple(t) => {
                self.handle_tuple_destructuring(&t.elts, value)?;
            }
            ast::Expr::List(l) => {
                self.handle_tuple_destructuring(&l.elts, value)?;
            }
            _ => {
                return Err(builder_error!(
                    UnsupportedStatement,
                    "Unsupported assignment target: {:?}",
                    target
                ))
            }
        }
        Ok(())
    }

    fn handle_tuple_destructuring(
        &mut self,
        elts: &[ast::Expr],
        value: Value,
    ) -> BuilderResult<()> {
        let val_ty = self.func.get_type(value);
        match val_ty {
            Type::Tuple(elt_types) => {
                if elts.len() != elt_types.len() {
                    return Err(builder_error!(
                        General,
                        "Cannot unpack tuple of size {} into {} targets",
                        elt_types.len(),
                        elts.len()
                    ));
                }
                for (i, target_elt) in elts.iter().enumerate() {
                    let elt_val = self.func.next_value();
                    push_inst!(self, InstructionKind::TupleExtract(elt_val, value, i));
                    self.func.set_type(elt_val, elt_types[i].clone());
                    self.handle_assignment_target(target_elt, elt_val)?;
                }
            }
            Type::NamedTuple(ref name) => {
                let fields = self
                    .func
                    .struct_layouts
                    .get(name)
                    .ok_or_else(|| {
                        builder_error!(General, "NamedTuple layout not found for {}", name)
                    })?
                    .clone();
                if elts.len() != fields.len() {
                    return Err(builder_error!(
                        General,
                        "Cannot unpack NamedTuple of size {} into {} targets",
                        fields.len(),
                        elts.len()
                    ));
                }
                for (i, target_elt) in elts.iter().enumerate() {
                    let (field_name, field_ty) = &fields[i];
                    let field_offset =
                        self.get_field_offset(name, field_name).ok_or_else(|| {
                            builder_error!(General, "Field offset not found for {}", field_name)
                        })?;
                    let elt_val = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::StructLoad(elt_val, value, field_offset)
                    );
                    self.func.set_type(elt_val, field_ty.clone());
                    self.handle_assignment_target(target_elt, elt_val)?;
                }
            }
            _ => {
                return Err(builder_error!(
                    General,
                    "Cannot unpack non-tuple type {:?}",
                    val_ty
                ));
            }
        }
        Ok(())
    }

    pub(super) fn resolve_attribute_path(
        &self,
        expr: ast::Expr,
    ) -> BuilderResult<(String, usize, Type)> {
        match expr {
            ast::Expr::Name(n) => {
                let name = n.id.to_string();
                // Find the variable's value to get its type
                let val = if let Some(blocks) = self.variable_defs.get(&name) {
                    let name_cloned = name.clone();
                    blocks
                        .get(&self.current_block)
                        .cloned()
                        .or_else(|| blocks.values().next().cloned()) // Fallback
                        .ok_or_else(|| builder_error!(UnboundVariable, "{}", name_cloned))?
                } else {
                    return Err(builder_error!(UnboundVariable, "{}", name));
                };
                let ty = self.func.get_type(val);
                Ok((name, 0, ty))
            }
            ast::Expr::Attribute(attr) => {
                if attr.attr.as_str() == "val" {
                    return self.resolve_attribute_path(*attr.value.clone());
                }

                let (root_name, base_offset, parent_ty) =
                    self.resolve_attribute_path(*attr.value)?;

                let curr_ty = &parent_ty;

                if let Type::Struct(struct_name) = curr_ty {
                    let field_offset = self
                        .get_field_offset(struct_name, attr.attr.as_str())
                        .ok_or_else(|| {
                            builder_error!(
                                AttributeNotFound,
                                struct_name.clone(),
                                attr.attr.to_string()
                            )
                        })?;

                    let fields = self.func.struct_layouts.get(struct_name).unwrap();
                    let field_ty = fields
                        .iter()
                        .find(|(f, _)| f == attr.attr.as_str())
                        .unwrap()
                        .1
                        .clone();

                    Ok((root_name, base_offset + field_offset, field_ty))
                } else {
                    Err(builder_error!(
                        General,
                        "Cannot resolve attribute '{}' on non-struct type {:?}",
                        attr.attr,
                        parent_ty
                    ))
                }
            }
            _ => Err(builder_error!(
                General,
                "Invalid attribute path: must start with a variable name, found {:?}",
                expr
            )),
        }
    }
}
