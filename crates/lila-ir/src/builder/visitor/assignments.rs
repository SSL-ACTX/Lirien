use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, Type, Value};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(super) fn handle_assignment_target(
        &mut self,
        target: &ast::Expr,
        value: Value,
    ) -> Result<(), String> {
        match target {
            ast::Expr::Name(name) => {
                self.write_variable(name.id.to_string(), self.current_block, value);
            }
            ast::Expr::Subscript(sub) => {
                let arr = self.visit_expr(*sub.value.clone())?;
                let idx = self.visit_expr(*sub.slice.clone())?;
                let dest_arr = self.func.next_value();
                let arr_ty = self.func.get_type(arr);
                match arr_ty {
                    Type::Buffer(inner) => {
                        self.add_instruction(InstructionKind::BufferStore(
                            dest_arr,
                            arr,
                            idx,
                            value,
                            *inner.clone(),
                        ));
                        self.func.set_type(dest_arr, Type::Buffer(inner));
                    }
                    Type::Array(inner, size) => {
                        self.add_instruction(InstructionKind::ArrayStore(
                            dest_arr,
                            arr,
                            idx,
                            value,
                            *inner.clone(),
                        ));
                        self.func.set_type(dest_arr, Type::Array(inner, size));
                    }
                    _ => {
                        self.add_instruction(InstructionKind::ArrayStore(
                            dest_arr,
                            arr,
                            idx,
                            value,
                            Type::Unknown,
                        ));
                    }
                }

                if let ast::Expr::Name(name) = &*sub.value {
                    self.write_variable(name.id.to_string(), self.current_block, dest_arr);
                }
            }
            ast::Expr::Attribute(attr) => {
                let (root_name, offset, leaf_ty) =
                    self.resolve_attribute_path(ast::Expr::Attribute(attr.clone()))?;
                let root_val = self.read_variable(root_name.clone(), self.current_block)?;
                let root_ty = self.func.get_type(root_val);

                let dest_obj = self.func.next_value();
                self.add_instruction(InstructionKind::StructSet(
                    dest_obj, root_val, offset, value, leaf_ty,
                ));
                self.func.set_type(dest_obj, root_ty);

                self.write_variable(root_name, self.current_block, dest_obj);
            }
            ast::Expr::Tuple(t) => {
                let tuple_ty = self.func.get_type(value);
                if let Type::Tuple(elt_types) = tuple_ty {
                    if t.elts.len() != elt_types.len() {
                        return Err(format!(
                            "Cannot unpack tuple of size {} into {} targets",
                            elt_types.len(),
                            t.elts.len()
                        ));
                    }
                    for (i, target_elt) in t.elts.iter().enumerate() {
                        let elt_val = self.func.next_value();
                        self.add_instruction(InstructionKind::TupleExtract(elt_val, value, i));
                        self.func.set_type(elt_val, elt_types[i].clone());
                        self.handle_assignment_target(target_elt, elt_val)?;
                    }
                } else {
                    return Err("Cannot unpack non-tuple type".to_string());
                }
            }
            _ => return Err(format!("Unsupported assignment target: {:?}", target)),
        }
        Ok(())
    }

    pub(super) fn resolve_attribute_path(
        &self,
        expr: ast::Expr,
    ) -> Result<(String, usize, Type), String> {
        match expr {
            ast::Expr::Name(n) => {
                let name = n.id.to_string();
                // Find the variable's value to get its type
                let val = if let Some(blocks) = self.variable_defs.get(&name) {
                    blocks
                        .get(&self.current_block)
                        .cloned()
                        .or_else(|| blocks.values().next().cloned()) // Fallback
                        .ok_or_else(|| format!("Variable '{}' not defined", name))?
                } else {
                    return Err(format!("Variable '{}' not defined", name));
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
                            format!(
                                "Field '{}' not found in struct '{}'",
                                attr.attr, struct_name
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
                    Err(format!(
                        "Cannot resolve attribute '{}' on non-struct type {:?}",
                        attr.attr, parent_ty
                    ))
                }
            }
            _ => Err(format!(
                "Invalid attribute path: must start with a variable name, found {:?}",
                expr
            )),
        }
    }
}
