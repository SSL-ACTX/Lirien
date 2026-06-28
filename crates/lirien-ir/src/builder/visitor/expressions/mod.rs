pub mod access;
pub mod binary;
pub mod calls;
pub mod literals;

use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, Type, Value};
use crate::{builder_error, push_inst};
use rustpython_ast as ast;
use rustpython_ast::Ranged;

impl CFGBuilder {
    pub fn visit_expr(&mut self, expr: ast::Expr) -> BuilderResult<Value> {
        let expr_offset = expr.range().start().to_usize();
        self.update_location(expr_offset);

        match expr {
            ast::Expr::BinOp(s) => self.visit_binop(s, expr_offset),
            ast::Expr::BoolOp(s) => self.visit_bool_op(s),
            ast::Expr::UnaryOp(s) => self.visit_unary_op(s),
            ast::Expr::Compare(s) => self.visit_compare(s, expr_offset),
            ast::Expr::Constant(c) => self.visit_constant(c),
            ast::Expr::Name(n) => self.visit_name(n),
            ast::Expr::Attribute(s) => self.visit_attribute(s, expr_offset),
            ast::Expr::Subscript(s) => self.visit_subscript(s),
            ast::Expr::Tuple(t) => self.visit_tuple(t),
            ast::Expr::Call(s) => self.visit_call(s),
            ast::Expr::Lambda(s) => self.visit_lambda(s),
            ast::Expr::ListComp(s) => self.visit_listcomp(s),
            _ => Err(builder_error!(
                General,
                "Expression type {:?} not yet supported",
                expr
            )),
        }
    }

    pub fn build_binop(
        &mut self,
        op: ast::Operator,
        mut lhs: Value,
        mut rhs: Value,
        dest: Value,
    ) -> BuilderResult<InstructionKind> {
        let mut l_ty = self.func.get_type(lhs);
        let mut r_ty = self.func.get_type(rhs);

        // Handle mixed scalar-SIMD operations by automatic splatting
        if l_ty.is_simd() && !r_ty.is_simd() {
            let mut scalar_val = rhs;
            let scalar_ty = r_ty.clone();
            let target_elt_ty = match l_ty {
                Type::F32X4 => Type::F32,
                Type::I32X4 => Type::I32,
                Type::F64X2 => Type::F64,
                Type::I64X2 => Type::I64,
                Type::I8X16 => Type::I8,
                Type::U8X16 => Type::U8,
                Type::I16X8 => Type::I16,
                Type::U16X8 => Type::U16,
                _ => unreachable!(),
            };

            let is_f_lit = self.func.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    if let InstructionKind::ConstFloat(d, _) = &i.kind {
                        *d == scalar_val
                    } else {
                        false
                    }
                })
            });
            let is_i_lit = self.func.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    if let InstructionKind::ConstInt(d, _) = &i.kind {
                        *d == scalar_val
                    } else {
                        false
                    }
                })
            });

            let is_scalar_float = scalar_ty.is_float() || is_f_lit;
            let is_scalar_int = scalar_ty.is_int() || is_i_lit;

            if scalar_ty != target_elt_ty {
                let converted = self.func.next_value();
                if target_elt_ty.is_float() && is_scalar_int && !is_scalar_float {
                    push_inst!(
                        self,
                        InstructionKind::IToF(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else if !target_elt_ty.is_float() && is_scalar_float && !is_scalar_int {
                    push_inst!(
                        self,
                        InstructionKind::FToI(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else if target_elt_ty.is_float() && is_scalar_float {
                    push_inst!(
                        self,
                        InstructionKind::FConv(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else {
                    push_inst!(self, InstructionKind::Assign(converted, scalar_val));
                }
                self.func.set_type(converted, target_elt_ty.clone());
                scalar_val = converted;
            }

            let splat_val = self.func.next_value();
            push_inst!(self, InstructionKind::SIMDSplat(splat_val, scalar_val));
            self.func.set_type(splat_val, l_ty.clone());
            rhs = splat_val;
            r_ty = l_ty.clone();
        } else if !l_ty.is_simd() && r_ty.is_simd() {
            let mut scalar_val = lhs;
            let scalar_ty = l_ty.clone();
            let target_elt_ty = match r_ty {
                Type::F32X4 => Type::F32,
                Type::I32X4 => Type::I32,
                Type::F64X2 => Type::F64,
                Type::I64X2 => Type::I64,
                Type::I8X16 => Type::I8,
                Type::U8X16 => Type::U8,
                Type::I16X8 => Type::I16,
                Type::U16X8 => Type::U16,
                _ => unreachable!(),
            };

            let is_f_lit = self.func.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    if let InstructionKind::ConstFloat(d, _) = &i.kind {
                        *d == scalar_val
                    } else {
                        false
                    }
                })
            });
            let is_i_lit = self.func.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    if let InstructionKind::ConstInt(d, _) = &i.kind {
                        *d == scalar_val
                    } else {
                        false
                    }
                })
            });

            let is_scalar_float = scalar_ty.is_float() || is_f_lit;
            let is_scalar_int = scalar_ty.is_int() || is_i_lit;

            if scalar_ty != target_elt_ty {
                let converted = self.func.next_value();
                if target_elt_ty.is_float() && is_scalar_int && !is_scalar_float {
                    push_inst!(
                        self,
                        InstructionKind::IToF(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else if !target_elt_ty.is_float() && is_scalar_float && !is_scalar_int {
                    push_inst!(
                        self,
                        InstructionKind::FToI(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else if target_elt_ty.is_float() && is_scalar_float {
                    push_inst!(
                        self,
                        InstructionKind::FConv(converted, scalar_val, target_elt_ty.clone(),)
                    );
                } else {
                    push_inst!(self, InstructionKind::Assign(converted, scalar_val));
                }
                self.func.set_type(converted, target_elt_ty.clone());
                scalar_val = converted;
            }

            let splat_val = self.func.next_value();
            push_inst!(self, InstructionKind::SIMDSplat(splat_val, scalar_val));
            self.func.set_type(splat_val, r_ty.clone());
            lhs = splat_val;
            l_ty = r_ty.clone();
        }

        // Handle mixed scalar types (e.g. i64 + f64)
        if !l_ty.is_simd() && !r_ty.is_simd() && !l_ty.is_tensor() && !r_ty.is_tensor() {
            if l_ty.is_float() && r_ty.is_int() {
                let converted = self.func.next_value();
                push_inst!(self, InstructionKind::IToF(converted, rhs, l_ty.clone()));
                self.func.set_type(converted, l_ty.clone());
                rhs = converted;
                r_ty = l_ty.clone();
            } else if l_ty.is_int() && r_ty.is_float() {
                let converted = self.func.next_value();
                push_inst!(self, InstructionKind::IToF(converted, lhs, r_ty.clone()));
                self.func.set_type(converted, r_ty.clone());
                lhs = converted;
                l_ty = r_ty.clone();
            } else if l_ty.is_float() && r_ty.is_float() && l_ty != r_ty {
                // Promote f32 to f64
                let dest_ty = if matches!(l_ty, Type::F64) || matches!(r_ty, Type::F64) {
                    Type::F64
                } else {
                    Type::F32
                };
                if l_ty != dest_ty {
                    let converted = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::FConv(converted, lhs, dest_ty.clone())
                    );
                    self.func.set_type(converted, dest_ty.clone());
                    lhs = converted;
                    l_ty = dest_ty.clone();
                }
                if r_ty != dest_ty {
                    let converted = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::FConv(converted, rhs, dest_ty.clone())
                    );
                    self.func.set_type(converted, dest_ty.clone());
                    rhs = converted;
                    r_ty = dest_ty.clone();
                }
            }
        }

        let is_float = l_ty.is_float() || r_ty.is_float();

        if let (Type::Tensor(t1, dims1), Type::Tensor(t2, dims2)) = (l_ty.clone(), r_ty.clone()) {
            if op != ast::Operator::MatMult {
                if t1 != t2 {
                    return Err(builder_error!(
                        General,
                        "Tensor arithmetic requires same base types"
                    ));
                }
                if dims1 != dims2 {
                    if let Some(res_dims) = self.get_broadcast_shape(&dims1, &dims2) {
                        if dims1 != res_dims {
                            let mut target_dim_values = Vec::new();
                            for (i, d_str) in res_dims.iter().enumerate() {
                                let len1 = dims1.len();
                                let len2 = dims2.len();
                                let max_len = res_dims.len();
                                let idx1 = i as i64 - (max_len as i64 - len1 as i64);
                                let idx2 = i as i64 - (max_len as i64 - len2 as i64);

                                let val = if idx1 >= 0 && dims1[idx1 as usize] == *d_str {
                                    self.resolve_dim(lhs, d_str, idx1 as usize)
                                } else if idx2 >= 0 && dims2[idx2 as usize] == *d_str {
                                    self.resolve_dim(rhs, d_str, idx2 as usize)
                                } else {
                                    let dest_const = self.func.next_value();
                                    let c_val = d_str.parse::<i64>().unwrap_or(1);
                                    push_inst!(self, InstructionKind::ConstInt(dest_const, c_val));
                                    self.func.set_type(dest_const, Type::I64);
                                    dest_const
                                };
                                target_dim_values.push(val);
                            }

                            let new_lhs = self.func.next_value();
                            push_inst!(
                                self,
                                InstructionKind::TensorBroadcast(new_lhs, lhs, target_dim_values,)
                            );
                            self.func
                                .set_type(new_lhs, Type::Tensor(t1.clone(), res_dims.clone()));
                            lhs = new_lhs;
                        }

                        if dims2 != res_dims {
                            let mut target_dim_values = Vec::new();
                            for (i, d_str) in res_dims.iter().enumerate() {
                                let len1 = dims1.len();
                                let len2 = dims2.len();
                                let max_len = res_dims.len();
                                let idx1 = i as i64 - (max_len as i64 - len1 as i64);
                                let idx2 = i as i64 - (max_len as i64 - len2 as i64);

                                let val = if idx1 >= 0 && dims1[idx1 as usize] == *d_str {
                                    self.resolve_dim(lhs, d_str, idx1 as usize)
                                } else if idx2 >= 0 && dims2[idx2 as usize] == *d_str {
                                    self.resolve_dim(rhs, d_str, idx2 as usize)
                                } else {
                                    let dest_const = self.func.next_value();
                                    let c_val = d_str.parse::<i64>().unwrap_or(1);
                                    push_inst!(self, InstructionKind::ConstInt(dest_const, c_val));
                                    self.func.set_type(dest_const, Type::I64);
                                    dest_const
                                };
                                target_dim_values.push(val);
                            }

                            let new_rhs = self.func.next_value();
                            push_inst!(
                                self,
                                InstructionKind::TensorBroadcast(new_rhs, rhs, target_dim_values,)
                            );
                            self.func
                                .set_type(new_rhs, Type::Tensor(t2.clone(), res_dims.clone()));
                            rhs = new_rhs;
                        }
                        l_ty = Type::Tensor(t1, res_dims);
                    } else {
                        return Err(builder_error!(
                            General,
                            "Tensor shape mismatch in element-wise operation: {:?} vs {:?}",
                            dims1,
                            dims2
                        ));
                    }
                }

                let kind = match op {
                    ast::Operator::Add => InstructionKind::TensorAdd(dest, lhs, rhs),
                    ast::Operator::Sub => InstructionKind::TensorSub(dest, lhs, rhs),
                    ast::Operator::Mult => InstructionKind::TensorMul(dest, lhs, rhs),
                    ast::Operator::Div => InstructionKind::TensorDiv(dest, lhs, rhs),
                    _ => {
                        return Err(builder_error!(
                            General,
                            "Operator {:?} not supported for Tensors",
                            op
                        ))
                    }
                };
                self.func.set_type(dest, l_ty.clone());
                return Ok(kind);
            }
        }

        if let Type::Tensor(_inner, _) = &l_ty {
            if !r_ty.is_tensor() {
                // Tensor-Scalar arithmetic
                let kind = match op {
                    ast::Operator::Add => InstructionKind::TensorScalarAdd(dest, lhs, rhs),
                    ast::Operator::Sub => InstructionKind::TensorScalarSub(dest, lhs, rhs),
                    ast::Operator::Mult => InstructionKind::TensorScalarMul(dest, lhs, rhs),
                    ast::Operator::Div => InstructionKind::TensorScalarDiv(dest, lhs, rhs),
                    _ => {
                        return Err(builder_error!(
                            General,
                            "Operator {:?} not supported for Tensor-Scalar",
                            op
                        ))
                    }
                };
                self.func.set_type(dest, l_ty.clone());
                return Ok(kind);
            }
        } else if let Type::Tensor(_inner, _) = &r_ty {
            if !l_ty.is_tensor() {
                // Scalar-Tensor arithmetic (e.g. 5.0 + tensor)
                // For Add and Mul, it's commutative
                let kind = match op {
                    ast::Operator::Add => InstructionKind::TensorScalarAdd(dest, rhs, lhs),
                    ast::Operator::Mult => InstructionKind::TensorScalarMul(dest, rhs, lhs),
                    _ => {
                        return Err(builder_error!(
                            General,
                            "Operator {:?} not supported for Scalar-Tensor",
                            op
                        ))
                    }
                };
                self.func.set_type(dest, r_ty.clone());
                return Ok(kind);
            }
        }

        let kind = match op {
            ast::Operator::Add => {
                if is_float {
                    InstructionKind::FAdd(dest, lhs, rhs)
                } else {
                    InstructionKind::Add(dest, lhs, rhs)
                }
            }
            ast::Operator::Sub => {
                if is_float {
                    InstructionKind::FSub(dest, lhs, rhs)
                } else {
                    InstructionKind::Sub(dest, lhs, rhs)
                }
            }
            ast::Operator::Mult => {
                if is_float {
                    InstructionKind::FMul(dest, lhs, rhs)
                } else {
                    InstructionKind::Mul(dest, lhs, rhs)
                }
            }
            ast::Operator::MatMult => InstructionKind::MatMult(dest, lhs, rhs),
            ast::Operator::Div => InstructionKind::FDiv(dest, lhs, rhs),
            ast::Operator::FloorDiv => InstructionKind::SDiv(dest, lhs, rhs),
            ast::Operator::Mod => InstructionKind::SRem(dest, lhs, rhs),
            ast::Operator::BitAnd => InstructionKind::And(dest, lhs, rhs),
            ast::Operator::BitOr => InstructionKind::Or(dest, lhs, rhs),
            ast::Operator::BitXor => InstructionKind::Xor(dest, lhs, rhs),
            ast::Operator::LShift => InstructionKind::Shl(dest, lhs, rhs),
            ast::Operator::RShift => InstructionKind::AShr(dest, lhs, rhs),
            _ => {
                return Err(builder_error!(
                    General,
                    "Operator {:?} not yet supported",
                    op
                ))
            }
        };

        if let (ast::Operator::MatMult, Type::Tensor(t1, dims1), Type::Tensor(t2, dims2)) =
            (op, &l_ty, &r_ty)
        {
            if t1 != t2 {
                return Err(builder_error!(
                    General,
                    "Matrix multiplication requires tensors of the same base type"
                ));
            }
            if dims1.len() != 2 || dims2.len() != 2 {
                return Err(builder_error!(
                    General,
                    "Matrix multiplication currently requires exactly 2D tensors"
                ));
            }
            // Resulting shape: [M, K] from [M, N] @ [N, K]
            let new_dims = vec![dims1[0].clone(), dims2[1].clone()];
            self.func.set_type(dest, Type::Tensor(t1.clone(), new_dims));
        } else if l_ty.is_simd() {
            self.func.set_type(dest, l_ty);
        } else if is_float {
            self.func.set_type(
                dest,
                if matches!(l_ty, Type::F64) || matches!(r_ty, Type::F64) {
                    Type::F64
                } else {
                    Type::F32
                },
            );
        } else {
            self.func.set_type(dest, l_ty);
        }

        Ok(kind)
    }

    pub(crate) fn visit_listcomp(&mut self, s: ast::ExprListComp) -> BuilderResult<Value> {
        if s.generators.len() != 1 {
            return Err(builder_error!(
                UnsupportedStatement,
                "List comprehensions with multiple generators are not yet supported"
            ));
        }

        let gen = &s.generators[0];
        if gen.is_async {
            return Err(builder_error!(
                UnsupportedStatement,
                "Async list comprehensions are not supported"
            ));
        }

        let target_name = match &gen.target {
            ast::Expr::Name(n) => n.id.to_string(),
            _ => {
                return Err(builder_error!(
                    UnsupportedStatement,
                    "List comprehension target must be a simple variable name"
                ))
            }
        };

        // 1. Determine if iter is range(...)
        let mut is_range = false;
        let mut range_start = None;
        let mut range_end = None;
        let mut range_step = None;

        if let ast::Expr::Call(range_call) = &gen.iter {
            if let ast::Expr::Name(n) = &*range_call.func {
                if n.id.as_str() == "range" {
                    is_range = true;
                    match range_call.args.len() {
                        1 => {
                            range_end = Some(self.visit_expr(range_call.args[0].clone())?);
                        }
                        2 => {
                            range_start = Some(self.visit_expr(range_call.args[0].clone())?);
                            range_end = Some(self.visit_expr(range_call.args[1].clone())?);
                        }
                        3 => {
                            range_start = Some(self.visit_expr(range_call.args[0].clone())?);
                            range_end = Some(self.visit_expr(range_call.args[1].clone())?);
                            range_step = Some(self.visit_expr(range_call.args[2].clone())?);
                        }
                        _ => return Err(builder_error!(General, "Unsupported range() signature")),
                    }
                }
            }
        }

        let prev_block = self.current_block;

        // Resolve range start/step values if it is a range
        let mut st_v = None;
        let mut target_ty = Type::I64;
        let mut iter_val = None;
        let mut iter_ty = Type::Unknown;
        let mut len_val = None;
        let mut idx_var_name = String::new();

        if is_range {
            let start_val = if let Some(v) = range_start {
                v
            } else {
                let zero = self.func.next_value();
                push_inst!(self, InstructionKind::ConstInt(zero, 0));
                self.func.set_type(zero, Type::I64);
                zero
            };

            let step_val = if let Some(v) = range_step {
                v
            } else {
                let one = self.func.next_value();
                push_inst!(self, InstructionKind::ConstInt(one, 1));
                self.func.set_type(one, Type::I64);
                one
            };
            st_v = Some(step_val);

            self.write_variable(target_name.clone(), prev_block, start_val);
        } else {
            // It's a collection
            let v = self.visit_expr(gen.iter.clone())?;
            let v = self.auto_load(v);
            iter_val = Some(v);
            iter_ty = self.func.get_type(v);

            target_ty = match &iter_ty {
                Type::List(inner)
                | Type::Array(inner, _)
                | Type::Buffer(inner)
                | Type::Tensor(inner, _) => (**inner).clone(),
                _ => {
                    return Err(builder_error!(
                        General,
                        "List comprehension iterable must be a collection or range(), found {:?}",
                        iter_ty
                    ))
                }
            };

            // Get length of the collection
            let l_val = self.func.next_value();
            let len_inst = match &iter_ty {
                Type::List(_) => InstructionKind::ListLen(l_val, v),
                Type::Buffer(_) => InstructionKind::BufferLen(l_val, v),
                Type::Array(_, Some(size)) => InstructionKind::ConstInt(l_val, *size as i64),
                _ => unreachable!(),
            };
            push_inst!(self, len_inst);
            self.func.set_type(l_val, Type::I64);
            len_val = Some(l_val);

            // Initialize index to 0
            let idx_val = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(idx_val, 0));
            self.func.set_type(idx_val, Type::I64);

            idx_var_name = format!("_comp_idx_{}", self.func.value_count);
            self.write_variable(idx_var_name.clone(), prev_block, idx_val);
        }

        // Initialize the empty list (element type is unknown yet)
        let list_val = self.func.next_value();
        self.func.set_type(list_val, Type::Unknown);

        let list_var_name = format!("_comp_list_{}", self.func.value_count);
        self.write_variable(list_var_name.clone(), prev_block, list_val);

        // Setup blocks
        let header_block = self.create_block();
        let body_block = self.create_block();
        let latch_block = self.create_block();
        let exit_block = self.create_block();

        push_inst!(self, InstructionKind::Jump(header_block));
        self.link_blocks(prev_block, header_block);

        // --- Header Block ---
        self.start_block(header_block);
        let curr_idx = if is_range {
            self.read_variable(target_name.clone(), header_block)?
        } else {
            self.read_variable(idx_var_name.clone(), header_block)?
        };

        let limit_val = if is_range {
            range_end.unwrap()
        } else {
            len_val.unwrap()
        };

        let cond = self.func.next_value();
        push_inst!(self, InstructionKind::SLt(cond, curr_idx, limit_val));
        self.func.set_type(cond, Type::Bool);

        push_inst!(self, InstructionKind::Branch(cond, body_block, exit_block));
        self.link_blocks(header_block, body_block);
        self.link_blocks(header_block, exit_block);

        // --- Body Block ---
        self.seal_block(body_block)?;
        self.start_block(body_block);

        let target_val = if is_range {
            curr_idx
        } else {
            let loaded_val = self.func.next_value();
            let load_inst = match &iter_ty {
                Type::List(_) => InstructionKind::ListLoad(loaded_val, iter_val.unwrap(), curr_idx),
                Type::Buffer(_) => {
                    InstructionKind::BufferLoad(loaded_val, iter_val.unwrap(), curr_idx)
                }
                Type::Array(_, _) => {
                    InstructionKind::ArrayLoad(loaded_val, iter_val.unwrap(), curr_idx)
                }
                _ => unreachable!(),
            };
            push_inst!(self, load_inst);
            self.func.set_type(loaded_val, target_ty.clone());
            loaded_val
        };

        self.write_variable(target_name.clone(), body_block, target_val);

        // Handle filters (ifs)
        let mut current_body_block = body_block;
        for filter_expr in &gen.ifs {
            let pass_block = self.create_block();
            let filter_cond = self.visit_expr(filter_expr.clone())?;
            let filter_cond = self.auto_load(filter_cond);

            push_inst!(
                self,
                InstructionKind::Branch(filter_cond, pass_block, latch_block)
            );
            self.link_blocks(current_body_block, pass_block);
            self.link_blocks(current_body_block, latch_block);

            self.seal_block(pass_block)?;
            self.start_block(pass_block);
            current_body_block = pass_block;
        }

        // Evaluate elt
        let elt_val = self.visit_expr((*s.elt).clone())?;
        let elt_val = self.auto_load(elt_val);
        let elt_ty = self.func.get_type(elt_val);

        // Append to list
        let curr_list = self.read_variable(list_var_name.clone(), current_body_block)?;
        let next_list_val = self.func.next_value();
        push_inst!(
            self,
            InstructionKind::ListAppend(next_list_val, curr_list, elt_val)
        );
        self.func
            .set_type(next_list_val, Type::List(Box::new(elt_ty.clone())));
        self.write_variable(list_var_name.clone(), current_body_block, next_list_val);

        push_inst!(self, InstructionKind::Jump(latch_block));
        self.link_blocks(current_body_block, latch_block);

        // --- Latch Block ---
        self.seal_block(latch_block)?;
        self.start_block(latch_block);

        if is_range {
            let curr_val = self.read_variable(target_name.clone(), latch_block)?;
            let next_val = self.func.next_value();
            let step_v = st_v.unwrap();
            push_inst!(self, InstructionKind::Add(next_val, curr_val, step_v));
            self.func.set_type(next_val, Type::I64);
            self.write_variable(target_name.clone(), latch_block, next_val);
        } else {
            let curr_val = self.read_variable(idx_var_name.clone(), latch_block)?;
            let next_val = self.func.next_value();
            let one = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(one, 1));
            self.func.set_type(one, Type::I64);
            push_inst!(self, InstructionKind::Add(next_val, curr_val, one));
            self.func.set_type(next_val, Type::I64);
            self.write_variable(idx_var_name.clone(), latch_block, next_val);
        }

        push_inst!(self, InstructionKind::Jump(header_block));
        self.link_blocks(latch_block, header_block);

        // --- Exit Block ---
        self.seal_block(header_block)?;
        self.start_block(exit_block);

        // Insert ListCreate in prev_block now that we know elt_ty
        self.insert_instruction_before_terminator(
            prev_block,
            InstructionKind::ListCreate(list_val, elt_ty.clone()),
        );
        self.func.set_type(list_val, Type::List(Box::new(elt_ty)));

        let final_list_val = self.read_variable(list_var_name, exit_block)?;
        self.seal_block(exit_block)?;
        Ok(final_list_val)
    }
}
