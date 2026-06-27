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
}
