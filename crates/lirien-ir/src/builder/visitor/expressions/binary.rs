use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::{push_inst, builder_error};
use crate::ir::{InstructionKind, Type, Value};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(crate) fn visit_binop(&mut self, s: ast::ExprBinOp, expr_offset: usize) -> BuilderResult<Value> {
        let mut lhs = self.visit_expr(*s.left)?;
        let mut rhs = self.visit_expr(*s.right)?;
        lhs = self.auto_load(lhs);
        rhs = self.auto_load(rhs);
        self.update_location(expr_offset);
        let dest = self.func.next_value();
        let kind = self.build_binop(s.op, lhs, rhs, dest)?;
        let op_str = match s.op {
            ast::Operator::Add => "+",
            ast::Operator::Sub => "-",
            ast::Operator::Mult => "*",
            ast::Operator::Div => "/",
            ast::Operator::Mod => "%",
            ast::Operator::BitAnd => "&",
            ast::Operator::BitOr => "|",
            ast::Operator::BitXor => "^",
            ast::Operator::LShift => "<<",
            ast::Operator::RShift => ">>",
            _ => "",
        };
        let inst = push_inst!(self, kind);
        if !op_str.is_empty() {
            inst.add_constraint(format!("(= {} ({} {} {}))", dest, op_str, lhs, rhs));
        }
        Ok(dest)
    }

    pub(crate) fn visit_bool_op(&mut self, s: ast::ExprBoolOp) -> BuilderResult<Value> {
        if s.values.is_empty() {
            return Err(builder_error!(General, "Empty BoolOp"));
        }

        let merge_block = self.create_block();
        let result_var = "bool_op_tmp".to_string();
        let mut last_val = self.visit_expr(s.values[0].clone())?;
        last_val = self.auto_load(last_val);

        for i in 1..s.values.len() {
            let next_block = self.create_block();
            self.write_variable(result_var.clone(), self.current_block, last_val);

            match s.op {
                ast::BoolOp::And => {
                    push_inst!(self, InstructionKind::Branch(
                        last_val,
                        next_block,
                        merge_block,
                    ));
                    self.link_blocks(self.current_block, next_block);
                    self.link_blocks(self.current_block, merge_block);
                }
                ast::BoolOp::Or => {
                    push_inst!(self, InstructionKind::Branch(
                        last_val,
                        merge_block,
                        next_block,
                    ));
                    self.link_blocks(self.current_block, merge_block);
                    self.link_blocks(self.current_block, next_block);
                }
            }

            self.seal_block(next_block)?;
            self.start_block(next_block);
            last_val = self.visit_expr(s.values[i].clone())?;
            last_val = self.auto_load(last_val);
        }

        self.write_variable(result_var.clone(), self.current_block, last_val);
        push_inst!(self, InstructionKind::Jump(merge_block));
        self.link_blocks(self.current_block, merge_block);

        self.seal_block(merge_block)?;
        self.start_block(merge_block);
        self.read_variable(result_var, merge_block)
    }

    pub(crate) fn visit_unary_op(&mut self, s: ast::ExprUnaryOp) -> BuilderResult<Value> {
        let mut operand = self.visit_expr(*s.operand)?;
        operand = self.auto_load(operand);
        let dest = self.func.next_value();
        let (kind, op_str) = match s.op {
            ast::UnaryOp::Not => (InstructionKind::Not(dest, operand), "not"),
            ast::UnaryOp::Invert => (InstructionKind::Not(dest, operand), "~"),
            ast::UnaryOp::USub => {
                (InstructionKind::Neg(dest, operand), "-")
            }
            ast::UnaryOp::UAdd => return Ok(operand),
        };
        self.func.set_type(dest, self.func.get_type(operand));
        let inst = push_inst!(self, kind);
        if op_str == "-" {
            inst.add_constraint(format!("(= {} (- 0 {}))", dest, operand));
        } else if !op_str.is_empty() {
            inst.add_constraint(format!("(= {} ({} {}))", dest, op_str, operand));
        }
        Ok(dest)
    }

    pub(crate) fn visit_compare(&mut self, s: ast::ExprCompare, expr_offset: usize) -> BuilderResult<Value> {
        if s.ops.len() != 1 || s.comparators.len() != 1 {
            return Err(builder_error!(General, "Complex comparisons not supported yet"));
        }
        let mut lhs = self.visit_expr(*s.left)?;
        let mut rhs = self.visit_expr(s.comparators[0].clone())?;
        lhs = self.auto_load(lhs);
        rhs = self.auto_load(rhs);
        self.update_location(expr_offset);
        let dest = self.func.next_value();

        let mut l_ty = self.func.get_type(lhs);
        let mut r_ty = self.func.get_type(rhs);

        // Strip refinement and literal wrappers to get base type
        while let Type::Refined(inner, _) | Type::Literal(inner, _) = l_ty {
            l_ty = *inner;
        }
        while let Type::Refined(inner, _) | Type::Literal(inner, _) = r_ty {
            r_ty = *inner;
        }

        if let Type::Optional(_) = l_ty {
            let tag_val = self.func.next_value();
            self.func.set_type(tag_val, Type::Bool);
            push_inst!(self, InstructionKind::StructLoad(tag_val, lhs, 0));
            lhs = tag_val;
            l_ty = Type::Bool;
        }
        if let Type::Optional(_) = r_ty {
            let tag_val = self.func.next_value();
            self.func.set_type(tag_val, Type::Bool);
            push_inst!(self, InstructionKind::StructLoad(tag_val, rhs, 0));
            rhs = tag_val;
            r_ty = Type::Bool;
        }

        // Unify Bool and I64 (e.g. comparing tag to None/0)
        if l_ty == Type::Bool && r_ty == Type::I64 {
            self.func.set_type(rhs, Type::Bool);
        } else if l_ty == Type::I64 && r_ty == Type::Bool {
            self.func.set_type(lhs, Type::Bool);
        }

        let is_float =
            matches!(l_ty, Type::F32 | Type::F64) || matches!(r_ty, Type::F32 | Type::F64);

        let (kind, op_str) = match s.ops[0] {
            ast::CmpOp::Eq | ast::CmpOp::Is => (InstructionKind::Eq(dest, lhs, rhs), "="),
            ast::CmpOp::NotEq | ast::CmpOp::IsNot => (InstructionKind::Ne(dest, lhs, rhs), "!="),
            ast::CmpOp::Lt => {
                if is_float {
                    (InstructionKind::FLt(dest, lhs, rhs), "<")
                } else {
                    (InstructionKind::SLt(dest, lhs, rhs), "<")
                }
            }
            ast::CmpOp::LtE => {
                if is_float {
                    (InstructionKind::FLe(dest, lhs, rhs), "<=")
                } else {
                    (InstructionKind::SLe(dest, lhs, rhs), "<=")
                }
            }
            ast::CmpOp::Gt => {
                if is_float {
                    (InstructionKind::FGt(dest, lhs, rhs), ">")
                } else {
                    (InstructionKind::SGt(dest, lhs, rhs), ">")
                }
            }
            ast::CmpOp::GtE => {
                if is_float {
                    (InstructionKind::FGe(dest, lhs, rhs), ">=")
                } else {
                    (InstructionKind::SGe(dest, lhs, rhs), ">=")
                }
            }
            _ => {
                return Err(builder_error!(
                    General,
                    "Comparison operator {:?} not yet supported",
                    s.ops[0]
                ))
            }
        };

        self.func.set_type(dest, Type::Bool);
        let inst = push_inst!(self, kind);
        inst.add_constraint(format!("(= {} ({} {} {}))", dest, op_str, lhs, rhs));
        Ok(dest)
    }
}
