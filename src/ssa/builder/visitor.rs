use super::metadata::{extract_refinement, parse_type};
use super::CFGBuilder;
use crate::ssa::ir::{InstructionKind, SourceLocation, Type, Value};
use rustpython_ast as ast;
use rustpython_ast::Ranged;

impl CFGBuilder {
    pub fn visit_function_def(&mut self, s: ast::StmtFunctionDef) -> Result<(), String> {
        self.update_location(s.range().start().to_usize());
        self.func.arg_count = s.args.args.len();

        if let Some(returns) = &s.returns {
            self.func.return_type = parse_type(returns, &self.type_aliases)?;
        }

        for arg in s.args.args {
            let val = self.func.next_value();
            if let Some(annotation) = &arg.def.annotation {
                let ty = parse_type(annotation, &self.type_aliases)?;
                self.func.set_type(val, ty);
                if let Some(refinement) =
                    extract_refinement(annotation, &self.type_aliases, &self.func.struct_layouts)
                {
                    self.func.set_refinement(val, refinement);
                }
            }
            self.write_variable(arg.def.arg.to_string(), self.current_block, val);
        }

        for stmt in s.body {
            self.visit_stmt(stmt)?;
        }

        // Ensure the last block has a return if it's not terminated
        if !self.is_terminated(self.current_block) {
            let ret_val = if self.func.return_type != Type::Unknown {
                let zero = self.func.next_value();
                match self.func.return_type {
                    Type::F32 | Type::F64 => {
                        self.add_instruction(InstructionKind::ConstFloat(zero, 0.0))
                    }
                    _ => self.add_instruction(InstructionKind::ConstInt(zero, 0)),
                }
                Some(zero)
            } else {
                None
            };
            self.add_instruction(InstructionKind::Return(ret_val));
        }

        Ok(())
    }

    pub fn update_location(&mut self, line: usize) {
        self.current_location = Some(SourceLocation { line, column: 0 });
    }

    pub fn visit_stmt(&mut self, stmt: ast::Stmt) -> Result<(), String> {
        self.update_location(stmt.range().start().to_usize());
        match stmt {
            ast::Stmt::Assign(s) => {
                if s.targets.len() != 1 {
                    return Err("Only single targets supported".to_string());
                }
                let target = &s.targets[0];
                let value = self.visit_expr(*s.value)?;

                self.handle_assignment_target(target, value)?;
                Ok(())
            }
            ast::Stmt::AugAssign(s) => {
                let target = *s.target;
                let value = self.visit_expr(*s.value)?;
                let lhs = self.visit_expr(target.clone())?;
                let dest = self.func.next_value();
                let kind = self.build_binop(s.op, lhs, value, dest)?;
                self.add_instruction(kind);
                self.handle_assignment_target(&target, dest)?;
                Ok(())
            }
            ast::Stmt::AnnAssign(s) => {
                if let Some(value_expr) = s.value {
                    let value = self.visit_expr(*value_expr)?;
                    self.handle_assignment_target(&s.target, value)?;
                }
                Ok(())
            }
            ast::Stmt::Return(s) => {
                let val = if let Some(expr) = s.value {
                    Some(self.visit_expr(*expr)?)
                } else {
                    None
                };
                self.add_instruction(InstructionKind::Return(val));
                Ok(())
            }
            ast::Stmt::If(s) => {
                let cond = self.visit_expr(*s.test)?;
                let prev_block = self.current_block;

                let true_block = self.create_block();
                let false_block = self.create_block();
                let merge_block = self.create_block();

                self.add_instruction(InstructionKind::Branch(cond, true_block, false_block));
                self.link_blocks(prev_block, true_block);
                self.link_blocks(prev_block, false_block);

                self.seal_block(true_block)?;
                self.seal_block(false_block)?;

                self.start_block(true_block);
                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    self.add_instruction(InstructionKind::Jump(merge_block));
                    self.link_blocks(self.current_block, merge_block);
                }

                self.start_block(false_block);
                for stmt in s.orelse {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    self.add_instruction(InstructionKind::Jump(merge_block));
                    self.link_blocks(self.current_block, merge_block);
                }

                self.seal_block(merge_block)?;
                self.start_block(merge_block);
                Ok(())
            }
            ast::Stmt::While(s) => {
                let header_block = self.create_block();
                let body_block = self.create_block();
                let exit_block = self.create_block();

                let prev_block = self.current_block;
                self.add_instruction(InstructionKind::Jump(header_block));
                self.link_blocks(prev_block, header_block);

                self.loop_stack.push((header_block, exit_block));

                self.start_block(header_block);
                let cond = self.visit_expr(*s.test)?;
                self.add_instruction(InstructionKind::Branch(cond, body_block, exit_block));
                self.link_blocks(header_block, body_block);
                self.link_blocks(header_block, exit_block);

                self.seal_block(body_block)?;
                self.start_block(body_block);
                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    self.add_instruction(InstructionKind::Jump(header_block));
                    self.link_blocks(self.current_block, header_block);
                }

                self.loop_stack.pop();
                self.seal_block(header_block)?;
                self.start_block(exit_block);
                self.seal_block(exit_block)?;
                Ok(())
            }
            ast::Stmt::For(s) => {
                let iter_expr = *s.iter;
                let target = *s.target;

                let mut is_enumerate = false;
                let mut enum_buf_expr = None;

                let (start_val, end_val, step_val, is_direct_iter) = if let ast::Expr::Call(call) =
                    iter_expr.clone()
                {
                    if let ast::Expr::Name(n) = *call.func {
                        if n.id.as_str() == "range" {
                            let (start, end, step) = match call.args.len() {
                                1 => (None, self.visit_expr(call.args[0].clone())?, None),
                                2 => (
                                    Some(self.visit_expr(call.args[0].clone())?),
                                    self.visit_expr(call.args[1].clone())?,
                                    None,
                                ),
                                3 => (
                                    Some(self.visit_expr(call.args[0].clone())?),
                                    self.visit_expr(call.args[1].clone())?,
                                    Some(self.visit_expr(call.args[2].clone())?),
                                ),
                                _ => return Err("Unsupported range() signature".to_string()),
                            };

                            let start_v = if let Some(v) = start {
                                v
                            } else {
                                let zero = self.func.next_value();
                                self.add_instruction(InstructionKind::ConstInt(zero, 0));
                                zero
                            };
                            let step_v = if let Some(v) = step {
                                v
                            } else {
                                let one = self.func.next_value();
                                self.add_instruction(InstructionKind::ConstInt(one, 1));
                                one
                            };
                            (start_v, end, step_v, false)
                        } else if n.id.as_str() == "enumerate" && call.args.len() == 1 {
                            is_enumerate = true;
                            enum_buf_expr = Some(call.args[0].clone());
                            let buf_val = self.visit_expr(call.args[0].clone())?;
                            let buf_ty = self.func.get_type(buf_val);
                            let zero = self.func.next_value();
                            self.add_instruction(InstructionKind::ConstInt(zero, 0));
                            let one = self.func.next_value();
                            self.add_instruction(InstructionKind::ConstInt(one, 1));
                            let len = self.func.next_value();
                            if let Type::Buffer(_) = buf_ty {
                                self.add_instruction(InstructionKind::BufferLen(len, buf_val));
                            } else if let Type::Array(_, Some(size)) = buf_ty {
                                self.add_instruction(InstructionKind::ConstInt(len, size as i64));
                            } else {
                                return Err("Cannot iterate over unknown size array".to_string());
                            }
                            self.func.set_type(len, Type::I64);
                            (zero, len, one, true)
                        } else {
                            return Err(format!("Unsupported function in for loop: {}", n.id));
                        }
                    } else {
                        return Err("Only range() or direct iteration supported".to_string());
                    }
                } else {
                    // Potential direct iteration: for x in buf
                    let buf_val = self.visit_expr(iter_expr.clone())?;
                    let buf_ty = self.func.get_type(buf_val);
                    match buf_ty {
                        Type::Buffer(_) | Type::Array(_, _) => {
                            let zero = self.func.next_value();
                            self.add_instruction(InstructionKind::ConstInt(zero, 0));
                            let one = self.func.next_value();
                            self.add_instruction(InstructionKind::ConstInt(one, 1));
                            let len = self.func.next_value();
                            if let Type::Buffer(_) = buf_ty {
                                self.add_instruction(InstructionKind::BufferLen(len, buf_val));
                            } else if let Type::Array(_, Some(size)) = buf_ty {
                                self.add_instruction(InstructionKind::ConstInt(len, size as i64));
                            } else {
                                return Err("Cannot iterate over unknown size array".to_string());
                            }
                            self.func.set_type(len, Type::I64);
                            (zero, len, one, true)
                        }
                        _ => return Err(format!("Cannot iterate over type {:?}", buf_ty)),
                    }
                };

                let header_block = self.create_block();
                let body_block = self.create_block();
                let exit_block = self.create_block();

                // Iterator variable (index)
                let idx_name = if is_direct_iter {
                    format!("_lila_idx_{}", self.func.value_count)
                } else if let ast::Expr::Name(n) = target.clone() {
                    n.id.to_string()
                } else {
                    return Err("Unsupported loop target".to_string());
                };

                self.write_variable(idx_name.clone(), self.current_block, start_val);

                let prev_block = self.current_block;
                self.add_instruction(InstructionKind::Jump(header_block));
                self.link_blocks(prev_block, header_block);

                self.loop_stack.push((header_block, exit_block));

                self.start_block(header_block);
                let curr_idx = self.read_variable(idx_name.clone(), header_block)?;
                let cond = self.func.next_value();

                // Determine if we should use SLt or SGt based on step if constant
                let mut use_sgt = false;
                // Try to find if step_val is a negative constant
                for block in &self.func.blocks {
                    for inst in &block.instructions {
                        if let InstructionKind::ConstInt(v, val) = inst.kind {
                            if v == step_val && val < 0 {
                                use_sgt = true;
                                break;
                            }
                        }
                    }
                }

                if use_sgt {
                    self.add_instruction(InstructionKind::SGt(cond, curr_idx, end_val));
                } else {
                    self.add_instruction(InstructionKind::SLt(cond, curr_idx, end_val));
                }
                self.add_instruction(InstructionKind::Branch(cond, body_block, exit_block));
                self.link_blocks(header_block, body_block);
                self.link_blocks(header_block, exit_block);

                self.seal_block(body_block)?;
                self.start_block(body_block);

                if is_direct_iter {
                    // For direct iter, load the value into the target variable
                    let buf_expr = if is_enumerate {
                        enum_buf_expr.unwrap()
                    } else {
                        iter_expr.clone()
                    };
                    let buf_val = self.visit_expr(buf_expr)?;
                    let buf_ty = self.func.get_type(buf_val);
                    let element = self.func.next_value();
                    match buf_ty {
                        Type::Buffer(inner) => {
                            self.add_instruction(InstructionKind::BufferLoad(
                                element, buf_val, curr_idx,
                            ));
                            self.func.set_type(element, *inner);
                        }
                        Type::Array(inner, _) => {
                            self.add_instruction(InstructionKind::ArrayLoad(
                                element, buf_val, curr_idx,
                            ));
                            self.func.set_type(element, *inner);
                        }
                        _ => unreachable!(),
                    }

                    if is_enumerate {
                        if let ast::Expr::Tuple(t) = target {
                            if t.elts.len() != 2 {
                                return Err(
                                    "enumerate() requires a tuple of 2 elements".to_string()
                                );
                            }
                            self.handle_assignment_target(&t.elts[0], curr_idx)?;
                            self.handle_assignment_target(&t.elts[1], element)?;
                        } else {
                            return Err("enumerate() requires a tuple target".to_string());
                        }
                    } else if let ast::Expr::Name(n) = target {
                        self.write_variable(n.id.to_string(), self.current_block, element);
                    } else {
                        return Err("Unsupported loop target".to_string());
                    }
                }

                for stmt_in_body in s.body {
                    self.visit_stmt(stmt_in_body)?;
                }

                if !self.is_terminated(self.current_block) {
                    let next_idx = self.func.next_value();
                    let updated_idx = self.read_variable(idx_name.clone(), self.current_block)?;
                    self.add_instruction(InstructionKind::Add(next_idx, updated_idx, step_val));
                    self.write_variable(idx_name, self.current_block, next_idx);
                    self.add_instruction(InstructionKind::Jump(header_block));
                    self.link_blocks(self.current_block, header_block);
                }

                self.loop_stack.pop();
                self.seal_block(header_block)?;
                self.start_block(exit_block);
                self.seal_block(exit_block)?;
                Ok(())
            }
            ast::Stmt::Break(_) => {
                if let Some((_, exit_block)) = self.loop_stack.last() {
                    let eb = *exit_block;
                    self.add_instruction(InstructionKind::Jump(eb));
                    self.link_blocks(self.current_block, eb);
                    Ok(())
                } else {
                    Err("break outside of loop".to_string())
                }
            }
            ast::Stmt::Continue(_) => {
                if let Some((header_block, _)) = self.loop_stack.last() {
                    let hb = *header_block;
                    self.add_instruction(InstructionKind::Jump(hb));
                    self.link_blocks(self.current_block, hb);
                    Ok(())
                } else {
                    Err("continue outside of loop".to_string())
                }
            }
            ast::Stmt::Expr(s) => {
                self.visit_expr(*s.value)?;
                Ok(())
            }
            _ => Err(format!("Statement type {:?} not yet supported", stmt)),
        }
    }

    pub fn visit_expr(&mut self, expr: ast::Expr) -> Result<Value, String> {
        match expr {
            ast::Expr::BinOp(s) => {
                let lhs = self.visit_expr(*s.left)?;
                let rhs = self.visit_expr(*s.right)?;
                let dest = self.func.next_value();
                let kind = self.build_binop(s.op, lhs, rhs, dest)?;
                self.add_instruction(kind);
                Ok(dest)
            }
            ast::Expr::BoolOp(s) => {
                if s.values.is_empty() {
                    return Err("Empty BoolOp".to_string());
                }

                let merge_block = self.create_block();
                let result_var = "bool_op_tmp".to_string();
                let mut last_val = self.visit_expr(s.values[0].clone())?;

                for i in 1..s.values.len() {
                    let next_block = self.create_block();
                    self.write_variable(result_var.clone(), self.current_block, last_val);

                    match s.op {
                        ast::BoolOp::And => {
                            self.add_instruction(InstructionKind::Branch(
                                last_val,
                                next_block,
                                merge_block,
                            ));
                            self.link_blocks(self.current_block, next_block);
                            self.link_blocks(self.current_block, merge_block);
                        }
                        ast::BoolOp::Or => {
                            self.add_instruction(InstructionKind::Branch(
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
                }

                self.write_variable(result_var.clone(), self.current_block, last_val);
                self.add_instruction(InstructionKind::Jump(merge_block));
                self.link_blocks(self.current_block, merge_block);

                self.seal_block(merge_block)?;
                self.start_block(merge_block);
                self.read_variable(result_var, merge_block)
            }
            ast::Expr::UnaryOp(s) => {
                let operand = self.visit_expr(*s.operand)?;
                let dest = self.func.next_value();
                let kind = match s.op {
                    ast::UnaryOp::Not => InstructionKind::Not(dest, operand),
                    ast::UnaryOp::Invert => InstructionKind::Not(dest, operand),
                    ast::UnaryOp::USub => {
                        let zero = self.func.next_value();
                        self.add_instruction(InstructionKind::ConstInt(zero, 0));
                        InstructionKind::Sub(dest, zero, operand)
                    }
                    ast::UnaryOp::UAdd => return Ok(operand),
                };
                self.add_instruction(kind);
                Ok(dest)
            }
            ast::Expr::Compare(s) => {
                if s.ops.len() != 1 || s.comparators.len() != 1 {
                    return Err("Complex comparisons not supported yet".to_string());
                }
                let lhs = self.visit_expr(*s.left)?;
                let rhs = self.visit_expr(s.comparators[0].clone())?;
                let dest = self.func.next_value();

                let l_ty = self.func.get_type(lhs);
                let r_ty = self.func.get_type(rhs);
                let is_float =
                    matches!(l_ty, Type::F32 | Type::F64) || matches!(r_ty, Type::F32 | Type::F64);

                let kind = match s.ops[0] {
                    ast::CmpOp::Eq => InstructionKind::Eq(dest, lhs, rhs),
                    ast::CmpOp::NotEq => InstructionKind::Ne(dest, lhs, rhs),
                    ast::CmpOp::Lt => {
                        if is_float {
                            InstructionKind::FLt(dest, lhs, rhs)
                        } else {
                            InstructionKind::SLt(dest, lhs, rhs)
                        }
                    }
                    ast::CmpOp::LtE => {
                        if is_float {
                            InstructionKind::FLe(dest, lhs, rhs)
                        } else {
                            InstructionKind::SLe(dest, lhs, rhs)
                        }
                    }
                    ast::CmpOp::Gt => {
                        if is_float {
                            InstructionKind::FGt(dest, lhs, rhs)
                        } else {
                            InstructionKind::SGt(dest, lhs, rhs)
                        }
                    }
                    ast::CmpOp::GtE => {
                        if is_float {
                            InstructionKind::FGe(dest, lhs, rhs)
                        } else {
                            InstructionKind::SGe(dest, lhs, rhs)
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Comparison operator {:?} not yet supported",
                            s.ops[0]
                        ))
                    }
                };

                self.func.set_type(dest, Type::Bool);
                self.add_instruction(kind);
                Ok(dest)
            }
            ast::Expr::Constant(c) => {
                let val = self.func.next_value();
                match c.value {
                    ast::Constant::Int(i) => {
                        let int_val = i.to_string().parse::<i64>().map_err(|_| "Int too large")?;
                        self.add_instruction(InstructionKind::ConstInt(val, int_val));
                        self.func.set_type(val, Type::I64);
                    }
                    ast::Constant::Float(f) => {
                        self.add_instruction(InstructionKind::ConstFloat(val, f));
                        self.func.set_type(val, Type::F64);
                    }
                    ast::Constant::Bool(b) => {
                        self.add_instruction(InstructionKind::ConstInt(val, if b { 1 } else { 0 }));
                        self.func.set_type(val, Type::Bool);
                    }
                    _ => return Err("Unsupported constant type".to_string()),
                }
                Ok(val)
            }
            ast::Expr::Name(n) => self.read_variable(n.id.to_string(), self.current_block),
            ast::Expr::Attribute(s) => {
                // Handle .val for Mut/Ref as before
                if let ast::Expr::Name(n) = *s.value.clone() {
                    if s.attr.as_str() == "val" {
                        return self.read_variable(n.id.to_string(), self.current_block);
                    }
                }

                let (root_name, offset, leaf_ty) =
                    self.resolve_attribute_path(ast::Expr::Attribute(s.clone()))?;
                let root_val = self.read_variable(root_name, self.current_block)?;

                let dest = self.func.next_value();
                if let Type::Struct(_) = leaf_ty {
                    self.add_instruction(InstructionKind::StructOffset(dest, root_val, offset));
                } else {
                    self.add_instruction(InstructionKind::StructLoad(dest, root_val, offset));
                }
                self.func.set_type(dest, leaf_ty);
                Ok(dest)
            }
            ast::Expr::Subscript(s) => {
                let arr = self.visit_expr(*s.value)?;
                let idx = self.visit_expr(*s.slice)?;
                let dest = self.func.next_value();
                match self.func.get_type(arr) {
                    Type::Buffer(inner) => {
                        self.add_instruction(InstructionKind::BufferLoad(dest, arr, idx));
                        self.func.set_type(dest, *inner);
                    }
                    Type::Array(inner, _) => {
                        self.add_instruction(InstructionKind::ArrayLoad(dest, arr, idx));
                        self.func.set_type(dest, *inner);
                    }
                    _ => {
                        self.add_instruction(InstructionKind::ArrayLoad(dest, arr, idx));
                    }
                }
                Ok(dest)
            }
            ast::Expr::Tuple(t) => {
                let mut elts = Vec::new();
                let mut elt_types = Vec::new();
                for elt in &t.elts {
                    let val = self.visit_expr(elt.clone())?;
                    elts.push(val);
                    elt_types.push(self.func.get_type(val));
                }
                let dest = self.func.next_value();
                // We'll use the TupleCreate instruction which the backend will handle.
                self.add_instruction(InstructionKind::TupleCreate(dest, elts));
                self.func.set_type(dest, Type::Tuple(elt_types));
                Ok(dest)
            }
            ast::Expr::Call(s) => {
                let (func_name, method_obj) = match *s.func {
                    ast::Expr::Name(n) => (n.id.to_string(), None),
                    ast::Expr::Attribute(attr) => {
                        if let ast::Expr::Name(n) = &*attr.value {
                            if n.id.as_str() == "math" {
                                (format!("math.{}", attr.attr), None)
                            } else {
                                let obj = self.visit_expr(*attr.value)?;
                                let mut curr_ty = self.func.get_type(obj);

                                // Unwrap Mut/Ref/Owned to get the base struct type
                                while let Type::Mut(inner) | Type::Ref(inner) | Type::Owned(inner) =
                                    curr_ty
                                {
                                    curr_ty = *inner;
                                }

                                if let Type::Struct(struct_name) = curr_ty {
                                    (format!("{}_{}", struct_name, attr.attr), Some(obj))
                                } else {
                                    return Err(format!(
                                        "Cannot call method '{}' on non-struct type {:?}",
                                        attr.attr,
                                        self.func.get_type(obj)
                                    ));
                                }
                            }
                        } else {
                            let obj = self.visit_expr(*attr.value)?;
                            let mut curr_ty = self.func.get_type(obj);

                            // Unwrap Mut/Ref/Owned to get the base struct type
                            while let Type::Mut(inner) | Type::Ref(inner) | Type::Owned(inner) =
                                curr_ty
                            {
                                curr_ty = *inner;
                            }

                            if let Type::Struct(struct_name) = curr_ty {
                                (format!("{}_{}", struct_name, attr.attr), Some(obj))
                            } else {
                                return Err(format!(
                                    "Cannot call method '{}' on non-struct type {:?}",
                                    attr.attr,
                                    self.func.get_type(obj)
                                ));
                            }
                        }
                    }
                    _ => return Err("Complex calls not supported yet".to_string()),
                };

                if func_name == "Ref" {
                    if s.args.len() != 1 {
                        return Err("Ref expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::Borrow(dest, arg));
                    let ty = self.func.get_type(arg);
                    self.func.set_type(dest, Type::Ref(Box::new(ty)));
                    return Ok(dest);
                } else if func_name == "Mut" {
                    if s.args.len() != 1 {
                        return Err("Mut expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::MutBorrow(dest, arg));
                    let ty = self.func.get_type(arg);
                    self.func.set_type(dest, Type::Mut(Box::new(ty)));
                    return Ok(dest);
                } else if func_name == "f64" || func_name == "float" {
                    if s.args.len() != 1 {
                        return Err("f64() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let arg_ty = self.func.get_type(arg);
                    if matches!(arg_ty, Type::F64) {
                        return Ok(arg);
                    }
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::IToF(dest, arg, Type::F64));
                    self.func.set_type(dest, Type::F64);
                    return Ok(dest);
                } else if func_name == "i64" || func_name == "int" {
                    if s.args.len() != 1 {
                        return Err("i64() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let arg_ty = self.func.get_type(arg);
                    if matches!(arg_ty, Type::I64) {
                        return Ok(arg);
                    }
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::FToI(dest, arg, Type::I64));
                    self.func.set_type(dest, Type::I64);
                    return Ok(dest);
                } else if func_name == "len" {
                    if s.args.len() != 1 {
                        return Err("len() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let ty = self.func.get_type(arg);
                    if let Type::Buffer(_) = ty {
                        let dest = self.func.next_value();
                        self.add_instruction(InstructionKind::BufferLen(dest, arg));
                        self.func.set_type(dest, Type::I64);
                        return Ok(dest);
                    }
                } else if func_name == "math.sqrt" {
                    if s.args.len() != 1 {
                        return Err("sqrt() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::FSqrt(dest, arg));
                    self.func.set_type(dest, Type::F64);
                    return Ok(dest);
                } else if func_name == "math.sin" {
                    if s.args.len() != 1 {
                        return Err("sin() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::FSin(dest, arg));
                    self.func.set_type(dest, Type::F64);
                    return Ok(dest);
                } else if func_name == "math.cos" {
                    if s.args.len() != 1 {
                        return Err("cos() expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::FCos(dest, arg));
                    self.func.set_type(dest, Type::F64);
                    return Ok(dest);
                } else if func_name == "math.pow" {
                    if s.args.len() != 2 {
                        return Err("pow() expects 2 arguments".to_string());
                    }
                    let b = self.visit_expr(s.args[0].clone())?;
                    let e = self.visit_expr(s.args[1].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::FPow(dest, b, e));
                    self.func.set_type(dest, Type::F64);
                    return Ok(dest);
                }

                let mut args = Vec::new();
                if let Some(obj) = method_obj {
                    args.push(obj);
                }
                for arg in s.args {
                    args.push(self.visit_expr(arg)?);
                }

                let dest = self.func.next_value();
                self.add_instruction(InstructionKind::Call(dest, func_name.clone(), args));

                // Look up return type in registry
                let mut ret_ty = Type::Unknown;
                if let Ok(registry) = crate::bridge::registry::GLOBAL_REGISTRY.lock() {
                    if let Some(sig) = registry.get(&func_name) {
                        ret_ty = sig.return_type.clone();
                    }
                }
                self.func.set_type(dest, ret_ty);
                Ok(dest)
            }
            _ => Err(format!("Expression type {:?} not yet supported", expr)),
        }
    }

    pub fn build_binop(
        &mut self,
        op: ast::Operator,
        lhs: Value,
        rhs: Value,
        dest: Value,
    ) -> Result<InstructionKind, String> {
        let l_ty = self.func.get_type(lhs);
        let r_ty = self.func.get_type(rhs);
        let is_float =
            matches!(l_ty, Type::F32 | Type::F64) || matches!(r_ty, Type::F32 | Type::F64);

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
            ast::Operator::Div => InstructionKind::FDiv(dest, lhs, rhs),
            ast::Operator::FloorDiv => InstructionKind::SDiv(dest, lhs, rhs),
            ast::Operator::Mod => InstructionKind::SRem(dest, lhs, rhs),
            ast::Operator::BitAnd => InstructionKind::And(dest, lhs, rhs),
            ast::Operator::BitOr => InstructionKind::Or(dest, lhs, rhs),
            ast::Operator::BitXor => InstructionKind::Xor(dest, lhs, rhs),
            ast::Operator::LShift => InstructionKind::Shl(dest, lhs, rhs),
            ast::Operator::RShift => InstructionKind::AShr(dest, lhs, rhs),
            _ => return Err(format!("Operator {:?} not yet supported", op)),
        };

        if is_float {
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

    pub fn handle_assignment_target(
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

    pub fn resolve_attribute_path(&self, expr: ast::Expr) -> Result<(String, usize, Type), String> {
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
                let (root_name, base_offset, parent_ty) =
                    self.resolve_attribute_path(*attr.value)?;

                let mut curr_ty = &parent_ty;
                while let Type::Mut(inner) | Type::Ref(inner) | Type::Owned(inner) = curr_ty {
                    curr_ty = inner;
                }

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
            _ => Err("Invalid attribute path: must start with a variable name".to_string()),
        }
    }
}
