use crate::builder::metadata::{extract_refinement, parse_type};
use crate::builder::CFGBuilder;
use crate::ir::{BlockId, InstructionKind, Type, Value};
use rustpython_ast as ast;
use rustpython_ast::Ranged;
use std::collections::HashMap;
use crate::builder::error::BuilderResult;
use crate::{push_inst, builder_error};

impl CFGBuilder {
    pub fn visit_function_def(&mut self, s: ast::StmtFunctionDef) -> BuilderResult<()> {
        self.update_location(s.range().start().to_usize());
        self.func.arg_count = s.args.args.len();

        if let Some(returns) = &s.returns {
            let ret_ty = parse_type(returns, &self.type_aliases, &self.named_tuple_names, &self.typed_dict_names, &self.enum_names)?;
            self.func.return_type = ret_ty;
            self.func.ret_refinement =
                extract_refinement(returns, &self.type_aliases, &self.func.struct_layouts, &self.named_tuple_names, &self.typed_dict_names, &self.enum_names)?;
        }

        for arg in s.args.args {
            let val = self.func.next_value();
            if let Some(annotation) = &arg.def.annotation {
                let ty = parse_type(annotation, &self.type_aliases, &self.named_tuple_names, &self.typed_dict_names, &self.enum_names)?;
                self.func.set_type(val, ty);
                if let Some(refinement) =
                    extract_refinement(annotation, &self.type_aliases, &self.func.struct_layouts, &self.named_tuple_names, &self.typed_dict_names, &self.enum_names)?
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
                        push_inst!(self, InstructionKind::ConstFloat(zero, 0.0));
                    }
                    _ => {
                        push_inst!(self, InstructionKind::ConstInt(zero, 0));
                    }
                }
                Some(zero)
            } else {
                None
            };
            push_inst!(self, InstructionKind::Return(ret_val));
        }

        Ok(())
    }

    pub fn visit_stmt(&mut self, stmt: ast::Stmt) -> BuilderResult<()> {
        self.update_location(stmt.range().start().to_usize());
        match stmt {
            ast::Stmt::Assign(s) => {
                if s.targets.len() != 1 {
                    return Err(builder_error!(UnsupportedStatement, "Only single targets supported"));
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
                push_inst!(self, kind);
                self.handle_assignment_target(&target, dest)?;
                Ok(())
            }
            ast::Stmt::AnnAssign(s) => {
                if let Some(value_expr) = s.value {
                    let mut value = self.visit_expr(*value_expr)?;
                    value = self.auto_load(value);

                    if let Ok(ann_ty) = parse_type(&s.annotation, &self.type_aliases, &self.named_tuple_names, &self.typed_dict_names, &self.enum_names) {
                        let val_ty = self.func.get_type(value);
                        if ann_ty.is_float() && val_ty.is_int() {
                            let converted = self.func.next_value();
                            push_inst!(self, InstructionKind::IToF(
                                converted,
                                value,
                                ann_ty.clone(),
                            ));
                            self.func.set_type(converted, ann_ty);
                            value = converted;
                        }
                    }

                    self.handle_assignment_target(&s.target, value)?;
                }
                Ok(())
            }
            ast::Stmt::Return(s) => {
                let mut val = if let Some(expr) = s.value {
                    let v = self.visit_expr(*expr)?;
                    Some(self.auto_load(v))
                } else {
                    None
                };

                // Auto-cast to return type if necessary
                if let Some(v) = val {
                    let val_ty = self.func.get_type(v);
                    let ret_ty = self.func.return_type.clone();
                    if ret_ty.is_float() && val_ty.is_int() {
                        let converted = self.func.next_value();
                        push_inst!(self, InstructionKind::IToF(converted, v, ret_ty.clone()));
                        self.func.set_type(converted, ret_ty);
                        val = Some(converted);
                    } else if ret_ty.is_int() && val_ty.is_float() {
                        let converted = self.func.next_value();
                        push_inst!(self, InstructionKind::FToI(converted, v, ret_ty.clone()));
                        self.func.set_type(converted, ret_ty);
                        val = Some(converted);
                    }
                }

                push_inst!(self, InstructionKind::Return(val));
                Ok(())
            }
            ast::Stmt::If(s) => {
                let cond = self.visit_expr(*s.test)?;
                let cond = self.auto_load(cond);

                // Constant pruning for If
                if let Some(val) = self.get_constant_int(cond) {
                    if val != 0 {
                        for stmt in s.body {
                            self.visit_stmt(stmt)?;
                        }
                    } else {
                        for stmt in s.orelse {
                            self.visit_stmt(stmt)?;
                        }
                    }
                    return Ok(());
                }

                let prev_block = self.current_block;

                let true_block = self.create_block();
                let false_block = self.create_block();
                let merge_block = self.create_block();

                push_inst!(self, InstructionKind::Branch(cond, true_block, false_block));
                self.link_blocks(prev_block, true_block);
                self.link_blocks(prev_block, false_block);

                self.seal_block(true_block)?;
                self.seal_block(false_block)?;

                self.start_block(true_block);
                push_inst!(self, InstructionKind::Nop())
                    .add_constraint(format!("(= {} true)", cond));
                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(merge_block));
                    self.link_blocks(self.current_block, merge_block);
                }

                self.start_block(false_block);
                push_inst!(self, InstructionKind::Nop())
                    .add_constraint(format!("(= {} false)", cond));
                for stmt in s.orelse {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(merge_block));
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
                push_inst!(self, InstructionKind::Jump(header_block));
                self.link_blocks(prev_block, header_block);

                self.loop_stack.push((header_block, exit_block));

                self.start_block(header_block);
                let cond = self.visit_expr(*s.test)?;
                push_inst!(self, InstructionKind::Branch(cond, body_block, exit_block));
                self.link_blocks(header_block, body_block);
                self.link_blocks(header_block, exit_block);

                self.seal_block(body_block)?;
                self.start_block(body_block);
                push_inst!(self, InstructionKind::Nop())
                    .add_constraint(format!("(= {} true)", cond));
                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(header_block));
                    self.link_blocks(self.current_block, header_block);
                }

                self.loop_stack.pop();
                self.seal_block(header_block)?;
                self.start_block(exit_block);
                push_inst!(self, InstructionKind::Nop())
                    .add_constraint(format!("(= {} false)", cond));
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
                                _ => return Err(builder_error!(UnsupportedStatement, "Unsupported range() signature")),
                            };

                            let start_v = if let Some(v) = start {
                                v
                            } else {
                                let zero = self.func.next_value();
                                push_inst!(self, InstructionKind::ConstInt(zero, 0));
                                zero
                            };
                            let step_v = if let Some(v) = step {
                                v
                            } else {
                                let one = self.func.next_value();
                                push_inst!(self, InstructionKind::ConstInt(one, 1));
                                one
                            };
                            (start_v, end, step_v, false)
                        } else if n.id.as_str() == "enumerate" && call.args.len() == 1 {
                            is_enumerate = true;
                            enum_buf_expr = Some(call.args[0].clone());
                            let buf_val = self.visit_expr(call.args[0].clone())?;
                            let buf_ty = self.func.get_type(buf_val);
                            let zero = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(zero, 0));
                            let one = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(one, 1));
                            let len = self.func.next_value();
                            if let Type::Buffer(_) = buf_ty {
                                push_inst!(self, InstructionKind::BufferLen(len, buf_val));
                            } else if let Type::Array(_, Some(size)) = buf_ty {
                                push_inst!(self, InstructionKind::ConstInt(len, size as i64));
                            } else {
                                return Err(builder_error!(General, "Cannot iterate over unknown size array"));
                            }
                            self.func.set_type(len, Type::I64);
                            (zero, len, one, true)
                        } else {
                            return Err(builder_error!(UnsupportedStatement, "Unsupported function in for loop: {}", n.id));
                        }
                    } else {
                        return Err(builder_error!(UnsupportedStatement, "Only range() or direct iteration supported"));
                    }
                } else {
                    // Potential direct iteration: for x in buf
                    let buf_val = self.visit_expr(iter_expr.clone())?;
                    let buf_ty = self.func.get_type(buf_val);
                    match buf_ty {
                        Type::Buffer(_) | Type::Array(_, _) => {
                            let zero = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(zero, 0));
                            let one = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(one, 1));
                            let len = self.func.next_value();
                            if let Type::Buffer(_) = buf_ty {
                                push_inst!(self, InstructionKind::BufferLen(len, buf_val));
                            } else if let Type::Array(_, Some(size)) = buf_ty {
                                push_inst!(self, InstructionKind::ConstInt(len, size as i64));
                            } else {
                                return Err(builder_error!(General, "Cannot iterate over unknown size array"));
                            }
                            self.func.set_type(len, Type::I64);
                            (zero, len, one, true)
                        }
                        _ => return Err(builder_error!(General, "Cannot iterate over type {:?}", buf_ty)),
                    }
                };

                // UNROLLING LOGIC
                let start_const = self.get_constant_int(start_val);
                let end_const = self.get_constant_int(end_val);
                let step_const = self.get_constant_int(step_val);

                if let (Some(start_c), Some(end_c), Some(step_c)) =
                    (start_const, end_const, step_const)
                {
                    let trip_count = if step_c > 0 {
                        if end_c > start_c {
                            (end_c - start_c + step_c - 1) / step_c
                        } else {
                            0
                        }
                    } else if step_c < 0 {
                        if start_c > end_c {
                            (start_c - end_c + (-step_c) - 1) / (-step_c)
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                    if step_c != 0 && (0..=128).contains(&trip_count) {
                        // UNROLL SAFETY: Check for total unrolled complexity
                        let body_stmt_count = s.body.len();
                        if trip_count as usize * body_stmt_count > 1024 {
                            // Too much code bloat, fall back to regular loop
                        } else {
                            // Unroll!
                            let idx_name = if is_direct_iter {
                                format!("_lirien_idx_{}", self.func.value_count)
                            } else if let ast::Expr::Name(n) = target.clone() {
                                n.id.to_string()
                            } else {
                                return Err(builder_error!(UnsupportedStatement, "Unsupported loop target"));
                            };

                            // UNROLL!
                            // We generate a dedicated sequence of blocks for each iteration.
                            // This ensures 'break' and 'continue' work correctly via the loop_stack.
                            let final_exit_block = self.create_block();
                            let mut current_idx_const = start_c;

                            for i in 0..trip_count {
                                let iteration_body_block = self.create_block();
                                let next_iteration_block = if i == trip_count - 1 {
                                    final_exit_block
                                } else {
                                    self.create_block()
                                };

                                // Connect previous block to this iteration's body
                                push_inst!(self, InstructionKind::Jump(iteration_body_block));
                                self.link_blocks(self.current_block, iteration_body_block);
                                self.seal_block(iteration_body_block)?;
                                self.start_block(iteration_body_block);

                                // Set up loop stack for this iteration:
                                // continue -> next_iteration_block (start of next iteration or final exit)
                                // break -> final_exit_block
                                self.loop_stack
                                    .push((next_iteration_block, final_exit_block));

                                let curr_idx = self.func.next_value();
                                push_inst!(self, InstructionKind::ConstInt(
                                    curr_idx,
                                    current_idx_const,
                                ));
                                self.func.set_type(curr_idx, Type::I64);
                                // Inject refinement for Z3 to know the exact loop index
                                push_inst!(self, InstructionKind::Nop()).add_constraint(
                                    format!("(= {} {})", curr_idx, current_idx_const),
                                );

                                self.write_variable(idx_name.clone(), self.current_block, curr_idx);

                                if is_direct_iter {
                                    let buf_expr = if is_enumerate {
                                        enum_buf_expr.clone().ok_or_else(|| builder_error!(General, "Missing enum buffer expression"))?
                                    } else {
                                        iter_expr.clone()
                                    };
                                    let buf_val = self.visit_expr(buf_expr)?;
                                    let buf_ty = self.func.get_type(buf_val);
                                    let element = self.func.next_value();
                                    match buf_ty {
                                        Type::Buffer(inner) => {
                                            push_inst!(self, InstructionKind::BufferLoad(
                                                element, buf_val, curr_idx,
                                            ));
                                            self.func.set_type(element, *inner);
                                        }
                                        Type::Array(inner, _) => {
                                            push_inst!(self, InstructionKind::ArrayLoad(
                                                element, buf_val, curr_idx,
                                            ));
                                            self.func.set_type(element, *inner);
                                        }
                                        _ => unreachable!(),
                                    }

                                    if is_enumerate {
                                        if let ast::Expr::Tuple(t) = target.clone() {
                                            if t.elts.len() != 2 {
                                                return Err(builder_error!(General, "enumerate() requires a tuple of 2 elements"));
                                            }
                                            self.handle_assignment_target(&t.elts[0], curr_idx)?;
                                            self.handle_assignment_target(&t.elts[1], element)?;
                                        }
                                    } else {
                                        self.handle_assignment_target(&target, element)?;
                                    }
                                }

                                for stmt in s.body.clone() {
                                    self.visit_stmt(stmt)?;
                                }

                                // If not terminated (no break/return/continue), jump to next iteration
                                if !self.is_terminated(self.current_block) {
                                    push_inst!(self, InstructionKind::Jump(
                                        next_iteration_block,
                                    ));
                                    self.link_blocks(self.current_block, next_iteration_block);
                                }

                                self.loop_stack.pop();

                                // Prepare for next iteration
                                if i < trip_count - 1 {
                                    self.seal_block(next_iteration_block)?;
                                    self.start_block(next_iteration_block);
                                }

                                current_idx_const += step_c;
                            }

                            self.start_block(final_exit_block);
                            self.seal_block(final_exit_block)?;
                            return Ok(());
                        }
                    }
                }

                let header_block = self.create_block();
                let body_block = self.create_block();
                let increment_block = self.create_block();
                let exit_block = self.create_block();

                // Iterator variable (index)
                let idx_name = if is_direct_iter {
                    format!("_lirien_idx_{}", self.func.value_count)
                } else if let ast::Expr::Name(n) = target.clone() {
                    n.id.to_string()
                } else {
                    return Err(builder_error!(UnsupportedStatement, "Unsupported loop target"));
                };

                self.write_variable(idx_name.clone(), self.current_block, start_val);

                let prev_block = self.current_block;
                push_inst!(self, InstructionKind::Jump(header_block));
                self.link_blocks(prev_block, header_block);

                self.loop_stack.push((increment_block, exit_block));

                self.start_block(header_block);
                let curr_idx = self.read_variable(idx_name.clone(), header_block)?;
                let cond = self.func.next_value();

                // Determine if we should use SLt or SGt based on step if constant
                let mut use_sgt = false;
                if let Some(val) = self.get_constant_int(step_val) {
                    if val < 0 {
                        use_sgt = true;
                    }
                }

                if use_sgt {
                    push_inst!(self, InstructionKind::SGt(cond, curr_idx, end_val));
                } else {
                    push_inst!(self, InstructionKind::SLt(cond, curr_idx, end_val));
                }
                push_inst!(self, InstructionKind::Branch(cond, body_block, exit_block));
                self.link_blocks(header_block, body_block);
                self.link_blocks(header_block, exit_block);

                self.seal_block(body_block)?;
                self.start_block(body_block);

                if is_direct_iter {
                    // For direct iter, load the value into the target variable
                    let buf_expr = if is_enumerate {
                        enum_buf_expr.ok_or_else(|| builder_error!(General, "Missing enum buffer expression"))?
                    } else {
                        iter_expr.clone()
                    };
                    let buf_val = self.visit_expr(buf_expr)?;
                    let buf_ty = self.func.get_type(buf_val);
                    let element = self.func.next_value();
                    match buf_ty {
                        Type::Buffer(inner) => {
                            push_inst!(self, InstructionKind::BufferLoad(
                                element, buf_val, curr_idx,
                            ));
                            self.func.set_type(element, *inner);
                        }
                        Type::Array(inner, _) => {
                            push_inst!(self, InstructionKind::ArrayLoad(
                                element, buf_val, curr_idx,
                            ));
                            self.func.set_type(element, *inner);
                        }
                        _ => unreachable!(),
                    }

                    if is_enumerate {
                        if let ast::Expr::Tuple(t) = target {
                            if t.elts.len() != 2 {
                                return Err(builder_error!(General, "enumerate() requires a tuple of 2 elements"));
                            }
                            self.handle_assignment_target(&t.elts[0], curr_idx)?;
                            self.handle_assignment_target(&t.elts[1], element)?;
                        } else {
                            return Err(builder_error!(UnsupportedStatement, "enumerate() requires a tuple target"));
                        }
                    } else if let ast::Expr::Name(n) = target {
                        self.write_variable(n.id.to_string(), self.current_block, element);
                    } else {
                        return Err(builder_error!(UnsupportedStatement, "Unsupported loop target"));
                    }
                }

                for stmt_in_body in s.body {
                    self.visit_stmt(stmt_in_body)?;
                }

                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(increment_block));
                    self.link_blocks(self.current_block, increment_block);
                }

                self.seal_block(increment_block)?;
                self.start_block(increment_block);
                let next_idx = self.func.next_value();
                let updated_idx = self.read_variable(idx_name.clone(), increment_block)?;
                push_inst!(self, InstructionKind::Add(next_idx, updated_idx, step_val));
                self.write_variable(idx_name, increment_block, next_idx);
                push_inst!(self, InstructionKind::Jump(header_block));
                self.link_blocks(increment_block, header_block);

                self.loop_stack.pop();
                self.seal_block(header_block)?;
                self.start_block(exit_block);
                self.seal_block(exit_block)?;
                Ok(())
            }
            ast::Stmt::Match(s) => {
                let subject_val = self.visit_expr(*s.subject)?;
                let subject_ty = self.func.get_type(subject_val);

                let enum_name = match subject_ty {
                    Type::Enum(name) => name,
                    _ => {
                        return Err(builder_error!(UnsupportedStatement, "match statement currently only supported for Enums, found {:?}", subject_ty));
                    }
                };

                let exit_block = self.create_block();
                let start_block = self.current_block;

                let tag_val = self.func.next_value();
                push_inst!(self, InstructionKind::EnumGetTag(tag_val, subject_val));
                self.func.set_type(tag_val, Type::U8);

                let variants = self
                    .func
                    .enum_layouts
                    .get(&enum_name)
                    .cloned()
                    .ok_or_else(|| builder_error!(General, "Unknown enum layout for '{}'", enum_name))?;

                // Group cases by tag
                let mut tag_to_cases: HashMap<usize, Vec<ast::MatchCase>> = HashMap::new();
                let mut global_default_case: Option<ast::MatchCase> = None;

                for case in s.cases {
                    match case.pattern {
                        ast::Pattern::MatchAs(ref p)
                            if p.pattern.is_none() && case.guard.is_none() =>
                        {
                            // Catch-all pattern without guard
                            if global_default_case.is_none() {
                                global_default_case = Some(case);
                            }
                            // Subsequent catch-alls are unreachable
                            break;
                        }
                        _ => {
                            let variant_res = match case.pattern {
                                ast::Pattern::MatchClass(ref p) => {
                                    let attr = match &*p.cls {
                                        ast::Expr::Attribute(a) => a,
                                        _ => {
                                            return Err(builder_error!(UnsupportedStatement, "Expected Enum.Variant pattern"));
                                        }
                                    };
                                    Some(attr.attr.to_string())
                                }
                                ast::Pattern::MatchValue(ref p) => {
                                    let attr = match &*p.value {
                                        ast::Expr::Attribute(a) => a,
                                        _ => {
                                            return Err(builder_error!(UnsupportedStatement, "Expected Enum.Variant pattern"));
                                        }
                                    };
                                    Some(attr.attr.to_string())
                                }
                                _ => None, // Might be MatchAs with a guard
                            };

                            if let Some(variant_name) = variant_res {
                                let tag_idx =
                                    variants.iter().position(|(name, _)| *name == variant_name);
                                if let Some(tag_idx) = tag_idx {
                                    tag_to_cases.entry(tag_idx).or_default().push(case);
                                } else {
                                    return Err(builder_error!(General, "Unknown variant '{}' for enum '{}'", variant_name, enum_name));
                                }
                            } else {
                                // This is a non-variant pattern (e.g. MatchAs with a name/guard).
                                // Python's `match` is strictly sequential.
                                // Currently, Lirien uses a single jump table for all variant-based cases.
                                // Interleaving generic patterns (like `case x if cond:`) between variants
                                // is not yet supported because it would require multiple sequential jump tables.
                                return Err(builder_error!(UnsupportedStatement, "Lirien currently requires all Enum variant patterns to come before any guarded catch-all patterns."));
                            }
                        }
                    }
                }

                let mut cases_map = HashMap::new();
                let default_block = self.create_block();

                // 1. Handle explicit variant cases
                for (tag_idx, tag_cases) in tag_to_cases {
                    let dispatch_block = self.create_block();
                    cases_map.insert(tag_idx, dispatch_block);

                    self.start_block(dispatch_block);
                    self.link_blocks(start_block, dispatch_block);

                    let mut current_chain_block = dispatch_block;

                    for (i, case) in tag_cases.iter().enumerate() {
                        let next_in_chain = if i < tag_cases.len() - 1 {
                            self.create_block()
                        } else {
                            default_block
                        };

                        let body_block = self.create_block();

                        // 1.1. Pattern destructuring (bindings)
                        let pattern_args = match &case.pattern {
                            ast::Pattern::MatchClass(p) => &p.patterns,
                            ast::Pattern::MatchValue(_) => &Vec::new(),
                            _ => unreachable!(),
                        };

                        if !pattern_args.is_empty() {
                            let payload = self.func.next_value();
                            push_inst!(self, InstructionKind::EnumExtract(
                                payload,
                                subject_val,
                                tag_idx,
                            ));
                            let variant_ty = variants[tag_idx].1.clone();
                            self.func.set_type(payload, variant_ty.clone());

                            if pattern_args.len() == 1 {
                                self.handle_nested_pattern(
                                    &pattern_args[0],
                                    payload,
                                    current_chain_block,
                                )?;
                            } else {
                                if let Type::Tuple(ref types) = variant_ty {
                                    for (j, p_arg) in pattern_args.iter().enumerate() {
                                        let elt = self.func.next_value();
                                        push_inst!(self, InstructionKind::TupleExtract(
                                            elt, payload, j,
                                        ));
                                        self.func.set_type(elt, types[j].clone());
                                        self.handle_nested_pattern(
                                            p_arg,
                                            elt,
                                            current_chain_block,
                                        )?;
                                    }
                                }
                            }
                        }

                        // 1.2. Guard check
                        if let Some(guard_expr) = &case.guard {
                            let cond = self.visit_expr(*guard_expr.clone())?;
                            push_inst!(self, InstructionKind::Branch(
                                cond,
                                body_block,
                                next_in_chain,
                            ));
                            self.link_blocks(current_chain_block, body_block);
                            self.link_blocks(current_chain_block, next_in_chain);
                        } else {
                            push_inst!(self, InstructionKind::Jump(body_block));
                            self.link_blocks(current_chain_block, body_block);
                        }
                        self.seal_block(current_chain_block)?;

                        // 1.3. Body
                        self.start_block(body_block);
                        self.seal_block(body_block)?;
                        for stmt in &case.body {
                            self.visit_stmt(stmt.clone())?;
                        }
                        if !self.is_terminated(self.current_block) {
                            push_inst!(self, InstructionKind::Jump(exit_block));
                            self.link_blocks(self.current_block, exit_block);
                        }
                        self.seal_block(body_block)?;

                        if i < tag_cases.len() - 1 {
                            self.start_block(next_in_chain);
                            current_chain_block = next_in_chain;
                        }
                    }
                }

                // 2. Handle global default (catch-all)
                self.start_block(default_block);
                self.link_blocks(start_block, default_block);
                if let Some(ref case) = global_default_case {
                    if let ast::Pattern::MatchAs(ref p) = case.pattern {
                        if let Some(name) = &p.name {
                            self.write_variable(name.to_string(), default_block, subject_val);
                        }
                    }
                    for stmt in &case.body {
                        self.visit_stmt(stmt.clone())?;
                    }
                    if !self.is_terminated(self.current_block) {
                        push_inst!(self, InstructionKind::Jump(exit_block));
                        self.link_blocks(self.current_block, exit_block);
                    }
                } else {
                    push_inst!(self, InstructionKind::Jump(exit_block));
                    self.link_blocks(default_block, exit_block);
                }
                self.seal_block(default_block)?;

                // Finalize start block
                self.start_block(start_block);
                push_inst!(self, InstructionKind::Match(
                    tag_val,
                    cases_map,
                    default_block,
                    global_default_case.is_none(),
                ));

                self.start_block(exit_block);
                self.seal_block(exit_block)?;
                Ok(())
            }
            ast::Stmt::With(s) => {
                for item in s.items {
                    let val = self.visit_expr(item.context_expr)?;
                    if let Some(vars) = item.optional_vars {
                        self.handle_assignment_target(&vars, val)?;
                    }
                }

                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }

                Ok(())
            }
            ast::Stmt::Break(_) => {
                if let Some((_, exit_block)) = self.loop_stack.last() {
                    let eb = *exit_block;
                    push_inst!(self, InstructionKind::Jump(eb));
                    self.link_blocks(self.current_block, eb);
                    Ok(())
                } else {
                    Err(builder_error!(General, "break outside of loop"))
                }
            }
            ast::Stmt::Continue(_) => {
                if let Some((header_block, _)) = self.loop_stack.last() {
                    let hb = *header_block;
                    push_inst!(self, InstructionKind::Jump(hb));
                    self.link_blocks(self.current_block, hb);
                    Ok(())
                } else {
                    Err(builder_error!(General, "continue outside of loop"))
                }
            }
            ast::Stmt::Expr(s) => {
                self.visit_expr(*s.value)?;
                Ok(())
            }
            ast::Stmt::FunctionDef(s) => self.visit_nested_function_def(s),
            _ => Err(builder_error!(UnsupportedStatement, "Statement type {:?} not yet supported", stmt)),
        }
    }

    pub fn visit_nested_function_def(&mut self, s: ast::StmtFunctionDef) -> BuilderResult<()> {
        use crate::builder::capture_analysis::CaptureVisitor;
        use rustpython_ast::Visitor;

        let next_val = self.func.next_value().0;
        let func_name = format!("{}_{}_{}", self.func.name, s.name, next_val);

        // 1. Capture Analysis
        let mut params = Vec::new();
        for arg in &s.args.args {
            params.push(arg.def.arg.to_string());
        }
        let mut capture_visitor = CaptureVisitor::new(params);
        for stmt in &s.body {
            capture_visitor.visit_stmt(stmt.clone());
        }

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

        // 2. Build Inner Function
        let mut inner_builder = self.new_sub_builder(func_name.clone());

        // Define arguments in inner function
        inner_builder.func.arg_count = 1 + s.args.args.len();
        inner_builder.func.value_count = inner_builder.func.arg_count;
        inner_builder
            .func
            .value_types
            .insert(Value(0), Type::Struct("ClosureEnv".to_string())); // ctx_ptr

        if let Some(returns) = &s.returns {
            inner_builder.func.return_type = parse_type(
                returns,
                &self.type_aliases,
                &self.named_tuple_names,
                &self.typed_dict_names,
                &self.enum_names,
            )?;
        }

        for (i, arg) in s.args.args.iter().enumerate() {
            let arg_ty = if let Some(annotation) = &arg.def.annotation {
                parse_type(
                    annotation,
                    &self.type_aliases,
                    &self.named_tuple_names,
                    &self.typed_dict_names,
                    &self.enum_names,
                )?
            } else {
                Type::Unknown
            };
            inner_builder.func.value_types.insert(Value(i + 1), arg_ty);
            inner_builder.write_variable(
                arg.def.arg.to_string(),
                inner_builder.current_block,
                Value(i + 1),
            );
        }

        // If there are captures, load them from ctx_ptr
        if !captures.is_empty() {
            let mut offset = 8; // Offset 0 is fn_ptr
            for (name, ty) in captures.iter().zip(capture_types.iter()) {
                let align = ty.align(&self.func.struct_layouts);
                offset = (offset + align - 1) & !(align - 1);

                let dest = inner_builder.func.next_value();
                push_inst!(inner_builder, InstructionKind::StructLoad(
                    dest,
                    Value(0),
                    offset,
                ));
                inner_builder.func.set_type(dest, ty.clone());
                inner_builder.write_variable(
                    name.0.clone(),
                    inner_builder.current_block,
                    dest,
                );

                offset += ty.size(&self.func.struct_layouts);
            }
        }

        // Visit body
        for stmt in s.body {
            inner_builder.visit_stmt(stmt)?;
        }

        // Ensure the last block has a return if it's not terminated
        if !inner_builder.is_terminated(inner_builder.current_block) {
            let ret_val = if inner_builder.func.return_type != Type::Unknown {
                let zero = inner_builder.func.next_value();
                match inner_builder.func.return_type {
                    Type::F32 | Type::F64 => {
                        push_inst!(inner_builder, InstructionKind::ConstFloat(zero, 0.0));
                    }
                    _ => {
                        push_inst!(inner_builder, InstructionKind::ConstInt(zero, 0));
                    }
                }
                Some(zero)
            } else {
                None
            };
            push_inst!(inner_builder, InstructionKind::Return(ret_val));
        }

        // Optimization for inner function
        crate::optimization::optimize(&mut inner_builder.func);

        // Store inner function for later compilation
        let inner_func = inner_builder.func;
        self.lambdas.push(inner_func.clone());
        // Collect nested lambdas from the sub-builder
        self.lambdas.extend(inner_builder.lambdas);

        // 3. Create Closure Instruction
        let dest = self.func.next_value();
        let capture_vals: Vec<Value> = captures.iter().map(|(_, v)| *v).collect();
        push_inst!(self, InstructionKind::Lambda(
            dest,
            func_name.clone(),
            capture_vals,
        ));

        let arg_types: Vec<Type> = (1..1 + s.args.args.len())
            .map(|i| inner_func.get_type(Value(i)))
            .collect();
        self.func.set_type(
            dest,
            Type::Closure(func_name.clone(), arg_types, Box::new(inner_func.return_type), Some(s.name.to_string())),
        );

        self.write_variable(s.name.to_string(), self.current_block, dest);

        Ok(())
    }

    fn handle_nested_pattern(
        &mut self,
        pattern: &ast::Pattern,
        val: Value,
        block: BlockId,
    ) -> BuilderResult<()> {
        let ty = self.func.get_type(val);
        if let Type::Pointer(inner) = ty {
            // Automatically dereference pointers for matching
            let deref_val = self.func.next_value();
            push_inst!(self, InstructionKind::PointerLoad(deref_val, val));
            self.func.set_type(deref_val, (*inner).clone());
            return self.handle_nested_pattern(pattern, deref_val, block);
        }

        match pattern {
            ast::Pattern::MatchAs(p) => {
                if p.pattern.is_some() {
                    return Err(builder_error!(UnsupportedStatement, "Nested patterns in MatchAs not yet supported"));
                }
                if let Some(name) = &p.name {
                    self.write_variable(name.to_string(), block, val);
                }
                Ok(())
            }
            ast::Pattern::MatchClass(p) => {
                // Nested struct or enum destructuring
                let ty = self.func.get_type(val);
                match ty {
                    Type::Struct(ref name) | Type::NamedTuple(ref name) => {
                        let fields = self
                            .func
                            .struct_layouts
                            .get(name)
                            .cloned()
                            .ok_or_else(|| builder_error!(General, "Unknown struct layout for '{}'", name))?;
                        if p.patterns.len() > fields.len() {
                            return Err(builder_error!(General, "Struct '{}' has {} fields, but pattern has {}", name, fields.len(), p.patterns.len()));
                        }
                        let mut current_offset = 0;
                        for (i, sub_pattern) in p.patterns.iter().enumerate() {
                            let field_ty = &fields[i].1;
                            let align = field_ty.align(&self.func.struct_layouts);
                            current_offset = (current_offset + align - 1) & !(align - 1);

                            let field_val = self.func.next_value();
                            if field_ty.is_composite() {
                                push_inst!(self, InstructionKind::StructOffset(
                                    field_val,
                                    val,
                                    current_offset,
                                ));
                            } else {
                                push_inst!(self, InstructionKind::StructLoad(
                                    field_val,
                                    val,
                                    current_offset,
                                ));
                            }
                            self.func.set_type(field_val, field_ty.clone());
                            self.handle_nested_pattern(sub_pattern, field_val, block)?;

                            current_offset += field_ty.size(&self.func.struct_layouts);
                        }
                        Ok(())
                    }
                    Type::Enum(ref name) => {
                        let variant_name = match &*p.cls {
                            ast::Expr::Attribute(a) => a.attr.to_string(),
                            _ => return Err(builder_error!(UnsupportedStatement, "Expected Enum.Variant pattern")),
                        };
                        let variants = self
                            .func
                            .enum_layouts
                            .get(name)
                            .cloned() // Clone to avoid borrowing self.func
                            .ok_or_else(|| builder_error!(General, "Unknown enum layout for '{}'", name))?;
                        let tag_idx = variants
                            .iter()
                            .position(|(n, _)| *n == variant_name)
                            .ok_or_else(|| {
                                builder_error!(General, "Unknown variant '{}' for enum '{}'", variant_name, name)
                            })?;

                        let payload = self.func.next_value();
                        push_inst!(self, InstructionKind::EnumExtract(payload, val, tag_idx));
                        let variant_ty = variants[tag_idx].1.clone();
                        self.func.set_type(payload, variant_ty.clone()); // Clone to avoid move

                        if p.patterns.len() == 1 {
                            self.handle_nested_pattern(&p.patterns[0], payload, block)?;
                        } else if !p.patterns.is_empty() {
                            if let Type::Tuple(ref types) = variant_ty {
                                if types.len() != p.patterns.len() {
                                    return Err(builder_error!(General, "Variant '{}' has {} fields, but pattern has {}", variant_name, types.len(), p.patterns.len()));
                                }
                                for (i, sub_p) in p.patterns.iter().enumerate() {
                                    let elt = self.func.next_value();
                                    push_inst!(self, InstructionKind::TupleExtract(
                                        elt, payload, i,
                                    ));
                                    self.func.set_type(elt, types[i].clone());
                                    self.handle_nested_pattern(sub_p, elt, block)?;
                                }
                            } else {
                                return Err(builder_error!(General, "Variant '{}' has a non-tuple payload, but pattern has {} fields", variant_name, p.patterns.len()));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(builder_error!(General, "Cannot destructure type {:?}", ty)),
                }
            }
            ast::Pattern::MatchValue(_) => {
                Err(builder_error!(UnsupportedStatement, "Literal matching not yet supported in nested patterns"))
            }
            _ => Err(builder_error!(UnsupportedStatement, "Unsupported nested pattern type: {:?}", pattern)),
        }
    }
}
