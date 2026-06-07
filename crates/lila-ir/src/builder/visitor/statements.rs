use crate::builder::metadata::{extract_refinement, parse_type};
use crate::builder::CFGBuilder;
use crate::ir::{BlockId, InstructionKind, Type, Value};
use rustpython_ast as ast;
use rustpython_ast::Ranged;
use std::collections::HashMap;

impl CFGBuilder {
    pub fn visit_function_def(&mut self, s: ast::StmtFunctionDef) -> Result<(), String> {
        self.update_location(s.range().start().to_usize());
        self.func.arg_count = s.args.args.len();

        if let Some(returns) = &s.returns {
            self.func.return_type = parse_type(returns, &self.type_aliases)?;
            self.func.ret_refinement =
                extract_refinement(returns, &self.type_aliases, &self.func.struct_layouts);
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
                        self.add_instruction(InstructionKind::ConstFloat(zero, 0.0));
                    }
                    _ => {
                        self.add_instruction(InstructionKind::ConstInt(zero, 0));
                    }
                }
                Some(zero)
            } else {
                None
            };
            self.add_instruction(InstructionKind::Return(ret_val));
        }

        Ok(())
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
                    let v = self.visit_expr(*expr)?;
                    Some(self.auto_load(v))
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
                self.add_instruction(InstructionKind::Nop)
                    .add_constraint(format!("(= {} true)", cond));
                for stmt in s.body {
                    self.visit_stmt(stmt)?;
                }
                if !self.is_terminated(self.current_block) {
                    self.add_instruction(InstructionKind::Jump(merge_block));
                    self.link_blocks(self.current_block, merge_block);
                }

                self.start_block(false_block);
                self.add_instruction(InstructionKind::Nop)
                    .add_constraint(format!("(= {} false)", cond));
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
                self.add_instruction(InstructionKind::Nop)
                    .add_constraint(format!("(= {} true)", cond));
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
                self.add_instruction(InstructionKind::Nop)
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
                let increment_block = self.create_block();
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
                    self.add_instruction(InstructionKind::Jump(increment_block));
                    self.link_blocks(self.current_block, increment_block);
                }

                self.seal_block(increment_block)?;
                self.start_block(increment_block);
                let next_idx = self.func.next_value();
                let updated_idx = self.read_variable(idx_name.clone(), increment_block)?;
                self.add_instruction(InstructionKind::Add(next_idx, updated_idx, step_val));
                self.write_variable(idx_name, increment_block, next_idx);
                self.add_instruction(InstructionKind::Jump(header_block));
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
                    Type::Struct(name) if self.func.enum_layouts.contains_key(&name) => {
                        // Fix the type misclassification
                        self.func.set_type(subject_val, Type::Enum(name.clone()));
                        name
                    }
                    _ => {
                        return Err(format!(
                            "match statement currently only supported for Enums, found {:?}",
                            subject_ty
                        ))
                    }
                };

                let exit_block = self.create_block();
                let start_block = self.current_block;

                let tag_val = self.func.next_value();
                self.add_instruction(InstructionKind::EnumGetTag(tag_val, subject_val));
                self.func.set_type(tag_val, Type::U8);

                let mut cases_map = HashMap::new();
                let mut default_block = self.create_block();
                let mut catch_all_found = false;

                let variants = self
                    .func
                    .enum_layouts
                    .get(&enum_name)
                    .cloned()
                    .ok_or_else(|| format!("Unknown enum layout for '{}'", enum_name))?;

                for case in s.cases {
                    if catch_all_found {
                        break;
                    }

                    match case.pattern {
                        ast::Pattern::MatchAs(p) => {
                            if p.pattern.is_some() {
                                return Err("Nested patterns not yet supported".to_string());
                            }
                            catch_all_found = true;
                            let body_block = self.create_block();
                            default_block = body_block;

                            self.start_block(body_block);
                            self.link_blocks(start_block, body_block);
                            if let Some(name) = p.name {
                                self.write_variable(name.to_string(), body_block, subject_val);
                            }
                            for stmt in case.body {
                                self.visit_stmt(stmt)?;
                            }
                            if !self.is_terminated(self.current_block) {
                                self.add_instruction(InstructionKind::Jump(exit_block));
                                self.link_blocks(self.current_block, exit_block);
                            }
                            self.seal_block(body_block)?;
                        }
                        _ => {
                            let (variant_name, pattern_args) = match case.pattern {
                                ast::Pattern::MatchClass(p) => {
                                    let attr = match *p.cls {
                                        ast::Expr::Attribute(a) => a,
                                        _ => {
                                            return Err("Expected Enum.Variant pattern".to_string())
                                        }
                                    };
                                    (attr.attr.to_string(), p.patterns)
                                }
                                ast::Pattern::MatchValue(p) => {
                                    let attr = match *p.value {
                                        ast::Expr::Attribute(a) => a,
                                        _ => {
                                            return Err("Expected Enum.Variant pattern".to_string())
                                        }
                                    };
                                    (attr.attr.to_string(), Vec::new())
                                }
                                _ => {
                                    return Err(format!(
                                        "Unsupported pattern type: {:?}",
                                        case.pattern
                                    ))
                                }
                            };

                            let tag_idx =
                                variants.iter().position(|(name, _)| *name == variant_name);
                            if let Some(tag_idx) = tag_idx {
                                if cases_map.contains_key(&tag_idx) {
                                    // Already handled by previous case
                                    continue;
                                }

                                let body_block = self.create_block();
                                cases_map.insert(tag_idx, body_block);

                                self.start_block(body_block);
                                self.link_blocks(start_block, body_block);

                                // Handle destructuring recursively
                                if !pattern_args.is_empty() {
                                    let payload = self.func.next_value();
                                    self.add_instruction(InstructionKind::EnumExtract(
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
                                            body_block,
                                        )?;
                                    } else {
                                        // Multi-element payload (tuple)
                                        if let Type::Tuple(ref types) = variant_ty {
                                            if types.len() != pattern_args.len() {
                                                return Err(format!(
                                                    "Variant '{}' has {} fields, but pattern has {}",
                                                    variant_name,
                                                    types.len(),
                                                    pattern_args.len()
                                                ));
                                            }
                                            for (i, p_arg) in pattern_args.iter().enumerate() {
                                                let elt = self.func.next_value();
                                                self.add_instruction(
                                                    InstructionKind::TupleExtract(elt, payload, i),
                                                );
                                                self.func.set_type(elt, types[i].clone());
                                                self.handle_nested_pattern(p_arg, elt, body_block)?;
                                            }
                                        } else {
                                            return Err(format!(
                                                "Variant '{}' has a non-tuple payload, but pattern has {} fields",
                                                variant_name,
                                                pattern_args.len()
                                            ));
                                        }
                                    }
                                }

                                for stmt in case.body {
                                    self.visit_stmt(stmt)?;
                                }
                                if !self.is_terminated(self.current_block) {
                                    self.add_instruction(InstructionKind::Jump(exit_block));
                                    self.link_blocks(self.current_block, exit_block);
                                }
                                self.seal_block(body_block)?;
                            } else {
                                return Err(format!(
                                    "Unknown variant '{}' for enum '{}'",
                                    variant_name, enum_name
                                ));
                            }
                        }
                    }
                }

                if !catch_all_found {
                    self.start_block(default_block);
                    self.link_blocks(start_block, default_block);
                    self.add_instruction(InstructionKind::Jump(exit_block));
                    self.link_blocks(default_block, exit_block);
                    self.seal_block(default_block)?;
                }

                // Finalize the Match instruction in the start block
                self.start_block(start_block);
                self.add_instruction(InstructionKind::Match(
                    tag_val,
                    cases_map,
                    default_block,
                    !catch_all_found,
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

    fn handle_nested_pattern(
        &mut self,
        pattern: &ast::Pattern,
        val: Value,
        block: BlockId,
    ) -> Result<(), String> {
        match pattern {
            ast::Pattern::MatchAs(p) => {
                if p.pattern.is_some() {
                    return Err("Nested patterns in MatchAs not yet supported".to_string());
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
                    Type::Struct(ref name) => {
                        let fields = self
                            .func
                            .struct_layouts
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("Unknown struct layout for '{}'", name))?;
                        if p.patterns.len() > fields.len() {
                            return Err(format!(
                                "Struct '{}' has {} fields, but pattern has {}",
                                name,
                                fields.len(),
                                p.patterns.len()
                            ));
                        }
                        let mut current_offset = 0;
                        for (i, sub_pattern) in p.patterns.iter().enumerate() {
                            let field_ty = &fields[i].1;
                            let align = field_ty.align(&self.func.struct_layouts);
                            current_offset = (current_offset + align - 1) & !(align - 1);

                            let field_val = self.func.next_value();
                            if field_ty.is_composite() {
                                self.add_instruction(InstructionKind::StructOffset(
                                    field_val,
                                    val,
                                    current_offset,
                                ));
                            } else {
                                self.add_instruction(InstructionKind::StructLoad(
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
                            _ => return Err("Expected Enum.Variant pattern".to_string()),
                        };
                        let variants = self
                            .func
                            .enum_layouts
                            .get(name)
                            .cloned() // Clone to avoid borrowing self.func
                            .ok_or_else(|| format!("Unknown enum layout for '{}'", name))?;
                        let tag_idx = variants
                            .iter()
                            .position(|(n, _)| *n == variant_name)
                            .ok_or_else(|| {
                                format!("Unknown variant '{}' for enum '{}'", variant_name, name)
                            })?;

                        let payload = self.func.next_value();
                        self.add_instruction(InstructionKind::EnumExtract(payload, val, tag_idx));
                        let variant_ty = variants[tag_idx].1.clone();
                        self.func.set_type(payload, variant_ty.clone()); // Clone to avoid move

                        if p.patterns.len() == 1 {
                            self.handle_nested_pattern(&p.patterns[0], payload, block)?;
                        } else if !p.patterns.is_empty() {
                            if let Type::Tuple(ref types) = variant_ty {
                                if types.len() != p.patterns.len() {
                                    return Err(format!(
                                        "Variant '{}' has {} fields, but pattern has {}",
                                        variant_name,
                                        types.len(),
                                        p.patterns.len()
                                    ));
                                }
                                for (i, sub_p) in p.patterns.iter().enumerate() {
                                    let elt = self.func.next_value();
                                    self.add_instruction(InstructionKind::TupleExtract(
                                        elt, payload, i,
                                    ));
                                    self.func.set_type(elt, types[i].clone());
                                    self.handle_nested_pattern(sub_p, elt, block)?;
                                }
                            } else {
                                return Err(format!(
                                    "Variant '{}' has a non-tuple payload, but pattern has {} fields",
                                    variant_name,
                                    p.patterns.len()
                                ));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(format!("Cannot destructure type {:?}", ty)),
                }
            }
            ast::Pattern::MatchValue(_) => {
                Err("Literal matching not yet supported in nested patterns".to_string())
            }
            _ => Err(format!("Unsupported nested pattern type: {:?}", pattern)),
        }
    }
}
