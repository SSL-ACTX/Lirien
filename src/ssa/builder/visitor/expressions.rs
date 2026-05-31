use crate::ssa::builder::CFGBuilder;
use crate::ssa::ir::{InstructionKind, Type, Value};
use rustpython_ast as ast;
use rustpython_ast::Ranged;

impl CFGBuilder {
    pub fn visit_expr(&mut self, expr: ast::Expr) -> Result<Value, String> {
        let expr_offset = expr.range().start().to_usize();
        self.update_location(expr_offset);

        match expr {
            ast::Expr::BinOp(s) => {
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
                let inst = self.add_instruction(kind);
                if !op_str.is_empty() {
                    inst.add_constraint(format!("(= {} ({} {} {}))", dest, op_str, lhs, rhs));
                }
                Ok(dest)
            }
            ast::Expr::BoolOp(s) => {
                if s.values.is_empty() {
                    return Err("Empty BoolOp".to_string());
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
                    last_val = self.auto_load(last_val);
                }

                self.write_variable(result_var.clone(), self.current_block, last_val);
                self.add_instruction(InstructionKind::Jump(merge_block));
                self.link_blocks(self.current_block, merge_block);

                self.seal_block(merge_block)?;
                self.start_block(merge_block);
                self.read_variable(result_var, merge_block)
            }
            ast::Expr::UnaryOp(s) => {
                let mut operand = self.visit_expr(*s.operand)?;
                operand = self.auto_load(operand);
                let dest = self.func.next_value();
                let (kind, op_str) = match s.op {
                    ast::UnaryOp::Not => (InstructionKind::Not(dest, operand), "not"),
                    ast::UnaryOp::Invert => (InstructionKind::Not(dest, operand), "~"),
                    ast::UnaryOp::USub => {
                        let zero = self.func.next_value();
                        self.add_instruction(InstructionKind::ConstInt(zero, 0));
                        (InstructionKind::Sub(dest, zero, operand), "-")
                    }
                    ast::UnaryOp::UAdd => return Ok(operand),
                };
                let inst = self.add_instruction(kind);
                if op_str == "-" {
                    inst.add_constraint(format!("(= {} (- 0 {}))", dest, operand));
                } else if !op_str.is_empty() {
                    inst.add_constraint(format!("(= {} ({} {}))", dest, op_str, operand));
                }
                Ok(dest)
            }
            ast::Expr::Compare(s) => {
                if s.ops.len() != 1 || s.comparators.len() != 1 {
                    return Err("Complex comparisons not supported yet".to_string());
                }
                let mut lhs = self.visit_expr(*s.left)?;
                let mut rhs = self.visit_expr(s.comparators[0].clone())?;
                lhs = self.auto_load(lhs);
                rhs = self.auto_load(rhs);
                self.update_location(expr_offset);
                let dest = self.func.next_value();

                let l_ty = self.func.get_type(lhs);
                let r_ty = self.func.get_type(rhs);
                let is_float =
                    matches!(l_ty, Type::F32 | Type::F64) || matches!(r_ty, Type::F32 | Type::F64);

                let (kind, op_str) = match s.ops[0] {
                    ast::CmpOp::Eq => (InstructionKind::Eq(dest, lhs, rhs), "="),
                    ast::CmpOp::NotEq => (InstructionKind::Ne(dest, lhs, rhs), "!="),
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
                        return Err(format!(
                            "Comparison operator {:?} not yet supported",
                            s.ops[0]
                        ))
                    }
                };

                self.func.set_type(dest, Type::Bool);
                let inst = self.add_instruction(kind);
                inst.add_constraint(format!("(= {} ({} {} {}))", dest, op_str, lhs, rhs));
                Ok(dest)
            }
            ast::Expr::Constant(c) => {
                let val = self.func.next_value();
                match c.value {
                    ast::Constant::Int(i) => {
                        let int_val = i.to_string().parse::<i64>().map_err(|_| "Int too large")?;
                        let inst = self.add_instruction(InstructionKind::ConstInt(val, int_val));
                        inst.add_constraint(format!("(= {} {})", val, int_val));
                        self.func.set_type(val, Type::I64);
                    }
                    ast::Constant::Float(f) => {
                        let inst = self.add_instruction(InstructionKind::ConstFloat(val, f));
                        inst.add_constraint(format!("(= {} {})", val, f));
                        self.func.set_type(val, Type::F64);
                    }
                    ast::Constant::Bool(b) => {
                        let inst = self
                            .add_instruction(InstructionKind::ConstInt(val, if b { 1 } else { 0 }));
                        inst.add_constraint(format!("(= {} {})", val, b));
                        self.func.set_type(val, Type::Bool);
                    }
                    _ => return Err("Unsupported constant type".to_string()),
                }
                Ok(val)
            }
            ast::Expr::Name(n) => self.read_variable(n.id.to_string(), self.current_block),
            ast::Expr::Attribute(s) => {
                let obj = self.visit_expr(*s.value.clone())?;
                let orig_ty = self.func.get_type(obj);
                let mut curr_ty = orig_ty.clone();

                // Handle .val unwrap for Refined types (no-op in IR)
                if s.attr.as_str() == "val" {
                    let mut temp_ty = curr_ty.clone();
                    while let Type::Hand(inner) | Type::Peek(inner) | Type::Held(inner) = temp_ty {
                        temp_ty = *inner;
                    }
                    if !matches!(temp_ty, Type::Struct(_)) {
                        return Ok(obj);
                    }
                }

                let mut is_ptr = false;
                while let Type::Hand(inner) | Type::Peek(inner) | Type::Held(inner) = curr_ty {
                    curr_ty = *inner;
                    is_ptr = true;
                }

                if let Type::Struct(struct_name) = curr_ty {
                    let field_offset = self
                        .get_field_offset(&struct_name, s.attr.as_str())
                        .ok_or_else(|| {
                            format!("Field '{}' not found in struct '{}'", s.attr, struct_name)
                        })?;

                    let fields = self.func.struct_layouts.get(&struct_name).unwrap();
                    let mut field_ty = fields
                        .iter()
                        .find(|(f, _)| f == s.attr.as_str())
                        .unwrap()
                        .1
                        .clone();

                    if matches!(field_ty, Type::Unknown) {
                        return Err(format!(
                            "Field '{}' has unknown type in struct '{}'",
                            s.attr, struct_name
                        ));
                    }

                    let dest = self.func.next_value();
                    self.update_location(expr_offset);

                    if is_ptr {
                        // If parent is a pointer, field access returns a pointer to the field
                        field_ty = match orig_ty {
                            Type::Hand(_) => Type::Hand(Box::new(field_ty)),
                            Type::Peek(_) => Type::Peek(Box::new(field_ty)),
                            Type::Held(_) => Type::Held(Box::new(field_ty)),
                            _ => field_ty, // Should not happen given is_ptr
                        };
                        self.add_instruction(InstructionKind::StructOffset(
                            dest,
                            obj,
                            field_offset,
                        ));
                    } else if field_ty.is_composite() {
                        self.add_instruction(InstructionKind::StructOffset(
                            dest,
                            obj,
                            field_offset,
                        ));
                    } else {
                        self.add_instruction(InstructionKind::StructLoad(dest, obj, field_offset));
                    }

                    self.func.set_type(dest, field_ty);
                    Ok(dest)
                } else {
                    Err(format!(
                        "Cannot resolve attribute '{}' on non-struct type {:?}",
                        s.attr,
                        self.func.get_type(obj)
                    ))
                }
            }
            ast::Expr::Subscript(s) => {
                let arr = self.visit_expr(*s.value)?;
                let mut idx = self.visit_expr(*s.slice)?;
                idx = self.auto_load(idx);
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
                                self.add_instruction(InstructionKind::TupleExtract(dest, arr, i));
                                self.func.set_type(dest, elt_ty);
                            } else {
                                return Err(format!("Tuple index out of bounds: {}", i));
                            }
                        } else {
                            return Err("Tuple index must be a constant".to_string());
                        }
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
                self.add_instruction(InstructionKind::TupleCreate(dest, elts));
                self.func.set_type(dest, Type::Tuple(elt_types));
                Ok(dest)
            }
            ast::Expr::Call(s) => {
                let expr_offset = s.range.start().to_usize();
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
                                let mut curr_ty = self.func.get_type(obj);

                                // Unwrap Hand/Peek/Held to get the base struct type
                                while let Type::Hand(_inner)
                                | Type::Peek(_inner)
                                | Type::Held(_inner) = curr_ty
                                {
                                    curr_ty = (*_inner).clone();
                                }

                                if let Type::Struct(struct_name) = curr_ty {
                                    (format!("{}_{}", struct_name, attr.attr), Some(obj), false)
                                } else if let Type::Enum(enum_name) = curr_ty {
                                    (format!("{}_{}", enum_name, attr.attr), Some(obj), false)
                                } else {
                                    return Err(format!(
                                        "Cannot call method '{}' on non-struct/enum type {:?}",
                                        attr.attr,
                                        self.func.get_type(obj)
                                    ));
                                }
                            }
                        } else {
                            let obj = self.visit_expr((*attr.value).clone())?;
                            let mut curr_ty = self.func.get_type(obj);

                            // Unwrap Hand/Peek/Held to get the base struct type
                            while let Type::Hand(inner) | Type::Peek(inner) | Type::Held(inner) =
                                curr_ty
                            {
                                curr_ty = *inner;
                            }

                            if let Type::Struct(struct_name) = curr_ty {
                                (format!("{}_{}", struct_name, attr.attr), Some(obj), false)
                            } else if let Type::Enum(enum_name) = curr_ty {
                                (format!("{}_{}", enum_name, attr.attr), Some(obj), false)
                            } else {
                                return Err(format!(
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

                    let (arg_types, ret_ty) = match fn_ty {
                        Type::Closure(_, params, ret) | Type::FnPointer(params, ret) => {
                            (params, *ret)
                        }
                        _ => (Vec::new(), Type::Unknown),
                    };

                    let mut args = Vec::new();
                    for (i, arg) in s.args.into_iter().enumerate() {
                        let mut v = self.visit_expr(arg)?;
                        if i < arg_types.len() {
                            let expected_ty = &arg_types[i];
                            if expected_ty.is_int()
                                || expected_ty.is_float()
                                || *expected_ty == Type::Bool
                            {
                                v = self.auto_load(v);
                            }
                        } else {
                            let ty = self.func.get_type(v);
                            if !matches!(ty, Type::Held(_)) {
                                v = self.auto_load(v);
                            }
                        }
                        args.push(v);
                    }

                    let dest = self.func.next_value();
                    self.update_location(expr_offset);
                    self.add_instruction(InstructionKind::IndirectCall(dest, fn_val, args));
                    self.func.set_type(dest, ret_ty);
                    return Ok(dest);
                }

                // Check for Enum Creation
                if let Some(enum_name) = func_name.split('_').next() {
                    if self.func.enum_layouts.contains_key(enum_name) && method_obj.is_none() {
                        let variant_name =
                            func_name.split('_').skip(1).collect::<Vec<_>>().join("_");
                        let variants = self.func.enum_layouts.get(enum_name).unwrap();
                        let tag_idx = variants
                            .iter()
                            .position(|(name, _)| name == &variant_name)
                            .ok_or_else(|| {
                                format!(
                                    "Unknown variant '{}' for enum '{}'",
                                    variant_name, enum_name
                                )
                            })?;

                        let payload = if s.args.is_empty() {
                            None
                        } else if s.args.len() == 1 {
                            Some(self.visit_expr(s.args[0].clone())?)
                        } else {
                            return Err(
                                "Enum variant constructor takes at most 1 argument".to_string()
                            );
                        };

                        let dest = self.func.next_value();
                        self.add_instruction(InstructionKind::EnumCreate(
                            dest,
                            enum_name.to_string(),
                            tag_idx,
                            payload,
                        ));
                        self.func.set_type(dest, Type::Enum(enum_name.to_string()));
                        return Ok(dest);
                    }
                }

                // Check for Enum Methods (is_Variant, as_Variant)
                if let Some(obj) = method_obj {
                    if let Type::Enum(enum_name) = self.func.get_type(obj) {
                        let method = func_name.strip_prefix(&format!("{}_", enum_name)).unwrap();
                        if method.starts_with("is_") {
                            let variant_name = method.strip_prefix("is_").unwrap();
                            let variants = self.func.enum_layouts.get(&enum_name).unwrap();
                            let tag_idx = variants
                                .iter()
                                .position(|(name, _)| name == variant_name)
                                .ok_or_else(|| {
                                    format!(
                                        "Unknown variant '{}' for enum '{}'",
                                        variant_name, enum_name
                                    )
                                })?;

                            let dest = self.func.next_value();
                            self.add_instruction(InstructionKind::EnumIsVariant(
                                dest, obj, tag_idx,
                            ));
                            self.func.set_type(dest, Type::Bool);
                            return Ok(dest);
                        } else if method.starts_with("as_") {
                            let variant_name = method.strip_prefix("as_").unwrap();
                            let variants = self.func.enum_layouts.get(&enum_name).unwrap();
                            let tag_idx = variants
                                .iter()
                                .position(|(name, _)| name == variant_name)
                                .ok_or_else(|| {
                                    format!(
                                        "Unknown variant '{}' for enum '{}'",
                                        variant_name, enum_name
                                    )
                                })?;

                            let payload_ty = variants[tag_idx].1.clone();
                            let dest = self.func.next_value();
                            self.add_instruction(InstructionKind::EnumExtract(dest, obj, tag_idx));
                            self.func.set_type(dest, payload_ty);
                            return Ok(dest);
                        }
                    }
                }

                if self.func.struct_layouts.contains_key(&func_name) {
                    let mut struct_args = Vec::new();
                    for arg in s.args.clone() {
                        struct_args.push(self.visit_expr(arg)?);
                    }
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::StructCreate(
                        dest,
                        func_name.clone(),
                        struct_args,
                    ));
                    self.func.set_type(dest, Type::Struct(func_name.clone()));
                    return Ok(dest);
                }

                if func_name == "Peek" {
                    if s.args.len() != 1 {
                        return Err("Peek expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::Peek(dest, arg));
                    let ty = self.func.get_type(arg);
                    self.func.set_type(dest, Type::Peek(Box::new(ty)));
                    return Ok(dest);
                } else if func_name == "Hand" {
                    if s.args.len() != 1 {
                        return Err("Hand expects 1 argument".to_string());
                    }
                    let arg = self.visit_expr(s.args[0].clone())?;
                    let dest = self.func.next_value();
                    self.add_instruction(InstructionKind::Hand(dest, arg));
                    let ty = self.func.get_type(arg);
                    self.func.set_type(dest, Type::Hand(Box::new(ty)));
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

                // Look up return type and arg types in registry or self
                let mut ret_ty = Type::Unknown;
                let mut arg_types = Vec::new();
                if func_name == self.func.name {
                    ret_ty = self.func.return_type.clone();
                    for i in 0..self.func.arg_count {
                        arg_types.push(self.func.get_type(Value(i)));
                    }
                } else if let Ok(registry) = crate::bridge::registry::GLOBAL_REGISTRY.lock() {
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
                        if expected_ty.is_int()
                            || expected_ty.is_float()
                            || *expected_ty == Type::Bool
                        {
                            v = self.auto_load(v);
                        }
                    } else {
                        // If signature unknown, only auto-load if it's not Held (to preserve moves)
                        let ty = self.func.get_type(v);
                        if !matches!(ty, Type::Held(_)) {
                            v = self.auto_load(v);
                        }
                    }
                    args.push(v);
                    arg_idx += 1;
                }

                let dest = self.func.next_value();
                self.update_location(expr_offset);
                self.add_instruction(InstructionKind::Call(dest, func_name.clone(), args));
                self.func.set_type(dest, ret_ty);
                Ok(dest)
            }
            ast::Expr::Lambda(s) => {
                use crate::ssa::builder::capture_analysis::CaptureVisitor;
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
                lambda_builder.func.arg_count = 1 + s.args.args.len();
                lambda_builder.func.value_count = lambda_builder.func.arg_count;
                lambda_builder
                    .func
                    .value_types
                    .insert(Value(0), Type::Peek(Box::new(Type::Unknown))); // ctx_ptr

                for (i, arg) in s.args.args.iter().enumerate() {
                    let arg_ty = if let Some(ann) = &arg.def.annotation {
                        crate::ssa::builder::metadata::parse_type(ann, &self.type_aliases)?
                    } else {
                        Type::Unknown
                    };
                    lambda_builder.func.value_types.insert(Value(i + 1), arg_ty);
                    lambda_builder.write_variable(
                        arg.def.arg.to_string(),
                        lambda_builder.current_block,
                        Value(i + 1),
                    );
                }

                // If there are captures, load them from ctx_ptr
                if !captures.is_empty() {
                    let mut offset = 8; // Offset 0 is fn_ptr
                    for (name, ty) in captures.iter().zip(capture_types.iter()) {
                        let align = ty.align(&self.func.struct_layouts);
                        offset = (offset + align - 1) & !(align - 1);

                        let dest = lambda_builder.func.next_value();
                        lambda_builder.add_instruction(InstructionKind::StructLoad(
                            dest,
                            Value(0),
                            offset,
                        ));
                        lambda_builder.func.set_type(dest, ty.clone());
                        lambda_builder.write_variable(
                            name.0.clone(),
                            lambda_builder.current_block,
                            dest,
                        );

                        offset += ty.size(&self.func.struct_layouts);
                    }
                }

                // Visit body
                let ret_val = lambda_builder.visit_expr(*s.body)?;
                lambda_builder.add_instruction(InstructionKind::Return(Some(ret_val)));
                lambda_builder.func.return_type = lambda_builder.func.get_type(ret_val);

                // Optimization for lambda
                crate::ssa::optimization::optimize(&mut lambda_builder.func);

                // Store lambda for later compilation
                let lambda_func = lambda_builder.func;
                self.lambdas.push(lambda_func.clone());
                // Collect nested lambdas from the sub-builder
                self.lambdas.extend(lambda_builder.lambdas);

                // 3. Create Closure Instruction
                let dest = self.func.next_value();
                let capture_vals: Vec<Value> = captures.iter().map(|(_, v)| *v).collect();
                self.add_instruction(InstructionKind::Lambda(
                    dest,
                    lambda_name.clone(),
                    capture_vals,
                ));

                let arg_types: Vec<Type> = (1..1 + s.args.args.len())
                    .map(|i| lambda_func.get_type(Value(i)))
                    .collect();
                self.func.set_type(
                    dest,
                    Type::Closure(lambda_name, arg_types, Box::new(lambda_func.return_type)),
                );

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
}
