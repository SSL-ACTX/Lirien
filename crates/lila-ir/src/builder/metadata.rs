use super::super::ir::Type;
use rustpython_ast as ast;
use rustpython_parser::Parse;
use std::collections::HashMap;

pub fn parse_type(expr: &ast::Expr, aliases: &HashMap<String, String>) -> Result<Type, String> {
    match expr {
        ast::Expr::Name(n) => {
            if let Some(alias) = aliases.get(n.id.as_str()) {
                let alias_expr = ast::Expr::parse(alias, "<alias>")
                    .map_err(|e| format!("Error parsing alias '{}': {}", n.id, e))?;
                return parse_type(&alias_expr, aliases);
            }

            match n.id.as_str() {
                "i8" => Ok(Type::I8),
                "u8" => Ok(Type::U8),
                "i16" => Ok(Type::I16),
                "u16" => Ok(Type::U16),
                "i32" => Ok(Type::I32),
                "u32" => Ok(Type::U32),
                "i64" | "int" => Ok(Type::I64),
                "u64" => Ok(Type::U64),
                "f32" => Ok(Type::F32),
                "f64" | "float" => Ok(Type::F64),
                "bool" => Ok(Type::Bool),
                "None" | "none" => Ok(Type::Unknown),
                _ => Ok(Type::Struct(n.id.to_string())),
            }
        }
        ast::Expr::Attribute(a) => match a.attr.as_str() {
            "i8" => Ok(Type::I8),
            "u8" => Ok(Type::U8),
            "i16" => Ok(Type::I16),
            "u16" => Ok(Type::U16),
            "i32" => Ok(Type::I32),
            "u32" => Ok(Type::U32),
            "i64" | "int" => Ok(Type::I64),
            "u64" => Ok(Type::U64),
            "f32" => Ok(Type::F32),
            "f64" | "float" => Ok(Type::F64),
            "bool" => Ok(Type::Bool),
            "None" | "none" => Ok(Type::Unknown),
            "Buffer" | "buffer" => Ok(Type::Buffer(Box::new(Type::I64))),
            _ => Ok(Type::Struct(a.attr.to_string())),
        },
        ast::Expr::Subscript(s) => {
            let base_str = match &*s.value {
                ast::Expr::Name(n) => n.id.as_str(),
                ast::Expr::Attribute(a) => a.attr.as_str(),
                _ => return Err(format!("Invalid type annotation base: {:?}", s.value)),
            };

            let base = if let Some(alias) = aliases.get(base_str) {
                alias.as_str()
            } else {
                base_str
            };

            match base.to_lowercase().as_str() {
                "array" => {
                    let inner = parse_type(&s.slice, aliases)?;
                    Ok(Type::Array(Box::new(inner), None))
                }
                "sizedarray" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let inner = parse_type(&t.elts[0], aliases)?;
                            if let ast::Expr::Constant(c) = &t.elts[1] {
                                if let ast::Constant::Int(i) = &c.value {
                                    let size = i
                                        .to_string()
                                        .parse::<usize>()
                                        .map_err(|_| "Invalid array size")?;
                                    return Ok(Type::Array(Box::new(inner), Some(size)));
                                }
                            }
                        }
                    }
                    let inner = parse_type(&s.slice, aliases)?;
                    Ok(Type::Array(Box::new(inner), None))
                }
                "buffer" => {
                    let inner = parse_type(&s.slice, aliases)?;
                    Ok(Type::Buffer(Box::new(inner)))
                }
                "box" => {
                    let inner = parse_type(&s.slice, aliases)?;
                    Ok(Type::Pointer(Box::new(inner)))
                }
                "fnpointer" | "callable" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let mut arg_types = Vec::new();
                            if let ast::Expr::List(args) = &t.elts[0] {
                                for arg in &args.elts {
                                    arg_types.push(parse_type(arg, aliases)?);
                                }
                            } else {
                                arg_types.push(parse_type(&t.elts[0], aliases)?);
                            }
                            let ret_type = parse_type(&t.elts[1], aliases)?;
                            return Ok(Type::FnPointer(arg_types, Box::new(ret_type)));
                        }
                    }
                    Err("FnPointer expects [[arg_types], ret_type]".to_string())
                }
                "closure" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let mut arg_types = Vec::new();
                            if let ast::Expr::List(args) = &t.elts[0] {
                                for arg in &args.elts {
                                    arg_types.push(parse_type(arg, aliases)?);
                                }
                            } else {
                                arg_types.push(parse_type(&t.elts[0], aliases)?);
                            }
                            let ret_type = parse_type(&t.elts[1], aliases)?;
                            return Ok(Type::Closure(
                                "".to_string(),
                                arg_types,
                                Box::new(ret_type),
                            ));
                        }
                    }
                    Err("Closure expects [[arg_types], ret_type]".to_string())
                }
                "tuple" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        let mut types = Vec::new();
                        for elt in &t.elts {
                            types.push(parse_type(elt, aliases)?);
                        }
                        Ok(Type::Tuple(types))
                    } else {
                        // Handle Tuple[i64]
                        let inner = parse_type(&s.slice, aliases)?;
                        Ok(Type::Tuple(vec![inner]))
                    }
                }
                "refined" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if !t.elts.is_empty() {
                            return parse_type(&t.elts[0], aliases);
                        }
                    }
                    let inner = parse_type(&s.slice, aliases)?;
                    Ok(inner)
                }
                "annotated" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if !t.elts.is_empty() {
                            return parse_type(&t.elts[0], aliases);
                        }
                    }
                    parse_type(&s.slice, aliases)
                }
                _ => Err(format!(
                    "Unsupported generic type: '{}' (lowered: '{}')",
                    base,
                    base.to_lowercase()
                )),
            }
        }
        ast::Expr::Constant(c) => match &c.value {
            ast::Constant::None => Ok(Type::Unknown),
            _ => Err("Unsupported constant in type annotation".to_string()),
        },
        ast::Expr::Tuple(t) => {
            let mut types = Vec::new();
            for elt in &t.elts {
                types.push(parse_type(elt, aliases)?);
            }
            Ok(Type::Tuple(types))
        }
        _ => Err(format!("Invalid type annotation: {:?}", expr)),
    }
}

pub fn extract_refinement(
    expr: &ast::Expr,
    aliases: &HashMap<String, String>,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
) -> Option<String> {
    match expr {
        ast::Expr::Subscript(s) => {
            if let ast::Expr::Name(n) = &*s.value {
                if n.id.as_str() == "Refined" {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let base_ty = parse_type(&t.elts[0], aliases).ok()?;
                            return expr_to_string_internal(
                                &t.elts[1],
                                None,
                                &base_ty,
                                struct_layouts,
                            )
                            .ok();
                        }
                    }
                }
            }
        }
        ast::Expr::Name(n) => {
            if let Some(alias) = aliases.get(n.id.as_str()) {
                if let Ok(alias_expr) = ast::Expr::parse(alias, "<alias>") {
                    return extract_refinement(&alias_expr, aliases, struct_layouts);
                }
            }
        }
        _ => {}
    }
    None
}

fn expr_to_string_internal(
    expr: &ast::Expr,
    arg_name: Option<&str>,
    base_ty: &Type,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
) -> Result<String, String> {
    match expr {
        ast::Expr::Lambda(l) => {
            let name = if !l.args.args.is_empty() {
                Some(l.args.args[0].def.arg.as_str())
            } else {
                None
            };
            expr_to_string_internal(&l.body, name, base_ty, struct_layouts)
        }
        ast::Expr::Compare(c) => {
            let left = expr_to_string_internal(&c.left, arg_name, base_ty, struct_layouts)?;
            let op = match c.ops[0] {
                ast::CmpOp::Eq => "=",
                ast::CmpOp::NotEq => "!=",
                ast::CmpOp::Lt => "<",
                ast::CmpOp::LtE => "<=",
                ast::CmpOp::Gt => ">",
                ast::CmpOp::GtE => ">=",
                _ => return Err("Unsupported operator in refinement".to_string()),
            };
            let right =
                expr_to_string_internal(&c.comparators[0], arg_name, base_ty, struct_layouts)?;
            Ok(format!("({} {} {})", op, left, right))
        }
        ast::Expr::BinOp(b) => {
            let left = expr_to_string_internal(&b.left, arg_name, base_ty, struct_layouts)?;
            let op = match b.op {
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
                _ => return Err(format!("Unsupported binop in refinement: {:?}", b.op)),
            };
            let right = expr_to_string_internal(&b.right, arg_name, base_ty, struct_layouts)?;
            Ok(format!("({} {} {})", op, left, right))
        }
        ast::Expr::BoolOp(b) => {
            let op = match b.op {
                ast::BoolOp::And => "and",
                ast::BoolOp::Or => "or",
            };
            let mut parts = Vec::new();
            for val in &b.values {
                parts.push(expr_to_string_internal(
                    val,
                    arg_name,
                    base_ty,
                    struct_layouts,
                )?);
            }
            Ok(format!("({} {})", op, parts.join(" ")))
        }
        ast::Expr::UnaryOp(u) => {
            let operand = expr_to_string_internal(&u.operand, arg_name, base_ty, struct_layouts)?;
            let op = match u.op {
                ast::UnaryOp::Not => "not",
                ast::UnaryOp::Invert => "~",
                ast::UnaryOp::USub => "-",
                _ => return Err(format!("Unsupported unary op in refinement: {:?}", u.op)),
            };
            Ok(format!("({} {})", op, operand))
        }
        ast::Expr::IfExp(i) => {
            let test = expr_to_string_internal(&i.test, arg_name, base_ty, struct_layouts)?;
            let body = expr_to_string_internal(&i.body, arg_name, base_ty, struct_layouts)?;
            let orelse = expr_to_string_internal(&i.orelse, arg_name, base_ty, struct_layouts)?;
            Ok(format!("(ite {} {} {})", test, body, orelse))
        }
        ast::Expr::Attribute(s) => {
            if let ast::Expr::Name(n) = &*s.value {
                if let Some(name) = arg_name {
                    if n.id.as_str() == name {
                        // Resolve field offset
                        if let Type::Struct(struct_name) = base_ty {
                            let fields = struct_layouts.get(struct_name).ok_or_else(|| {
                                format!("Struct '{}' not found in layouts", struct_name)
                            })?;
                            let mut offset = 0;
                            for (f_name, f_ty) in fields {
                                let align = f_ty.align(struct_layouts);
                                offset = (offset + align - 1) & !(align - 1);
                                if f_name == s.attr.as_str() {
                                    return Ok(format!("(select {{v}} {})", offset));
                                }
                                offset += f_ty.size(struct_layouts);
                            }
                            return Err(format!(
                                "Field '{}' not found in struct '{}'",
                                s.attr, struct_name
                            ));
                        }
                    }
                }
            }
            Err("Unsupported attribute access in refinement".to_string())
        }
        ast::Expr::Name(n) => {
            if let Some(name) = arg_name {
                if n.id.as_str() == name {
                    return Ok("{v}".to_string());
                }
            }
            Ok(n.id.to_string())
        }
        ast::Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => Ok(i.to_string()),
            ast::Constant::Float(f) => Ok(f.to_string()),
            ast::Constant::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            _ => Err("Unsupported constant in refinement".to_string()),
        },
        _ => Err(format!(
            "Expression {:?} not supported in refinements",
            expr
        )),
    }
}
