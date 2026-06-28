use super::super::ir::Type;
use super::error::{BuilderError, BuilderResult};
use rustpython_ast as ast;
use rustpython_parser::Parse;
use std::collections::{HashMap, HashSet};

pub fn parse_type(
    expr: &ast::Expr,
    aliases: &HashMap<String, String>,
    named_tuple_names: &HashSet<String>,
    typed_dict_names: &HashSet<String>,
    enum_names: &HashSet<String>,
) -> BuilderResult<Type> {
    match expr {
        ast::Expr::Name(n) => {
            if let Some(alias) = aliases.get(n.id.as_str()) {
                let alias_expr = ast::Expr::parse(alias, "<alias>").map_err(|e| {
                    BuilderError::General(format!("Error parsing alias '{}': {}", n.id, e), None)
                })?;
                return parse_type(
                    &alias_expr,
                    aliases,
                    named_tuple_names,
                    typed_dict_names,
                    enum_names,
                );
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
                "f32x4" => Ok(Type::F32X4),
                "i32x4" => Ok(Type::I32X4),
                "f64x2" => Ok(Type::F64X2),
                "i64x2" => Ok(Type::I64X2),
                "i8x16" => Ok(Type::I8X16),
                "u8x16" => Ok(Type::U8X16),
                "i16x8" => Ok(Type::I16X8),
                "u16x8" => Ok(Type::U16X8),
                "bool" => Ok(Type::Bool),
                "str" => Ok(Type::Str),
                "None" | "none" => Ok(Type::Unknown),
                _ => {
                    if named_tuple_names.contains(n.id.as_str()) {
                        Ok(Type::NamedTuple(n.id.to_string()))
                    } else if typed_dict_names.contains(n.id.as_str()) {
                        Ok(Type::TypedDict(n.id.to_string()))
                    } else if enum_names.contains(n.id.as_str()) {
                        Ok(Type::Enum(n.id.to_string()))
                    } else {
                        Ok(Type::Struct(n.id.to_string()))
                    }
                }
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
            "f32x4" => Ok(Type::F32X4),
            "i32x4" => Ok(Type::I32X4),
            "f64x2" => Ok(Type::F64X2),
            "i64x2" => Ok(Type::I64X2),
            "i8x16" => Ok(Type::I8X16),
            "u8x16" => Ok(Type::U8X16),
            "i16x8" => Ok(Type::I16X8),
            "u16x8" => Ok(Type::U16X8),
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            "None" | "none" => Ok(Type::Unknown),
            "Buffer" | "buffer" => Ok(Type::Buffer(Box::new(Type::I64))),
            _ => {
                if named_tuple_names.contains(a.attr.as_str()) {
                    Ok(Type::NamedTuple(a.attr.to_string()))
                } else if typed_dict_names.contains(a.attr.as_str()) {
                    Ok(Type::TypedDict(a.attr.to_string()))
                } else if enum_names.contains(a.attr.as_str()) {
                    Ok(Type::Enum(a.attr.to_string()))
                } else {
                    Ok(Type::Struct(a.attr.to_string()))
                }
            }
        },
        ast::Expr::Subscript(s) => {
            let base_str = match &*s.value {
                ast::Expr::Name(n) => n.id.as_str(),
                ast::Expr::Attribute(a) => a.attr.as_str(),
                _ => {
                    return Err(BuilderError::General(
                        format!("Invalid type annotation base: {:?}", s.value),
                        None,
                    ))
                }
            };

            let base = if let Some(alias) = aliases.get(base_str) {
                alias.as_str()
            } else {
                base_str
            };

            match base.to_lowercase().as_str() {
                "array" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(Type::Array(Box::new(inner), None))
                }
                "sizedarray" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let inner = parse_type(
                                &t.elts[0],
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            )?;
                            if let ast::Expr::Constant(c) = &t.elts[1] {
                                if let ast::Constant::Int(i) = &c.value {
                                    let size = i.to_string().parse::<usize>().map_err(|_| {
                                        BuilderError::General(
                                            "Invalid array size".to_string(),
                                            None,
                                        )
                                    })?;
                                    return Ok(Type::Array(Box::new(inner), Some(size)));
                                }
                            }
                        }
                    }
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(Type::Array(Box::new(inner), None))
                }
                "buffer" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(Type::Buffer(Box::new(inner)))
                }
                "list" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(Type::List(Box::new(inner)))
                }
                "tensor" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.is_empty() {
                            return Err(BuilderError::General(
                                "Tensor requires at least a base type".to_string(),
                                None,
                            ));
                        }
                        let inner = parse_type(
                            &t.elts[0],
                            aliases,
                            named_tuple_names,
                            typed_dict_names,
                            enum_names,
                        )?;
                        let mut dims = Vec::new();
                        for dim_expr in t.elts.iter().skip(1) {
                            if let ast::Expr::Constant(c) = dim_expr {
                                match &c.value {
                                    ast::Constant::Str(s) => dims.push(s.to_string()),
                                    ast::Constant::Int(i) => dims.push(i.to_string()),
                                    ast::Constant::Ellipsis => dims.push("...".to_string()),
                                    _ => return Err(BuilderError::General("Tensor dimensions must be strings (e.g., \"M\"), integers, or Ellipsis (...)".to_string(), None)),
                                }
                            } else {
                                return Err(BuilderError::General(
                                    "Tensor dimensions must be string constants or Ellipsis"
                                        .to_string(),
                                    None,
                                ));
                            }
                        }
                        Ok(Type::Tensor(Box::new(inner), dims))
                    } else {
                        let inner = parse_type(
                            &s.slice,
                            aliases,
                            named_tuple_names,
                            typed_dict_names,
                            enum_names,
                        )?;
                        Ok(Type::Tensor(Box::new(inner), Vec::new()))
                    }
                }
                "literal" => {
                    if let ast::Expr::Constant(c) = &*s.slice {
                        if let ast::Constant::Int(i) = &c.value {
                            let val = i.to_string().parse::<i64>().map_err(|_| {
                                BuilderError::General("Invalid literal value".to_string(), None)
                            })?;
                            return Ok(Type::Literal(Box::new(Type::I64), val));
                        }
                    }
                    Err(BuilderError::General(
                        "Literal expects an integer constant".to_string(),
                        None,
                    ))
                }
                "box" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(Type::Pointer(Box::new(inner)))
                }
                "nullable" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    match inner {
                        Type::Pointer(p) => Ok(Type::NullablePointer(p)),
                        Type::NullablePointer(p) => Ok(Type::NullablePointer(p)),
                        other => {
                            if other.is_pointer_like() {
                                Ok(Type::NullablePointer(Box::new(other)))
                            } else {
                                Ok(Type::Optional(Box::new(other)))
                            }
                        }
                    }
                }
                "fnpointer" | "callable" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() >= 2 {
                            let mut arg_types = Vec::new();
                            if let ast::Expr::List(args) = &t.elts[0] {
                                for arg in &args.elts {
                                    arg_types.push(parse_type(
                                        arg,
                                        aliases,
                                        named_tuple_names,
                                        typed_dict_names,
                                        enum_names,
                                    )?);
                                }
                            } else {
                                arg_types.push(parse_type(
                                    &t.elts[0],
                                    aliases,
                                    named_tuple_names,
                                    typed_dict_names,
                                    enum_names,
                                )?);
                            }
                            let ret_type = parse_type(
                                &t.elts[1],
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            )?;
                            let target = if t.elts.len() > 2 {
                                if let ast::Expr::Constant(c) = &t.elts[2] {
                                    match &c.value {
                                        ast::Constant::Str(s) => Some(s.to_string()),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            return Ok(Type::FnPointer(arg_types, Box::new(ret_type), target));
                        }
                    }
                    Err(BuilderError::General(
                        "FnPointer expects [[arg_types], ret_type, optional_target]".to_string(),
                        None,
                    ))
                }
                "closure" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() >= 2 {
                            let mut arg_types = Vec::new();
                            if let ast::Expr::List(args) = &t.elts[0] {
                                for arg in &args.elts {
                                    arg_types.push(parse_type(
                                        arg,
                                        aliases,
                                        named_tuple_names,
                                        typed_dict_names,
                                        enum_names,
                                    )?);
                                }
                            } else {
                                arg_types.push(parse_type(
                                    &t.elts[0],
                                    aliases,
                                    named_tuple_names,
                                    typed_dict_names,
                                    enum_names,
                                )?);
                            }
                            let ret_type = parse_type(
                                &t.elts[1],
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            )?;
                            let target = if t.elts.len() > 2 {
                                if let ast::Expr::Constant(c) = &t.elts[2] {
                                    match &c.value {
                                        ast::Constant::Str(s) => Some(s.to_string()),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            return Ok(Type::Closure(
                                "".to_string(),
                                arg_types,
                                Box::new(ret_type),
                                target,
                            ));
                        }
                    }
                    Err(BuilderError::General(
                        "Closure expects [[arg_types], ret_type, optional_target]".to_string(),
                        None,
                    ))
                }
                "tuple" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        let mut types = Vec::new();
                        for elt in &t.elts {
                            types.push(parse_type(
                                elt,
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            )?);
                        }
                        Ok(Type::Tuple(types))
                    } else {
                        // Handle Tuple[i64]
                        let inner = parse_type(
                            &s.slice,
                            aliases,
                            named_tuple_names,
                            typed_dict_names,
                            enum_names,
                        )?;
                        Ok(Type::Tuple(vec![inner]))
                    }
                }
                "refined" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if !t.elts.is_empty() {
                            return parse_type(
                                &t.elts[0],
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            );
                        }
                    }
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    Ok(inner)
                }
                "annotated" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if !t.elts.is_empty() {
                            return parse_type(
                                &t.elts[0],
                                aliases,
                                named_tuple_names,
                                typed_dict_names,
                                enum_names,
                            );
                        }
                    }
                    parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )
                }
                "optional" => {
                    let inner = parse_type(
                        &s.slice,
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    match inner {
                        Type::Pointer(p) => Ok(Type::NullablePointer(p)),
                        Type::NullablePointer(p) => Ok(Type::NullablePointer(p)),
                        other => {
                            if other.is_pointer_like() {
                                Ok(Type::NullablePointer(Box::new(other)))
                            } else {
                                Ok(Type::Optional(Box::new(other)))
                            }
                        }
                    }
                }
                "union" => {
                    if let ast::Expr::Tuple(t) = &*s.slice {
                        if t.elts.len() == 2 {
                            let mut inner_ty = None;
                            let mut has_none = false;
                            for elt in &t.elts {
                                if let ast::Expr::Constant(c) = elt {
                                    if matches!(c.value, ast::Constant::None) {
                                        has_none = true;
                                        continue;
                                    }
                                }
                                inner_ty = Some(parse_type(
                                    elt,
                                    aliases,
                                    named_tuple_names,
                                    typed_dict_names,
                                    enum_names,
                                )?);
                            }
                            if has_none {
                                if let Some(inner) = inner_ty {
                                    return match inner {
                                        Type::Pointer(p) => Ok(Type::NullablePointer(p)),
                                        Type::NullablePointer(p) => Ok(Type::NullablePointer(p)),
                                        other => {
                                            if other.is_pointer_like() {
                                                Ok(Type::NullablePointer(Box::new(other)))
                                            } else {
                                                Ok(Type::Optional(Box::new(other)))
                                            }
                                        }
                                    };
                                }
                            }
                        }
                    }
                    Err(BuilderError::General(
                        "Only Union[T, None] is supported".to_string(),
                        None,
                    ))
                }
                _ => Err(BuilderError::General(
                    format!(
                        "Unsupported generic type: '{}' (lowered: '{}')",
                        base,
                        base.to_lowercase()
                    ),
                    None,
                )),
            }
        }
        ast::Expr::Constant(c) => {
            match &c.value {
                ast::Constant::Str(_) => Ok(Type::Unknown), // String literals in types are usually metadata
                ast::Constant::Int(_) => Ok(Type::I64),
                ast::Constant::Float(_) => Ok(Type::F64),
                ast::Constant::Bool(_) => Ok(Type::Bool),
                ast::Constant::None => Ok(Type::Unknown),
                ast::Constant::Ellipsis => Ok(Type::Unknown),
                _ => Err(BuilderError::General(
                    format!("Unsupported constant in type annotation: {:?}", c.value),
                    None,
                )),
            }
        }
        ast::Expr::Tuple(t) => {
            let mut types = Vec::new();
            for elt in &t.elts {
                types.push(parse_type(
                    elt,
                    aliases,
                    named_tuple_names,
                    typed_dict_names,
                    enum_names,
                )?);
            }
            Ok(Type::Tuple(types))
        }
        ast::Expr::BinOp(b) if matches!(b.op, ast::Operator::BitOr) => {
            let lhs = parse_type(
                &b.left,
                aliases,
                named_tuple_names,
                typed_dict_names,
                enum_names,
            )?;
            let rhs = parse_type(
                &b.right,
                aliases,
                named_tuple_names,
                typed_dict_names,
                enum_names,
            )?;

            let (inner, has_none) = match (&lhs, &rhs) {
                (Type::Unknown, other) | (other, Type::Unknown) => (other.clone(), true),
                _ => (Type::Unknown, false),
            };

            if has_none && inner != Type::Unknown {
                match inner {
                    Type::Pointer(p) => Ok(Type::NullablePointer(p)),
                    Type::NullablePointer(p) => Ok(Type::NullablePointer(p)),
                    other => {
                        if other.is_pointer_like() {
                            Ok(Type::NullablePointer(Box::new(other)))
                        } else {
                            Ok(Type::Optional(Box::new(other)))
                        }
                    }
                }
            } else {
                Err(BuilderError::General(
                    "Only T | None is supported for unions".to_string(),
                    None,
                ))
            }
        }
        _ => Err(BuilderError::General(
            format!("Invalid type annotation: {:?}", expr),
            None,
        )),
    }
}

pub fn extract_refinement(
    expr: &ast::Expr,
    aliases: &HashMap<String, String>,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
    named_tuple_names: &HashSet<String>,
    typed_dict_names: &HashSet<String>,
    enum_names: &HashSet<String>,
) -> BuilderResult<Option<String>> {
    match expr {
        ast::Expr::Subscript(s) => {
            let base_name = match &*s.value {
                ast::Expr::Name(n) => n.id.as_str(),
                _ => "",
            };
            if base_name == "Refined" || base_name == "Annotated" {
                if let ast::Expr::Tuple(t) = &*s.slice {
                    let base_ty = parse_type(
                        &t.elts[0],
                        aliases,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    )?;
                    for elt in t.elts.iter().skip(1) {
                        match elt {
                            ast::Expr::Constant(c) => match &c.value {
                                ast::Constant::Str(s) => {
                                    let refinement_expr = ast::Expr::parse(s, "<refinement>")
                                        .map_err(|e| BuilderError::General(e.to_string(), None))?;
                                    return Ok(Some(
                                        expr_to_string_internal(
                                            &refinement_expr,
                                            None,
                                            &base_ty,
                                            struct_layouts,
                                        )?
                                        .0,
                                    ));
                                }
                                ast::Constant::Ellipsis => return Ok(Some("...".to_string())),
                                _ => {}
                            },
                            _ => {
                                // Try to extract from non-constant expression (e.g. lambda)
                                if let Ok((s, _)) =
                                    expr_to_string_internal(elt, None, &base_ty, struct_layouts)
                                {
                                    return Ok(Some(s));
                                }
                            }
                        }
                    }
                }
            }
        }
        ast::Expr::Name(n) => {
            if let Some(alias) = aliases.get(n.id.as_str()) {
                if let Ok(alias_expr) = ast::Expr::parse(alias, "<alias>") {
                    return extract_refinement(
                        &alias_expr,
                        aliases,
                        struct_layouts,
                        named_tuple_names,
                        typed_dict_names,
                        enum_names,
                    );
                }
            }
        }
        _ => {}
    }
    Ok(None)
}

pub fn expr_to_string(
    expr: &ast::Expr,
    arg_name: Option<&str>,
    base_ty: &Type,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
) -> BuilderResult<String> {
    Ok(expr_to_string_internal(expr, arg_name, base_ty, struct_layouts)?.0)
}

fn expr_to_string_internal(
    expr: &ast::Expr,
    arg_name: Option<&str>,
    base_ty: &Type,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
) -> BuilderResult<(String, Type)> {
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
            let (mut left, _) =
                expr_to_string_internal(&c.left, arg_name, base_ty, struct_layouts)?;
            let mut parts = Vec::new();

            for i in 0..c.ops.len() {
                let op = match c.ops[i] {
                    ast::CmpOp::Eq => "=",
                    ast::CmpOp::NotEq => "!=",
                    ast::CmpOp::Lt => "<",
                    ast::CmpOp::LtE => "<=",
                    ast::CmpOp::Gt => ">",
                    ast::CmpOp::GtE => ">=",
                    _ => {
                        return Err(BuilderError::General(
                            "Unsupported operator in refinement".to_string(),
                            None,
                        ))
                    }
                };
                let (right, _) =
                    expr_to_string_internal(&c.comparators[i], arg_name, base_ty, struct_layouts)?;
                parts.push(format!("({} {} {})", op, left, right));
                left = right;
            }

            if parts.len() == 1 {
                Ok((parts[0].clone(), Type::Bool))
            } else {
                Ok((format!("(and {})", parts.join(" ")), Type::Bool))
            }
        }
        ast::Expr::BinOp(b) => {
            let (left, l_ty) = expr_to_string_internal(&b.left, arg_name, base_ty, struct_layouts)?;
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
                _ => {
                    return Err(BuilderError::General(
                        format!("Unsupported binop in refinement: {:?}", b.op),
                        None,
                    ))
                }
            };
            let (right, _) = expr_to_string_internal(&b.right, arg_name, base_ty, struct_layouts)?;
            Ok((format!("({} {} {})", op, left, right), l_ty))
        }
        ast::Expr::BoolOp(b) => {
            let op = match b.op {
                ast::BoolOp::And => "and",
                ast::BoolOp::Or => "or",
            };
            let mut parts = Vec::new();
            for val in &b.values {
                let (s, _) = expr_to_string_internal(val, arg_name, base_ty, struct_layouts)?;
                parts.push(s);
            }
            Ok((format!("({} {})", op, parts.join(" ")), Type::Bool))
        }
        ast::Expr::UnaryOp(u) => {
            let (operand, ty) =
                expr_to_string_internal(&u.operand, arg_name, base_ty, struct_layouts)?;
            let op = match u.op {
                ast::UnaryOp::Not => "not",
                ast::UnaryOp::Invert => "~",
                ast::UnaryOp::USub => "-",
                _ => {
                    return Err(BuilderError::General(
                        format!("Unsupported unary op in refinement: {:?}", u.op),
                        None,
                    ))
                }
            };
            Ok((
                format!("({} {})", op, operand),
                if op == "not" { Type::Bool } else { ty },
            ))
        }
        ast::Expr::IfExp(i) => {
            let (test, _) = expr_to_string_internal(&i.test, arg_name, base_ty, struct_layouts)?;
            let (body, ty) = expr_to_string_internal(&i.body, arg_name, base_ty, struct_layouts)?;
            let (orelse, _) =
                expr_to_string_internal(&i.orelse, arg_name, base_ty, struct_layouts)?;
            Ok((format!("(ite {} {} {})", test, body, orelse), ty))
        }
        ast::Expr::Call(c) => {
            let (func_name, method_obj) = match &*c.func {
                ast::Expr::Name(n) => (n.id.to_string(), None),
                ast::Expr::Attribute(a) => {
                    let (obj, _) =
                        expr_to_string_internal(&a.value, arg_name, base_ty, struct_layouts)?;
                    (a.attr.to_string(), Some(obj))
                }
                _ => {
                    return Err(BuilderError::General(
                        "Only named functions supported in refinements".to_string(),
                        None,
                    ))
                }
            };

            let mut args = Vec::new();
            if let Some(obj) = method_obj {
                args.push(obj);
            }
            for arg in &c.args {
                let (s, _) = expr_to_string_internal(arg, arg_name, base_ty, struct_layouts)?;
                args.push(s);
            }
            Ok((format!("({} {})", func_name, args.join(" ")), Type::I64)) // Defaulting to I64 for calls
        }
        ast::Expr::Attribute(s) => {
            let (base_expr, current_ty) =
                expr_to_string_internal(&s.value, arg_name, base_ty, struct_layouts)?;

            if let Type::Struct(struct_name) = current_ty {
                let fields = struct_layouts.get(&struct_name).ok_or_else(|| {
                    BuilderError::General(
                        format!("Struct '{}' not found in layouts", struct_name),
                        None,
                    )
                })?;
                let mut offset = 0;
                for (f_name, f_ty) in fields {
                    let align = f_ty.align(struct_layouts);
                    offset = (offset + align - 1) & !(align - 1);
                    if f_name == s.attr.as_str() {
                        return Ok((format!("(select {} {})", base_expr, offset), f_ty.clone()));
                    }
                    offset += f_ty.size(struct_layouts);
                }
                return Err(BuilderError::General(
                    format!("Field '{}' not found in struct '{}'", s.attr, struct_name),
                    None,
                ));
            }

            // If it's not a direct attribute of a known struct, treat it as a symbolic name
            Ok((format!("({} {})", s.attr, base_expr), Type::Unknown))
        }
        ast::Expr::Subscript(s) => {
            let (base, b_ty) =
                expr_to_string_internal(&s.value, arg_name, base_ty, struct_layouts)?;
            let (slice, _) = expr_to_string_internal(&s.slice, arg_name, base_ty, struct_layouts)?;
            let inner_ty = match b_ty {
                Type::Array(t, _) | Type::Buffer(t) | Type::List(t) | Type::Tensor(t, _) => *t,
                _ => Type::Unknown,
            };
            Ok((format!("(select {} {})", base, slice), inner_ty))
        }
        ast::Expr::Name(n) => {
            if n.id.as_str() == "v" {
                return Ok(("{v}".to_string(), base_ty.clone()));
            }
            if let Some(name) = arg_name {
                if n.id.as_str() == name {
                    return Ok(("{v}".to_string(), base_ty.clone()));
                }
            }
            Ok((n.id.to_string(), Type::Unknown))
        }
        ast::Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => Ok((i.to_string(), Type::I64)),
            ast::Constant::Float(f) => Ok((f.to_string(), Type::F64)),
            ast::Constant::Bool(b) => {
                Ok((if *b { "true" } else { "false" }.to_string(), Type::Bool))
            }
            ast::Constant::Str(s) => Ok((format!("\"{}\"", s), Type::Unknown)),
            _ => Err(BuilderError::General(
                "Unsupported constant in refinement".to_string(),
                None,
            )),
        },
        _ => Err(BuilderError::General(
            format!("Expression {:?} not supported in refinements", expr),
            None,
        )),
    }
}
