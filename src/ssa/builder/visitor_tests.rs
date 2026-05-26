#[cfg(test)]
mod tests {
    use crate::ssa::builder::CFGBuilder;
    use crate::ssa::ir::{InstructionKind, Type};
    use rustpython_ast as ast;
    use std::collections::HashMap;

    #[test]
    fn test_struct_tuple_field_ir() {
        let mut builder = CFGBuilder::new(
            "test".to_string(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let struct_name = "S".to_string();
        builder.func.struct_layouts.insert(
            struct_name.clone(),
            vec![("t".to_string(), Type::Tuple(vec![Type::I64, Type::I64]))],
        );

        let obj = builder.func.next_value();
        builder
            .func
            .set_type(obj, Type::Struct(struct_name.clone()));
        builder.write_variable("s".to_string(), builder.current_block, obj);

        // n.t
        let attr = ast::Expr::Attribute(ast::ExprAttribute {
            value: Box::new(ast::Expr::Name(ast::ExprName {
                id: "s".into(),
                ctx: ast::ExprContext::Load,
                range: Default::default(),
            })),
            attr: "t".into(),
            ctx: ast::ExprContext::Load,
            range: Default::default(),
        });

        let res = builder.visit_expr(attr).unwrap();

        // Should generate StructOffset because Tuple is composite
        let mut found_offset = false;
        for block in &builder.func.blocks {
            for inst in &block.instructions {
                if let InstructionKind::StructOffset(d, o, offset) = inst.kind {
                    if d == res && o == obj && offset == 0 {
                        found_offset = true;
                    }
                }
            }
        }
        assert!(
            found_offset,
            "Should have generated StructOffset for Tuple field"
        );
    }

    #[test]
    fn test_struct_tuple_string_layout() {
        let mut layouts = HashMap::new();
        layouts.insert(
            "Nested".to_string(),
            vec![
                ("t".to_string(), "Tuple[i64, i64]".to_string()),
                ("val".to_string(), "i64".to_string()),
            ],
        );

        let mut builder =
            CFGBuilder::new("test".to_string(), layouts, HashMap::new(), HashMap::new());

        let obj = builder.func.next_value();
        builder
            .func
            .set_type(obj, Type::Struct("Nested".to_string()));
        builder.write_variable("n".to_string(), builder.current_block, obj);

        // n.t
        let attr = ast::Expr::Attribute(ast::ExprAttribute {
            value: Box::new(ast::Expr::Name(ast::ExprName {
                id: "n".into(),
                ctx: ast::ExprContext::Load,
                range: Default::default(),
            })),
            attr: "t".into(),
            ctx: ast::ExprContext::Load,
            range: Default::default(),
        });

        let res = builder.visit_expr(attr).unwrap();

        // Should generate StructOffset because Tuple is composite
        let mut found_offset = false;
        for block in &builder.func.blocks {
            for inst in &block.instructions {
                if let InstructionKind::StructOffset(d, o, offset) = inst.kind {
                    if d == res && o == obj && offset == 0 {
                        found_offset = true;
                    }
                }
            }
        }
        assert!(
            found_offset,
            "Should have generated StructOffset for string-defined Tuple field"
        );

        let res_ty = builder.func.get_type(res);
        assert!(matches!(res_ty, Type::Tuple(_)));
    }
}
