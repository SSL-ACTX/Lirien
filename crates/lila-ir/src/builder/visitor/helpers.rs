use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, SourceLocation, Type, Value};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(super) fn get_constant_int(&self, val: Value) -> Option<i64> {
        for block in &self.func.blocks {
            for inst in &block.instructions {
                match inst.kind {
                    InstructionKind::ConstInt(v, val_const) if v == val => {
                        return Some(val_const);
                    }
                    InstructionKind::Sub(v, lhs, rhs) if v == val => {
                        if let (Some(l), Some(r)) =
                            (self.get_constant_int(lhs), self.get_constant_int(rhs))
                        {
                            return Some(l - r);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    pub(super) fn update_location(&mut self, offset: usize) {
        self.current_location = Some(SourceLocation { offset });
    }

    pub(super) fn auto_load(&mut self, val: Value) -> Value {
        val
    }

    pub(super) fn collect_variable_names(&self, expr: &ast::Expr, names: &mut Vec<String>) {
        match expr {
            ast::Expr::Name(n) => names.push(n.id.to_string()),
            ast::Expr::Tuple(t) => {
                for elt in &t.elts {
                    self.collect_variable_names(elt, names);
                }
            }
            _ => {}
        }
    }
}
