use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, SourceLocation, Value};

impl CFGBuilder {
    pub(super) fn get_constant_int(&self, val: Value) -> Option<i64> {
        // Check if the type system already knows this is a literal
        if let Some(crate::ir::Type::Literal(_, v)) = self.func.value_types.get(&val) {
            return Some(*v);
        }

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
                    InstructionKind::Neg(v, src) if v == val => {
                        if let Some(s) = self.get_constant_int(src) {
                            return Some(-s);
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
}
