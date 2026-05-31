use crate::ssa::builder::CFGBuilder;
use crate::ssa::ir::{InstructionKind, SourceLocation, Type, Value};

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
        let ty = self.func.get_type(val);
        if let Type::Hand(inner) | Type::Peek(inner) | Type::Held(inner) = ty {
            if inner.is_int() || inner.is_float() || *inner == Type::Bool {
                let dest = self.func.next_value();
                self.add_instruction(InstructionKind::StructLoad(dest, val, 0));
                self.func.set_type(dest, *inner);
                return dest;
            }
        }
        val
    }
}
