use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, SourceLocation, Value};
use crate::push_inst;

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

    pub(super) fn resolve_dim(&mut self, tensor: Value, dim_str: &str, index: usize) -> Value {
        if let Ok(val) = dim_str.parse::<i64>() {
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(dest, val));
            self.func.set_type(dest, crate::ir::Type::I64);
            dest
        } else {
            let dest = self.func.next_value();
            push_inst!(self, InstructionKind::TensorDim(dest, tensor, index));
            self.func.set_type(dest, crate::ir::Type::I64);
            dest
        }
    }

    pub(super) fn get_broadcast_shape(
        &self,
        dims1: &[String],
        dims2: &[String],
    ) -> Option<Vec<String>> {
        let mut res = Vec::new();
        let len1 = dims1.len();
        let len2 = dims2.len();
        let max_len = std::cmp::max(len1, len2);

        for i in 0..max_len {
            let idx1 = i as i64 - (max_len as i64 - len1 as i64);
            let idx2 = i as i64 - (max_len as i64 - len2 as i64);

            let d1 = if idx1 < 0 {
                "1"
            } else {
                &dims1[idx1 as usize]
            };
            let d2 = if idx2 < 0 {
                "1"
            } else {
                &dims2[idx2 as usize]
            };

            if d1 == d2 {
                res.push(d1.to_string());
            } else if d1 == "1" {
                res.push(d2.to_string());
            } else if d2 == "1" {
                res.push(d1.to_string());
            } else {
                return None;
            }
        }
        Some(res)
    }
}
