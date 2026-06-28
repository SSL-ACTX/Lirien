use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::ir::{InstructionKind, SourceLocation, Type, Value};
use crate::push_inst;

impl CFGBuilder {
    pub(crate) fn dummy_value(&mut self, ty: &Type) -> BuilderResult<Value> {
        let val = self.func.next_value();
        match ty {
            Type::I64
            | Type::I32
            | Type::I16
            | Type::I8
            | Type::U64
            | Type::U32
            | Type::U16
            | Type::U8 => {
                push_inst!(self, InstructionKind::ConstInt(val, 0));
            }
            Type::F64 | Type::F32 => {
                push_inst!(self, InstructionKind::ConstFloat(val, 0.0));
            }
            Type::Bool => {
                let zero = self.func.next_value();
                push_inst!(self, InstructionKind::ConstInt(zero, 0));
                self.func.set_type(zero, Type::I64);
                push_inst!(self, InstructionKind::Ne(val, zero, zero)); // false
            }
            Type::Pointer(_)
            | Type::NullablePointer(_)
            | Type::Optional(_)
            | Type::List(_)
            | Type::Buffer(_)
            | Type::Tensor(_, _)
            | Type::Array(_, _) => {
                push_inst!(self, InstructionKind::ConstInt(val, 0)); // null pointer / 0
            }
            _ => {
                push_inst!(self, InstructionKind::ConstInt(val, 0));
            }
        }
        self.func.set_type(val, ty.clone());
        Ok(val)
    }

    pub(crate) fn check_and_propagate_exception(&mut self) -> BuilderResult<()> {
        let exc_ptr = Value(0);
        let exc_val = self.func.next_value();
        push_inst!(self, InstructionKind::PointerLoad(exc_val, exc_ptr));
        self.func.set_type(exc_val, Type::I64);

        let zero = self.func.next_value();
        push_inst!(self, InstructionKind::ConstInt(zero, 0));
        self.func.set_type(zero, Type::I64);

        let has_exc = self.func.next_value();
        push_inst!(self, InstructionKind::Ne(has_exc, exc_val, zero));
        self.func.set_type(has_exc, Type::Bool);

        let next_block = self.create_block();
        let handler_block = if let Some(&handler) = self.try_stack.last() {
            handler
        } else {
            let ret_block = self.create_block();
            let prev_block = self.current_block;

            self.start_block(ret_block);
            let ret_val = if self.func.return_type != Type::Unknown
                && self.func.return_type != Type::Tuple(vec![])
            {
                Some(self.dummy_value(&self.func.return_type.clone())?)
            } else {
                None
            };
            push_inst!(self, InstructionKind::Return(ret_val));
            self.seal_block(ret_block)?;

            self.current_block = prev_block;
            ret_block
        };

        push_inst!(
            self,
            InstructionKind::Branch(has_exc, handler_block, next_block)
        );
        self.link_blocks(self.current_block, handler_block);
        self.link_blocks(self.current_block, next_block);

        self.seal_block(next_block)?;
        self.start_block(next_block);
        Ok(())
    }

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

            let d1 = if idx1 < 0 { "1" } else { &dims1[idx1 as usize] };
            let d2 = if idx2 < 0 { "1" } else { &dims2[idx2 as usize] };

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
