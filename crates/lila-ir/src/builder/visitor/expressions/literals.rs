use crate::builder::error::BuilderResult;
use crate::builder::CFGBuilder;
use crate::{push_inst, builder_error};
use crate::ir::{InstructionKind, Type, Value};
use rustpython_ast as ast;

impl CFGBuilder {
    pub(crate) fn visit_constant(&mut self, c: ast::ExprConstant) -> BuilderResult<Value> {
        let val = self.func.next_value();
        match c.value {
            ast::Constant::Int(i) => {
                let int_val = i.to_string().parse::<i64>().map_err(|_| builder_error!(General, "Int too large"))?;
                push_inst!(self, InstructionKind::ConstInt(val, int_val));
                self.func.set_type(val, Type::I64);
            }
            ast::Constant::Float(f) => {
                push_inst!(self, InstructionKind::ConstFloat(val, f));
                self.func.set_type(val, Type::Unknown);
            }
            ast::Constant::Bool(b) => {
                let inst = push_inst!(self, InstructionKind::ConstInt(val, if b { 1 } else { 0 }));
                inst.add_constraint(format!("(= {} {})", val, b));
                self.func.set_type(val, Type::Bool);
            }
            ast::Constant::None => {
                push_inst!(self, InstructionKind::ConstInt(val, 0));
                self.func.set_type(val, Type::I64);
            }
            _ => return Err(builder_error!(General, "Unsupported constant type")),
        }
        Ok(val)
    }

    pub(crate) fn visit_tuple(&mut self, t: ast::ExprTuple) -> BuilderResult<Value> {
        let mut elts = Vec::new();
        let mut elt_types = Vec::new();
        for elt in t.elts {
            let val = self.visit_expr(elt)?;
            elts.push(val);
            elt_types.push(self.func.get_type(val));
        }
        let dest = self.func.next_value();
        push_inst!(self, InstructionKind::TupleCreate(dest, elts));
        self.func.set_type(dest, Type::Tuple(elt_types));
        Ok(dest)
    }

    pub(crate) fn visit_name(&mut self, n: ast::ExprName) -> BuilderResult<Value> {
        self.read_variable(n.id.to_string(), self.current_block)
    }
}
