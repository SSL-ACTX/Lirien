use super::instruction::Instruction;
use super::types::{BlockId, Type, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: BlockId,
    pub value_count: usize,
    pub block_count: usize,
    pub arg_count: usize,
    pub return_type: Type,
    pub ret_refinement: Option<String>,
    pub value_types: HashMap<Value, Type>,
    pub refinements: HashMap<Value, String>,
    pub struct_layouts: HashMap<String, Vec<(String, Type)>>,
    pub enum_layouts: HashMap<String, Vec<(String, Type)>>,
}

impl Function {
    pub fn new(name: String) -> Self {
        Self {
            name,
            blocks: Vec::new(),
            entry_block: BlockId(0),
            value_count: 0,
            block_count: 0,
            arg_count: 0,
            return_type: Type::Unknown,
            ret_refinement: None,
            value_types: HashMap::new(),
            refinements: HashMap::new(),
            struct_layouts: HashMap::new(),
            enum_layouts: HashMap::new(),
        }
    }

    pub fn set_refinement(&mut self, val: Value, refinement: String) {
        self.refinements.insert(val, refinement);
    }

    pub fn next_value(&mut self) -> Value {
        let val = Value(self.value_count);
        self.value_count += 1;
        val
    }

    pub fn set_type(&mut self, val: Value, ty: Type) {
        self.value_types.insert(val, ty);
    }

    pub fn get_type(&self, val: Value) -> Type {
        self.value_types.get(&val).cloned().unwrap_or(Type::Unknown)
    }

    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.block_count);
        self.block_count += 1;
        id
    }

    pub fn dump(&self) {
        println!("function {} {{", self.name);
        for block in &self.blocks {
            println!("{}:", block.id);
            for inst in &block.instructions {
                println!("{}", inst);
            }
        }
        println!("}}");
    }
}
