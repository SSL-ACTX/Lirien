//! Representation of JIT functions and basic blocks.
//!
//! This module defines [`Function`] and [`BasicBlock`], which are the primary structures
//! representing the compiler's Control Flow Graph (CFG) in Static Single Assignment (SSA) form.

use super::instruction::Instruction;
use super::types::{BlockId, Type, Value, SourceLocation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a user-defined loop invariant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopInvariant {
    /// The block ID of the loop header where the invariant holds.
    pub header_block: BlockId,
    /// The Z3-compatible predicate string of the invariant.
    pub predicate: String,
    /// Optional Python source location mapping.
    pub location: Option<SourceLocation>,
}

/// A basic block containing a contiguous sequence of instructions.
///
/// Within a basic block, execution is straight-line: control enters at the first instruction
/// and leaves at the last one. Predecessors and successors track control-flow edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    /// Unique identifier for the block.
    pub id: BlockId,
    /// Ordered list of instructions inside the block.
    pub instructions: Vec<Instruction>,
    /// Predecessor basic block IDs.
    pub predecessors: Vec<BlockId>,
    /// Successor basic block IDs.
    pub successors: Vec<BlockId>,
}

/// A JIT-compilable function defined in the Lirien Intermediate Representation.
///
/// Contains basic blocks, value definitions, type mapping, refinement predicates,
/// and metadata about parameters, return types, and memory layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// The unique name of the function.
    pub name: String,
    /// The basic blocks composing the function's control flow graph.
    pub blocks: Vec<BasicBlock>,
    /// The entry block of the function.
    pub entry_block: BlockId,
    /// Counter for generated SSA [`Value`]s.
    pub value_count: usize,
    /// Counter for generated [`BlockId`]s.
    pub block_count: usize,
    /// Number of arguments accepted by the function.
    pub arg_count: usize,
    /// Expected return type.
    pub return_type: Type,
    /// Optional Z3-compatible refinement predicate for the return value.
    pub ret_refinement: Option<String>,
    /// Type map of all SSA [`Value`]s defined in the function.
    pub value_types: HashMap<Value, Type>,
    /// Z3-compatible refinement predicates for individual [`Value`]s.
    pub refinements: HashMap<Value, String>,
    /// Memory layouts for custom structs used in this function.
    pub struct_layouts: HashMap<String, Vec<(String, Type)>>,
    /// Memory layouts for custom enums used in this function.
    pub enum_layouts: HashMap<String, Vec<(String, Type)>>,
    /// Function preconditions.
    pub preconditions: Vec<String>,
    /// Function postconditions.
    pub postconditions: Vec<String>,
    /// User loop invariants.
    pub loop_invariants: Vec<LoopInvariant>,
}

impl Function {
    /// Creates a new empty `Function` with the given name.
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
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            loop_invariants: Vec::new(),
        }
    }

    /// Associates a Z3 refinement predicate string with a given SSA value.
    pub fn set_refinement(&mut self, val: Value, refinement: String) {
        self.refinements.insert(val, refinement);
    }

    /// Allocates and returns the next unique SSA [`Value`].
    pub fn next_value(&mut self) -> Value {
        let val = Value(self.value_count);
        self.value_count += 1;
        val
    }

    /// Associates a type with a given SSA value.
    pub fn set_type(&mut self, val: Value, ty: Type) {
        self.value_types.insert(val, ty);
    }

    /// Retrieves the type of a given SSA value, defaulting to [`Type::Unknown`].
    pub fn get_type(&self, val: Value) -> Type {
        self.value_types.get(&val).cloned().unwrap_or(Type::Unknown)
    }

    /// Allocates and returns the next unique [`BlockId`].
    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.block_count);
        self.block_count += 1;
        id
    }

    /// Prints a text representation of the function's CFG and instructions to standard output.
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

