//! Abstract SSA construction manager.
//!
//! This module defines the [`SsaManager`] trait, which abstracts over CFG construction
//! and Braun-style SSA variable resolution (handling reads/writes of local variables,
//! recursively resolving variables across joins, inserting Phi nodes, and sealing blocks).

use super::super::ir::{BlockId, Instruction, Value};

/// Interface for building control flow graphs and constructing SSA form.
pub trait SsaManager {
    /// Writes the current definition value of a variable in a specific basic block.
    fn write_variable(&mut self, variable: String, block: BlockId, value: Value);

    /// Reads the active SSA value of a variable in a specific basic block.
    ///
    /// If the variable is not defined in the block, this triggers search/recursive lookup.
    ///
    /// # Errors
    /// Returns an error if the variable is read before it has been defined.
    fn read_variable(&mut self, variable: String, block: BlockId) -> Result<Value, String>;

    /// Recursively searches predecessors for a variable definition to insert a Phi node.
    ///
    /// Used when a variable is read from a block but has no local definition.
    ///
    /// # Errors
    /// Returns an error if resolution fails or hits undefined variables.
    fn read_variable_recursive(
        &mut self,
        variable: String,
        block: BlockId,
    ) -> Result<Value, String>;

    /// Populates operands for a Phi instruction at a block join point.
    ///
    /// # Errors
    /// Returns an error if reading operand variables from predecessor blocks fails.
    fn add_phi_operands(
        &mut self,
        variable: String,
        phi_val: Value,
        block: BlockId,
    ) -> Result<Value, String>;

    /// Seals a block, signaling that all of its predecessors are known.
    ///
    /// Once a block is sealed, incomplete Phi nodes can be resolved.
    ///
    /// # Errors
    /// Returns an error if sealing fails.
    fn seal_block(&mut self, block: BlockId) -> Result<(), String>;

    /// Retrieves the predecessor block IDs of the specified block.
    fn get_predecessors(&self, block_id: BlockId) -> Vec<BlockId>;

    /// Appends an instruction to the active basic block.
    fn add_instruction(&mut self, inst: Instruction);

    /// Appends an instruction to a specific basic block.
    fn add_instruction_to_block(&mut self, block_id: BlockId, inst: Instruction);

    /// Allocates and returns a new empty basic block ID.
    fn create_block(&mut self) -> BlockId;

    /// Sets the specified block as the active insertion point.
    fn start_block(&mut self, id: BlockId);

    /// Adds a directed control-flow edge between two basic blocks.
    fn link_blocks(&mut self, from: BlockId, to: BlockId);

    /// Returns `true` if the specified block ends with a terminator instruction.
    fn is_terminated(&self, block_id: BlockId) -> bool;
}
