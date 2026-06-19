use super::super::ir::{BlockId, Instruction, Value};

pub trait SsaManager {
    fn write_variable(&mut self, variable: String, block: BlockId, value: Value);
    fn read_variable(&mut self, variable: String, block: BlockId) -> Result<Value, String>;
    fn read_variable_recursive(
        &mut self,
        variable: String,
        block: BlockId,
    ) -> Result<Value, String>;
    fn add_phi_operands(
        &mut self,
        variable: String,
        phi_val: Value,
        block: BlockId,
    ) -> Result<Value, String>;
    fn seal_block(&mut self, block: BlockId) -> Result<(), String>;
    fn get_predecessors(&self, block_id: BlockId) -> Vec<BlockId>;
    fn add_instruction(&mut self, inst: Instruction);
    fn add_instruction_to_block(&mut self, block_id: BlockId, inst: Instruction);
    fn create_block(&mut self) -> BlockId;
    fn start_block(&mut self, id: BlockId);
    fn link_blocks(&mut self, from: BlockId, to: BlockId);
    fn is_terminated(&self, block_id: BlockId) -> bool;
}

// We will implement this on CFGBuilder in mod.rs or keep it as helper methods.
