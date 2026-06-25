//! AST to SSA-CFG compiler implementation.
//!
//! This module coordinates the transformation pipeline, parsing Python AST, building
//! basic blocks, resolving variable reads and writes to generate single-assignment values,
//! and resolving loop phis.

pub mod capture_analysis;
pub mod error;
pub mod macros;
pub mod metadata;
pub mod ssa_manager;
pub mod visitor;
#[cfg(test)]
pub mod visitor_tests;

use self::error::{BuilderError, BuilderResult};
use super::ir::{
    BasicBlock, BlockId, Function, Instruction, InstructionKind, SourceLocation, Type, Value,
};
use rustpython_ast as ast;
use rustpython_parser::Parse;
use std::collections::{HashMap, HashSet};

/// Core builder for constructing the Control Flow Graph in Static Single Assignment form.
pub struct CFGBuilder {
    /// The function IR being built.
    pub func: Function,
    /// The current basic block index.
    pub current_block: BlockId,
    /// Active Python source code location for error reporting and mapping.
    pub current_location: Option<SourceLocation>,
    /// User defined type aliases.
    pub type_aliases: HashMap<String, String>,
    /// Set of known named tuple layouts.
    pub named_tuple_names: HashSet<String>,
    /// Set of known typed dictionary layouts.
    pub typed_dict_names: HashSet<String>,
    /// Set of known enum layouts.
    pub enum_names: HashSet<String>,
    /// Map of variable names to their active SSA value within each block.
    pub variable_defs: HashMap<String, HashMap<BlockId, Value>>,
    /// Unfinished Phi nodes from join points that are resolved once loop predecessors are sealed.
    pub incomplete_phis: HashMap<BlockId, HashMap<String, Value>>,
    /// Sealed basic blocks (all predecessor blocks are known).
    pub sealed_blocks: HashSet<BlockId>,
    /// Active loop block tracking for break and continue statements: (header, exit).
    pub loop_stack: Vec<(BlockId, BlockId)>,
    /// Collected lambda helper functions compiled from the AST.
    pub lambdas: Vec<Function>,
}

impl CFGBuilder {
    /// Retrieves the layout byte size of the specified type.
    pub fn get_type_size(&self, ty: &Type) -> usize {
        ty.size(&self.func.struct_layouts)
    }

    pub fn get_type_align(&self, ty: &Type) -> usize {
        ty.align(&self.func.struct_layouts)
    }

    pub fn get_field_offset(&self, struct_name: &str, field_name: &str) -> Option<usize> {
        let fields = self.func.struct_layouts.get(struct_name)?;
        let mut offset = 0;
        for (f_name, f_ty) in fields {
            let align = f_ty.align(&self.func.struct_layouts);
            offset = (offset + align - 1) & !(align - 1);
            if f_name == field_name {
                return Some(offset);
            }
            offset += f_ty.size(&self.func.struct_layouts);
        }
        None
    }

    pub fn new(
        name: String,
        layouts: HashMap<String, Vec<(String, String)>>,
        enum_layouts_raw: HashMap<String, Vec<(String, String)>>,
        type_aliases: HashMap<String, String>,
        named_tuple_layouts: HashMap<String, Vec<(String, String)>>,
        typed_dict_layouts: HashMap<String, Vec<(String, String)>>,
    ) -> Self {
        let mut named_tuple_names = HashSet::new();
        for name in named_tuple_layouts.keys() {
            named_tuple_names.insert(name.clone());
        }

        let mut typed_dict_names = HashSet::new();
        for name in typed_dict_layouts.keys() {
            typed_dict_names.insert(name.clone());
        }

        let mut enum_names = HashSet::new();
        for name in enum_layouts_raw.keys() {
            enum_names.insert(name.clone());
        }

        let mut struct_layouts = HashMap::new();
        for (s_name, fields) in layouts {
            let mut field_types = Vec::new();
            for (f_name, f_ty_str) in fields {
                let ty = if let Ok(expr) = ast::Expr::parse(&f_ty_str, "<field>") {
                    metadata::parse_type(&expr, &type_aliases, &named_tuple_names, &typed_dict_names, &enum_names).unwrap_or(Type::Unknown)
                } else {
                    match f_ty_str.as_str() {
                        "i8" => Type::I8,
                        "u8" => Type::U8,
                        "i16" => Type::I16,
                        "u16" => Type::U16,
                        "i32" => Type::I32,
                        "u32" => Type::U32,
                        "i64" => Type::I64,
                        "u64" => Type::U64,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "bool" => Type::Bool,
                        _ => {
                            if f_ty_str == "unknown" {
                                Type::Unknown
                            } else if enum_names.contains(&f_ty_str) {
                                Type::Enum(f_ty_str.clone())
                            } else if named_tuple_names.contains(&f_ty_str) {
                                Type::NamedTuple(f_ty_str.clone())
                            } else if typed_dict_names.contains(&f_ty_str) {
                                Type::TypedDict(f_ty_str.clone())
                            } else {
                                Type::Struct(f_ty_str.clone())
                            }
                        }
                    }
                };
                field_types.push((f_name, ty));
            }
            struct_layouts.insert(s_name, field_types);
        }

        for (nt_name, fields) in named_tuple_layouts {
            let mut field_types = Vec::new();
            for (f_name, f_ty_str) in fields {
                let ty = if let Ok(expr) = ast::Expr::parse(&f_ty_str, "<field>") {
                    metadata::parse_type(&expr, &type_aliases, &named_tuple_names, &typed_dict_names, &enum_names).unwrap_or(Type::Unknown)
                } else {
                    match f_ty_str.as_str() {
                        "i8" => Type::I8,
                        "u8" => Type::U8,
                        "i16" => Type::I16,
                        "u16" => Type::U16,
                        "i32" => Type::I32,
                        "u32" => Type::U32,
                        "i64" => Type::I64,
                        "u64" => Type::U64,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "bool" => Type::Bool,
                        _ => {
                            if f_ty_str == "unknown" {
                                Type::Unknown
                            } else if enum_names.contains(&f_ty_str) {
                                Type::Enum(f_ty_str.clone())
                            } else if named_tuple_names.contains(&f_ty_str) {
                                Type::NamedTuple(f_ty_str.clone())
                            } else if typed_dict_names.contains(&f_ty_str) {
                                Type::TypedDict(f_ty_str.clone())
                            } else {
                                Type::Struct(f_ty_str.clone())
                            }
                        }
                    }
                };
                field_types.push((f_name, ty));
            }
            struct_layouts.insert(nt_name, field_types);
        }

        for (td_name, fields) in typed_dict_layouts {
            let mut field_types = Vec::new();
            for (f_name, f_ty_str) in fields {
                let ty = if let Ok(expr) = ast::Expr::parse(&f_ty_str, "<field>") {
                    metadata::parse_type(&expr, &type_aliases, &named_tuple_names, &typed_dict_names, &enum_names).unwrap_or(Type::Unknown)
                } else {
                    match f_ty_str.as_str() {
                        "i8" => Type::I8,
                        "u8" => Type::U8,
                        "i16" => Type::I16,
                        "u16" => Type::U16,
                        "i32" => Type::I32,
                        "u32" => Type::U32,
                        "i64" => Type::I64,
                        "u64" => Type::U64,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "bool" => Type::Bool,
                        _ => {
                            if f_ty_str == "unknown" {
                                Type::Unknown
                            } else if enum_names.contains(&f_ty_str) {
                                Type::Enum(f_ty_str.clone())
                            } else if named_tuple_names.contains(&f_ty_str) {
                                Type::NamedTuple(f_ty_str.clone())
                            } else if typed_dict_names.contains(&f_ty_str) {
                                Type::TypedDict(f_ty_str.clone())
                            } else {
                                Type::Struct(f_ty_str.clone())
                            }
                        }
                    }
                };
                field_types.push((f_name, ty));
            }
            struct_layouts.insert(td_name, field_types);
        }

        let mut enum_layouts = HashMap::new();
        for (e_name, variants) in enum_layouts_raw {
            let mut variant_types = Vec::new();
            for (v_name, v_ty_str) in variants {
                let ty = if v_ty_str == "None" {
                    Type::Unknown
                } else if let Ok(expr) = ast::Expr::parse(&v_ty_str, "<variant>") {
                    metadata::parse_type(&expr, &type_aliases, &named_tuple_names, &typed_dict_names, &enum_names).unwrap_or(Type::Unknown)
                } else {
                    match v_ty_str.as_str() {
                        "i8" => Type::I8,
                        "u8" => Type::U8,
                        "i16" => Type::I16,
                        "u16" => Type::U16,
                        "i32" => Type::I32,
                        "u32" => Type::U32,
                        "i64" => Type::I64,
                        "u64" => Type::U64,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "bool" => Type::Bool,
                        _ => {
                            if v_ty_str == "unknown" {
                                Type::Unknown
                            } else if enum_names.contains(&v_ty_str) {
                                Type::Enum(v_ty_str.clone())
                            } else if named_tuple_names.contains(&v_ty_str) {
                                Type::NamedTuple(v_ty_str.clone())
                            } else if typed_dict_names.contains(&v_ty_str) {
                                Type::TypedDict(v_ty_str.clone())
                            } else {
                                Type::Struct(v_ty_str.clone())
                            }
                        }
                    }
                };
                variant_types.push((v_name, ty));
            }
            enum_layouts.insert(e_name, variant_types);
        }

        let mut builder = Self {
            func: Function {
                struct_layouts,
                enum_layouts,
                ..Function::new(name)
            },
            current_block: BlockId(0),
            current_location: None,
            type_aliases,
            named_tuple_names,
            typed_dict_names,
            enum_names,
            variable_defs: HashMap::new(),
            incomplete_phis: HashMap::new(),
            sealed_blocks: HashSet::new(),
            loop_stack: Vec::new(),
            lambdas: Vec::new(),
        };


        let entry = builder.create_block();
        builder.current_block = entry;
        builder.sealed_blocks.insert(entry);
        builder
    }

    pub fn build(&mut self, suite: ast::Suite) -> BuilderResult<()> {
        for stmt in suite {
            if let ast::Stmt::FunctionDef(s) = stmt {
                if s.name.as_str() == self.func.name {
                    self.visit_function_def(s)?;
                    return Ok(());
                }
            }
        }
        Err(BuilderError::General(
            format!("Function '{}' not found in source", self.func.name),
            None,
        ))
    }

    // SSA Management Methods
    pub fn write_variable(&mut self, variable: String, block: BlockId, value: Value) {
        self.variable_defs
            .entry(variable)
            .or_default()
            .insert(block, value);
    }

    pub fn read_variable(&mut self, variable: String, block: BlockId) -> BuilderResult<Value> {
        if let Some(defs) = self.variable_defs.get(&variable) {
            if let Some(val) = defs.get(&block) {
                return Ok(*val);
            }
        }
        self.read_variable_recursive(variable, block)
    }

    pub fn read_variable_recursive(
        &mut self,
        variable: String,
        block: BlockId,
    ) -> BuilderResult<Value> {
        let mut val: Value;

        if !self.sealed_blocks.contains(&block) {
            val = self.func.next_value();
            self.incomplete_phis
                .entry(block)
                .or_default()
                .insert(variable.clone(), val);
            self.add_instruction_to_block(block, InstructionKind::Phi(val, HashMap::new()));

            // Try to set initial type from current definition if available
            if let Some(defs) = self.variable_defs.get(&variable) {
                if let Some(prev_val) = defs.values().next() {
                    let ty = self.func.get_type(*prev_val);
                    if ty != Type::Unknown {
                        self.func.set_type(val, ty);
                    }
                }
            }
        } else {
            let predecessors = self.get_predecessors(block);
            if predecessors.is_empty() {
                return Err(BuilderError::UnboundVariable(
                    variable,
                    self.current_location,
                ));
            } else if predecessors.len() == 1 {
                val = self.read_variable(variable.clone(), predecessors[0])?;
            } else {
                val = self.func.next_value();
                self.write_variable(variable.clone(), block, val);
                self.add_instruction_to_block(block, InstructionKind::Phi(val, HashMap::new()));
                val = self.add_phi_operands(variable.clone(), val, block)?;
            }
        }

        self.write_variable(variable, block, val);
        Ok(val)
    }

    pub fn add_phi_operands(
        &mut self,
        variable: String,
        phi_val: Value,
        block: BlockId,
    ) -> BuilderResult<Value> {
        let predecessors = self.get_predecessors(block);
        let mut operands = HashMap::new();
        let mut phi_type = Type::Unknown;

        for pred in predecessors {
            let val = self.read_variable(variable.clone(), pred)?;
            operands.insert(pred, val);

            if phi_type == Type::Unknown {
                phi_type = self.func.get_type(val);
            }
        }

        if phi_type != Type::Unknown {
            self.func.set_type(phi_val, phi_type);
        }

        if let Some(b) = self.func.blocks.iter_mut().find(|b| b.id == block) {
            for inst in &mut b.instructions {
                if let InstructionKind::Phi(v, ops) = &mut inst.kind {
                    if *v == phi_val {
                        *ops = operands;
                        break;
                    }
                }
            }
        }
        Ok(phi_val)
    }

    pub fn seal_block(&mut self, block: BlockId) -> BuilderResult<()> {
        self.sealed_blocks.insert(block);
        let phis = self.incomplete_phis.remove(&block).unwrap_or_default();
        for (variable, phi_val) in phis {
            self.add_phi_operands(variable, phi_val, block)?;
        }
        Ok(())
    }

    pub fn get_predecessors(&self, block_id: BlockId) -> Vec<BlockId> {
        self.func
            .blocks
            .iter()
            .find(|b| b.id == block_id)
            .map(|b| b.predecessors.clone())
            .unwrap_or_default()
    }

    pub fn add_instruction(&mut self, kind: InstructionKind) -> &mut Instruction {
        let block_id = self.current_block;
        self.add_instruction_to_block(block_id, kind)
    }

    pub fn add_instruction_to_block(
        &mut self,
        block_id: BlockId,
        kind: InstructionKind,
    ) -> &mut Instruction {
        if let Some(block) = self.func.blocks.iter_mut().find(|b| b.id == block_id) {
            let inst = Instruction::new(kind, self.current_location);
            if let InstructionKind::Phi(_, _) = &inst.kind {
                block.instructions.insert(0, inst);
                &mut block.instructions[0]
            } else {
                block.instructions.push(inst);
                block.instructions.last_mut().unwrap()
            }
        } else {
            panic!("Block {} not found", block_id);
        }
    }

    pub fn create_block(&mut self) -> BlockId {
        let id = self.func.next_block();
        self.func.blocks.push(BasicBlock {
            id,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        });
        id
    }

    pub fn start_block(&mut self, id: BlockId) {
        self.current_block = id;
    }

    pub fn link_blocks(&mut self, from: BlockId, to: BlockId) {
        if let Some(block) = self.func.blocks.iter_mut().find(|b| b.id == from) {
            if !block.successors.contains(&to) {
                block.successors.push(to);
            }
        }
        if let Some(block) = self.func.blocks.iter_mut().find(|b| b.id == to) {
            if !block.predecessors.contains(&from) {
                block.predecessors.push(from);
            }
        }
    }

    pub fn is_terminated(&self, block_id: BlockId) -> bool {
        if let Some(block) = self.func.blocks.iter().find(|b| b.id == block_id) {
            if let Some(last) = block.instructions.last() {
                return matches!(
                    &last.kind,
                    InstructionKind::Jump(_)
                        | InstructionKind::Branch(_, _, _)
                        | InstructionKind::Return(_)
                );
            }
        }
        false
    }

    pub fn new_sub_builder(&self, name: String) -> Self {
        let mut builder = Self {
            func: Function {
                struct_layouts: self.func.struct_layouts.clone(),
                enum_layouts: self.func.enum_layouts.clone(),
                ..Function::new(name)
            },
            current_block: BlockId(0),
            current_location: self.current_location,
            type_aliases: self.type_aliases.clone(),
            named_tuple_names: self.named_tuple_names.clone(),
            typed_dict_names: self.typed_dict_names.clone(),
            enum_names: self.enum_names.clone(),
            variable_defs: HashMap::new(),
            incomplete_phis: HashMap::new(),
            sealed_blocks: HashSet::new(),
            loop_stack: Vec::new(),
            lambdas: Vec::new(),
        };

        let entry = builder.create_block();
        builder.current_block = entry;
        builder.sealed_blocks.insert(entry);
        builder
    }
}
