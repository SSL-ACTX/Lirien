use crate::ir::Type;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SerializedSignature {
    pub arg_types: Vec<Type>,
    pub arg_refinements: HashMap<usize, String>,
    pub return_type: Type,
    pub return_refinement: Option<String>,
}

impl From<&FunctionSignature> for SerializedSignature {
    fn from(sig: &FunctionSignature) -> Self {
        Self {
            arg_types: sig.arg_types.clone(),
            arg_refinements: sig.arg_refinements.clone(),
            return_type: sig.return_type.clone(),
            return_refinement: sig.return_refinement.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub arg_types: Vec<Type>,
    pub arg_refinements: HashMap<usize, String>,
    pub return_type: Type,
    pub return_refinement: Option<String>,
    pub pointer: usize,
}

pub struct Registry {
    pub functions: HashMap<String, FunctionSignature>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn register(&mut self, sig: FunctionSignature) {
        self.functions.insert(sig.name.clone(), sig);
    }

    pub fn get(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name)
    }
}

lazy_static! {
    pub static ref GLOBAL_REGISTRY: Mutex<Registry> = Mutex::new(Registry::new());
}
