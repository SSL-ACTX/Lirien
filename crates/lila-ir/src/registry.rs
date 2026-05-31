use crate::ir::Type;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

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
