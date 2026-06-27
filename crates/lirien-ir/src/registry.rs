//! Function registry for runtime symbol resolution.
//!
//! This module defines the global registry of JIT-compiled functions,
//! mapping function names to their types, Z3 refinements, and raw machine pointers.

use crate::ir::Type;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// A serialized representation of a function's signature.
///
/// This is used to pass type annotations and refinement constraints
/// across the FFI boundaries (e.g. to Python decorators).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SerializedSignature {
    /// Types of the function arguments.
    pub arg_types: Vec<Type>,
    /// Refinement constraint predicates mapped by parameter index.
    pub arg_refinements: HashMap<usize, String>,
    /// The return type of the function.
    pub return_type: Type,
    /// The refinement constraint predicate for the return value, if any.
    pub return_refinement: Option<String>,
    /// Precondition constraint S-expressions.
    pub preconditions: Vec<String>,
    /// Postcondition constraint S-expressions.
    pub postconditions: Vec<String>,
}

impl From<&FunctionSignature> for SerializedSignature {
    fn from(sig: &FunctionSignature) -> Self {
        Self {
            arg_types: sig.arg_types.clone(),
            arg_refinements: sig.arg_refinements.clone(),
            return_type: sig.return_type.clone(),
            return_refinement: sig.return_refinement.clone(),
            preconditions: sig.preconditions.clone(),
            postconditions: sig.postconditions.clone(),
        }
    }
}

/// Information about a registered JIT-compiled function.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// The name of the function.
    pub name: String,
    /// Types of the function arguments.
    pub arg_types: Vec<Type>,
    /// Refinement constraint predicates mapped by parameter index.
    pub arg_refinements: HashMap<usize, String>,
    /// The return type of the function.
    pub return_type: Type,
    /// The refinement constraint predicate for the return value, if any.
    pub return_refinement: Option<String>,
    /// Precondition constraint S-expressions.
    pub preconditions: Vec<String>,
    /// Postcondition constraint S-expressions.
    pub postconditions: Vec<String>,
    /// A raw machine code pointer to the compiled function.
    pub pointer: usize,
}

/// A registry mapping function names to their signatures.
pub struct Registry {
    /// Internal map of name to [`FunctionSignature`].
    pub functions: HashMap<String, FunctionSignature>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Creates a new empty `Registry`.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Registers a function signature.
    pub fn register(&mut self, sig: FunctionSignature) {
        self.functions.insert(sig.name.clone(), sig);
    }

    /// Retrieves a function signature by name, if it exists.
    pub fn get(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name)
    }
}

lazy_static! {
    /// The global, thread-safe registry of compiled Lirien functions.
    pub static ref GLOBAL_REGISTRY: Mutex<Registry> = Mutex::new(Registry::new());
}

