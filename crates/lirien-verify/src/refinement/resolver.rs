//! Refinement variable resolution maps.
//!
//! This module defines the [`Resolver`] helper, which maps variable identifier names
//! (e.g. `"v0"`, `"v1"`) from liquid type refinement predicates back to actual Z3 SMT variables.

use lirien_ir::ir::Value;
use std::collections::HashMap;
use z3::ast::{Array, Bool, Float, Int, BV};

/// Resolver maps variable name strings (e.g. `"v3"`) to active Z3 SMT variables.
pub struct Resolver<'a> {
    /// Maps values to solver Int variables.
    pub ints: &'a HashMap<Value, Int>,
    /// Maps values to solver Float variables.
    pub floats: &'a HashMap<Value, Float>,
    /// Maps values to solver Bit-Vector (BV) variables.
    pub bvs: &'a HashMap<Value, BV>,
    /// Maps values to solver memory Arrays.
    pub arrays: &'a HashMap<Value, Array>,
}

impl<'a> Resolver<'a> {
    /// Resolves a boolean variable or constant from its name.
    pub fn resolve_bool(&self, name: &str) -> Option<Bool> {

        if name == "true" {
            return Some(Bool::from_bool(true));
        }
        if name == "false" {
            return Some(Bool::from_bool(false));
        }
        if let Some(stripped) = name.strip_prefix('v') {
            if let Ok(id) = stripped.parse::<usize>() {
                let v = Value(id);
                // In Lirien, booleans are often modeled as BV1 or Int(0/1)
                if let Some(bv) = self.bvs.get(&v) {
                    if bv.get_size() == 1 {
                        let zero = BV::from_i64(0, 1);
                        return Some(bv.eq(&zero).not());
                    }
                }
                if let Some(i) = self.ints.get(&v) {
                    let zero = Int::from_i64(0);
                    return Some(i.eq(&zero).not());
                }
            }
        }
        None
    }

    pub fn resolve_int(&self, name: &str) -> Option<Int> {
        if let Some(stripped) = name.strip_prefix('v') {
            if let Ok(id) = stripped.parse::<usize>() {
                let val = Value(id);
                if let Some(i) = self.ints.get(&val) {
                    return Some(i.clone());
                }
                if let Some(bv) = self.bvs.get(&val) {
                    return Some(bv.to_int(true));
                }
            }
        }
        None
    }

    /// Resolves a bit-vector variable from its name.
    pub fn resolve_bv(&self, name: &str) -> Option<BV> {
        if let Some(stripped) = name.strip_prefix('v') {
            if let Ok(id) = stripped.parse::<usize>() {
                return self.bvs.get(&Value(id)).cloned();
            }
        }
        None
    }

    /// Resolves a floating-point variable from its name.
    pub fn resolve_float(&self, name: &str) -> Option<Float> {
        if let Some(stripped) = name.strip_prefix('v') {
            if let Ok(id) = stripped.parse::<usize>() {
                return self.floats.get(&Value(id)).cloned();
            }
        }
        None
    }
}
