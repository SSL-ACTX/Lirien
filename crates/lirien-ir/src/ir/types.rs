//! Primitive types, composite structures, SSA values, and locations.
//!
//! This module contains the core type system representation (`Type`),
//! variables/values (`Value`), memory access tracking (`AccessPath`),
//! and block/source mapping structures (`BlockId`, `SourceLocation`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Representation of types supported by the Lirien compiler.
///
/// Includes hardware primitives, fixed and variable size composites,
/// SIMD vector types, closures, and verification-only types like `Refined` and `Literal`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer (corresponds to standard native Python `int`).
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// Single-precision 32-bit floating-point number.
    F32,
    /// Double-precision 64-bit floating-point number (corresponds to standard native Python `float`).
    F64,
    /// Boolean type.
    Bool,

    // SIMD 128-bit vectors
    /// 4x 32-bit floats SIMD vector.
    F32X4,
    /// 4x 32-bit integers SIMD vector.
    I32X4,
    /// 2x 64-bit floats SIMD vector.
    F64X2,
    /// 2x 64-bit integers SIMD vector.
    I64X2,
    /// 16x 8-bit signed integers SIMD vector.
    I8X16,
    /// 16x 8-bit unsigned integers SIMD vector.
    U8X16,
    /// 8x 16-bit signed integers SIMD vector.
    I16X8,
    /// 8x 16-bit unsigned integers SIMD vector.
    U16X8,

    /// Flat array of a given type, optionally with a fixed compile-time size.
    Array(Box<Type>, Option<usize>),
    /// Memory buffer containing elements of a given type (typically fat pointer with length).
    Buffer(Box<Type>),
    /// Multidimensional tensor layout with list of axis dimensions.
    Tensor(Box<Type>, Vec<String>),
    /// Custom struct identified by its name.
    Struct(String),
    /// Typed dictionary identified by its name.
    TypedDict(String),
    /// Named tuple identified by its name.
    NamedTuple(String),
    /// Custom enum identified by its name.
    Enum(String),
    /// Heterogeneous tuple type.
    Tuple(Vec<Type>),
    /// A raw memory pointer to a type.
    Pointer(Box<Type>),
    /// A pointer that may contain NULL.
    NullablePointer(Box<Type>),
    /// Optional type, logically containing a tag and a payload.
    Optional(Box<Type>),
    /// Growable dynamic list with element type.
    List(Box<Type>),
    /// Function pointer type containing argument types, return type, and description.
    FnPointer(Vec<Type>, Box<Type>, Option<String>),
    /// Closure type containing closure name, captured variables, return type, and description.
    Closure(String, Vec<Type>, Box<Type>, Option<String>),
    /// A type refined by a Z3 path constraint predicate.
    Refined(Box<Type>, String),
    /// A literal type carrying a concrete compile-time value.
    Literal(Box<Type>, i64),
    /// Unknown or unresolved type.
    Unknown,
}

impl Type {
    /// Returns `true` if this type represents a scalar or SIMD float.
    pub fn is_float(&self) -> bool {
        match self {
            Type::F32 | Type::F64 | Type::F32X4 | Type::F64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float(),
            _ => false,
        }
    }

    /// Returns `true` if this type represents a 32-bit scalar float.
    pub fn is_float32(&self) -> bool {
        match self {
            Type::F32 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float32(),
            _ => false,
        }
    }

    /// Returns `true` if this type represents a 64-bit scalar float.
    pub fn is_float64(&self) -> bool {
        match self {
            Type::F64 | Type::F64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float64(),
            _ => false,
        }
    }

    /// Returns `true` if this type represents a scalar or SIMD integer.
    pub fn is_int(&self) -> bool {
        match self {
            Type::I8
            | Type::U8
            | Type::I16
            | Type::U16
            | Type::I32
            | Type::U32
            | Type::I64
            | Type::U64
            | Type::I32X4
            | Type::I64X2
            | Type::I8X16
            | Type::U8X16
            | Type::I16X8
            | Type::U16X8 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_int(),
            _ => false,
        }
    }

    /// Returns the integer bit width of the type, if applicable.
    pub fn int_bit_width(&self) -> Option<u32> {
        match self {
            Type::I8 | Type::U8 => Some(8),
            Type::I16 | Type::U16 => Some(16),
            Type::I32 | Type::U32 | Type::F32 => Some(32),
            Type::I64 | Type::U64 | Type::F64 => Some(64),
            Type::Bool => Some(1),
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.int_bit_width(),
            _ => None,
        }
    }

    /// Returns `true` if this is a signed integer type.
    pub fn is_signed(&self) -> bool {
        match self {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_signed(),
            _ => false,
        }
    }

    /// Extracts the core base type, stripping away refinements or literals.
    pub fn base_type(&self) -> &Type {
        match self {
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.base_type(),
            other => other,
        }
    }

    /// Returns `true` if this is an unsigned integer type.
    pub fn is_unsigned(&self) -> bool {
        match self {
            Type::U8 | Type::U16 | Type::U32 | Type::U64 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_unsigned(),
            _ => false,
        }
    }

    /// Returns `true` if this type is pointer-like (i.e. resides in memory or is a callable pointer).
    pub fn is_pointer_like(&self) -> bool {
        match self {
            Type::Buffer(_)
            | Type::List(_)
            | Type::Array(_, _)
            | Type::Pointer(_)
            | Type::NullablePointer(_)
            | Type::FnPointer(..)
            | Type::Closure(..) => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_pointer_like(),
            _ => false,
        }
    }

    /// Returns `true` if this is a SIMD vector type.
    pub fn is_simd(&self) -> bool {
        match self {
            Type::F32X4
            | Type::I32X4
            | Type::F64X2
            | Type::I64X2
            | Type::I8X16
            | Type::U8X16
            | Type::I16X8
            | Type::U16X8 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_simd(),
            _ => false,
        }
    }

    /// Returns `true` if this is a tensor type.
    pub fn is_tensor(&self) -> bool {
        match self {
            Type::Tensor(_, _) => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_tensor(),
            _ => false,
        }
    }

    /// Returns `true` if this is a composite/aggregate type (e.g. struct, tuple, or fixed-size aggregate).
    pub fn is_composite(&self) -> bool {
        match self {
            Type::Struct(_)
            | Type::TypedDict(_)
            | Type::NamedTuple(_)
            | Type::Tuple(_)
            | Type::Enum(_)
            | Type::Optional(_) => true,
            Type::Array(inner, Some(_)) => {
                // If the inner type is not a primitive, we treat fixed arrays as composite for offsets
                !inner.is_int() && !inner.is_float()
            }
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_composite(),
            _ => false,
        }
    }

    /// Computes the exact byte size of the type according to the target struct layout configuration.
    pub fn size(&self, struct_layouts: &HashMap<String, Vec<(String, Type)>>) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 | Type::Pointer(_) | Type::NullablePointer(_) => 8,
            Type::F32X4
            | Type::I32X4
            | Type::F64X2
            | Type::I64X2
            | Type::I8X16
            | Type::U8X16
            | Type::I16X8
            | Type::U16X8 => 16,
            Type::FnPointer(..) | Type::Closure(..) => 8,
            Type::Array(inner, Some(s)) => s * inner.size(struct_layouts),
            Type::Array(_, None) => 8, // Pointer to array
            Type::Buffer(_) => 16,     // Fat Pointer: (ptr, len)
            Type::List(_) => 8,        // Pointer to list header struct
            Type::Struct(name) | Type::TypedDict(name) | Type::NamedTuple(name) => {
                if let Some(fields) = struct_layouts.get(name) {
                    let mut offset = 0;
                    for (_, f_ty) in fields {
                        let align = f_ty.align(struct_layouts);
                        offset = (offset + align - 1) & !(align - 1);
                        offset += f_ty.size(struct_layouts);
                    }
                    let total_align = self.align(struct_layouts);
                    (offset + total_align - 1) & !(total_align - 1)
                } else {
                    0
                }
            }
            Type::Tuple(types) => {
                let mut offset = 0;
                for f_ty in types {
                    let align = f_ty.align(struct_layouts);
                    offset = (offset + align - 1) & !(align - 1);
                    offset += f_ty.size(struct_layouts);
                }
                let total_align = self.align(struct_layouts);
                (offset + total_align - 1) & !(total_align - 1)
            }
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.size(struct_layouts),
            Type::Optional(inner) => {
                let tag_size = 1;
                let payload_align = inner.align(struct_layouts);
                let payload_size = inner.size(struct_layouts);
                let mut offset = tag_size;
                offset = (offset + payload_align - 1) & !(payload_align - 1);
                offset += payload_size;
                let total_align = self.align(struct_layouts);
                (offset + total_align - 1) & !(total_align - 1)
            }
            Type::Enum(name) => {
                if let Some(variants) = struct_layouts.get(name) {
                    let mut max_payload_size = 0;
                    for (_, f_ty) in variants {
                        let sz = f_ty.size(struct_layouts);
                        if sz > max_payload_size {
                            max_payload_size = sz;
                        }
                    }
                    let tag_size = 1;
                    let payload_align = {
                        let mut max_align = 1;
                        for (_, f_ty) in variants {
                            let align = f_ty.align(struct_layouts);
                            if align > max_align {
                                max_align = align;
                            }
                        }
                        max_align
                    };

                    let mut offset = tag_size;
                    // Pad for payload alignment
                    offset = (offset + payload_align - 1) & !(payload_align - 1);
                    offset += max_payload_size;

                    let total_align = self.align(struct_layouts);
                    (offset + total_align - 1) & !(total_align - 1)
                } else {
                    0
                }
            }
            _ => 8,
        }
    }

    /// Computes the alignment requirement of the type (in bytes).
    pub fn align(&self, struct_layouts: &HashMap<String, Vec<(String, Type)>>) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 | Type::Pointer(_) | Type::NullablePointer(_) => 8,
            Type::F32X4
            | Type::I32X4
            | Type::F64X2
            | Type::I64X2
            | Type::I8X16
            | Type::U8X16
            | Type::I16X8
            | Type::U16X8 => 16,
            Type::FnPointer(..) | Type::Closure(..) => 8,
            Type::Array(inner, Some(_)) => inner.align(struct_layouts),
            Type::Array(_, None) => 8,
            Type::Struct(name) | Type::TypedDict(name) | Type::NamedTuple(name) => {
                if let Some(fields) = struct_layouts.get(name) {
                    let mut max_align = 1;
                    for (_, f_ty) in fields {
                        let align = f_ty.align(struct_layouts);
                        if align > max_align {
                            max_align = align;
                        }
                    }
                    max_align
                } else {
                    1
                }
            }
            Type::Enum(name) => {
                if let Some(variants) = struct_layouts.get(name) {
                    let mut max_align = 1; // tag is u8 (align 1)
                    for (_, f_ty) in variants {
                        let align = f_ty.align(struct_layouts);
                        if align > max_align {
                            max_align = align;
                        }
                    }
                    max_align
                } else {
                    1
                }
            }
            Type::Tuple(types) => {
                let mut max_align = 1;
                for f_ty in types {
                    let align = f_ty.align(struct_layouts);
                    if align > max_align {
                        max_align = align;
                    }
                }
                max_align
            }
            Type::Refined(inner, _) | Type::Literal(inner, _) | Type::Optional(inner) => {
                inner.align(struct_layouts)
            }
            _ => 8,
        }
    }
}

/// An identifier representing a unique SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Value(pub usize);

/// An element along a structured object memory access path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathElement {
    /// Field offset index inside a struct or composite layout.
    Field(usize),
    /// Subscript index using an SSA variable value.
    Index(Value),
}

/// A list of path elements describing the exact nested location accessed inside an aggregate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessPath(pub Vec<PathElement>);

impl AccessPath {
    /// Extends this access path with an additional [`PathElement`].
    pub fn extend(&self, el: PathElement) -> Self {
        let mut new_path = self.0.clone();
        new_path.push(el);
        AccessPath(new_path)
    }

    /// Returns `true` if `self` is a prefix of `other`.
    pub fn is_prefix_of(&self, other: &AccessPath) -> bool {
        if self.0.len() > other.0.len() {
            return false;
        }
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }

    /// Returns `true` if two access paths overlap (one is a prefix of the other).
    pub fn overlaps(&self, other: &AccessPath) -> bool {
        self.is_prefix_of(other) || other.is_prefix_of(self)
    }

    /// Computes the Lowest Common Ancestor (LCA) path of two access paths.
    pub fn lca(&self, other: &AccessPath) -> Self {
        let mut common = Vec::new();
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if a == b {
                common.push(a.clone());
            } else {
                break;
            }
        }
        AccessPath(common)
    }
}

/// Unique identifier for a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

/// Tracks byte/character offsets mapping JIT IR instructions to the Python source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    /// Offset inside the source file.
    pub offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_composite() {
        assert!(
            Type::Struct("S".to_string()).is_composite(),
            "Struct should be composite"
        );
        assert!(
            Type::Tuple(vec![Type::I64]).is_composite(),
            "Tuple should be composite"
        );
        assert!(
            Type::Enum("E".to_string()).is_composite(),
            "Enum should be composite"
        );
        assert!(
            Type::Array(Box::new(Type::I64), Some(10)).is_composite(),
            "Sized array should be composite"
        );
        assert!(!Type::I64.is_composite(), "i64 should not be composite");
        assert!(
            !Type::Array(Box::new(Type::I64), None).is_composite(),
            "Unsized array should not be composite"
        );
    }
}
