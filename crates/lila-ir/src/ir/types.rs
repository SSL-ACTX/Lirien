use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Bool,
    // SIMD 128-bit vectors
    F32X4,
    I32X4,
    F64X2,
    I64X2,
    I8X16,
    U8X16,
    I16X8,
    U16X8,
    Array(Box<Type>, Option<usize>),
    Buffer(Box<Type>),
    Tensor(Box<Type>, Vec<String>),
    Struct(String),
    TypedDict(String),
    NamedTuple(String),
    Enum(String),
    Tuple(Vec<Type>),
    Pointer(Box<Type>),
    NullablePointer(Box<Type>),
    FnPointer(Vec<Type>, Box<Type>, Option<String>),
    Closure(String, Vec<Type>, Box<Type>, Option<String>),
    Refined(Box<Type>, String),
    Literal(Box<Type>, i64),
    Unknown,
}

impl Type {
    pub fn is_float(&self) -> bool {
        match self {
            Type::F32 | Type::F64 | Type::F32X4 | Type::F64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float(),
            _ => false,
        }
    }

    pub fn is_float32(&self) -> bool {
        match self {
            Type::F32 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float32(),
            _ => false,
        }
    }

    pub fn is_float64(&self) -> bool {
        match self {
            Type::F64 | Type::F64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_float64(),
            _ => false,
        }
    }

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

    pub fn is_signed(&self) -> bool {
        match self {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_signed(),
            _ => false,
        }
    }

    pub fn base_type(&self) -> &Type {
        match self {
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.base_type(),
            other => other,
        }
    }

    pub fn is_unsigned(&self) -> bool {
        match self {
            Type::U8 | Type::U16 | Type::U32 | Type::U64 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_unsigned(),
            _ => false,
        }
    }

    pub fn is_pointer_like(&self) -> bool {
        match self {
            Type::Buffer(_)
            | Type::Array(_, _)
            | Type::Pointer(_)
            | Type::NullablePointer(_)
            | Type::FnPointer(..)
            | Type::Closure(..) => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_pointer_like(),
            _ => false,
        }
    }

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

    pub fn is_tensor(&self) -> bool {
        match self {
            Type::Tensor(_, _) => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_tensor(),
            _ => false,
        }
    }

    pub fn is_composite(&self) -> bool {
        match self {
            Type::Struct(_) | Type::TypedDict(_) | Type::NamedTuple(_) | Type::Tuple(_) | Type::Enum(_) => true,
            Type::Array(inner, Some(_)) => {
                // If the inner type is not a primitive, we treat fixed arrays as composite for offsets
                !inner.is_int() && !inner.is_float()
            }
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_composite(),
            _ => false,
        }
    }

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
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.align(struct_layouts),
            _ => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Value(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathElement {
    Field(usize),
    Index(Value),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessPath(pub Vec<PathElement>);

impl AccessPath {
    pub fn extend(&self, el: PathElement) -> Self {
        let mut new_path = self.0.clone();
        new_path.push(el);
        AccessPath(new_path)
    }

    pub fn is_prefix_of(&self, other: &AccessPath) -> bool {
        if self.0.len() > other.0.len() {
            return false;
        }
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }

    pub fn overlaps(&self, other: &AccessPath) -> bool {
        self.is_prefix_of(other) || other.is_prefix_of(self)
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
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
