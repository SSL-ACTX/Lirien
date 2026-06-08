#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

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
    Array(Box<Type>, Option<usize>),
    Buffer(Box<Type>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Type>),
    Pointer(Box<Type>),
    FnPointer(Vec<Type>, Box<Type>),
    Closure(String, Vec<Type>, Box<Type>),
    Refined(Box<Type>, String),
    Literal(Box<Type>, i64),
    Unknown,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I8 => write!(f, "i8"),
            Type::U8 => write!(f, "u8"),
            Type::I16 => write!(f, "i16"),
            Type::U16 => write!(f, "u16"),
            Type::I32 => write!(f, "i32"),
            Type::U32 => write!(f, "u32"),
            Type::I64 => write!(f, "i64"),
            Type::U64 => write!(f, "u64"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Bool => write!(f, "bool"),
            Type::F32X4 => write!(f, "f32x4"),
            Type::I32X4 => write!(f, "i32x4"),
            Type::F64X2 => write!(f, "f64x2"),
            Type::I64X2 => write!(f, "i64x2"),
            Type::Array(t, s) => match s {
                Some(size) => write!(f, "Array<{}, {}>", t, size),
                None => write!(f, "Array<{}>", t),
            },
            Type::Buffer(t) => write!(f, "Buffer<{}>", t),
            Type::Struct(name) => write!(f, "Struct<{}>", name),
            Type::Enum(name) => write!(f, "Enum<{}>", name),
            Type::Tuple(types) => {
                let inner: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "Tuple<{}>", inner.join(", "))
            }
            Type::Pointer(t) => write!(f, "Box<{}>", t),
            Type::FnPointer(args, ret) => {
                let inner: Vec<String> = args.iter().map(|t| t.to_string()).collect();
                write!(f, "Fn({}) -> {}", inner.join(", "), ret)
            }
            Type::Closure(name, args, ret) => {
                let inner: Vec<String> = args.iter().map(|t| t.to_string()).collect();
                write!(f, "Closure {}({}) -> {}", name, inner.join(", "), ret)
            }
            Type::Refined(inner, constraint) => write!(f, "Refined<{}, {}>", inner, constraint),
            Type::Literal(inner, val) => write!(f, "Literal<{}, {}>", inner, val),
            Type::Unknown => write!(f, "unknown"),
        }
    }
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
            Type::F32 | Type::F32X4 => true,
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
            | Type::I64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_int(),
            _ => false,
        }
    }

    pub fn int_bit_width(&self) -> Option<u32> {
        match self {
            Type::I8 | Type::U8 => Some(8),
            Type::I16 | Type::U16 => Some(16),
            Type::I32 | Type::U32 => Some(32),
            Type::I64 | Type::U64 => Some(64),
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
            | Type::FnPointer(_, _)
            | Type::Closure(_, _, _) => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_pointer_like(),
            _ => false,
        }
    }

    pub fn is_simd(&self) -> bool {
        match self {
            Type::F32X4 | Type::I32X4 | Type::F64X2 | Type::I64X2 => true,
            Type::Refined(inner, _) | Type::Literal(inner, _) => inner.is_simd(),
            _ => false,
        }
    }

    pub fn is_composite(&self) -> bool {
        match self {
            Type::Struct(_) | Type::Tuple(_) | Type::Enum(_) => true,
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
            Type::I64 | Type::U64 | Type::F64 | Type::Pointer(_) => 8,
            Type::F32X4 | Type::I32X4 | Type::F64X2 | Type::I64X2 => 16,
            Type::FnPointer(_, _) | Type::Closure(_, _, _) => 8,
            Type::Array(inner, Some(s)) => s * inner.size(struct_layouts),
            Type::Array(_, None) => 8, // Pointer to array
            Type::Buffer(_) => 16,     // Fat Pointer: (ptr, len)
            Type::Struct(name) => {
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
            Type::I64 | Type::U64 | Type::F64 | Type::Pointer(_) => 8,
            Type::F32X4 | Type::I32X4 | Type::F64X2 | Type::I64X2 => 16,
            Type::FnPointer(_, _) | Type::Closure(_, _, _) => 8,
            Type::Array(inner, Some(_)) => inner.align(struct_layouts),
            Type::Array(_, None) => 8,
            Type::Struct(name) => {
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

impl fmt::Display for PathElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathElement::Field(i) => write!(f, ".{}", i),
            PathElement::Index(v) => write!(f, "[{}]", v),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessPath(pub Vec<PathElement>);

impl fmt::Display for AccessPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for el in &self.0 {
            write!(f, "{}", el)?;
        }
        Ok(())
    }
}

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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub offset: usize,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "offset {}", self.offset)
    }
}

#[macro_export]
macro_rules! lila_instructions {
    ($mac:ident) => {
        $mac! {
            // Integer Arithmetic
            Add(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = add {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            Sub(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = sub {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            Mul(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = mul {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            SDiv(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = sdiv {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            UDiv(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = udiv {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            SRem(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = srem {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            URem(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = urem {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },

            // Bitwise
            And(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = and {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            Or(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = or {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            Xor(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = xor {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            Shl(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = shl {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            LShr(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = lshr {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            AShr(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = ashr {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Bitwise
            },
            Not(dest: Value, src: Value) {
                display: "{} = not {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Bitwise
            },

            // Float Arithmetic
            FAdd(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fadd {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Float
            },
            FSub(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fsub {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Float
            },
            FMul(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fmul {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Float
            },
            FDiv(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fdiv {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Float
            },
            FSqrt(dest: Value, src: Value) {
                display: "{} = sqrt {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },

            // SIMD
            SIMDSplat(dest: Value, src: Value) {
                display: "{} = splat {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: SIMD
            },
            SIMDExtractLane(dest: Value, vector: Value, lane: usize) {
                display: "{} = extract_lane {}[{}]",
                def: Some(*dest),
                uses: [*vector],
                side_effects: false,
                category: SIMD
            },
            SIMDInsertLane(dest: Value, vector: Value, scalar: Value, lane: usize) {
                display: "{} = insert_lane {}[{}] <- {}",
                def: Some(*dest),
                uses: [*vector, *scalar],
                side_effects: false,
                category: SIMD
            },
            FSin(dest: Value, src: Value) {
                display: "{} = sin {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FCos(dest: Value, src: Value) {
                display: "{} = cos {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FPow(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = pow {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Float
            },

            // Comparisons
            Eq(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = eq {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            Ne(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = ne {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            SLt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = slt {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            SLe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = sle {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            SGt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = sgt {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            SGe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = sge {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            ULt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = ult {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            ULe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = ule {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            UGt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = ugt {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            UGe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = uge {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            FLt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = flt {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            FLe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fle {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            FGt(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fgt {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },
            FGe(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = fge {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Comparison
            },

            IToF(dest: Value, src: Value, ty: Type) {
                display: "{} = itof {} to {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Conversion
            },
            FToI(dest: Value, src: Value, ty: Type) {
                display: "{} = ftoi {} to {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Conversion
            },
            FConv(dest: Value, src: Value, ty: Type) {
                display: "{} = fconv {} to {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Conversion
            },

            ConstInt(dest: Value, val: i64) {
                display: "{} = const_int {}",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Constant
            },
            ConstFloat(dest: Value, val: f64) {
                display: "{} = const_float {}",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Constant
            },
            Assign(dest: Value, src: Value) {
                display: "{} = assign {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Arithmetic
            },
            Jump(target: BlockId) {
                display: "jump {}",
                def: None,
                uses: [],
                side_effects: true,
                category: ControlFlow
            },
            Branch(cond: Value, true_block: BlockId, false_block: BlockId) {
                display: "br {}, {}, {}",
                def: None,
                uses: [*cond],
                side_effects: true,
                category: ControlFlow
            },
            Return(val: Option<Value>) {
                display: "ret",
                def: None,
                uses: [],
                side_effects: true,
                category: ControlFlow
            },
            Phi(dest: Value, mappings: HashMap<BlockId, Value>) {
                display: "{} = phi",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: ControlFlow
            },
            Call(dest: Value, func: String, args: Vec<Value>) {
                display: "{} = call {}(...)",
                def: Some(*dest),
                uses: [],
                side_effects: true,
                category: Arithmetic
            },
            ArrayLoad(dest: Value, arr: Value, idx: Value) {
                display: "{} = load {}[{}]",
                def: Some(*dest),
                uses: [*arr, *idx],
                side_effects: false,
                category: Memory
            },
            ArrayStore(dest: Value, arr: Value, idx: Value, val: Value, ty: Type) {
                display: "{} = store {}[{}] <- {} (as {})",
                def: Some(*dest),
                uses: [*arr, *idx, *val],
                side_effects: true,
                category: Memory
            },
            BufferLoad(dest: Value, buf: Value, idx: Value) {
                display: "{} = bufload {}[{}]",
                def: Some(*dest),
                uses: [*buf, *idx],
                side_effects: false,
                category: Memory
            },
            BufferStore(dest: Value, buf: Value, idx: Value, val: Value, ty: Type) {
                display: "{} = bufstore {}[{}] <- {} (as {})",
                def: Some(*dest),
                uses: [*buf, *idx, *val],
                side_effects: true,
                category: Memory
            },
            BufferLen(dest: Value, buf: Value) {
                display: "{} = buflen {}",
                def: Some(*dest),
                uses: [*buf],
                side_effects: false,
                category: Memory
            },
            StructCreate(dest: Value, name: String, args: Vec<Value>) {
                display: "{} = struct {} (...)",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Memory
            },
            StructLoad(dest: Value, obj: Value, offset: usize) {
                display: "{} = load {} + {}",
                def: Some(*dest),
                uses: [*obj],
                side_effects: false,
                category: Memory
            },
            StructOffset(dest: Value, obj: Value, offset: usize) {
                display: "{} = offset {} + {}",
                def: Some(*dest),
                uses: [*obj],
                side_effects: false,
                category: Memory
            },
            StructSet(dest: Value, obj: Value, offset: usize, val: Value, ty: Type) {
                display: "{} = set {} + {} <- {} (as {})",
                def: Some(*dest),
                uses: [*obj, *val],
                side_effects: true,
                category: Memory
            },

            // Enums
            EnumCreate(dest: Value, name: String, tag_idx: usize, payload: Option<Value>) {
                display: "{} = enum {}::{}",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Enum
            },
            EnumIsVariant(dest: Value, obj: Value, tag_idx: usize) {
                display: "{} = is_variant {} == {}",
                def: Some(*dest),
                uses: [*obj],
                side_effects: false,
                category: Enum
            },
            EnumGetTag(dest: Value, obj: Value) {
                display: "{} = get_tag {}",
                def: Some(*dest),
                uses: [*obj],
                side_effects: false,
                category: Enum
            },
            EnumExtract(dest: Value, obj: Value, tag_idx: usize) {
                display: "{} = extract_variant {} as {}",
                def: Some(*dest),
                uses: [*obj],
                side_effects: false,
                category: Enum
            },

            Match(selector: Value, cases: HashMap<usize, BlockId>, default: BlockId, is_strict: bool) {
                display: "match {}",
                def: None,
                uses: [*selector],
                side_effects: true,
                category: ControlFlow
            },

            // Tuples
            TupleCreate(dest: Value, elts: Vec<Value>) {
                display: "{} = tuple(...)",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Tuple
            },
            TupleExtract(dest: Value, tuple_val: Value, index: usize) {
                display: "{} = extract {}[{}] (tuple)",
                def: Some(*dest),
                uses: [*tuple_val],
                side_effects: false,
                category: Tuple
            },

            // Heap
            Alloc(dest: Value, ty: Type) {
                display: "{} = alloc {}",
                def: Some(*dest),
                uses: [],
                side_effects: true,
                category: Memory
            },
            PointerLoad(dest: Value, ptr: Value) {
                display: "{} = pload *{}",
                def: Some(*dest),
                uses: [*ptr],
                side_effects: false,
                category: Memory
            },
            PointerStore(ptr: Value, val: Value) {
                display: "pstore *{} = {}",
                def: None,
                uses: [*ptr, *val],
                side_effects: true,
                category: Memory
            },

            Lambda(dest: Value, name: String, captures: Vec<Value>) {
                display: "{} = lambda {}(...)",
                def: Some(*dest),
                uses: [],
                side_effects: true,
                category: HigherOrder
            },
            IndirectCall(dest: Value, ptr: Value, args: Vec<Value>) {
                display: "{} = icall {}(...)",
                def: Some(*dest),
                uses: [*ptr],
                side_effects: true,
                category: HigherOrder
            },

            ParallelFor(index_var: Value, start: Value, stop: Value, step: Value, body_block: BlockId, exit_block: BlockId, captures: Vec<Value>) {
                display: "pfor",
                def: None,
                uses: [*start, *stop, *step],
                side_effects: true,
                category: Parallel
            },
            Nop() {
                display: "nop",
                def: None,
                uses: [],
                side_effects: false,
                category: Arithmetic
            }
        }
    }
}

macro_rules! generate_instruction_kind {
    ($($name:ident($($arg_name:ident : $arg_ty:ty),*) { $($rest:tt)* }),* $(,)?) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum InstructionKind {
            $($name($($arg_ty),*)),*
        }
    }
}

lila_instructions!(generate_instruction_kind);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub kind: InstructionKind,
    pub location: Option<SourceLocation>,
    pub constraints: Vec<String>,
}

impl Instruction {
    pub fn new(kind: InstructionKind, location: Option<SourceLocation>) -> Self {
        Self {
            kind,
            location,
            constraints: Vec::new(),
        }
    }

    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn add_constraint(&mut self, constraint: String) -> &mut Self {
        self.constraints.push(constraint);
        self
    }

    #[allow(unused_variables)]
    pub fn get_def(&self) -> Option<Value> {
        macro_rules! match_def {
            ($($name:ident($($arg_name:ident : $arg_ty:ty),*) {
                display: $display:expr,
                def: $def:expr,
                uses: [$($uses:expr),*],
                side_effects: $side_effects:expr,
                category: $category:ident
            }),* $(,)?) => {
                match &self.kind {
                    $(InstructionKind::$name($($arg_name),*) => $def),*
                }
            }
        }
        lila_instructions!(match_def)
    }

    #[allow(unused_variables)]
    pub fn get_uses(&self) -> Vec<Value> {
        macro_rules! match_uses {
            ($($name:ident($($arg_name:ident : $arg_ty:ty),*) {
                display: $display:expr,
                def: $def:expr,
                uses: [$($uses:expr),*],
                side_effects: $side_effects:expr,
                category: $category:ident
            }),* $(,)?) => {
                match &self.kind {
                    $(InstructionKind::$name($($arg_name),*) => {
                        let mut operands = Vec::new();
                        $(operands.push($uses);)*
                        // Special handling for instructions with Vec<Value>
                        match &self.kind {
                            InstructionKind::Call(_, _, args) => {
                                for v in args { operands.push(*v); }
                            }
                            InstructionKind::StructCreate(_, _, args) => {
                                for v in args { operands.push(*v); }
                            }
                            InstructionKind::TupleCreate(_, elts) => {
                                for v in elts { operands.push(*v); }
                            }
                            InstructionKind::Lambda(_, _, captures) => {
                                for v in captures { operands.push(*v); }
                            }
                            InstructionKind::IndirectCall(_, _, args) => {
                                for v in args { operands.push(*v); }
                            }
                            InstructionKind::ParallelFor(_, _, _, _, _, _, captures) => {
                                for v in captures { operands.push(*v); }
                            }
                            InstructionKind::EnumCreate(_, _, _, Some(v)) => {
                                operands.push(*v);
                            }
                            InstructionKind::Return(Some(v)) => {
                                operands.push(*v);
                            }
                            _ => {}
                        }
                        operands
                    }),*
                }
            }
        }
        lila_instructions!(match_uses)
    }

    #[allow(unused_variables)]
    pub fn has_side_effects(&self) -> bool {
        macro_rules! match_side_effects {
            ($($name:ident($($arg_name:ident : $arg_ty:ty),*) {
                display: $display:expr,
                def: $def:expr,
                uses: [$($uses:expr),*],
                side_effects: $side_effects:expr,
                category: $category:ident
            }),* $(,)?) => {
                match &self.kind {
                    $(InstructionKind::$name(..) => $side_effects),*
                }
            }
        }
        lila_instructions!(match_side_effects)
    }

    #[allow(unused_variables)]
    pub fn visit<V: InstructionVisitor<R>, R>(&self, visitor: &mut V) -> R {
        macro_rules! match_visit {
            ($($name:ident($($arg_name:ident : $arg_ty:ty),*) {
                display: $display:expr,
                def: $def:expr,
                uses: [$($uses:expr),*],
                side_effects: $side_effects:expr,
                category: $category:ident
            }),* $(,)?) => {
                match &self.kind {
                    $(InstructionKind::$name($($arg_name),*) => visitor.$name($($arg_name),*)),*
                }
            }
        }
        lila_instructions!(match_visit)
    }
}

macro_rules! define_visitor_methods {
    ($($name:ident($($arg_name:ident : $arg_ty:ty),*) {
        display: $display:expr,
        def: $def:expr,
        uses: [$($uses:expr),*],
        side_effects: $side_effects:expr,
        category: $category:ident
    }),* $(,)?) => {
        $(
            #[allow(non_snake_case, clippy::ptr_arg, clippy::too_many_arguments)]
            fn $name(&mut self, $($arg_name: &$arg_ty),*) -> R;
        )*
    }
}

pub trait InstructionVisitor<R> {
    lila_instructions!(define_visitor_methods);
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_str = self
            .location
            .map(|l| format!("  ; at {}", l))
            .unwrap_or_default();
        let constraints_str = if self.constraints.is_empty() {
            String::new()
        } else {
            format!("  ; constraints: [{}]", self.constraints.join(", "))
        };

        match &self.kind {
            InstructionKind::Add(d, l, r) => write!(
                f,
                "  {} = add {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Sub(d, l, r) => write!(
                f,
                "  {} = sub {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Mul(d, l, r) => write!(
                f,
                "  {} = mul {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::SDiv(d, l, r) => write!(
                f,
                "  {} = sdiv {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::UDiv(d, l, r) => write!(
                f,
                "  {} = udiv {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::SRem(d, l, r) => write!(
                f,
                "  {} = srem {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::URem(d, l, r) => write!(
                f,
                "  {} = urem {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::And(d, l, r) => write!(
                f,
                "  {} = and {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Or(d, l, r) => {
                write!(f, "  {} = or {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::Xor(d, l, r) => write!(
                f,
                "  {} = xor {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Shl(d, l, r) => write!(
                f,
                "  {} = shl {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::LShr(d, l, r) => write!(
                f,
                "  {} = lshr {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::AShr(d, l, r) => write!(
                f,
                "  {} = ashr {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Not(d, s) => {
                write!(f, "  {} = not {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FAdd(d, l, r) => write!(
                f,
                "  {} = fadd {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FSub(d, l, r) => write!(
                f,
                "  {} = fsub {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FMul(d, l, r) => write!(
                f,
                "  {} = fmul {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FDiv(d, l, r) => write!(
                f,
                "  {} = fdiv {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FSqrt(d, s) => {
                write!(f, "  {} = sqrt {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::SIMDSplat(d, s) => {
                write!(f, "  {} = splat {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::SIMDExtractLane(d, v, l) => write!(
                f,
                "  {} = extract_lane {}[{}] {}{}",
                d, v, l, loc_str, constraints_str
            ),
            InstructionKind::SIMDInsertLane(d, v, s, l) => write!(
                f,
                "  {} = insert_lane {}[{}] <- {} {}{}",
                d, v, l, s, loc_str, constraints_str
            ),
            InstructionKind::FSin(d, s) => {
                write!(f, "  {} = sin {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FCos(d, s) => {
                write!(f, "  {} = cos {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FPow(d, b, e) => write!(
                f,
                "  {} = pow {}, {}{}{}",
                d, b, e, loc_str, constraints_str
            ),
            InstructionKind::Eq(d, l, r) => {
                write!(f, "  {} = eq {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::Ne(d, l, r) => {
                write!(f, "  {} = ne {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::SLt(d, l, r) => write!(
                f,
                "  {} = slt {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::SLe(d, l, r) => write!(
                f,
                "  {} = sle {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::SGt(d, l, r) => write!(
                f,
                "  {} = sgt {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::SGe(d, l, r) => write!(
                f,
                "  {} = sge {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::ULt(d, l, r) => write!(
                f,
                "  {} = ult {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::ULe(d, l, r) => write!(
                f,
                "  {} = ule {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::UGt(d, l, r) => write!(
                f,
                "  {} = ugt {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::UGe(d, l, r) => write!(
                f,
                "  {} = uge {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FLt(d, l, r) => write!(
                f,
                "  {} = flt {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FLe(d, l, r) => write!(
                f,
                "  {} = fle {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FGt(d, l, r) => write!(
                f,
                "  {} = fgt {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::FGe(d, l, r) => write!(
                f,
                "  {} = fge {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::IToF(d, s, t) => write!(
                f,
                "  {} = itof {} to {}{}{}",
                d, s, t, loc_str, constraints_str
            ),
            InstructionKind::FToI(d, s, t) => write!(
                f,
                "  {} = ftoi {} to {}{}{}",
                d, s, t, loc_str, constraints_str
            ),
            InstructionKind::FConv(d, s, t) => write!(
                f,
                "  {} = fconv {} to {}{}{}",
                d, s, t, loc_str, constraints_str
            ),
            InstructionKind::ConstInt(d, v) => {
                write!(f, "  {} = const_int {}{}{}", d, v, loc_str, constraints_str)
            }
            InstructionKind::ConstFloat(d, v) => write!(
                f,
                "  {} = const_float {}{}{}",
                d, v, loc_str, constraints_str
            ),
            InstructionKind::Assign(d, s) => {
                write!(f, "  {} = assign {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::Jump(b) => write!(f, "  jump {}{}{}", b, loc_str, constraints_str),
            InstructionKind::Branch(c, t, e) => {
                write!(f, "  br {}, {}, {}{}{}", c, t, e, loc_str, constraints_str)
            }
            InstructionKind::Return(v) => match v {
                Some(val) => write!(f, "  ret {}{}{}", val, loc_str, constraints_str),
                None => write!(f, "  ret{}{}", loc_str, constraints_str),
            },
            InstructionKind::Phi(d, m) => {
                let mappings: Vec<String> =
                    m.iter().map(|(b, v)| format!("{}: {}", b, v)).collect();
                write!(
                    f,
                    "  {} = phi [{}]{}{}",
                    d,
                    mappings.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::Call(d, func, args) => {
                let args_str: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = call {}({}){}{}",
                    d,
                    func,
                    args_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::ArrayLoad(d, arr, idx) => write!(
                f,
                "  {} = load {}[{}]{}{}",
                d, arr, idx, loc_str, constraints_str
            ),
            InstructionKind::ArrayStore(d, arr, idx, val, ty) => write!(
                f,
                "  {} = store {}[{}] <- {} (as {}){}{}",
                d, arr, idx, val, ty, loc_str, constraints_str
            ),
            InstructionKind::BufferLoad(d, buf, idx) => write!(
                f,
                "  {} = bufload {}[{}]{}{}",
                d, buf, idx, loc_str, constraints_str
            ),
            InstructionKind::BufferStore(d, buf, idx, val, ty) => write!(
                f,
                "  {} = bufstore {}[{}] <- {} (as {}){}{}",
                d, buf, idx, val, ty, loc_str, constraints_str
            ),
            InstructionKind::BufferLen(d, buf) => {
                write!(f, "  {} = buflen {}{}{}", d, buf, loc_str, constraints_str)
            }
            InstructionKind::StructCreate(d, name, args) => {
                let args_str: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = struct {} ({}){}{}",
                    d,
                    name,
                    args_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::StructLoad(d, obj, offset) => write!(
                f,
                "  {} = load {} + {}{}{}",
                d, obj, offset, loc_str, constraints_str
            ),
            InstructionKind::StructOffset(d, obj, offset) => write!(
                f,
                "  {} = offset {} + {}{}{}",
                d, obj, offset, loc_str, constraints_str
            ),
            InstructionKind::StructSet(d, obj, offset, val, ty) => write!(
                f,
                "  {} = set {} + {} <- {} (as {}){}{}",
                d, obj, offset, val, ty, loc_str, constraints_str
            ),
            InstructionKind::EnumCreate(d, name, tag_idx, payload) => {
                if let Some(p) = payload {
                    write!(
                        f,
                        "  {} = enum {}::{} ({}){}{}",
                        d, name, tag_idx, p, loc_str, constraints_str
                    )
                } else {
                    write!(
                        f,
                        "  {} = enum {}::{}{}{}",
                        d, name, tag_idx, loc_str, constraints_str
                    )
                }
            }
            InstructionKind::EnumIsVariant(d, e, tag_idx) => write!(
                f,
                "  {} = is_variant {} == {}{}{}",
                d, e, tag_idx, loc_str, constraints_str
            ),
            InstructionKind::EnumGetTag(d, o) => {
                write!(f, "  {} = get_tag {}{}{}", d, o, loc_str, constraints_str)
            }
            InstructionKind::EnumExtract(d, o, i) => write!(
                f,
                "  {} = extract_variant {} as {}{}{}",
                d, o, i, loc_str, constraints_str
            ),
            InstructionKind::Match(s, cases, default, is_strict) => {
                let mut case_strings: Vec<String> = cases
                    .iter()
                    .map(|(tag, block)| format!("{}: {}", tag, block))
                    .collect();
                case_strings.sort();
                write!(
                    f,
                    "  match {}, default {} [ {} ]{}{}{}",
                    s,
                    default,
                    case_strings.join(", "),
                    if *is_strict { " (strict)" } else { "" },
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::TupleCreate(d, e) => {
                let args_str: Vec<String> = e.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = tuple({}){}{}",
                    d,
                    args_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::TupleExtract(d, t, i) => write!(
                f,
                "  {} = extract {}[{}] (tuple){}{}",
                d, t, i, loc_str, constraints_str
            ),
            InstructionKind::Alloc(d, t) => {
                write!(f, "  {} = alloc {}{}{}", d, t, loc_str, constraints_str)
            }
            InstructionKind::PointerLoad(d, p) => {
                write!(f, "  {} = pload *{}{}{}", d, p, loc_str, constraints_str)
            }
            InstructionKind::PointerStore(p, v) => {
                write!(f, "  pstore *{} = {}{}{}", p, v, loc_str, constraints_str)
            }
            InstructionKind::Lambda(d, name, args) => {
                let caps: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = lambda {}({}){}{}",
                    d,
                    name,
                    caps.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::IndirectCall(d, ptr, args) => {
                let args_str: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = icall {}({}){}{}",
                    d,
                    ptr,
                    args_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::ParallelFor(
                index_var,
                start,
                stop,
                step,
                body_block,
                exit_block,
                captures,
            ) => {
                let captures_str: Vec<String> = captures.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  pfor {} in range({}, {}, {}) body: {:?}, exit: {:?}, captures: [{}]",
                    index_var,
                    start,
                    stop,
                    step,
                    body_block,
                    exit_block,
                    captures_str.join(", ")
                )
            }
            InstructionKind::Nop() => write!(f, "  nop{}{}", loc_str, constraints_str),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: BlockId,
    pub value_count: usize,
    pub block_count: usize,
    pub arg_count: usize,
    pub return_type: Type,
    pub ret_refinement: Option<String>,
    pub value_types: HashMap<Value, Type>,
    pub refinements: HashMap<Value, String>,
    pub struct_layouts: HashMap<String, Vec<(String, Type)>>,
    pub enum_layouts: HashMap<String, Vec<(String, Type)>>,
}

impl Function {
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
        }
    }

    pub fn set_refinement(&mut self, val: Value, refinement: String) {
        self.refinements.insert(val, refinement);
    }

    pub fn next_value(&mut self) -> Value {
        let val = Value(self.value_count);
        self.value_count += 1;
        val
    }

    pub fn set_type(&mut self, val: Value, ty: Type) {
        self.value_types.insert(val, ty);
    }

    pub fn get_type(&self, val: Value) -> Type {
        self.value_types.get(&val).cloned().unwrap_or(Type::Unknown)
    }

    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.block_count);
        self.block_count += 1;
        id
    }

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
