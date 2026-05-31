use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Held(Box<Type>),
    Peek(Box<Type>),
    Hand(Box<Type>),
    Array(Box<Type>, Option<usize>),
    Buffer(Box<Type>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Type>),
    FnPointer(Vec<Type>, Box<Type>),
    Closure(String, Vec<Type>, Box<Type>),
    Refined(Box<Type>, String),
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
            Type::Held(t) => write!(f, "Held<{}>", t),
            Type::Peek(t) => write!(f, "Peek<{}>", t),
            Type::Hand(t) => write!(f, "Hand<{}>", t),
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
            Type::FnPointer(args, ret) => {
                let inner: Vec<String> = args.iter().map(|t| t.to_string()).collect();
                write!(f, "Fn({}) -> {}", inner.join(", "), ret)
            }
            Type::Closure(name, args, ret) => {
                let inner: Vec<String> = args.iter().map(|t| t.to_string()).collect();
                write!(f, "Closure {}({}) -> {}", name, inner.join(", "), ret)
            }
            Type::Refined(inner, constraint) => write!(f, "Refined<{}, {}>", inner, constraint),
            Type::Unknown => write!(f, "unknown"),
        }
    }
}

impl Type {
    pub fn is_float(&self) -> bool {
        match self {
            Type::F32 | Type::F64 => true,
            Type::Refined(inner, _) => inner.is_float(),
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
            | Type::U64 => true,
            Type::Refined(inner, _) => inner.is_int(),
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
            Type::Refined(inner, _) => inner.int_bit_width(),
            _ => None,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            Type::I8 | Type::I16 | Type::I32 | Type::I64 => true,
            Type::Refined(inner, _) => inner.is_signed(),
            _ => false,
        }
    }

    pub fn is_pointer_like(&self) -> bool {
        match self {
            Type::Peek(_)
            | Type::Hand(_)
            | Type::Held(_)
            | Type::Buffer(_)
            | Type::Array(_, _)
            | Type::FnPointer(_, _)
            | Type::Closure(_, _, _) => true,
            Type::Refined(inner, _) => inner.is_pointer_like(),
            _ => false,
        }
    }

    pub fn is_composite(&self) -> bool {
        match self {
            Type::Struct(_) | Type::Tuple(_) | Type::Enum(_) => true,
            Type::Array(_, Some(_)) => true,
            Type::Refined(inner, _) => inner.is_composite(),
            _ => false,
        }
    }

    pub fn size(&self, struct_layouts: &HashMap<String, Vec<(String, Type)>>) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Peek(_)
            | Type::Hand(_)
            | Type::Held(_)
            | Type::FnPointer(_, _)
            | Type::Closure(_, _, _) => 8,
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
            Type::Refined(inner, _) => inner.size(struct_layouts),
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
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Peek(_)
            | Type::Hand(_)
            | Type::Held(_)
            | Type::FnPointer(_, _)
            | Type::Closure(_, _, _) => 8,
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
            Type::Refined(inner, _) => inner.align(struct_layouts),
            _ => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
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
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    pub offset: usize,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "offset {}", self.offset)
    }
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub kind: InstructionKind,
    pub location: Option<SourceLocation>,
    pub constraints: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum InstructionKind {
    // Integer Arithmetic
    Add(Value, Value, Value),
    Sub(Value, Value, Value),
    Mul(Value, Value, Value),
    SDiv(Value, Value, Value),
    UDiv(Value, Value, Value),
    SRem(Value, Value, Value),
    URem(Value, Value, Value),

    // Bitwise
    And(Value, Value, Value),
    Or(Value, Value, Value),
    Xor(Value, Value, Value),
    Shl(Value, Value, Value),
    LShr(Value, Value, Value),
    AShr(Value, Value, Value),
    Not(Value, Value),

    // Float Arithmetic
    FAdd(Value, Value, Value),
    FSub(Value, Value, Value),
    FMul(Value, Value, Value),
    FDiv(Value, Value, Value),
    FSqrt(Value, Value),
    FSin(Value, Value),
    FCos(Value, Value),
    FPow(Value, Value, Value),

    // Comparisons
    Eq(Value, Value, Value),
    Ne(Value, Value, Value),
    SLt(Value, Value, Value),
    SLe(Value, Value, Value),
    SGt(Value, Value, Value),
    SGe(Value, Value, Value),
    ULt(Value, Value, Value),
    ULe(Value, Value, Value),
    UGt(Value, Value, Value),
    UGe(Value, Value, Value),
    FLt(Value, Value, Value),
    FLe(Value, Value, Value),
    FGt(Value, Value, Value),
    FGe(Value, Value, Value),

    IToF(Value, Value, Type),
    FToI(Value, Value, Type),

    ConstInt(Value, i64),
    ConstFloat(Value, f64),
    Jump(BlockId),
    Branch(Value, BlockId, BlockId),
    Return(Option<Value>),
    Phi(Value, HashMap<BlockId, Value>),
    Call(Value, String, Vec<Value>),
    Peek(Value, Value),
    Hand(Value, Value),
    ArrayLoad(Value, Value, Value),
    ArrayStore(Value, Value, Value, Value, Type),
    BufferLoad(Value, Value, Value),
    BufferStore(Value, Value, Value, Value, Type),
    BufferLen(Value, Value),
    StructCreate(Value, String, Vec<Value>),
    StructLoad(Value, Value, usize),
    StructOffset(Value, Value, usize),
    StructSet(Value, Value, usize, Value, Type),

    // Enums
    EnumCreate(Value, String, usize, Option<Value>), // dest, name, tag_idx, payload
    EnumIsVariant(Value, Value, usize),              // dest, enum_val, tag_idx
    EnumExtract(Value, Value, usize),                // dest, enum_val, tag_idx

    // Tuples
    TupleCreate(Value, Vec<Value>),
    TupleExtract(Value, Value, usize), // dest, tuple_val, index

    Lambda(Value, String, Vec<Value>), // dest, func_name, captured_vals
    IndirectCall(Value, Value, Vec<Value>), // dest, fn_ptr_val, args

    Release(Value), // Explicitly release a borrow/permission
    ParallelFor {
        index_var: Value,
        start: Value,
        stop: Value,
        step: Value,
        body_block: BlockId,
        exit_block: BlockId,
        captures: Vec<Value>,
    },
    Nop,
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
            InstructionKind::Add(d, l, r) => {
                write!(
                    f,
                    "  {} = add {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::Sub(d, l, r) => {
                write!(
                    f,
                    "  {} = sub {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::Mul(d, l, r) => {
                write!(
                    f,
                    "  {} = mul {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::SDiv(d, l, r) => {
                write!(
                    f,
                    "  {} = sdiv {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::UDiv(d, l, r) => {
                write!(
                    f,
                    "  {} = udiv {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::SRem(d, l, r) => {
                write!(
                    f,
                    "  {} = srem {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::URem(d, l, r) => {
                write!(
                    f,
                    "  {} = urem {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }

            InstructionKind::And(d, l, r) => {
                write!(
                    f,
                    "  {} = and {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::Or(d, l, r) => {
                write!(f, "  {} = or {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::Xor(d, l, r) => {
                write!(
                    f,
                    "  {} = xor {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::Shl(d, l, r) => {
                write!(
                    f,
                    "  {} = shl {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::LShr(d, l, r) => {
                write!(
                    f,
                    "  {} = lshr {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::AShr(d, l, r) => {
                write!(
                    f,
                    "  {} = ashr {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::Not(d, s) => {
                write!(f, "  {} = not {}{}{}", d, s, loc_str, constraints_str)
            }

            InstructionKind::FAdd(d, l, r) => {
                write!(
                    f,
                    "  {} = fadd {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FSub(d, l, r) => {
                write!(
                    f,
                    "  {} = fsub {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FMul(d, l, r) => {
                write!(
                    f,
                    "  {} = fmul {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FDiv(d, l, r) => {
                write!(
                    f,
                    "  {} = fdiv {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FSqrt(d, s) => {
                write!(f, "  {} = sqrt {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FSin(d, s) => {
                write!(f, "  {} = sin {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FCos(d, s) => {
                write!(f, "  {} = cos {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::FPow(d, b, e) => {
                write!(
                    f,
                    "  {} = pow {}, {}{}{}",
                    d, b, e, loc_str, constraints_str
                )
            }

            InstructionKind::Eq(d, l, r) => {
                write!(f, "  {} = eq {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::Ne(d, l, r) => {
                write!(f, "  {} = ne {}, {}{}{}", d, l, r, loc_str, constraints_str)
            }
            InstructionKind::SLt(d, l, r) => {
                write!(
                    f,
                    "  {} = slt {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::SLe(d, l, r) => {
                write!(
                    f,
                    "  {} = sle {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::SGt(d, l, r) => {
                write!(
                    f,
                    "  {} = sgt {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::SGe(d, l, r) => {
                write!(
                    f,
                    "  {} = sge {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::ULt(d, l, r) => {
                write!(
                    f,
                    "  {} = ult {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::ULe(d, l, r) => {
                write!(
                    f,
                    "  {} = ule {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::UGt(d, l, r) => {
                write!(
                    f,
                    "  {} = ugt {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::UGe(d, l, r) => {
                write!(
                    f,
                    "  {} = uge {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FLt(d, l, r) => {
                write!(
                    f,
                    "  {} = flt {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FLe(d, l, r) => {
                write!(
                    f,
                    "  {} = fle {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FGt(d, l, r) => {
                write!(
                    f,
                    "  {} = fgt {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }
            InstructionKind::FGe(d, l, r) => {
                write!(
                    f,
                    "  {} = fge {}, {}{}{}",
                    d, l, r, loc_str, constraints_str
                )
            }

            InstructionKind::IToF(d, s, t) => {
                write!(
                    f,
                    "  {} = itof {} to {}{}{}",
                    d, s, t, loc_str, constraints_str
                )
            }
            InstructionKind::FToI(d, s, t) => {
                write!(
                    f,
                    "  {} = ftoi {} to {}{}{}",
                    d, s, t, loc_str, constraints_str
                )
            }

            InstructionKind::ConstInt(d, v) => {
                write!(f, "  {} = const_int {}{}{}", d, v, loc_str, constraints_str)
            }
            InstructionKind::ConstFloat(d, v) => {
                write!(
                    f,
                    "  {} = const_float {}{}{}",
                    d, v, loc_str, constraints_str
                )
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
            InstructionKind::Peek(d, s) => {
                write!(f, "  {} = peek {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::Hand(d, s) => {
                write!(f, "  {} = hand {}{}{}", d, s, loc_str, constraints_str)
            }
            InstructionKind::ArrayLoad(d, arr, idx) => {
                write!(
                    f,
                    "  {} = load {}[{}]{}{}",
                    d, arr, idx, loc_str, constraints_str
                )
            }
            InstructionKind::ArrayStore(d, arr, idx, val, ty) => {
                write!(
                    f,
                    "  {} = store {}[{}] <- {} (as {}){}{}",
                    d, arr, idx, val, ty, loc_str, constraints_str
                )
            }
            InstructionKind::BufferLoad(d, buf, idx) => {
                write!(
                    f,
                    "  {} = bufload {}[{}]{}{}",
                    d, buf, idx, loc_str, constraints_str
                )
            }
            InstructionKind::BufferStore(d, buf, idx, val, ty) => {
                write!(
                    f,
                    "  {} = bufstore {}[{}] <- {} (as {}){}{}",
                    d, buf, idx, val, ty, loc_str, constraints_str
                )
            }
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
            InstructionKind::StructLoad(d, obj, offset) => {
                write!(
                    f,
                    "  {} = load {} + {}{}{}",
                    d, obj, offset, loc_str, constraints_str
                )
            }
            InstructionKind::StructOffset(d, obj, offset) => {
                write!(
                    f,
                    "  {} = offset {} + {}{}{}",
                    d, obj, offset, loc_str, constraints_str
                )
            }
            InstructionKind::StructSet(d, obj, offset, val, ty) => {
                write!(
                    f,
                    "  {} = set {} + {} <- {} (as {}){}{}",
                    d, obj, offset, val, ty, loc_str, constraints_str
                )
            }
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
            InstructionKind::EnumIsVariant(d, e, tag_idx) => {
                write!(
                    f,
                    "  {} = is_variant {} == {}{}{}",
                    d, e, tag_idx, loc_str, constraints_str
                )
            }
            InstructionKind::EnumExtract(d, e, tag_idx) => {
                write!(
                    f,
                    "  {} = extract_variant {} as {}{}{}",
                    d, e, tag_idx, loc_str, constraints_str
                )
            }
            InstructionKind::TupleCreate(d, elts) => {
                let args_str: Vec<String> = elts.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = tuple({}){}{}",
                    d,
                    args_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::TupleExtract(d, t, idx) => {
                write!(
                    f,
                    "  {} = extract {}[{}] (tuple){}{}",
                    d, t, idx, loc_str, constraints_str
                )
            }
            InstructionKind::Lambda(d, name, captures) => {
                let caps: Vec<String> = captures.iter().map(|v| v.to_string()).collect();
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
            InstructionKind::Release(v) => {
                write!(f, "  release {}{}{}", v, loc_str, constraints_str)
            }
            InstructionKind::ParallelFor {
                index_var,
                start,
                stop,
                step,
                body_block,
                exit_block,
                captures,
            } => {
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
            InstructionKind::Nop => write!(f, "  nop{}{}", loc_str, constraints_str),
        }
    }
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

    pub fn get_def(&self) -> Option<Value> {
        match &self.kind {
            InstructionKind::Add(d, _, _)
            | InstructionKind::Sub(d, _, _)
            | InstructionKind::Mul(d, _, _)
            | InstructionKind::SDiv(d, _, _)
            | InstructionKind::UDiv(d, _, _)
            | InstructionKind::SRem(d, _, _)
            | InstructionKind::URem(d, _, _)
            | InstructionKind::And(d, _, _)
            | InstructionKind::Or(d, _, _)
            | InstructionKind::Xor(d, _, _)
            | InstructionKind::Shl(d, _, _)
            | InstructionKind::LShr(d, _, _)
            | InstructionKind::AShr(d, _, _)
            | InstructionKind::Not(d, _)
            | InstructionKind::FAdd(d, _, _)
            | InstructionKind::FSub(d, _, _)
            | InstructionKind::FMul(d, _, _)
            | InstructionKind::FDiv(d, _, _)
            | InstructionKind::Eq(d, _, _)
            | InstructionKind::Ne(d, _, _)
            | InstructionKind::SLt(d, _, _)
            | InstructionKind::SLe(d, _, _)
            | InstructionKind::SGt(d, _, _)
            | InstructionKind::SGe(d, _, _)
            | InstructionKind::ULt(d, _, _)
            | InstructionKind::ULe(d, _, _)
            | InstructionKind::UGt(d, _, _)
            | InstructionKind::UGe(d, _, _)
            | InstructionKind::FLt(d, _, _)
            | InstructionKind::FLe(d, _, _)
            | InstructionKind::FGt(d, _, _)
            | InstructionKind::FGe(d, _, _)
            | InstructionKind::ConstInt(d, _)
            | InstructionKind::ConstFloat(d, _)
            | InstructionKind::Phi(d, _)
            | InstructionKind::Call(d, _, _)
            | InstructionKind::Peek(d, _)
            | InstructionKind::Hand(d, _)
            | InstructionKind::ArrayLoad(d, _, _)
            | InstructionKind::ArrayStore(d, _, _, _, _)
            | InstructionKind::StructLoad(d, _, _)
            | InstructionKind::StructOffset(d, _, _)
            | InstructionKind::StructSet(d, _, _, _, _)
            | InstructionKind::StructCreate(d, _, _)
            | InstructionKind::EnumCreate(d, _, _, _)
            | InstructionKind::EnumIsVariant(d, _, _)
            | InstructionKind::EnumExtract(d, _, _)
            | InstructionKind::TupleCreate(d, _)
            | InstructionKind::TupleExtract(d, _, _)
            | InstructionKind::Lambda(d, _, _)
            | InstructionKind::IndirectCall(d, _, _)
            | InstructionKind::BufferLoad(d, _, _)
            | InstructionKind::BufferStore(d, _, _, _, _)
            | InstructionKind::BufferLen(d, _) => Some(*d),
            _ => None,
        }
    }

    pub fn get_uses(&self) -> Vec<Value> {
        let mut operands = Vec::new();
        match &self.kind {
            InstructionKind::Add(_, l, r)
            | InstructionKind::Sub(_, l, r)
            | InstructionKind::Mul(_, l, r)
            | InstructionKind::SDiv(_, l, r)
            | InstructionKind::UDiv(_, l, r)
            | InstructionKind::SRem(_, l, r)
            | InstructionKind::URem(_, l, r)
            | InstructionKind::And(_, l, r)
            | InstructionKind::Or(_, l, r)
            | InstructionKind::Xor(_, l, r)
            | InstructionKind::Shl(_, l, r)
            | InstructionKind::LShr(_, l, r)
            | InstructionKind::AShr(_, l, r)
            | InstructionKind::FAdd(_, l, r)
            | InstructionKind::FSub(_, l, r)
            | InstructionKind::FMul(_, l, r)
            | InstructionKind::FDiv(_, l, r)
            | InstructionKind::Eq(_, l, r)
            | InstructionKind::Ne(_, l, r)
            | InstructionKind::SLt(_, l, r)
            | InstructionKind::SLe(_, l, r)
            | InstructionKind::SGt(_, l, r)
            | InstructionKind::SGe(_, l, r)
            | InstructionKind::ULt(_, l, r)
            | InstructionKind::ULe(_, l, r)
            | InstructionKind::UGt(_, l, r)
            | InstructionKind::UGe(_, l, r)
            | InstructionKind::FLt(_, l, r)
            | InstructionKind::FLe(_, l, r)
            | InstructionKind::FGt(_, l, r)
            | InstructionKind::FGe(_, l, r) => {
                operands.push(*l);
                operands.push(*r);
            }
            InstructionKind::Not(_, s)
            | InstructionKind::FSqrt(_, s)
            | InstructionKind::FSin(_, s)
            | InstructionKind::FCos(_, s)
            | InstructionKind::IToF(_, s, _)
            | InstructionKind::FToI(_, s, _) => {
                operands.push(*s);
            }
            InstructionKind::FPow(_, b, e) => {
                operands.push(*b);
                operands.push(*e);
            }
            InstructionKind::Branch(c, _, _) => {
                operands.push(*c);
            }
            InstructionKind::Return(Some(v)) => {
                operands.push(*v);
            }
            InstructionKind::Phi(_, _) => {
                // Phi uses are handled specially by liveness analysis
                // as they depend on the predecessor block.
            }
            InstructionKind::Call(_, _, args) => {
                for v in args {
                    operands.push(*v);
                }
            }
            InstructionKind::Peek(_, s) | InstructionKind::Hand(_, s) => {
                operands.push(*s);
            }
            InstructionKind::ArrayLoad(_, arr, idx) => {
                operands.push(*arr);
                operands.push(*idx);
            }
            InstructionKind::ArrayStore(_, arr, idx, val, _) => {
                operands.push(*arr);
                operands.push(*idx);
                operands.push(*val);
            }
            InstructionKind::BufferLoad(_, buf, idx) => {
                operands.push(*buf);
                operands.push(*idx);
            }
            InstructionKind::BufferStore(_, buf, idx, val, _) => {
                operands.push(*buf);
                operands.push(*idx);
                operands.push(*val);
            }
            InstructionKind::BufferLen(_, buf) => {
                operands.push(*buf);
            }
            InstructionKind::StructCreate(_, _, args) => {
                for v in args {
                    operands.push(*v);
                }
            }
            InstructionKind::StructLoad(_, obj, _) | InstructionKind::StructOffset(_, obj, _) => {
                operands.push(*obj);
            }
            InstructionKind::StructSet(_, obj, _, val, _) => {
                operands.push(*obj);
                operands.push(*val);
            }
            InstructionKind::EnumCreate(_, _, _, Some(p)) => {
                operands.push(*p);
            }
            InstructionKind::EnumCreate(_, _, _, None) => {}
            InstructionKind::EnumIsVariant(_, obj, _) | InstructionKind::EnumExtract(_, obj, _) => {
                operands.push(*obj);
            }
            InstructionKind::TupleCreate(_, elts) => {
                for v in elts {
                    operands.push(*v);
                }
            }
            InstructionKind::TupleExtract(_, t, _) => {
                operands.push(*t);
            }
            InstructionKind::Lambda(_, _, captures) => {
                for v in captures {
                    operands.push(*v);
                }
            }
            InstructionKind::IndirectCall(_, ptr, args) => {
                operands.push(*ptr);
                for v in args {
                    operands.push(*v);
                }
            }
            InstructionKind::Release(v) => {
                operands.push(*v);
            }
            _ => {}
        }
        operands
    }
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
}

#[derive(Debug, Clone)]
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
