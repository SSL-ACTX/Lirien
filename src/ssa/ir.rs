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
    Owned(Box<Type>),
    Ref(Box<Type>),
    Mut(Box<Type>),
    Array(Box<Type>, Option<usize>),
    Buffer(Box<Type>),
    Struct(String),
    Tuple(Vec<Type>),
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
            Type::Owned(t) => write!(f, "Owned<{}>", t),
            Type::Ref(t) => write!(f, "Ref<{}>", t),
            Type::Mut(t) => write!(f, "Mut<{}>", t),
            Type::Array(t, s) => match s {
                Some(size) => write!(f, "Array<{}, {}>", t, size),
                None => write!(f, "Array<{}>", t),
            },
            Type::Buffer(t) => write!(f, "Buffer<{}>", t),
            Type::Struct(name) => write!(f, "Struct<{}>", name),
            Type::Tuple(types) => {
                let inner: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "Tuple<{}>", inner.join(", "))
            }
            Type::Unknown => write!(f, "unknown"),
        }
    }
}

impl Type {
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    pub fn size(&self, struct_layouts: &HashMap<String, Vec<(String, Type)>>) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Ref(_) | Type::Mut(_) | Type::Owned(_) => 8,
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
            _ => 8,
        }
    }

    pub fn align(&self, struct_layouts: &HashMap<String, Vec<(String, Type)>>) -> usize {
        match self {
            Type::I8 | Type::U8 | Type::Bool => 1,
            Type::I16 | Type::U16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
            Type::Ref(_) | Type::Mut(_) | Type::Owned(_) => 8,
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
            _ => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub usize);

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
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub kind: InstructionKind,
    pub location: Option<SourceLocation>,
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
    Borrow(Value, Value),
    MutBorrow(Value, Value),
    ArrayLoad(Value, Value, Value),
    ArrayStore(Value, Value, Value, Value, Type),
    BufferLoad(Value, Value, Value),
    BufferStore(Value, Value, Value, Value, Type),
    BufferLen(Value, Value),
    StructLoad(Value, Value, usize),
    StructOffset(Value, Value, usize),
    StructSet(Value, Value, usize, Value, Type),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_str = self
            .location
            .map(|l| format!("  ; at {}", l))
            .unwrap_or_default();
        match &self.kind {
            InstructionKind::Add(d, l, r) => write!(f, "  {} = add {}, {}{}", d, l, r, loc_str),
            InstructionKind::Sub(d, l, r) => write!(f, "  {} = sub {}, {}{}", d, l, r, loc_str),
            InstructionKind::Mul(d, l, r) => write!(f, "  {} = mul {}, {}{}", d, l, r, loc_str),
            InstructionKind::SDiv(d, l, r) => write!(f, "  {} = sdiv {}, {}{}", d, l, r, loc_str),
            InstructionKind::UDiv(d, l, r) => write!(f, "  {} = udiv {}, {}{}", d, l, r, loc_str),
            InstructionKind::SRem(d, l, r) => write!(f, "  {} = srem {}, {}{}", d, l, r, loc_str),
            InstructionKind::URem(d, l, r) => write!(f, "  {} = urem {}, {}{}", d, l, r, loc_str),

            InstructionKind::And(d, l, r) => write!(f, "  {} = and {}, {}{}", d, l, r, loc_str),
            InstructionKind::Or(d, l, r) => write!(f, "  {} = or {}, {}{}", d, l, r, loc_str),
            InstructionKind::Xor(d, l, r) => write!(f, "  {} = xor {}, {}{}", d, l, r, loc_str),
            InstructionKind::Shl(d, l, r) => write!(f, "  {} = shl {}, {}{}", d, l, r, loc_str),
            InstructionKind::LShr(d, l, r) => write!(f, "  {} = lshr {}, {}{}", d, l, r, loc_str),
            InstructionKind::AShr(d, l, r) => write!(f, "  {} = ashr {}, {}{}", d, l, r, loc_str),
            InstructionKind::Not(d, s) => write!(f, "  {} = not {}{}", d, s, loc_str),

            InstructionKind::FAdd(d, l, r) => write!(f, "  {} = fadd {}, {}{}", d, l, r, loc_str),
            InstructionKind::FSub(d, l, r) => write!(f, "  {} = fsub {}, {}{}", d, l, r, loc_str),
            InstructionKind::FMul(d, l, r) => write!(f, "  {} = fmul {}, {}{}", d, l, r, loc_str),
            InstructionKind::FDiv(d, l, r) => write!(f, "  {} = fdiv {}, {}{}", d, l, r, loc_str),
            InstructionKind::FSqrt(d, s) => write!(f, "  {} = sqrt {}{}", d, s, loc_str),
            InstructionKind::FSin(d, s) => write!(f, "  {} = sin {}{}", d, s, loc_str),
            InstructionKind::FCos(d, s) => write!(f, "  {} = cos {}{}", d, s, loc_str),
            InstructionKind::FPow(d, b, e) => write!(f, "  {} = pow {}, {}{}", d, b, e, loc_str),

            InstructionKind::Eq(d, l, r) => write!(f, "  {} = eq {}, {}{}", d, l, r, loc_str),
            InstructionKind::Ne(d, l, r) => write!(f, "  {} = ne {}, {}{}", d, l, r, loc_str),
            InstructionKind::SLt(d, l, r) => write!(f, "  {} = slt {}, {}{}", d, l, r, loc_str),
            InstructionKind::SLe(d, l, r) => write!(f, "  {} = sle {}, {}{}", d, l, r, loc_str),
            InstructionKind::SGt(d, l, r) => write!(f, "  {} = sgt {}, {}{}", d, l, r, loc_str),
            InstructionKind::SGe(d, l, r) => write!(f, "  {} = sge {}, {}{}", d, l, r, loc_str),
            InstructionKind::ULt(d, l, r) => write!(f, "  {} = ult {}, {}{}", d, l, r, loc_str),
            InstructionKind::ULe(d, l, r) => write!(f, "  {} = ule {}, {}{}", d, l, r, loc_str),
            InstructionKind::UGt(d, l, r) => write!(f, "  {} = ugt {}, {}{}", d, l, r, loc_str),
            InstructionKind::UGe(d, l, r) => write!(f, "  {} = uge {}, {}{}", d, l, r, loc_str),
            InstructionKind::FLt(d, l, r) => write!(f, "  {} = flt {}, {}{}", d, l, r, loc_str),
            InstructionKind::FLe(d, l, r) => write!(f, "  {} = fle {}, {}{}", d, l, r, loc_str),
            InstructionKind::FGt(d, l, r) => write!(f, "  {} = fgt {}, {}{}", d, l, r, loc_str),
            InstructionKind::FGe(d, l, r) => write!(f, "  {} = fge {}, {}{}", d, l, r, loc_str),

            InstructionKind::IToF(d, s, t) => write!(f, "  {} = itof {} to {}{}", d, s, t, loc_str),
            InstructionKind::FToI(d, s, t) => write!(f, "  {} = ftoi {} to {}{}", d, s, t, loc_str),

            InstructionKind::ConstInt(d, v) => write!(f, "  {} = const_int {}{}", d, v, loc_str),
            InstructionKind::ConstFloat(d, v) => {
                write!(f, "  {} = const_float {}{}", d, v, loc_str)
            }
            InstructionKind::Jump(b) => write!(f, "  jump {}{}", b, loc_str),
            InstructionKind::Branch(c, t, e) => write!(f, "  br {}, {}, {}{}", c, t, e, loc_str),
            InstructionKind::Return(v) => match v {
                Some(val) => write!(f, "  ret {}{}", val, loc_str),
                None => write!(f, "  ret{}", loc_str),
            },
            InstructionKind::Phi(d, m) => {
                let mappings: Vec<String> =
                    m.iter().map(|(b, v)| format!("{}: {}", b, v)).collect();
                write!(f, "  {} = phi [{}]{}", d, mappings.join(", "), loc_str)
            }
            InstructionKind::Call(d, func, args) => {
                let args_str: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = call {}({}){}",
                    d,
                    func,
                    args_str.join(", "),
                    loc_str
                )
            }
            InstructionKind::Borrow(d, s) => write!(f, "  {} = borrow {}{}", d, s, loc_str),
            InstructionKind::MutBorrow(d, s) => write!(f, "  {} = mut_borrow {}{}", d, s, loc_str),
            InstructionKind::ArrayLoad(d, arr, idx) => {
                write!(f, "  {} = load {}[{}]{}", d, arr, idx, loc_str)
            }
            InstructionKind::ArrayStore(d, arr, idx, val, ty) => {
                write!(
                    f,
                    "  {} = store {}[{}] <- {} (as {}){}",
                    d, arr, idx, val, ty, loc_str
                )
            }
            InstructionKind::BufferLoad(d, buf, idx) => {
                write!(f, "  {} = bufload {}[{}]{}", d, buf, idx, loc_str)
            }
            InstructionKind::BufferStore(d, buf, idx, val, ty) => {
                write!(
                    f,
                    "  {} = bufstore {}[{}] <- {} (as {}){}",
                    d, buf, idx, val, ty, loc_str
                )
            }
            InstructionKind::BufferLen(d, buf) => {
                write!(f, "  {} = buflen {}{}", d, buf, loc_str)
            }
            InstructionKind::StructLoad(d, obj, offset) => {
                write!(f, "  {} = load {} + {}{}", d, obj, offset, loc_str)
            }
            InstructionKind::StructOffset(d, obj, offset) => {
                write!(f, "  {} = offset {} + {}{}", d, obj, offset, loc_str)
            }
            InstructionKind::StructSet(d, obj, offset, val, ty) => {
                write!(
                    f,
                    "  {} = set {} + {} <- {} (as {}){}",
                    d, obj, offset, val, ty, loc_str
                )
            }
        }
    }
}

impl Instruction {
    pub fn new(kind: InstructionKind, location: Option<SourceLocation>) -> Self {
        Self { kind, location }
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
    pub value_types: HashMap<Value, Type>,
    pub refinements: HashMap<Value, String>,
    pub struct_layouts: HashMap<String, Vec<(String, Type)>>,
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
            value_types: HashMap::new(),
            refinements: HashMap::new(),
            struct_layouts: HashMap::new(),
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
