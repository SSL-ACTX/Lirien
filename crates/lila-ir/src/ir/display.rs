use super::instruction::{Instruction, InstructionKind};
use super::types::{AccessPath, BlockId, PathElement, SourceLocation, Type, Value};
use std::fmt;

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
            Type::I8X16 => write!(f, "i8x16"),
            Type::U8X16 => write!(f, "u8x16"),
            Type::I16X8 => write!(f, "i16x8"),
            Type::U16X8 => write!(f, "u16x8"),
            Type::Array(t, size) => match size {
                Some(size) => write!(f, "Array<{}, {}>", t, size),
                None => write!(f, "Array<{}>", t),
            },
            Type::Buffer(t) => write!(f, "Buffer<{}>", t),
            Type::Tensor(t, dims) => write!(f, "Tensor<{}, {}>", t, dims.join(", ")),
            Type::Struct(name) => write!(f, "Struct<{}>", name),
            Type::Enum(name) => write!(f, "Enum<{}>", name),
            Type::Tuple(types) => {
                let inner: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "Tuple<{}>", inner.join(", "))
            }
            Type::Pointer(t) => write!(f, "Box<{}>", t),
            Type::NullablePointer(t) => write!(f, "Box<{}>?", t),
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

impl fmt::Display for PathElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathElement::Field(i) => write!(f, ".{}", i),
            PathElement::Index(v) => write!(f, "[{}]", v),
        }
    }
}

impl fmt::Display for AccessPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for el in &self.0 {
            write!(f, "{}", el)?;
        }
        Ok(())
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.0)
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "offset {}", self.offset)
    }
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
            InstructionKind::Abs(d, s) => write!(
                f,
                "  {} = abs {}{}{}",
                d, s, loc_str, constraints_str
            ),
            InstructionKind::Neg(d, s) => write!(
                f,
                "  {} = neg {}{}{}",
                d, s, loc_str, constraints_str
            ),
            InstructionKind::Min(d, l, r) => write!(
                f,
                "  {} = min {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Max(d, l, r) => write!(
                f,
                "  {} = max {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::Avg(d, l, r) => write!(
                f,
                "  {} = avg {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::MatMult(d, l, r) => write!(
                f,
                "  {} = matmult {}, {}{}{}",
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
            InstructionKind::TensorLoad(d, t, indices) => {
                let idx_str: Vec<String> = indices.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = tload {}[{}]{}{}",
                    d,
                    t,
                    idx_str.join(", "),
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::TensorStore(d, t, indices, v) => {
                let idx_str: Vec<String> = indices.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = tstore {}[{}] <- {}{}{}",
                    d,
                    t,
                    idx_str.join(", "),
                    v,
                    loc_str,
                    constraints_str
                )
            }
            InstructionKind::TensorAdd(d, l, r) => write!(
                f,
                "  {} = tadd {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::TensorSub(d, l, r) => write!(
                f,
                "  {} = tsub {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::TensorMul(d, l, r) => write!(
                f,
                "  {} = tmul {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::TensorDiv(d, l, r) => write!(
                f,
                "  {} = tdiv {}, {}{}{}",
                d, l, r, loc_str, constraints_str
            ),
            InstructionKind::TensorScalarAdd(d, t, s) => write!(
                f,
                "  {} = tsadd {}, {}{}{}",
                d, t, s, loc_str, constraints_str
            ),
            InstructionKind::TensorScalarSub(d, t, s) => write!(
                f,
                "  {} = tssub {}, {}{}{}",
                d, t, s, loc_str, constraints_str
            ),
            InstructionKind::TensorScalarMul(d, t, s) => write!(
                f,
                "  {} = tsmul {}, {}{}{}",
                d, t, s, loc_str, constraints_str
            ),
            InstructionKind::TensorScalarDiv(d, t, s) => write!(
                f,
                "  {} = tsdiv {}, {}{}{}",
                d, t, s, loc_str, constraints_str
            ),
            InstructionKind::TensorSum(d, t) => write!(
                f,
                "  {} = tsum {}{}{}",
                d, t, loc_str, constraints_str
            ),
            InstructionKind::TensorMax(d, t) => write!(
                f,
                "  {} = tmax {}{}{}",
                d, t, loc_str, constraints_str
            ),
            InstructionKind::TensorMin(d, t) => write!(
                f,
                "  {} = tmin {}{}{}",
                d, t, loc_str, constraints_str
            ),
            InstructionKind::TensorDim(d, t, i) => write!(
                f,
                "  {} = tdim {}[{}] {}{}",
                d, t, i, loc_str, constraints_str
            ),
            InstructionKind::TensorBroadcast(d, s, dims) => {
                let dims_str: Vec<String> = dims.iter().map(|v| v.to_string()).collect();
                write!(
                    f,
                    "  {} = tbroadcast {} to [{}] {}{}",
                    d,
                    s,
                    dims_str.join(", "),
                    loc_str,
                    constraints_str
                )
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
            InstructionKind::EnumAsVariant(d, e, tag_idx) => write!(
                f,
                "  {} = as_variant {} == {}{}{}",
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