//! Definition of Lirien JIT IR instructions.
//!
//! This module defines the complete instruction set of the Lirien Intermediate
//! Representation, including integer/float arithmetic, bitwise operators, memory access,
//! control flow, closures, SIMD, tensors, and verification constraints.
//!
//! Instructions are defined via a declarative macro [`crate::lirien_instructions!`] to keep
//! code generation, serialization, visitation, def-use analysis, and display formatting unified.

use super::types::{BlockId, SourceLocation, Type, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a fused algebraic expression, typically used inside tensor fusion operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusedExpr {
    /// Fused tensor input variable.
    Input(Value),
    /// Fused scalar constant or variable.
    Scalar(Value),
    /// Fused addition.
    Add(Box<FusedExpr>, Box<FusedExpr>),
    /// Fused subtraction.
    Sub(Box<FusedExpr>, Box<FusedExpr>),
    /// Fused multiplication.
    Mul(Box<FusedExpr>, Box<FusedExpr>),
    /// Fused division.
    Div(Box<FusedExpr>, Box<FusedExpr>),
}

#[macro_export]
macro_rules! lirien_instructions {
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
            Abs(dest: Value, src: Value) {
                display: "{} = abs {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Arithmetic
            },
            Neg(dest: Value, src: Value) {
                display: "{} = neg {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Arithmetic
            },
            Min(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = min {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            Max(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = max {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            Avg(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = avg {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            MatMult(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = matmult {}, {}",
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
            FTan(dest: Value, src: Value) {
                display: "{} = tan {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FAsin(dest: Value, src: Value) {
                display: "{} = asin {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FAcos(dest: Value, src: Value) {
                display: "{} = acos {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FAtan(dest: Value, src: Value) {
                display: "{} = atan {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FExp(dest: Value, src: Value) {
                display: "{} = exp {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FLog(dest: Value, src: Value) {
                display: "{} = log {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FLog10(dest: Value, src: Value) {
                display: "{} = log10 {}",
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
            FFloor(dest: Value, src: Value) {
                display: "{} = floor {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FCeil(dest: Value, src: Value) {
                display: "{} = ceil {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FTrunc(dest: Value, src: Value) {
                display: "{} = trunc {}",
                def: Some(*dest),
                uses: [*src],
                side_effects: false,
                category: Float
            },
            FNearest(dest: Value, src: Value) {
                display: "{} = nearest {}",
                def: Some(*dest),
                uses: [*src],
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
            ArraySlice(dest: Value, arr: Value, start_idx: Value, step: Value) {
                display: "{} = slice {}[{}: step {}]",
                def: Some(*dest),
                uses: [*arr, *start_idx, *step],
                side_effects: false,
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
            ListCreate(dest: Value, element_type: Type) {
                display: "{} = list_create (as {})",
                def: Some(*dest),
                uses: [],
                side_effects: true,
                category: Memory
            },
            ListAppend(dest: Value, list: Value, val: Value) {
                display: "{} = list_append {}, {}",
                def: Some(*dest),
                uses: [*list, *val],
                side_effects: true,
                category: Memory
            },
            ListLen(dest: Value, list: Value) {
                display: "{} = list_len {}",
                def: Some(*dest),
                uses: [*list],
                side_effects: false,
                category: Memory
            },
            ListLoad(dest: Value, list: Value, index: Value) {
                display: "{} = list_load {}[{}]",
                def: Some(*dest),
                uses: [*list, *index],
                side_effects: false,
                category: Memory
            },
            ListStore(dest: Value, list: Value, index: Value, val: Value) {
                display: "{} = list_store {}[{}] <- {}",
                def: Some(*dest),
                uses: [*list, *index, *val],
                side_effects: true,
                category: Memory
            },
            ConstStr(dest: Value, val: String) {
                display: "{} = const_str {:?}",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Constant
            },
            StrLen(dest: Value, string: Value) {
                display: "{} = strlen {}",
                def: Some(*dest),
                uses: [*string],
                side_effects: false,
                category: Memory
            },
            StrConcat(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = strconcat {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Memory
            },
            StrCompare(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = strcompare {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Memory
            },
            StrIndex(dest: Value, string: Value, index: Value) {
                display: "{} = strindex {}[{}]",
                def: Some(*dest),
                uses: [*string, *index],
                side_effects: false,
                category: Memory
            },
            StrSlice(dest: Value, string: Value, start: Value, end: Value) {
                display: "{} = strslice {}[{}:{}]",
                def: Some(*dest),
                uses: [*string, *start, *end],
                side_effects: false,
                category: Memory
            },
            TensorLoad(dest: Value, tensor: Value, indices: Vec<Value>) {
                display: "{} = tload {}[...] ",
                def: Some(*dest),
                uses: [*tensor],
                side_effects: false,
                category: Memory
            },
            TensorStore(dest: Value, tensor: Value, indices: Vec<Value>, val: Value) {
                display: "{} = tstore {}[...] <- {}",
                def: Some(*dest),
                uses: [*tensor, *val],
                side_effects: true,
                category: Memory
            },
            TensorAdd(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = tadd {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            TensorSub(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = tsub {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            TensorMul(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = tmul {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            TensorDiv(dest: Value, lhs: Value, rhs: Value) {
                display: "{} = tdiv {}, {}",
                def: Some(*dest),
                uses: [*lhs, *rhs],
                side_effects: false,
                category: Arithmetic
            },
            TensorScalarAdd(dest: Value, tensor: Value, scalar: Value) {
                display: "{} = tsadd {}, {}",
                def: Some(*dest),
                uses: [*tensor, *scalar],
                side_effects: false,
                category: Arithmetic
            },
            TensorScalarSub(dest: Value, tensor: Value, scalar: Value) {
                display: "{} = tssub {}, {}",
                def: Some(*dest),
                uses: [*tensor, *scalar],
                side_effects: false,
                category: Arithmetic
            },
            TensorScalarMul(dest: Value, tensor: Value, scalar: Value) {
                display: "{} = tsmul {}, {}",
                def: Some(*dest),
                uses: [*tensor, *scalar],
                side_effects: false,
                category: Arithmetic
            },
            TensorScalarDiv(dest: Value, tensor: Value, scalar: Value) {
                display: "{} = tsdiv {}, {}",
                def: Some(*dest),
                uses: [*tensor, *scalar],
                side_effects: false,
                category: Arithmetic
            },
            TensorSum(dest: Value, tensor: Value) {
                display: "{} = tsum {}",
                def: Some(*dest),
                uses: [*tensor],
                side_effects: false,
                category: Arithmetic
            },
            TensorMax(dest: Value, tensor: Value) {
                display: "{} = tmax {}",
                def: Some(*dest),
                uses: [*tensor],
                side_effects: false,
                category: Arithmetic
            },
            TensorMin(dest: Value, tensor: Value) {
                display: "{} = tmin {}",
                def: Some(*dest),
                uses: [*tensor],
                side_effects: false,
                category: Arithmetic
            },
            TensorDim(dest: Value, tensor: Value, index: usize) {
                display: "{} = tdim {}[{}]",
                def: Some(*dest),
                uses: [*tensor],
                side_effects: false,
                category: Memory
            },
            TensorBroadcast(dest: Value, src: Value, target_dims: Vec<Value>) {
                display: "{} = tbroadcast {} to [...]",
                def: Some(*dest),
                uses: [*src],
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
            EnumAsVariant(dest: Value, obj: Value, tag_idx: usize) {
                display: "{} = as_variant {} == {}",
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
            TensorFused(dest: Value, inputs: Vec<Value>, expr: FusedExpr) {
                display: "{} = tfused (...) ",
                def: Some(*dest),
                uses: [],
                side_effects: false,
                category: Arithmetic
            },
            Assert(test: Value, msg: Option<String>) {
                display: "assert {} (msg: {:?})",
                def: None,
                uses: [*test],
                side_effects: true,
                category: Arithmetic
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
        /// Represents the kind of instruction and its typed fields in Lirien JIT IR.
        ///
        /// This enum is macro-generated from the list of instruction definitions
        /// in [`crate::lirien_instructions!`].
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum InstructionKind {
            $(
                #[allow(missing_docs)]
                $name($($arg_ty),*)
            ),*
        }
    }
}

lirien_instructions!(generate_instruction_kind);

/// A wrapper struct for a Lirien JIT IR instruction.
///
/// Combines the [`InstructionKind`] payload with source mapping ([`SourceLocation`])
/// and refinement logic verification preconditions (`constraints`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    /// The specific operation and its operands.
    pub kind: InstructionKind,
    /// Location of the corresponding line in Python source code.
    pub location: Option<SourceLocation>,
    /// Refinement type verification predicates or preconditions.
    pub constraints: Vec<String>,
}

impl Instruction {
    /// Creates a new `Instruction` of the specified kind, with optional source location.
    pub fn new(kind: InstructionKind, location: Option<SourceLocation>) -> Self {
        Self {
            kind,
            location,
            constraints: Vec::new(),
        }
    }

    /// Builder pattern method to attach constraints to an instruction.
    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Appends a refinement constraint to this instruction.
    pub fn add_constraint(&mut self, constraint: String) -> &mut Self {
        self.constraints.push(constraint);
        self
    }

    /// Returns the defined SSA value if this instruction defines one, or `None` if it does not.
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
        lirien_instructions!(match_def)
    }

    /// Returns a vector of all SSA values consumed/used as operands by this instruction.
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
                            InstructionKind::TensorLoad(_, _, indices) => {
                                for v in indices { operands.push(*v); }
                            }
                            InstructionKind::TensorStore(_, _, indices, _) => {
                                for v in indices { operands.push(*v); }
                            }
                            InstructionKind::TensorFused(_, inputs, _) => {
                                for v in inputs { operands.push(*v); }
                            }
                            InstructionKind::TensorBroadcast(_, src, dims) => {
                                operands.push(*src);
                                for v in dims { operands.push(*v); }
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
                            InstructionKind::EnumExtract(_, obj, _) => {
                                operands.push(*obj);
                            }
                            InstructionKind::EnumAsVariant(_, obj, _) => {
                                operands.push(*obj);
                            }
                            InstructionKind::EnumIsVariant(_, obj, _) => {
                                operands.push(*obj);
                            }
                            InstructionKind::EnumGetTag(_, obj) => {
                                operands.push(*obj);
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
        lirien_instructions!(match_uses)
    }

    /// Returns `true` if this instruction has side effects (e.g. store, JUMP, call).
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
        lirien_instructions!(match_side_effects)
    }

    /// Helper method to accept a visitor pattern implementing [`InstructionVisitor`].
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
        lirien_instructions!(match_visit)
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

/// Visitor pattern interface for visiting all possible IR instructions.
pub trait InstructionVisitor<R> {
    lirien_instructions!(define_visitor_methods);
}
