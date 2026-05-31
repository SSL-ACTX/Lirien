use crate::ssa::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    Finite(f64),
    NegInf,
    PosInf,
}

impl Bound {
    pub fn is_finite(&self) -> bool {
        matches!(self, Bound::Finite(_))
    }

    pub fn min(&self, other: Self) -> Self {
        match (*self, other) {
            (Bound::NegInf, _) | (_, Bound::NegInf) => Bound::NegInf,
            (Bound::PosInf, x) | (x, Bound::PosInf) => x,
            (Bound::Finite(x), Bound::Finite(y)) => Bound::Finite(x.min(y)),
        }
    }

    pub fn max(&self, other: Self) -> Self {
        match (*self, other) {
            (Bound::PosInf, _) | (_, Bound::PosInf) => Bound::PosInf,
            (Bound::NegInf, x) | (x, Bound::NegInf) => x,
            (Bound::Finite(x), Bound::Finite(y)) => Bound::Finite(x.max(y)),
        }
    }
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Bound::NegInf, Bound::NegInf) => Some(std::cmp::Ordering::Equal),
            (Bound::NegInf, _) => Some(std::cmp::Ordering::Less),
            (_, Bound::NegInf) => Some(std::cmp::Ordering::Greater),
            (Bound::PosInf, Bound::PosInf) => Some(std::cmp::Ordering::Equal),
            (Bound::PosInf, _) => Some(std::cmp::Ordering::Greater),
            (_, Bound::PosInf) => Some(std::cmp::Ordering::Less),
            (Bound::Finite(x), Bound::Finite(y)) => x.partial_cmp(y),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub low: Bound,
    pub high: Bound,
}

impl Interval {
    pub fn everything() -> Self {
        Interval {
            low: Bound::NegInf,
            high: Bound::PosInf,
        }
    }

    pub fn from_const(val: f64) -> Self {
        Interval {
            low: Bound::Finite(val),
            high: Bound::Finite(val),
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        Interval {
            low: self.low.min(other.low),
            high: self.high.max(other.high),
        }
    }

    pub fn widen(&self, next: &Self) -> Self {
        let low = if next.low < self.low {
            Bound::NegInf
        } else {
            self.low
        };
        let high = if next.high > self.high {
            Bound::PosInf
        } else {
            self.high
        };
        Interval { low, high }
    }

    pub fn add(&self, other: &Self) -> Self {
        let low = match (self.low, other.low) {
            (Bound::NegInf, _) | (_, Bound::NegInf) => Bound::NegInf,
            (Bound::PosInf, _) | (_, Bound::PosInf) => Bound::PosInf,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(a + b),
        };
        let high = match (self.high, other.high) {
            (Bound::PosInf, _) | (_, Bound::PosInf) => Bound::PosInf,
            (Bound::NegInf, _) | (_, Bound::NegInf) => Bound::NegInf,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(a + b),
        };
        Interval { low, high }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let low = match (self.low, other.high) {
            (Bound::NegInf, _) | (_, Bound::PosInf) => Bound::NegInf,
            (Bound::PosInf, _) | (_, Bound::NegInf) => Bound::PosInf,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(a - b),
        };
        let high = match (self.high, other.low) {
            (Bound::PosInf, _) | (_, Bound::NegInf) => Bound::PosInf,
            (Bound::NegInf, _) | (_, Bound::PosInf) => Bound::NegInf,
            (Bound::Finite(a), Bound::Finite(b)) => Bound::Finite(a - b),
        };
        let mut res = Interval { low, high };
        // If x >= 2 and y == 1, then x - y >= 1.
        if let Bound::Finite(low_x) = self.low {
            if low_x >= 2.0 {
                if let Bound::Finite(1.0) = other.low {
                    if let Bound::Finite(1.0) = other.high {
                        res.low = res.low.max(Bound::Finite(1.0));
                    }
                }
            }
        }
        res
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self.low, self.high, other.low, other.high) {
            (Bound::Finite(a), Bound::Finite(b), Bound::Finite(c), Bound::Finite(d)) => {
                let p1 = a * c;
                let p2 = a * d;
                let p3 = b * c;
                let p4 = b * d;
                let min = p1.min(p2).min(p3).min(p4);
                let max = p1.max(p2).max(p3).max(p4);
                Interval {
                    low: Bound::Finite(min),
                    high: Bound::Finite(max),
                }
            }
            // Handle cases with infinity but known signs
            _ => {
                let mut low = Bound::NegInf;
                let high = Bound::PosInf;

                if self.is_strictly_positive() && other.is_strictly_positive() {
                    // [a, b] * [c, d] where a, c >= 1
                    low = match (self.low, other.low) {
                        (Bound::Finite(a), Bound::Finite(c)) => Bound::Finite(a * c),
                        (Bound::Finite(a), _) if a >= 1.0 => Bound::Finite(a),
                        (_, Bound::Finite(c)) if c >= 1.0 => Bound::Finite(c),
                        _ => Bound::Finite(1.0),
                    };
                } else if self.is_non_negative() && other.is_non_negative() {
                    // [a, b] * [c, d] where a, c >= 0 => [a*c, b*d]
                    low = match (self.low, other.low) {
                        (Bound::Finite(a), Bound::Finite(c)) => Bound::Finite(a * c),
                        _ => Bound::Finite(0.0),
                    };
                } else if self.is_strictly_positive() && other.is_non_negative() {
                    low = Bound::Finite(0.0);
                } else if self.is_non_negative() && other.is_strictly_positive() {
                    low = Bound::Finite(0.0);
                }

                Interval { low, high }
            }
        }
    }

    pub fn is_non_negative(&self) -> bool {
        match self.low {
            Bound::Finite(l) => l >= 0.0,
            Bound::PosInf => true,
            Bound::NegInf => false,
        }
    }

    pub fn clamp(&mut self, ty: Type) {
        if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if matches!(ty, Type::U8 | Type::U16 | Type::U32 | Type::U64) {
                (0.0, ((1u128 << bit_width) as f64) - 1.0)
            } else {
                (
                    -((1u128 << (bit_width - 1)) as f64),
                    ((1u128 << (bit_width - 1)) as f64) - 1.0,
                )
            };
            if let Bound::Finite(l) = self.low {
                if l < min {
                    self.low = Bound::NegInf;
                } else if l > max {
                    self.low = Bound::Finite(max);
                }
            }
            if let Bound::Finite(h) = self.high {
                if h > max {
                    self.high = Bound::PosInf;
                } else if h < min {
                    self.high = Bound::Finite(min);
                }
            }
        }
    }

    pub fn is_strictly_positive(&self) -> bool {
        match self.low {
            Bound::Finite(l) => l > 0.0,
            Bound::PosInf => true,
            Bound::NegInf => false,
        }
    }

    pub fn is_strictly_negative(&self) -> bool {
        match self.high {
            Bound::Finite(h) => h < 0.0,
            Bound::NegInf => true,
            Bound::PosInf => false,
        }
    }
}

pub struct IntervalAnalysisResults {
    pub intervals: HashMap<Value, Interval>,
    pub block_narrowing: HashMap<(Value, BlockId), Interval>,
}

pub fn analyze(func: &Function) -> IntervalAnalysisResults {
    let mut intervals: HashMap<Value, Interval> = HashMap::new();
    let mut block_narrowing: HashMap<(Value, BlockId), Interval> = HashMap::new();

    // Initialize parameter intervals from types and refinements
    for i in 0..func.arg_count {
        let val = Value(i);
        let ty = func.get_type(val);
        let mut interval = if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if matches!(ty, Type::U8 | Type::U16 | Type::U32 | Type::U64) {
                (0.0, ((1u128 << bit_width) as f64) - 1.0)
            } else {
                (
                    -((1u128 << (bit_width - 1)) as f64),
                    ((1u128 << (bit_width - 1)) as f64) - 1.0,
                )
            };
            Interval {
                low: Bound::Finite(min),
                high: Bound::Finite(max),
            }
        } else {
            Interval::everything()
        };

        // Apply refinement bounds if available
        if let Some(ref_str) = func.refinements.get(&val) {
            let (l, h) = parse_refinement_bounds(ref_str);
            if let Bound::Finite(lv) = l {
                interval.low = interval.low.max(Bound::Finite(lv));
            }
            if let Bound::Finite(hv) = h {
                interval.high = interval.high.min(Bound::Finite(hv));
            }
        }
        intervals.insert(val, interval);
    }

    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 50 {
        changed = false;
        iterations += 1;

        // Propagate narrowed intervals from predecessors
        for block in &func.blocks {
            let preds = &block.predecessors;
            if preds.len() == 1 {
                let pred_id = preds[0];
                // Inherit narrowed intervals from the single predecessor
                let pred_narrowing: Vec<(Value, Interval)> = block_narrowing
                    .iter()
                    .filter(|((_, b), _)| *b == pred_id)
                    .map(|((v, _), i)| (*v, i.clone()))
                    .collect();

                for (v, i) in pred_narrowing {
                    if !block_narrowing.contains_key(&(v, block.id)) {
                        block_narrowing.insert((v, block.id), i);
                        changed = true;
                    }
                }
            }

            for inst in &block.instructions {
                let updated = match &inst.kind {
                    InstructionKind::ConstInt(d, val) => {
                        update_interval(&mut intervals, *d, Interval::from_const(*val as f64))
                    }
                    InstructionKind::ConstFloat(d, val) => {
                        update_interval(&mut intervals, *d, Interval::from_const(*val))
                    }
                    InstructionKind::Call(d, target, _) => {
                        let mut res = Interval::everything();
                        let ty = func.get_type(*d);
                        if let Some(bit_width) = ty.int_bit_width() {
                            let (min, max) =
                                if matches!(ty, Type::U8 | Type::U16 | Type::U32 | Type::U64) {
                                    (0.0, ((1u128 << bit_width) as f64) - 1.0)
                                } else {
                                    (
                                        -((1u128 << (bit_width - 1)) as f64),
                                        ((1u128 << (bit_width - 1)) as f64) - 1.0,
                                    )
                                };
                            res = Interval {
                                low: Bound::Finite(min),
                                high: Bound::Finite(max),
                            };
                        }

                        // For recursive calls, use the function's own return refinement
                        if target == &func.name {
                            if let Some(ref_str) = &func.ret_refinement {
                                let (l, h) = parse_refinement_bounds(ref_str);
                                if let Bound::Finite(lv) = l {
                                    res.low = res.low.max(Bound::Finite(lv));
                                }
                                if let Bound::Finite(hv) = h {
                                    res.high = res.high.min(Bound::Finite(hv));
                                }
                            }
                        } else {
                            // For external calls, try to look up in registry
                            if let Ok(reg) = crate::bridge::registry::GLOBAL_REGISTRY.lock() {
                                if let Some(sig) = reg.get(target) {
                                    if let Some(ref_str) = &sig.return_refinement {
                                        let (l, h) = parse_refinement_bounds(ref_str);
                                        if let Bound::Finite(lv) = l {
                                            res.low = res.low.max(Bound::Finite(lv));
                                        }
                                        if let Bound::Finite(hv) = h {
                                            res.high = res.high.min(Bound::Finite(hv));
                                        }
                                    }
                                }
                            }
                        }
                        update_interval(&mut intervals, *d, res)
                    }
                    InstructionKind::Add(d, l, r) | InstructionKind::FAdd(d, l, r) => {
                        let li = block_narrowing
                            .get(&(*l, block.id))
                            .or_else(|| intervals.get(l))
                            .cloned();
                        let ri = block_narrowing
                            .get(&(*r, block.id))
                            .or_else(|| intervals.get(r))
                            .cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.add(&ri);
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Sub(d, l, r) | InstructionKind::FSub(d, l, r) => {
                        let li = block_narrowing
                            .get(&(*l, block.id))
                            .or_else(|| intervals.get(l))
                            .cloned();
                        let ri = block_narrowing
                            .get(&(*r, block.id))
                            .or_else(|| intervals.get(r))
                            .cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.sub(&ri);
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Mul(d, l, r) | InstructionKind::FMul(d, l, r) => {
                        let li = block_narrowing
                            .get(&(*l, block.id))
                            .or_else(|| intervals.get(l))
                            .cloned();
                        let ri = block_narrowing
                            .get(&(*r, block.id))
                            .or_else(|| intervals.get(r))
                            .cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.mul(&ri);
                            // Special optimization for x * x
                            if l == r {
                                match res.low {
                                    Bound::Finite(l) if l < 0.0 => {
                                        res.low = Bound::Finite(0.0);
                                    }
                                    Bound::NegInf => {
                                        res.low = Bound::Finite(0.0);
                                    }
                                    _ => {}
                                }
                            }
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Phi(d, mappings) => {
                        let mut joined: Option<Interval> = None;
                        for (pred_id, src_val) in mappings {
                            let src_int = block_narrowing
                                .get(&(*src_val, *pred_id))
                                .or_else(|| intervals.get(src_val));

                            if let Some(si) = src_int {
                                joined = match joined {
                                    Some(j) => Some(j.join(si)),
                                    None => Some(si.clone()),
                                };
                            }
                        }
                        if let Some(mut new_int) = joined {
                            if iterations > 10 {
                                if let Some(old_int) = intervals.get(d) {
                                    new_int = old_int.widen(&new_int);
                                }
                            }
                            update_interval(&mut intervals, *d, new_int)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Branch(cond, t_block, f_block) => {
                        if let Some(comp_inst) = find_comparison(block, *cond) {
                            match &comp_inst.kind {
                                InstructionKind::SLt(_, l, r) | InstructionKind::FLt(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Lt,
                                    );
                                }
                                InstructionKind::SLe(_, l, r) | InstructionKind::FLe(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Le,
                                    );
                                }
                                InstructionKind::SGt(_, l, r) | InstructionKind::FGt(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Gt,
                                    );
                                }
                                InstructionKind::SGe(_, l, r) | InstructionKind::FGe(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Ge,
                                    );
                                }
                                InstructionKind::Eq(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Eq,
                                    );
                                }
                                InstructionKind::Ne(_, l, r) => {
                                    narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        Comparison::Ne,
                                    );
                                }
                                _ => {}
                            }
                        }
                        false
                    }
                    InstructionKind::Jump(target) => {
                        // Propagate narrowed intervals to jump target
                        let current_narrowing: Vec<(Value, Interval)> = block_narrowing
                            .iter()
                            .filter(|((_, b), _)| *b == block.id)
                            .map(|((v, _), i)| (*v, i.clone()))
                            .collect();

                        for (v, i) in current_narrowing {
                            if !block_narrowing.contains_key(&(v, *target)) {
                                block_narrowing.insert((v, *target), i);
                                changed = true;
                            }
                        }
                        false
                    }
                    _ => false,
                };
                if updated {
                    changed = true;
                }
            }
        }
    }

    IntervalAnalysisResults {
        intervals,
        block_narrowing,
    }
}

enum Comparison {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

fn find_comparison(
    block: &crate::ssa::ir::BasicBlock,
    cond_val: Value,
) -> Option<&crate::ssa::ir::Instruction> {
    block.instructions.iter().find(|i| match &i.kind {
        InstructionKind::SLt(d, _, _)
        | InstructionKind::SLe(d, _, _)
        | InstructionKind::SGt(d, _, _)
        | InstructionKind::SGe(d, _, _)
        | InstructionKind::FLt(d, _, _)
        | InstructionKind::FLe(d, _, _)
        | InstructionKind::FGt(d, _, _)
        | InstructionKind::FGe(d, _, _)
        | InstructionKind::Eq(d, _, _)
        | InstructionKind::Ne(d, _, _) => *d == cond_val,
        _ => false,
    })
}

fn narrow_branch_intervals(
    intervals: &HashMap<Value, Interval>,
    l: Value,
    r: Value,
    t_block: BlockId,
    f_block: BlockId,
    narrowing: &mut HashMap<(Value, BlockId), Interval>,
    comp: Comparison,
) {
    let li = intervals.get(&l).cloned().unwrap_or(Interval::everything());
    let ri = intervals.get(&r).cloned().unwrap_or(Interval::everything());

    match comp {
        Comparison::Lt => {
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = new_li.high.min(Bound::Finite(rv - 1.0)); // < rv => <= rv-1
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = new_li.low.max(Bound::Finite(rv)); // not < rv => >= rv
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Le => {
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = new_li.high.min(Bound::Finite(rv));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = new_li.low.max(Bound::Finite(rv + 1.0)); // not <= rv => >= rv+1
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Gt => {
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = new_li.low.max(Bound::Finite(rv + 1.0)); // > rv => >= rv+1
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = new_li.high.min(Bound::Finite(rv)); // not > rv => <= rv
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Ge => {
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = new_li.low.max(Bound::Finite(rv));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = new_li.high.min(Bound::Finite(rv - 1.0)); // not >= rv => <= rv-1
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Eq => {
            // If l == r, then in t_block, l's interval is narrowed by r's, and vice versa.
            let mut new_li = li.clone();
            new_li.low = new_li.low.max(ri.low);
            new_li.high = new_li.high.min(ri.high);
            narrowing.insert((l, t_block), new_li);

            let mut new_ri = ri.clone();
            new_ri.low = new_ri.low.max(li.low);
            new_ri.high = new_ri.high.min(li.high);
            narrowing.insert((r, t_block), new_ri);
        }
        Comparison::Ne => {
            // Ne is harder to narrow unless it's a constant and we're at a bound.
            if let (Bound::Finite(rv_low), Bound::Finite(rv_high)) = (ri.low, ri.high) {
                if rv_low == rv_high {
                    // if l != rv
                    if li.low == Bound::Finite(rv_low) {
                        let mut new_li = li.clone();
                        new_li.low = Bound::Finite(rv_low + 1.0);
                        narrowing.insert((l, t_block), new_li);
                    } else if li.high == Bound::Finite(rv_high) {
                        let mut new_li = li.clone();
                        new_li.high = Bound::Finite(rv_high - 1.0);
                        narrowing.insert((l, t_block), new_li);
                    }

                    // And for f_block (where l == rv)
                    let mut eq_li = li.clone();
                    eq_li.low = Bound::Finite(rv_low);
                    eq_li.high = Bound::Finite(rv_high);
                    narrowing.insert((l, f_block), eq_li);
                }
            }
        }
    }
}

fn update_interval(
    intervals: &mut HashMap<Value, Interval>,
    val: Value,
    new_int: Interval,
) -> bool {
    if let Some(old_int) = intervals.get(&val) {
        if *old_int != new_int {
            intervals.insert(val, new_int);
            return true;
        }
    } else {
        intervals.insert(val, new_int);
        return true;
    }
    false
}

fn parse_refinement_bounds(ref_str: &str) -> (Bound, Bound) {
    if let Ok((l, h)) = eval_refinement_sexpr(ref_str.trim()) {
        (l, h)
    } else {
        (Bound::NegInf, Bound::PosInf)
    }
}

fn eval_refinement_sexpr(sexpr: &str) -> Result<(Bound, Bound), String> {
    if !sexpr.starts_with('(') {
        return Err("Not an S-expression".to_string());
    }

    let inner = &sexpr[1..sexpr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty S-expression".to_string());
    }

    match parts[0] {
        "and" | "&" => {
            let mut low = Bound::NegInf;
            let mut high = Bound::PosInf;
            for part in &parts[1..] {
                if let Ok((l, h)) = eval_refinement_sexpr(part) {
                    low = low.max(l);
                    high = high.min(h);
                }
            }
            Ok((low, high))
        }
        "or" | "|" => {
            let mut low = Bound::PosInf;
            let mut high = Bound::NegInf;
            for part in &parts[1..] {
                if let Ok((l, h)) = eval_refinement_sexpr(part) {
                    low = low.min(l);
                    high = high.max(h);
                }
            }
            if low == Bound::PosInf || high == Bound::NegInf {
                Ok((Bound::NegInf, Bound::PosInf))
            } else {
                Ok((low, high))
            }
        }
        "=" | "==" => {
            if parts.len() != 3 {
                return Err("= expects 2 args".to_string());
            }
            let val = if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                parts[2].parse::<f64>().ok()
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                parts[1].parse::<f64>().ok()
            } else {
                None
            };
            if let Some(v) = val {
                Ok((Bound::Finite(v), Bound::Finite(v)))
            } else {
                Ok((Bound::NegInf, Bound::PosInf))
            }
        }
        "<" => {
            if parts.len() != 3 {
                return Err("< expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v - 1.0)));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::Finite(v + 1.0), Bound::PosInf));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        "<=" => {
            if parts.len() != 3 {
                return Err("<= expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v)));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::Finite(v), Bound::PosInf));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        ">" => {
            if parts.len() != 3 {
                return Err("> expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::Finite(v + 1.0), Bound::PosInf));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v - 1.0)));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        ">=" => {
            if parts.len() != 3 {
                return Err(">= expects 2 args".to_string());
            }
            if parts[1] == "{v}" || parts[1] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[2].parse::<f64>() {
                    return Ok((Bound::Finite(v), Bound::PosInf));
                }
            } else if parts[2] == "{v}" || parts[2] == "VALUE_PLACEHOLDER" {
                if let Ok(v) = parts[1].parse::<f64>() {
                    return Ok((Bound::NegInf, Bound::Finite(v)));
                }
            }
            Ok((Bound::NegInf, Bound::PosInf))
        }
        _ => Ok((Bound::NegInf, Bound::PosInf)),
    }
}

fn split_sexpr_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut current_start = 0;
    let mut depth = 0;
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            ' ' if depth == 0 => {
                let part = s[current_start..i].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }
    let last_part = s[current_start..].trim();
    if !last_part.is_empty() {
        parts.push(last_part);
    }
    parts
}
