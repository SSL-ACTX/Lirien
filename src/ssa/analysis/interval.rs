use crate::ssa::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Finite(i128),
    NegInf,
    PosInf,
}

impl Bound {
    pub fn is_finite(&self) -> bool {
        matches!(self, Bound::Finite(_))
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

impl Ord for Bound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

    pub fn from_const(val: i64) -> Self {
        Interval {
            low: Bound::Finite(val as i128),
            high: Bound::Finite(val as i128),
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        Interval {
            low: std::cmp::min(self.low, other.low),
            high: std::cmp::max(self.high, other.high),
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
        Interval { low, high }
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self.low, self.high, other.low, other.high) {
            (Bound::Finite(a), Bound::Finite(b), Bound::Finite(c), Bound::Finite(d)) => {
                let p1 = a * c;
                let p2 = a * d;
                let p3 = b * c;
                let p4 = b * d;
                Interval {
                    low: Bound::Finite(std::cmp::min(std::cmp::min(p1, p2), std::cmp::min(p3, p4))),
                    high: Bound::Finite(std::cmp::max(
                        std::cmp::max(p1, p2),
                        std::cmp::max(p3, p4),
                    )),
                }
            }
            _ => Interval::everything(),
        }
    }

    pub fn clamp(&mut self, ty: Type) {
        if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if matches!(ty, Type::U8 | Type::U16 | Type::U32 | Type::U64) {
                (0, (1i128 << bit_width) - 1)
            } else {
                (-(1i128 << (bit_width - 1)), (1i128 << (bit_width - 1)) - 1)
            };
            if let Bound::Finite(l) = self.low {
                if l < min {
                    self.low = Bound::NegInf;
                }
                if l > max {
                    self.low = Bound::Finite(max);
                }
            }
            if let Bound::Finite(h) = self.high {
                if h > max {
                    self.high = Bound::PosInf;
                }
                if h < min {
                    self.high = Bound::Finite(min);
                }
            }
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

    // Initialize parameter intervals
    for i in 0..func.arg_count {
        let val = Value(i);
        let ty = func.get_type(val);
        if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if matches!(ty, Type::U8 | Type::U16 | Type::U32 | Type::U64) {
                (0, (1i128 << bit_width) - 1)
            } else {
                (-(1i128 << (bit_width - 1)), (1i128 << (bit_width - 1)) - 1)
            };
            intervals.insert(
                val,
                Interval {
                    low: Bound::Finite(min),
                    high: Bound::Finite(max),
                },
            );
        } else {
            intervals.insert(val, Interval::everything());
        }
    }

    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 50 {
        changed = false;
        iterations += 1;

        for block in &func.blocks {
            for inst in &block.instructions {
                let updated = match &inst.kind {
                    InstructionKind::ConstInt(d, val) => {
                        update_interval(&mut intervals, *d, Interval::from_const(*val))
                    }
                    InstructionKind::Add(d, l, r) => {
                        let li = intervals.get(l).cloned();
                        let ri = intervals.get(r).cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.add(&ri);
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Sub(d, l, r) => {
                        let li = intervals.get(l).cloned();
                        let ri = intervals.get(r).cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.sub(&ri);
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Mul(d, l, r) => {
                        let li = intervals.get(l).cloned();
                        let ri = intervals.get(r).cloned();
                        if let (Some(li), Some(ri)) = (li, ri) {
                            let mut res = li.mul(&ri);
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
                                InstructionKind::SLt(_, l, r) => {
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
                                InstructionKind::SLe(_, l, r) => {
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
                                InstructionKind::SGt(_, l, r) => {
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
                                InstructionKind::SGe(_, l, r) => {
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
                                _ => {}
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
                new_li.high = std::cmp::min(new_li.high, Bound::Finite(rv - 1));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = std::cmp::max(new_li.low, Bound::Finite(rv));
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Le => {
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = std::cmp::min(new_li.high, Bound::Finite(rv));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = std::cmp::max(new_li.low, Bound::Finite(rv + 1));
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Gt => {
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = std::cmp::max(new_li.low, Bound::Finite(rv + 1));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = std::cmp::min(new_li.high, Bound::Finite(rv));
                narrowing.insert((l, f_block), new_li);
            }
        }
        Comparison::Ge => {
            if let Bound::Finite(rv) = ri.low {
                let mut new_li = li.clone();
                new_li.low = std::cmp::max(new_li.low, Bound::Finite(rv));
                narrowing.insert((l, t_block), new_li);
            }
            if let Bound::Finite(rv) = ri.high {
                let mut new_li = li.clone();
                new_li.high = std::cmp::min(new_li.high, Bound::Finite(rv - 1));
                narrowing.insert((l, f_block), new_li);
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
