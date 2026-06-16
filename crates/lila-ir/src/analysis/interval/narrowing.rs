use crate::ir::{BlockId, InstructionKind, Value};
use std::collections::HashMap;
use super::types::{Bound, Interval};

pub enum Comparison {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

pub fn find_comparison(
    block: &crate::ir::BasicBlock,
    cond_val: Value,
) -> Option<&crate::ir::Instruction> {
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

pub fn narrow_branch_intervals(
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
