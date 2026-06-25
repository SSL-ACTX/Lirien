//! Abstract interpretation and interval range analysis pass.
//!
//! This module analyzes values inside a [`Function`]'s SSA Control Flow Graph to infer
//! their numeric ranges. The results are used for bound verification and optimizations.

pub mod narrowing;
pub mod ops;
pub mod refinement;
pub mod types;

pub use types::{Bound, Interval, IntervalAnalysisResults};

use crate::ir::{Function, InstructionKind, Value};
use std::collections::HashMap;

/// Performs numeric interval/range analysis on a [`Function`].
///
/// Infer ranges for SSA values by iteratively propagating bounds through instructions
/// (using abstract arithmetic operators) until a fixed point is reached (or 50 iterations pass).
///
/// It initializes argument ranges based on their annotated type width, refines them with Z3 liquid type
/// constraints (if available), and narrows ranges through conditional branch edges.
pub fn analyze(func: &Function) -> IntervalAnalysisResults {

    let mut intervals: HashMap<Value, Interval> = HashMap::new();
    let mut block_narrowing: HashMap<(Value, crate::ir::BlockId), Interval> = HashMap::new();

    // Initialize parameter intervals from types and refinements
    for i in 0..func.arg_count {
        let val = Value(i);
        let ty = func.get_type(val);
        let mut interval = if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if ty.is_unsigned() {
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
            let (l, h) = refinement::parse_refinement_bounds(ref_str);
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
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        block_narrowing.entry((v, block.id))
                    {
                        e.insert(i);
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
                            let (min, max) = if ty.is_unsigned() {
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
                                let (l, h) = refinement::parse_refinement_bounds(ref_str);
                                if let Bound::Finite(lv) = l {
                                    res.low = res.low.max(Bound::Finite(lv));
                                }
                                if let Bound::Finite(hv) = h {
                                    res.high = res.high.min(Bound::Finite(hv));
                                }
                            }
                        } else {
                            // For external calls, try to look up in registry
                            if let Ok(reg) = crate::registry::GLOBAL_REGISTRY.lock() {
                                if let Some(sig) = reg.get(target) {
                                    if let Some(ref_str) = &sig.return_refinement {
                                        let (l, h) = refinement::parse_refinement_bounds(ref_str);
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
                    InstructionKind::Neg(d, s) => {
                        let si = block_narrowing
                            .get(&(*s, block.id))
                            .or_else(|| intervals.get(s));
                        if let Some(si) = si {
                            let mut res = Interval {
                                low: match si.high {
                                    Bound::Finite(h) => Bound::Finite(-h),
                                    Bound::PosInf => Bound::NegInf,
                                    Bound::NegInf => Bound::PosInf,
                                },
                                high: match si.low {
                                    Bound::Finite(l) => Bound::Finite(-l),
                                    Bound::NegInf => Bound::PosInf,
                                    Bound::PosInf => Bound::NegInf,
                                },
                            };
                            res.clamp(func.get_type(*d));
                            update_interval(&mut intervals, *d, res)
                        } else {
                            false
                        }
                    }
                    InstructionKind::Abs(d, s) => {
                        let si = block_narrowing
                            .get(&(*s, block.id))
                            .or_else(|| intervals.get(s));
                        if let Some(si) = si {
                            let low = if si.low >= Bound::Finite(0.0) {
                                si.low
                            } else if si.high <= Bound::Finite(0.0) {
                                match si.high {
                                    Bound::Finite(h) => Bound::Finite(-h),
                                    _ => Bound::Finite(0.0), // Should not happen if high <= 0
                                }
                            } else {
                                Bound::Finite(0.0)
                            };

                            let high = match (si.low, si.high) {
                                (Bound::Finite(l), Bound::Finite(h)) => Bound::Finite(l.abs().max(h.abs())),
                                (Bound::NegInf, _) | (_, Bound::PosInf) => Bound::PosInf,
                                (Bound::Finite(l), _) => Bound::Finite(l.abs()),
                                (_, Bound::Finite(h)) => Bound::Finite(h.abs()),
                                (Bound::PosInf, Bound::NegInf) => Bound::Finite(0.0), // Invalid interval
                            };

                            let mut res = Interval { low, high };
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
                        if let Some(comp_inst) = narrowing::find_comparison(block, *cond) {
                            match &comp_inst.kind {
                                InstructionKind::SLt(_, l, r) | InstructionKind::FLt(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Lt,
                                    );
                                }
                                InstructionKind::SLe(_, l, r) | InstructionKind::FLe(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Le,
                                    );
                                }
                                InstructionKind::SGt(_, l, r) | InstructionKind::FGt(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Gt,
                                    );
                                }
                                InstructionKind::SGe(_, l, r) | InstructionKind::FGe(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Ge,
                                    );
                                }
                                InstructionKind::Eq(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Eq,
                                    );
                                }
                                InstructionKind::Ne(_, l, r) => {
                                    narrowing::narrow_branch_intervals(
                                        &intervals,
                                        *l,
                                        *r,
                                        *t_block,
                                        *f_block,
                                        &mut block_narrowing,
                                        narrowing::Comparison::Ne,
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
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                block_narrowing.entry((v, *target))
                            {
                                e.insert(i);
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
