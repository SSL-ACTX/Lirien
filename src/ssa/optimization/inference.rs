use crate::ssa::analysis::interval::{analyze, Bound, IntervalAnalysisResults};
use crate::ssa::ir::{Function, Type};

pub fn embed_intervals(func: &mut Function) {
    let analysis = analyze(func);
    for (val, interval) in analysis.intervals {
        let ty = func.get_type(val);
        if ty == Type::Unknown {
            continue;
        }

        let mut constraints = Vec::new();
        let is_float = ty.is_float();
        if is_float {
            // Do not embed float bounds into string refinements to avoid IEEE-754 decimal precision loss.
            // Z3 will still receive the binary-precise float bounds via `assert_derived_intervals`.
            continue;
        }

        if let Bound::Finite(low) = interval.low {
            constraints.push(format!("(>= {{v}} {})", low as i64));
        }
        if let Bound::Finite(high) = interval.high {
            constraints.push(format!("(<= {{v}} {})", high as i64));
        }

        if !constraints.is_empty() {
            let combined = if constraints.len() == 2 {
                format!("(and {} {})", constraints[0], constraints[1])
            } else {
                constraints[0].clone()
            };
            
            if let Some(existing) = func.refinements.get(&val) {
                func.set_refinement(val, format!("(and {} {})", existing, combined));
            } else {
                func.set_refinement(val, combined);
            }
        }
    }
}
