use crate::ssa::ir::{BlockId, Function, InstructionKind, Value};
use std::collections::{HashMap, HashSet};

pub struct LivenessAnalysisResults {
    pub live_in: HashMap<BlockId, HashSet<Value>>,
    pub live_out: HashMap<BlockId, HashSet<Value>>,
    pub inst_live_out: HashMap<(usize, usize), HashSet<Value>>,
}

pub fn analyze_liveness(func: &Function) -> LivenessAnalysisResults {
    let mut live_in: HashMap<BlockId, HashSet<Value>> = HashMap::new();
    let mut live_out: HashMap<BlockId, HashSet<Value>> = HashMap::new();

    for block in &func.blocks {
        live_in.insert(block.id, HashSet::new());
        live_out.insert(block.id, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in func.blocks.iter().rev() {
            // 1. Compute live_out: union of live_in of successors,
            // but handling Phi nodes specially (only operands from THIS block are live)
            let mut out = HashSet::new();
            for succ_id in &block.successors {
                if let Some(succ_block) = func.blocks.iter().find(|b| b.id == *succ_id) {
                    // Non-Phi live-ins from successor
                    if let Some(in_set) = live_in.get(succ_id) {
                        for &v in in_set {
                            // Only include if NOT defined by a Phi in successor
                            let is_phi_def = succ_block.instructions.iter().any(|inst| {
                                if let InstructionKind::Phi(d, _) = &inst.kind {
                                    *d == v
                                } else {
                                    false
                                }
                            });
                            if !is_phi_def {
                                out.insert(v);
                            }
                        }
                    }
                    // Phi operands for THIS block edge
                    for inst in &succ_block.instructions {
                        if let InstructionKind::Phi(_, mappings) = &inst.kind {
                            if let Some(&v) = mappings.get(&block.id) {
                                out.insert(v);
                            }
                        }
                    }
                }
            }
            live_out.insert(block.id, out.clone());

            // 2. Compute live_in: (live_out - defs) + uses
            let mut current_live = out;
            for inst in block.instructions.iter().rev() {
                if let Some(def) = inst.get_def() {
                    current_live.remove(&def);
                }
                // Only non-phi uses here
                if !matches!(inst.kind, InstructionKind::Phi(..)) {
                    for u in inst.get_uses() {
                        current_live.insert(u);
                    }
                }
            }

            if let Some(old_in) = live_in.get(&block.id) {
                if current_live != *old_in {
                    live_in.insert(block.id, current_live);
                    changed = true;
                }
            }
        }
    }

    let mut inst_live_out: HashMap<(usize, usize), HashSet<Value>> = HashMap::new();
    for block in &func.blocks {
        let mut current_live = live_out.get(&block.id).unwrap().clone();
        for (idx, inst) in block.instructions.iter().enumerate().rev() {
            inst_live_out.insert((block.id.0, idx), current_live.clone());
            tracing::debug!(target: "lila::liveness", "Block {} Inst {}: {:?}, live_out: {:?}", block.id.0, idx, inst.kind, current_live);

            if let Some(def) = inst.get_def() {
                current_live.remove(&def);
            }
            if !matches!(inst.kind, InstructionKind::Phi(..)) {
                for u in inst.get_uses() {
                    current_live.insert(u);
                }
            }
        }
    }

    LivenessAnalysisResults {
        live_in,
        live_out,
        inst_live_out,
    }
}
