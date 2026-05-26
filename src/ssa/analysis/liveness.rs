use crate::ssa::ir::{BlockId, Function, Value};
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
            let mut out = HashSet::new();
            for succ in &block.successors {
                if let Some(in_set) = live_in.get(succ) {
                    for v in in_set {
                        out.insert(*v);
                    }
                }
            }
            live_out.insert(block.id, out.clone());

            let mut current_live = out;
            for inst in block.instructions.iter().rev() {
                if let Some(def) = inst.get_def() {
                    current_live.remove(&def);
                }
                for u in inst.get_uses() {
                    current_live.insert(u);
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
            tracing::trace!(
                target: "lila::liveness",
                "Block {:?}, inst {}: live_out = {:?}",
                block.id, idx, current_live
            );
            if let Some(def) = inst.get_def() {
                current_live.remove(&def);
            }
            for u in inst.get_uses() {
                current_live.insert(u);
            }
        }
    }

    LivenessAnalysisResults {
        live_in,
        live_out,
        inst_live_out,
    }
}
