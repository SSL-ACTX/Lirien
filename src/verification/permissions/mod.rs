use crate::ssa::analysis::liveness::LivenessAnalysisResults;
use crate::ssa::ir::{BlockId, Function, InstructionKind, Type, Value};
use std::collections::{HashMap, HashSet};
use z3::ast::{Bool, Real};
use z3::{Context, SatResult, Solver};

pub struct PermissionVerifier<'a> {
    func: &'a Function,
    // Maps each pointer Value to its "Root" object (the original Owned value)
    roots: HashMap<Value, Value>,
    // List of instructions that perform a "move" (consume 1.0 permission)
    moves: Vec<(Value, BlockId, usize)>,
    uid: usize,
}

impl<'a> PermissionVerifier<'a> {
    pub fn new(func: &'a Function) -> Self {
        let mut verifier = Self {
            func,
            roots: HashMap::new(),
            moves: Vec::new(),
            uid: 0,
        };
        verifier.build_root_map();
        verifier.identify_moves();
        verifier
    }

    pub fn set_uid(&mut self, uid: usize) {
        self.uid = uid;
    }

    fn identify_moves(&mut self) {
        for block in &self.func.blocks {
            for (idx, inst) in block.instructions.iter().enumerate() {
                match &inst.kind {
                    InstructionKind::Call(_, _, args) => {
                        for &arg in args {
                            let ty = self.func.get_type(arg);
                            if matches!(ty, Type::Owned(_)) {
                                if let Some(&root) = self.roots.get(&arg) {
                                    self.moves.push((root, block.id, idx));
                                }
                            }
                        }
                    }
                    InstructionKind::Return(Some(v)) => {
                        let ty = self.func.get_type(*v);
                        if matches!(ty, Type::Owned(_)) {
                            if let Some(&root) = self.roots.get(v) {
                                self.moves.push((root, block.id, idx));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn build_root_map(&mut self) {
        for i in 0..self.func.arg_count {
            let v = Value(i);
            self.roots.insert(v, v);
        }
        for i in 0..self.func.value_count {
            let v = Value(i);
            if let Type::Owned(_) = self.func.get_type(v) {
                self.roots.insert(v, v);
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.func.blocks {
                for inst in &block.instructions {
                    if let Some(def) = inst.get_def() {
                        let current_root = self.roots.get(&def).copied();
                        let mut new_root = None;
                        match &inst.kind {
                            InstructionKind::Reference(_, src)
                            | InstructionKind::MutReference(_, src)
                            | InstructionKind::StructOffset(_, src, _)
                            | InstructionKind::StructLoad(_, src, _)
                            | InstructionKind::ArrayLoad(_, src, _)
                            | InstructionKind::BufferLoad(_, src, _)
                            | InstructionKind::TupleExtract(_, src, _)
                            | InstructionKind::EnumExtract(_, src, _) => {
                                new_root = self.roots.get(src).copied();
                            }
                            InstructionKind::Phi(_, mappings) => {
                                for v in mappings.values() {
                                    if let Some(r) = self.roots.get(v) {
                                        new_root = Some(*r);
                                        break;
                                    }
                                }
                            }
                            _ => {
                                let ty = self.func.get_type(def);
                                if matches!(
                                    ty,
                                    Type::Owned(_) | Type::Ref(_) | Type::Mut(_) | Type::Buffer(_)
                                ) {
                                    new_root = Some(def);
                                }
                            }
                        }
                        if new_root.is_some() && new_root != current_root {
                            self.roots.insert(def, new_root.unwrap());
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    pub fn generate_assertions(
        &self,
        _main_solver: &Solver,
        liveness: &LivenessAnalysisResults,
        _value_perms: &HashMap<Value, Real>,
        _block_conditions: &HashMap<BlockId, Bool>,
    ) -> Result<(), String> {
        let _ctx = Context::thread_local();
        let solver = Solver::new();

        let zero = Real::from_rational(0, 1);
        let one = Real::from_rational(1, 1);

        // 1. Declare structural block/edge conditions for pure CFG analysis
        let mut block_conds = HashMap::new();
        let mut edge_conds = HashMap::new();

        for block in &self.func.blocks {
            let b_cond = Bool::new_const(format!(
                "{}_pblock_{}_{}",
                self.func.name, block.id.0, self.uid
            ));
            block_conds.insert(block.id, b_cond);

            if let Some(last) = block.instructions.last() {
                match &last.kind {
                    InstructionKind::Jump(target) => {
                        let e_cond = Bool::new_const(format!(
                            "{}_pedge_{}_{}_{}",
                            self.func.name, block.id.0, target.0, self.uid
                        ));
                        edge_conds.insert((block.id, *target), e_cond);
                    }
                    InstructionKind::Branch(_, t, f) => {
                        let et = Bool::new_const(format!(
                            "{}_pedge_{}_{}_{}",
                            self.func.name, block.id.0, t.0, self.uid
                        ));
                        edge_conds.insert((block.id, *t), et);
                        let ef = Bool::new_const(format!(
                            "{}_pedge_{}_{}_{}",
                            self.func.name, block.id.0, f.0, self.uid
                        ));
                        edge_conds.insert((block.id, *f), ef);
                    }
                    _ => {}
                }
            }
        }

        // 2. Assert structural CFG constraints (purely flow-sensitive, no logic)
        if let Some(entry_cond) = block_conds.get(&self.func.entry_block) {
            solver.assert(&entry_cond.eq(&Bool::from_bool(true)));
        }

        for block in &self.func.blocks {
            let b_cond = block_conds.get(&block.id).unwrap();

            // Edges from this block imply block reached
            for &succ in &block.successors {
                if let Some(e_cond) = edge_conds.get(&(block.id, succ)) {
                    solver.assert(&e_cond.implies(b_cond));
                }
            }

            // Block reached (except entry) implies at least one predecessor edge
            if block.id != self.func.entry_block {
                let mut incoming = Vec::new();
                for ((_src, dst), e_cond) in &edge_conds {
                    if *dst == block.id {
                        incoming.push(e_cond);
                    }
                }
                if !incoming.is_empty() {
                    let or_incoming = Bool::or(incoming.as_slice());
                    solver.assert(&b_cond.implies(&or_incoming));
                }
            }
        }

        // 3. Setup UID-isolated shared weights
        let sw_denom = (self.func.value_count + self.moves.len() + 2) as i64;
        let sw_name = format!("{}_{}_sw", self.func.name, self.uid);
        let sw = Real::new_const(sw_name.as_str());
        solver.assert(&sw.eq(&Real::from_rational(1, sw_denom)));

        let all_roots: HashSet<Value> = self.roots.values().cloned().collect();

        // 4. Verify safety at every instruction using the structural model
        for block in &self.func.blocks {
            let block_reached = block_conds.get(&block.id).unwrap();

            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                let live_out = liveness
                    .inst_live_out
                    .get(&(block.id.0, inst_idx))
                    .expect("Liveness missing");
                let mut active_values = live_out.clone();
                for u in inst.get_uses() {
                    active_values.insert(u);
                }
                if let Some(d) = inst.get_def() {
                    active_values.insert(d);
                }

                for &root in &all_roots {
                    let mut terms = Vec::new();

                    for &v in &active_values {
                        if self.roots.get(&v) == Some(&root) {
                            match self.func.get_type(v) {
                                Type::Owned(_) | Type::Mut(_) => {
                                    terms.push(one.clone());
                                }
                                Type::Ref(_) => {
                                    terms.push(sw.clone());
                                }
                                _ => {}
                            }
                        }
                    }

                    for (m_root, m_block, m_idx) in &self.moves {
                        if *m_root == root {
                            let m_reached = block_conds.get(m_block).unwrap();
                            let is_before = if m_block == &block.id {
                                if *m_idx < inst_idx {
                                    true
                                } else {
                                    self.is_in_cycle(*m_block)
                                }
                            } else {
                                self.is_reachable(*m_block, block.id)
                            };

                            if is_before {
                                terms.push(m_reached.ite(&one, &zero));
                            }
                        }
                    }

                    if !terms.is_empty() {
                        let sum = if terms.len() == 1 {
                            terms[0].clone()
                        } else {
                            Real::add(terms.iter().collect::<Vec<_>>().as_slice())
                        };

                        solver.push();
                        solver.assert(block_reached);
                        solver.assert(&sum.gt(&one));
                        let check_res = solver.check();
                        solver.pop(1);

                        if check_res == SatResult::Sat {
                            tracing::error!(
                                target: "lila::verify::perm",
                                "Memory safety violation detected at {:?}:{} for root {} in function '{}' (session {})",
                                block.id, inst_idx, root, self.func.name, self.uid
                            );
                            return Err(format!(
                                "Memory safety violation: No valid fractional permission partitioning exists for root {} at instruction {} in block {:?}. (Possible aliasing, Use-after-move, or Move-in-Loop)",
                                root, inst_idx, block.id
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn is_reachable(&self, from: BlockId, to: BlockId) -> bool {
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        if let Some(block) = self.func.blocks.iter().find(|b| b.id == from) {
            for &succ in &block.successors {
                queue.push_back(succ);
            }
        }
        while let Some(curr) = queue.pop_front() {
            if curr == to {
                return true;
            }
            if visited.insert(curr) {
                if let Some(block) = self.func.blocks.iter().find(|b| b.id == curr) {
                    for &succ in &block.successors {
                        queue.push_back(succ);
                    }
                }
            }
        }
        false
    }

    fn is_in_cycle(&self, block_id: BlockId) -> bool {
        self.is_reachable(block_id, block_id)
    }
}
