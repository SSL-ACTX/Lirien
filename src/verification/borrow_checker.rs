use crate::ssa::ir::{BlockId, Function, Instruction, InstructionKind, Type, Value};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct BorrowChecker<'a> {
    func: &'a Function,
}

impl<'a> BorrowChecker<'a> {
    pub fn new(func: &'a Function) -> Self {
        Self { func }
    }

    pub fn check(&self) -> Result<(), String> {
        tracing::debug!(
            target: "lila::verify::borrow",
            "CFG-aware check for '{}'...",
            self.func.name
        );

        let mut live_in: HashMap<BlockId, HashSet<Value>> = HashMap::new();
        let mut live_out: HashMap<BlockId, HashSet<Value>> = HashMap::new();

        for block in &self.func.blocks {
            live_in.insert(block.id, HashSet::new());
            live_out.insert(block.id, HashSet::new());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for block in self.func.blocks.iter().rev() {
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
                    if let Some(def) = self.get_def(inst) {
                        current_live.remove(&def);
                    }
                    for u in self.get_uses(inst) {
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

        let mut inst_live_out: HashMap<*const Instruction, HashSet<Value>> = HashMap::new();
        for block in &self.func.blocks {
            let mut current_live = live_out.get(&block.id).unwrap().clone();
            for inst in block.instructions.iter().rev() {
                inst_live_out.insert(inst as *const Instruction, current_live.clone());
                if let Some(def) = self.get_def(inst) {
                    current_live.remove(&def);
                }
                for u in self.get_uses(inst) {
                    current_live.insert(u);
                }
            }
        }

        let mut imm_borrows: HashMap<Value, Value> = HashMap::new();
        let mut mut_borrows: HashMap<Value, Value> = HashMap::new();

        for block in &self.func.blocks {
            for inst in &block.instructions {
                if let InstructionKind::Borrow(d, s) = &inst.kind {
                    imm_borrows.insert(*d, *s);
                } else if let InstructionKind::MutBorrow(d, s) = &inst.kind {
                    mut_borrows.insert(*d, *s);
                }
            }
        }

        let mut moved_in: HashMap<BlockId, HashSet<Value>> = HashMap::new();
        let mut moved_out: HashMap<BlockId, HashSet<Value>> = HashMap::new();

        for block in &self.func.blocks {
            moved_in.insert(block.id, HashSet::new());
            moved_out.insert(block.id, HashSet::new());
        }

        let mut worklist = VecDeque::new();
        for block in &self.func.blocks {
            worklist.push_back(block.id);
        }

        let mut iterations = 0;
        let max_iterations = self.func.blocks.len() * self.func.blocks.len() + 10;

        while let Some(block_id) = worklist.pop_front() {
            iterations += 1;
            if iterations > max_iterations {
                return Err("Borrow checker fixed-point iteration exceeded limit".to_string());
            }

            let block = self.func.blocks.iter().find(|b| b.id == block_id).unwrap();

            let mut current_moved = HashSet::new();
            for pred_id in &block.predecessors {
                if let Some(out) = moved_out.get(pred_id) {
                    for v in out {
                        current_moved.insert(*v);
                    }
                }
            }
            moved_in.insert(block_id, current_moved.clone());

            for inst in &block.instructions {
                if let InstructionKind::Phi(_, mappings) = &inst.kind {
                    for (pred_id, val) in mappings {
                        if let Some(out) = moved_out.get(pred_id) {
                            if out.contains(val) {
                                return Err(format!(
                                    "Use-after-move of value {} in Phi from {}",
                                    val, pred_id
                                ));
                            }
                        }
                    }
                } else {
                    self.check_move_operands(inst, &current_moved)?;
                }

                let live_now_full = inst_live_out.get(&(inst as *const Instruction)).unwrap();
                let mut live_now = live_now_full.clone();
                if let Some(def) = self.get_def(inst) {
                    live_now.remove(&def);
                }

                let active_imm: HashSet<Value> = imm_borrows
                    .iter()
                    .filter(|(r, _)| live_now.contains(*r))
                    .map(|(_, v)| *v)
                    .collect();
                let active_mut: HashSet<Value> = mut_borrows
                    .iter()
                    .filter(|(r, _)| live_now.contains(*r))
                    .map(|(_, v)| *v)
                    .collect();

                match &inst.kind {
                    InstructionKind::Borrow(_, src) if active_mut.contains(src) => {
                        return Err(format!(
                            "Cannot borrow {} immutably because it is already borrowed mutably",
                            src
                        ));
                    }
                    InstructionKind::MutBorrow(_, src) => {
                        if active_mut.contains(src) {
                            return Err(format!(
                                "Cannot borrow {} mutably because it is already borrowed mutably",
                                src
                            ));
                        }
                        if active_imm.contains(src) {
                            return Err(format!(
                                "Cannot borrow {} mutably because it is already borrowed immutably",
                                src
                            ));
                        }
                    }
                    InstructionKind::Call(_, _, args) => {
                        for arg in args {
                            let ty = self.func.get_type(*arg);
                            if matches!(ty, Type::Owned(_)) {
                                current_moved.insert(*arg);
                            }
                        }
                    }
                    InstructionKind::StructSet(_, obj, _, _, _)
                    | InstructionKind::ArrayStore(_, obj, _, _, _) => {
                        if active_imm.contains(obj) {
                            return Err(format!(
                                "Cannot borrow {} mutably because it is already borrowed immutably",
                                obj
                            ));
                        }
                        if active_mut.contains(obj) {
                            return Err(format!(
                                "Cannot borrow {} mutably because it is already borrowed mutably",
                                obj
                            ));
                        }
                    }
                    _ => {}
                }
            }

            let old_out = moved_out.get(&block_id).unwrap();
            if &current_moved != old_out {
                moved_out.insert(block_id, current_moved);
                for succ_id in &block.successors {
                    if !worklist.contains(succ_id) {
                        worklist.push_back(*succ_id);
                    }
                }
            }
        }

        Ok(())
    }

    fn get_def(&self, inst: &Instruction) -> Option<Value> {
        match &inst.kind {
            InstructionKind::Add(d, _, _)
            | InstructionKind::Sub(d, _, _)
            | InstructionKind::Mul(d, _, _)
            | InstructionKind::SDiv(d, _, _)
            | InstructionKind::UDiv(d, _, _)
            | InstructionKind::SRem(d, _, _)
            | InstructionKind::URem(d, _, _)
            | InstructionKind::And(d, _, _)
            | InstructionKind::Or(d, _, _)
            | InstructionKind::Xor(d, _, _)
            | InstructionKind::Shl(d, _, _)
            | InstructionKind::LShr(d, _, _)
            | InstructionKind::AShr(d, _, _)
            | InstructionKind::Not(d, _)
            | InstructionKind::FAdd(d, _, _)
            | InstructionKind::FSub(d, _, _)
            | InstructionKind::FMul(d, _, _)
            | InstructionKind::FDiv(d, _, _)
            | InstructionKind::Eq(d, _, _)
            | InstructionKind::Ne(d, _, _)
            | InstructionKind::SLt(d, _, _)
            | InstructionKind::SLe(d, _, _)
            | InstructionKind::SGt(d, _, _)
            | InstructionKind::SGe(d, _, _)
            | InstructionKind::ULt(d, _, _)
            | InstructionKind::ULe(d, _, _)
            | InstructionKind::UGt(d, _, _)
            | InstructionKind::UGe(d, _, _)
            | InstructionKind::FLt(d, _, _)
            | InstructionKind::FLe(d, _, _)
            | InstructionKind::FGt(d, _, _)
            | InstructionKind::FGe(d, _, _)
            | InstructionKind::ConstInt(d, _)
            | InstructionKind::ConstFloat(d, _)
            | InstructionKind::Phi(d, _)
            | InstructionKind::Call(d, _, _)
            | InstructionKind::Borrow(d, _)
            | InstructionKind::MutBorrow(d, _)
            | InstructionKind::ArrayLoad(d, _, _)
            | InstructionKind::ArrayStore(d, _, _, _, _)
            | InstructionKind::StructLoad(d, _, _)
            | InstructionKind::StructOffset(d, _, _)
            | InstructionKind::StructSet(d, _, _, _, _) => Some(*d),
            _ => None,
        }
    }

    fn get_uses(&self, inst: &Instruction) -> Vec<Value> {
        let mut operands = Vec::new();
        match &inst.kind {
            InstructionKind::Add(_, l, r)
            | InstructionKind::Sub(_, l, r)
            | InstructionKind::Mul(_, l, r)
            | InstructionKind::SDiv(_, l, r)
            | InstructionKind::UDiv(_, l, r)
            | InstructionKind::SRem(_, l, r)
            | InstructionKind::URem(_, l, r)
            | InstructionKind::And(_, l, r)
            | InstructionKind::Or(_, l, r)
            | InstructionKind::Xor(_, l, r)
            | InstructionKind::Shl(_, l, r)
            | InstructionKind::LShr(_, l, r)
            | InstructionKind::AShr(_, l, r)
            | InstructionKind::FAdd(_, l, r)
            | InstructionKind::FSub(_, l, r)
            | InstructionKind::FMul(_, l, r)
            | InstructionKind::FDiv(_, l, r)
            | InstructionKind::Eq(_, l, r)
            | InstructionKind::Ne(_, l, r)
            | InstructionKind::SLt(_, l, r)
            | InstructionKind::SLe(_, l, r)
            | InstructionKind::SGt(_, l, r)
            | InstructionKind::SGe(_, l, r)
            | InstructionKind::ULt(_, l, r)
            | InstructionKind::ULe(_, l, r)
            | InstructionKind::UGt(_, l, r)
            | InstructionKind::UGe(_, l, r)
            | InstructionKind::FLt(_, l, r)
            | InstructionKind::FLe(_, l, r)
            | InstructionKind::FGt(_, l, r)
            | InstructionKind::FGe(_, l, r) => {
                operands.push(*l);
                operands.push(*r);
            }
            InstructionKind::Not(_, s) => {
                operands.push(*s);
            }
            InstructionKind::Branch(c, _, _) => {
                operands.push(*c);
            }
            InstructionKind::Return(Some(v)) => {
                operands.push(*v);
            }
            InstructionKind::Phi(_, mappings) => {
                for v in mappings.values() {
                    operands.push(*v);
                }
            }
            InstructionKind::Call(_, _, args) => {
                for v in args {
                    operands.push(*v);
                }
            }
            InstructionKind::Borrow(_, s) | InstructionKind::MutBorrow(_, s) => {
                operands.push(*s);
            }
            InstructionKind::ArrayLoad(_, arr, idx) => {
                operands.push(*arr);
                operands.push(*idx);
            }
            InstructionKind::ArrayStore(_, arr, idx, val, _) => {
                operands.push(*arr);
                operands.push(*idx);
                operands.push(*val);
            }
            InstructionKind::StructLoad(_, obj, _) | InstructionKind::StructOffset(_, obj, _) => {
                operands.push(*obj);
            }
            InstructionKind::StructSet(_, obj, _, val, _) => {
                operands.push(*obj);
                operands.push(*val);
            }
            _ => {}
        }
        operands
    }

    fn check_move_operands(
        &self,
        inst: &Instruction,
        moved: &HashSet<Value>,
    ) -> Result<(), String> {
        for op in self.get_uses(inst) {
            if moved.contains(&op) {
                return Err(format!("Use-after-move of value {}", op));
            }
        }
        Ok(())
    }
}
