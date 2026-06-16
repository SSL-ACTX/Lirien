use crate::ir::Value;
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

    pub fn is_non_negative(&self) -> bool {
        match self.low {
            Bound::Finite(l) => l >= 0.0,
            Bound::PosInf => true,
            Bound::NegInf => false,
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
    pub block_narrowing: HashMap<(Value, crate::ir::BlockId), Interval>,
}
