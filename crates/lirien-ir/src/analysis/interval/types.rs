//! Range and interval types for abstract interpretation.
//!
//! This module defines [`Bound`] and [`Interval`] types which represent numeric bounds
//! (finite values or infinities) used during abstract interpretation to infer value ranges.

use crate::ir::Value;
use std::collections::HashMap;

/// Represents a value boundary (either a finite value, positive infinity, or negative infinity).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// A concrete finite numeric boundary.
    Finite(f64),
    /// Negative infinity.
    NegInf,
    /// Positive infinity.
    PosInf,
}

impl Bound {
    /// Returns `true` if this is a finite numeric boundary.
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

/// Represents a numeric interval/range with lower and upper boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    /// Lower boundary.
    pub low: Bound,
    /// Upper boundary.
    pub high: Bound,
}

impl Interval {
    /// Creates an unbounded interval `[-inf, +inf]`.
    pub fn everything() -> Self {
        Interval {
            low: Bound::NegInf,
            high: Bound::PosInf,
        }
    }

    /// Creates a singleton interval `[val, val]`.
    pub fn from_const(val: f64) -> Self {
        Interval {
            low: Bound::Finite(val),
            high: Bound::Finite(val),
        }
    }

    /// Returns `true` if the interval is non-negative (`low >= 0`).
    pub fn is_non_negative(&self) -> bool {
        match self.low {
            Bound::Finite(l) => l >= 0.0,
            Bound::PosInf => true,
            Bound::NegInf => false,
        }
    }

    /// Returns `true` if the interval is strictly positive (`low > 0`).
    pub fn is_strictly_positive(&self) -> bool {
        match self.low {
            Bound::Finite(l) => l > 0.0,
            Bound::PosInf => true,
            Bound::NegInf => false,
        }
    }

    /// Returns `true` if the interval is strictly negative (`high < 0`).
    pub fn is_strictly_negative(&self) -> bool {
        match self.high {
            Bound::Finite(h) => h < 0.0,
            Bound::NegInf => true,
            Bound::PosInf => false,
        }
    }
}

/// The result mapping of the interval/range inference pass.
pub struct IntervalAnalysisResults {
    /// Inferred value ranges for each SSA variable.
    pub intervals: HashMap<Value, Interval>,
    /// Block-specific range narrowings (e.g. inside conditional branches).
    pub block_narrowing: HashMap<(Value, crate::ir::BlockId), Interval>,
}

