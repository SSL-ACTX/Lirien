use crate::ir::Type;
use super::types::{Bound, Interval};

impl Interval {
    pub fn join(&self, other: &Self) -> Self {
        Interval {
            low: self.low.min(other.low),
            high: self.high.max(other.high),
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
        let mut res = Interval { low, high };
        // If x >= 2 and y == 1, then x - y >= 1.
        if let Bound::Finite(low_x) = self.low {
            if low_x >= 2.0 {
                if let Bound::Finite(1.0) = other.low {
                    if let Bound::Finite(1.0) = other.high {
                        res.low = res.low.max(Bound::Finite(1.0));
                    }
                }
            }
        }
        res
    }

    pub fn mul(&self, other: &Self) -> Self {
        match (self.low, self.high, other.low, other.high) {
            (Bound::Finite(a), Bound::Finite(b), Bound::Finite(c), Bound::Finite(d)) => {
                let p1 = a * c;
                let p2 = a * d;
                let p3 = b * c;
                let p4 = b * d;
                let min = p1.min(p2).min(p3).min(p4);
                let max = p1.max(p2).max(p3).max(p4);
                Interval {
                    low: Bound::Finite(min),
                    high: Bound::Finite(max),
                }
            }
            // Handle cases with infinity but known signs
            _ => {
                let mut low = Bound::NegInf;
                let high = Bound::PosInf;

                if self.is_strictly_positive() && other.is_strictly_positive() {
                    // [a, b] * [c, d] where a, c >= 1
                    low = match (self.low, other.low) {
                        (Bound::Finite(a), Bound::Finite(c)) => Bound::Finite(a * c),
                        (Bound::Finite(a), _) if a >= 1.0 => Bound::Finite(a),
                        (_, Bound::Finite(c)) if c >= 1.0 => Bound::Finite(c),
                        _ => Bound::Finite(1.0),
                    };
                } else if self.is_non_negative() && other.is_non_negative() {
                    // [a, b] * [c, d] where a, c >= 0 => [a*c, b*d]
                    low = match (self.low, other.low) {
                        (Bound::Finite(a), Bound::Finite(c)) => Bound::Finite(a * c),
                        _ => Bound::Finite(0.0),
                    };
                } else if (self.is_strictly_positive() && other.is_non_negative())
                    || (self.is_non_negative() && other.is_strictly_positive())
                {
                    low = Bound::Finite(0.0);
                }

                Interval { low, high }
            }
        }
    }

    pub fn clamp(&mut self, ty: Type) {
        if let Some(bit_width) = ty.int_bit_width() {
            let (min, max) = if ty.is_unsigned() {
                (0.0, ((1u128 << bit_width) as f64) - 1.0)
            } else {
                (
                    -((1u128 << (bit_width - 1)) as f64),
                    ((1u128 << (bit_width - 1)) as f64) - 1.0,
                )
            };
            if let Bound::Finite(l) = self.low {
                if l < min {
                    self.low = Bound::NegInf;
                } else if l > max {
                    self.low = Bound::Finite(max);
                }
            }
            if let Bound::Finite(h) = self.high {
                if h > max {
                    self.high = Bound::PosInf;
                } else if h < min {
                    self.high = Bound::Finite(min);
                }
            }
        }
    }
}
