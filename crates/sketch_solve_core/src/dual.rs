//! Forward-mode automatic differentiation with one derivative slot. Residuals
//! are written once, generically over [`Real`], and evaluated with `f64` for
//! values or with [`Dual`] (seeded on one variable) for Jacobian columns.

use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Real:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    fn c(v: f64) -> Self;
    fn val(self) -> f64;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn abs(self) -> Self;
    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }
}

impl Real for f64 {
    fn c(v: f64) -> Self {
        v
    }
    fn val(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub d: f64,
}

impl Dual {
    pub fn var(v: f64) -> Dual {
        Dual { v, d: 1.0 }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { v: self.v + o.v, d: self.d + o.d }
    }
}
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { v: self.v - o.v, d: self.d - o.d }
    }
}
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { v: self.v * o.v, d: self.d * o.v + self.v * o.d }
    }
}
impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { v: self.v / o.v, d: (self.d * o.v - self.v * o.d) / (o.v * o.v) }
    }
}
impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { v: -self.v, d: -self.d }
    }
}

impl Real for Dual {
    fn c(v: f64) -> Self {
        Dual { v, d: 0.0 }
    }
    fn val(self) -> f64 {
        self.v
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        // Guard the derivative at zero: treat sqrt'(0) as a large finite slope.
        let d = if s > 0.0 { self.d / (2.0 * s) } else { self.d * 1e8 };
        Dual { v: s, d }
    }
    fn sin(self) -> Self {
        Dual { v: self.v.sin(), d: self.d * self.v.cos() }
    }
    fn cos(self) -> Self {
        Dual { v: self.v.cos(), d: -self.d * self.v.sin() }
    }
    fn atan2(self, x: Self) -> Self {
        let y = self;
        let den = x.v * x.v + y.v * y.v;
        let d = if den > 0.0 { (x.v * y.d - y.v * x.d) / den } else { 0.0 };
        Dual { v: y.v.atan2(x.v), d }
    }
    fn abs(self) -> Self {
        if self.v < 0.0 { -self } else { self }
    }
}
