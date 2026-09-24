//! Evaluator for the subset of Onshape FeatureScript expressions that appear
//! in feature and sketch parameters (`BTMParameterQuantity.expression`).
//!
//! Supported: numeric literals with an optional unit suffix (`in`, `mm`,
//! `cm`, `m`, `deg`, `rad`, also as a bare identifier, so `(#Depth) in` and
//! `-0.5*m` both work), `#Variable` references, `true` / `false`, unary `-`
//! and `!`, `^ * / + -`, comparisons, `== !=`, `&& ||`, the ternary `? :`,
//! array literals `[a, b, c]` with indexing `arr[i]`, and the functions
//! `floor ceil round abs sqrt min max sin cos tan asin acos atan atan2`.
//!
//! Values carry a dimension: lengths are stored in metres, angles in
//! radians, and `+ - == <` reject mismatched dimensions so a missing unit
//! surfaces as an error instead of a silently wrong number. Indexing rejects
//! non-integer indices the way FeatureScript does.

use std::collections::HashMap;
use std::fmt;

/// Physical dimension of a number as (length exponent, angle exponent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim {
    pub length: i8,
    pub angle: i8,
}

impl Dim {
    pub const NONE: Dim = Dim { length: 0, angle: 0 };
    pub const LENGTH: Dim = Dim { length: 1, angle: 0 };
    pub const ANGLE: Dim = Dim { length: 0, angle: 1 };
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Num { v: f64, dim: Dim },
    Bool(bool),
    Array(Vec<Value>),
    Str(String),
}

impl Value {
    pub fn num(v: f64) -> Value {
        Value::Num { v, dim: Dim::NONE }
    }
    pub fn length_m(v: f64) -> Value {
        Value::Num { v, dim: Dim::LENGTH }
    }
    pub fn angle_rad(v: f64) -> Value {
        Value::Num { v, dim: Dim::ANGLE }
    }
    /// The number in metres, if this is a length.
    pub fn as_length_m(&self) -> Result<f64, Error> {
        match self {
            Value::Num { v, dim } if *dim == Dim::LENGTH => Ok(*v),
            Value::Num { v, dim } if *dim == Dim::NONE && *v == 0.0 => Ok(0.0),
            other => Err(Error(format!("expected a length, got {other}"))),
        }
    }
    pub fn as_angle_rad(&self) -> Result<f64, Error> {
        match self {
            Value::Num { v, dim } if *dim == Dim::ANGLE => Ok(*v),
            Value::Num { v, dim } if *dim == Dim::NONE && *v == 0.0 => Ok(0.0),
            other => Err(Error(format!("expected an angle, got {other}"))),
        }
    }
    pub fn as_number(&self) -> Result<f64, Error> {
        match self {
            Value::Num { v, dim } if *dim == Dim::NONE => Ok(*v),
            other => Err(Error(format!("expected a plain number, got {other}"))),
        }
    }
    pub fn as_bool(&self) -> Result<bool, Error> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(Error(format!("expected a boolean, got {other}"))),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num { v, dim } => match *dim {
                Dim::NONE => write!(f, "{v}"),
                Dim::LENGTH => write!(f, "{v} m"),
                Dim::ANGLE => write!(f, "{v} rad"),
                d => write!(f, "{v} m^{} rad^{}", d.length, d.angle),
            },
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Array(a) => {
                write!(f, "[")?;
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, "]")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

/// Variable lookup for `#Name` references.
pub trait Scope {
    fn get(&self, name: &str) -> Option<Value>;
}

impl Scope for HashMap<String, Value> {
    fn get(&self, name: &str) -> Option<Value> {
        HashMap::get(self, name).cloned()
    }
}

// ---------------------------------------------------------------- AST

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Num(f64),
    Bool(bool),
    Str(String),
    Var(String),
    Ident(String),
    Array(Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Pow,
    Mul,
    Div,
    Add,
    Sub,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

// ---------------------------------------------------------------- lexer

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Var(String),
    Str(String),
    Op(&'static str),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Question,
    Colon,
}

fn lex(src: &str) -> Result<Vec<Tok>, Error> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit()) {
            let s = i;
            while i < b.len() && ((b[i] as char).is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && (b[j] as char).is_ascii_digit() {
                    i = j;
                    while i < b.len() && (b[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let t = &src[s..i];
            out.push(Tok::Num(t.parse().map_err(|_| Error(format!("bad number {t:?}")))?));
            continue;
        }
        if c == '#' {
            let s = i + 1;
            i += 1;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            if i == s {
                return Err(Error("'#' without a variable name".into()));
            }
            out.push(Tok::Var(src[s..i].to_string()));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(src[s..i].to_string()));
            continue;
        }
        if c == '"' {
            let s = i + 1;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            if i >= b.len() {
                return Err(Error("unterminated string".into()));
            }
            out.push(Tok::Str(src[s..i].to_string()));
            i += 1;
            continue;
        }
        let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
        let op2 = match two {
            "<=" => Some("<="),
            ">=" => Some(">="),
            "==" => Some("=="),
            "!=" => Some("!="),
            "&&" => Some("&&"),
            "||" => Some("||"),
            _ => None,
        };
        if let Some(op) = op2 {
            out.push(Tok::Op(op));
            i += 2;
            continue;
        }
        let t = match c {
            '(' => Tok::LParen,
            ')' => Tok::RParen,
            '[' => Tok::LBracket,
            ']' => Tok::RBracket,
            ',' => Tok::Comma,
            '?' => Tok::Question,
            ':' => Tok::Colon,
            '+' => Tok::Op("+"),
            '-' => Tok::Op("-"),
            '*' => Tok::Op("*"),
            '/' => Tok::Op("/"),
            '^' => Tok::Op("^"),
            '<' => Tok::Op("<"),
            '>' => Tok::Op(">"),
            '!' => Tok::Op("!"),
            other => return Err(Error(format!("unexpected character {other:?} in {src:?}"))),
        };
        out.push(t);
        i += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------- parser

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn expect(&mut self, t: Tok) -> Result<(), Error> {
        match self.next() {
            Some(ref x) if *x == t => Ok(()),
            other => Err(Error(format!("expected {t:?}, got {other:?}"))),
        }
    }

    fn ternary(&mut self) -> Result<Expr, Error> {
        let c = self.binary(0)?;
        if self.peek() == Some(&Tok::Question) {
            self.next();
            let a = self.ternary()?;
            self.expect(Tok::Colon)?;
            let b = self.ternary()?;
            return Ok(Expr::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    /// Precedence climbing; `min_prec` is the lowest binding power accepted.
    fn binary(&mut self, min_prec: u8) -> Result<Expr, Error> {
        let mut lhs = self.unary()?;
        loop {
            let (op, prec, right_assoc) = match self.peek() {
                Some(Tok::Op(o)) => match *o {
                    "||" => (BinOp::Or, 1, false),
                    "&&" => (BinOp::And, 2, false),
                    "==" => (BinOp::Eq, 3, false),
                    "!=" => (BinOp::Ne, 3, false),
                    "<" => (BinOp::Lt, 4, false),
                    "<=" => (BinOp::Le, 4, false),
                    ">" => (BinOp::Gt, 4, false),
                    ">=" => (BinOp::Ge, 4, false),
                    "+" => (BinOp::Add, 5, false),
                    "-" => (BinOp::Sub, 5, false),
                    "*" => (BinOp::Mul, 6, false),
                    "/" => (BinOp::Div, 6, false),
                    "^" => (BinOp::Pow, 8, true),
                    _ => break,
                },
                _ => break,
            };
            if prec < min_prec {
                break;
            }
            self.next();
            let rhs = self.binary(if right_assoc { prec } else { prec + 1 })?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, Error> {
        match self.peek() {
            Some(Tok::Op("-")) => {
                self.next();
                // Unary minus binds tighter than * but looser than ^, like FS.
                let e = self.binary(7)?;
                Ok(Expr::Neg(Box::new(e)))
            }
            Some(Tok::Op("!")) => {
                self.next();
                let e = self.unary()?;
                Ok(Expr::Not(Box::new(e)))
            }
            _ => self.postfix(),
        }
    }

    fn postfix(&mut self) -> Result<Expr, Error> {
        let mut e = self.primary()?;
        loop {
            match self.peek() {
                Some(Tok::LBracket) => {
                    self.next();
                    let idx = self.ternary()?;
                    self.expect(Tok::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(idx));
                }
                // A unit identifier directly after a value multiplies: `2 in`, `(#Depth) in`.
                Some(Tok::Ident(name)) if unit_value(name).is_some() => {
                    let name = name.clone();
                    self.next();
                    e = Expr::Bin(BinOp::Mul, Box::new(e), Box::new(Expr::Ident(name)));
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr, Error> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Expr::Num(v)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Var(v)) => Ok(Expr::Var(v)),
            Some(Tok::Ident(name)) => {
                if name == "true" {
                    return Ok(Expr::Bool(true));
                }
                if name == "false" {
                    return Ok(Expr::Bool(false));
                }
                if self.peek() == Some(&Tok::LParen) {
                    self.next();
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.ternary()?);
                            if self.peek() == Some(&Tok::Comma) {
                                self.next();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::RParen)?;
                    return Ok(Expr::Call(name, args));
                }
                Ok(Expr::Ident(name))
            }
            Some(Tok::LParen) => {
                let e = self.ternary()?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::LBracket) => {
                let mut items = Vec::new();
                if self.peek() != Some(&Tok::RBracket) {
                    loop {
                        items.push(self.ternary()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.next();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Tok::RBracket)?;
                Ok(Expr::Array(items))
            }
            other => Err(Error(format!("unexpected token {other:?}"))),
        }
    }
}

/// Parse an expression string into an AST.
pub fn parse(src: &str) -> Result<Expr, Error> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    let e = p.ternary()?;
    if p.pos != p.toks.len() {
        return Err(Error(format!("trailing tokens after expression in {src:?}: {:?}", &p.toks[p.pos..])));
    }
    Ok(e)
}

// ---------------------------------------------------------------- units

fn unit_value(name: &str) -> Option<Value> {
    Some(match name {
        "m" | "meter" | "meters" => Value::length_m(1.0),
        "cm" | "centimeter" => Value::length_m(0.01),
        "mm" | "millimeter" => Value::length_m(0.001),
        "in" | "inch" | "inches" => Value::length_m(0.0254),
        "ft" | "foot" | "feet" => Value::length_m(0.3048),
        "yd" | "yard" => Value::length_m(0.9144),
        "deg" | "degree" | "degrees" => Value::angle_rad(std::f64::consts::PI / 180.0),
        "rad" | "radian" | "radians" => Value::angle_rad(1.0),
        _ => return None,
    })
}

// ---------------------------------------------------------------- eval

fn num(v: &Value, what: &str) -> Result<(f64, Dim), Error> {
    match v {
        Value::Num { v, dim } => Ok((*v, *dim)),
        other => Err(Error(format!("{what}: expected a number, got {other}"))),
    }
}

fn same_dim(a: Dim, av: f64, b: Dim, bv: f64, what: &str) -> Result<Dim, Error> {
    if a == b {
        return Ok(a);
    }
    // FeatureScript allows a dimensionless 0 against any unit.
    if a == Dim::NONE && av == 0.0 {
        return Ok(b);
    }
    if b == Dim::NONE && bv == 0.0 {
        return Ok(a);
    }
    Err(Error(format!("{what}: dimension mismatch ({av} with dim {a:?} vs {bv} with dim {b:?})")))
}

fn truthy(v: &Value) -> Result<bool, Error> {
    v.as_bool()
}

/// Evaluate an AST against a variable scope.
pub fn eval(e: &Expr, scope: &dyn Scope) -> Result<Value, Error> {
    Ok(match e {
        Expr::Num(v) => Value::num(*v),
        Expr::Bool(b) => Value::Bool(*b),
        Expr::Str(s) => Value::Str(s.clone()),
        Expr::Var(name) => scope.get(name).ok_or_else(|| Error(format!("undefined variable #{name}")))?,
        Expr::Ident(name) => unit_value(name).ok_or_else(|| Error(format!("unknown identifier {name}")))?,
        Expr::Array(items) => Value::Array(items.iter().map(|x| eval(x, scope)).collect::<Result<_, _>>()?),
        Expr::Index(arr, idx) => {
            let a = eval(arr, scope)?;
            let i = eval(idx, scope)?;
            let items = match a {
                Value::Array(items) => items,
                other => return Err(Error(format!("indexing a non-array {other}"))),
            };
            let (iv, dim) = num(&i, "array index")?;
            if dim != Dim::NONE {
                return Err(Error(format!("array index has a unit: {i}")));
            }
            if iv.fract().abs() > 1e-9 {
                return Err(Error(format!("array index {iv} is not an integer")));
            }
            let k = iv.round();
            if k < 0.0 || k as usize >= items.len() {
                return Err(Error(format!("array index {k} out of range 0..{}", items.len())));
            }
            items[k as usize].clone()
        }
        Expr::Call(name, args) => call(name, &args.iter().map(|x| eval(x, scope)).collect::<Result<Vec<_>, _>>()?)?,
        Expr::Neg(x) => {
            let (v, dim) = num(&eval(x, scope)?, "unary minus")?;
            Value::Num { v: -v, dim }
        }
        Expr::Not(x) => Value::Bool(!truthy(&eval(x, scope)?)?),
        Expr::Cond(c, a, b) => {
            if truthy(&eval(c, scope)?)? {
                eval(a, scope)?
            } else {
                eval(b, scope)?
            }
        }
        Expr::Bin(op, l, r) => {
            // Short-circuit logic first.
            match op {
                BinOp::And => {
                    return Ok(Value::Bool(truthy(&eval(l, scope)?)? && truthy(&eval(r, scope)?)?));
                }
                BinOp::Or => {
                    return Ok(Value::Bool(truthy(&eval(l, scope)?)? || truthy(&eval(r, scope)?)?));
                }
                _ => {}
            }
            let lv = eval(l, scope)?;
            let rv = eval(r, scope)?;
            match (op, &lv, &rv) {
                (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => return Ok(Value::Bool(a == b)),
                (BinOp::Ne, Value::Bool(a), Value::Bool(b)) => return Ok(Value::Bool(a != b)),
                (BinOp::Eq, Value::Str(a), Value::Str(b)) => return Ok(Value::Bool(a == b)),
                (BinOp::Ne, Value::Str(a), Value::Str(b)) => return Ok(Value::Bool(a != b)),
                _ => {}
            }
            let (a, ad) = num(&lv, "binary operator")?;
            let (b, bd) = num(&rv, "binary operator")?;
            match op {
                BinOp::Add => Value::Num { v: a + b, dim: same_dim(ad, a, bd, b, "+")? },
                BinOp::Sub => Value::Num { v: a - b, dim: same_dim(ad, a, bd, b, "-")? },
                BinOp::Mul => Value::Num { v: a * b, dim: Dim { length: ad.length + bd.length, angle: ad.angle + bd.angle } },
                BinOp::Div => Value::Num { v: a / b, dim: Dim { length: ad.length - bd.length, angle: ad.angle - bd.angle } },
                BinOp::Pow => {
                    if bd != Dim::NONE {
                        return Err(Error("exponent has a unit".into()));
                    }
                    if ad != Dim::NONE && b.fract() != 0.0 {
                        return Err(Error("non-integer power of a dimensioned value".into()));
                    }
                    let k = b as i8;
                    Value::Num { v: a.powf(b), dim: Dim { length: ad.length * k, angle: ad.angle * k } }
                }
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                    same_dim(ad, a, bd, b, "comparison")?;
                    Value::Bool(match op {
                        BinOp::Lt => a < b,
                        BinOp::Le => a <= b,
                        BinOp::Gt => a > b,
                        BinOp::Ge => a >= b,
                        BinOp::Eq => a == b,
                        BinOp::Ne => a != b,
                        _ => unreachable!(),
                    })
                }
                BinOp::And | BinOp::Or => unreachable!(),
            }
        }
    })
}

fn call(name: &str, args: &[Value]) -> Result<Value, Error> {
    let arg = |i: usize| -> Result<(f64, Dim), Error> {
        num(args.get(i).ok_or_else(|| Error(format!("{name}: missing argument {i}")))?, name)
    };
    let plain = |i: usize| -> Result<f64, Error> {
        let (v, d) = arg(i)?;
        if d != Dim::NONE {
            return Err(Error(format!("{name}: argument {i} has a unit")));
        }
        Ok(v)
    };
    let angle = |i: usize| -> Result<f64, Error> {
        let (v, d) = arg(i)?;
        if d == Dim::ANGLE || (d == Dim::NONE && v == 0.0) {
            Ok(v)
        } else {
            Err(Error(format!("{name}: argument {i} is not an angle")))
        }
    };
    Ok(match name {
        "floor" => {
            let (v, d) = arg(0)?;
            Value::Num { v: v.floor(), dim: d }
        }
        "ceil" => {
            let (v, d) = arg(0)?;
            Value::Num { v: v.ceil(), dim: d }
        }
        "round" => {
            let (v, d) = arg(0)?;
            Value::Num { v: v.round(), dim: d }
        }
        "abs" => {
            let (v, d) = arg(0)?;
            Value::Num { v: v.abs(), dim: d }
        }
        "sqrt" => {
            let (v, d) = arg(0)?;
            if d.length % 2 != 0 || d.angle % 2 != 0 {
                return Err(Error("sqrt of an odd dimension".into()));
            }
            Value::Num { v: v.sqrt(), dim: Dim { length: d.length / 2, angle: d.angle / 2 } }
        }
        "min" | "max" => {
            if args.is_empty() {
                return Err(Error(format!("{name}: no arguments")));
            }
            let (mut best, mut dim) = arg(0)?;
            for i in 1..args.len() {
                let (v, d) = arg(i)?;
                dim = same_dim(dim, best, d, v, name)?;
                best = if name == "min" { best.min(v) } else { best.max(v) };
            }
            Value::Num { v: best, dim }
        }
        "sin" => Value::num(angle(0)?.sin()),
        "cos" => Value::num(angle(0)?.cos()),
        "tan" => Value::num(angle(0)?.tan()),
        "asin" => Value::angle_rad(plain(0)?.asin()),
        "acos" => Value::angle_rad(plain(0)?.acos()),
        "atan" => Value::angle_rad(plain(0)?.atan()),
        "atan2" => {
            let (y, yd) = arg(0)?;
            let (x, xd) = arg(1)?;
            same_dim(yd, y, xd, x, name)?;
            Value::angle_rad(y.atan2(x))
        }
        _ => return Err(Error(format!("unknown function {name}"))),
    })
}

/// Parse and evaluate in one step.
pub fn evaluate(src: &str, scope: &dyn Scope) -> Result<Value, Error> {
    eval(&parse(src)?, scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> HashMap<String, Value> {
        let mut s = HashMap::new();
        s.insert("W".to_string(), Value::num(18.0));
        s.insert("D".to_string(), Value::num(20.0));
        s.insert("Flag".to_string(), Value::Bool(true));
        s.insert("Len".to_string(), Value::length_m(0.0254));
        s
    }

    fn ev(src: &str) -> Value {
        evaluate(src, &scope()).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    #[test]
    fn units_and_arithmetic() {
        assert_eq!(ev("2 in"), Value::length_m(0.0508));
        assert_eq!(ev("(#D - 2) in"), Value::length_m(18.0 * 0.0254));
        assert_eq!(ev("-0.5*m"), Value::length_m(-0.5));
        assert_eq!(ev("#Len + 1 in").as_length_m().unwrap(), 2.0 * 0.0254);
        assert_eq!(ev("2^3 * 2"), Value::num(16.0));
        assert_eq!(ev("-2^2"), Value::num(-4.0));
        assert_eq!(ev("1/tan(45 deg)").as_number().unwrap().round(), 1.0);
        assert!((ev("13/12").as_number().unwrap() - 13.0 / 12.0).abs() < 1e-15);
    }

    #[test]
    fn logic_ternary_arrays() {
        assert_eq!(ev("(#W + 3) <= #D - 1 ? 1 : 2"), Value::num(2.0));
        assert_eq!(ev("(#W + 3) <= #D + 1 ? 1 : 2"), Value::num(1.0));
        assert_eq!(ev("(#Flag == true) ? 5 : 6"), Value::num(5.0));
        assert_eq!(ev("!#Flag || #W >= 24"), Value::Bool(false));
        assert_eq!(ev("[1, 2, 4.5][#W - 16]"), Value::num(4.5));
        assert_eq!(ev("([0, 0.625][floor(#D - 18.5)]) in"), Value::length_m(0.625 * 0.0254));
        assert_eq!(ev("#W >= 24 && #D > 3"), Value::Bool(false));
    }

    #[test]
    fn errors() {
        assert!(evaluate("[1,2][0.5]", &scope()).is_err());
        assert!(evaluate("1 in + 2", &scope()).is_err());
        assert!(evaluate("#Missing", &scope()).is_err());
        assert!(evaluate("1 +", &scope()).is_err());
        assert!(evaluate("(1", &scope()).is_err());
    }

    #[test]
    fn zero_matches_any_dimension() {
        assert_eq!(ev("0 + 2 in"), Value::length_m(0.0508));
        assert_eq!(ev("max(0, 3 mm)"), Value::length_m(0.003));
    }
}
