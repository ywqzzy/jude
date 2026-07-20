use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

#[pyclass(name = "Expression")]
#[derive(Clone, Debug)]
pub struct Expression {
    kind: ExprKind,
}

#[derive(Clone, Debug)]
enum ExprKind {
    Column(String),
    Literal(LiteralValue),
    RawSql(String),
    /// SQL fragment rendered verbatim without wrapping parentheses (used for
    /// order modifiers like `x DESC NULLS LAST` and predicates like `x IS NULL`).
    Bare(String),
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    Aliased {
        expr: Box<Expression>,
        alias: String,
    },
    Star,
}

#[derive(Clone, Debug)]
pub enum LiteralValue {
    Null,
    Boolean(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Expression {
    pub fn render_sql(&self) -> String {
        match &self.kind {
            ExprKind::Column(name) => quote_identifier(name),
            ExprKind::Literal(l) => render_literal(l),
            ExprKind::RawSql(sql) => format!("({sql})"),
            ExprKind::Bare(sql) => sql.clone(),
            ExprKind::FunctionCall { name, args } => {
                // Render binary/unary operators using infix/prefix SQL syntax
                // rather than function-call syntax.
                if args.len() == 2 && is_infix_operator(name) {
                    return format!(
                        "({} {} {})",
                        args[0].render_sql(),
                        name,
                        args[1].render_sql()
                    );
                }
                if args.len() == 1 && name == "NOT" {
                    return format!("(NOT {})", args[0].render_sql());
                }
                if args.len() == 1 && name == "-" {
                    return format!("(-{})", args[0].render_sql());
                }
                let args_sql = args
                    .iter()
                    .map(|a| a.render_sql())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{name}({args_sql})")
            }
            ExprKind::Aliased { expr, alias } => {
                format!("{} AS {}", expr.render_sql(), quote_identifier(alias))
            }
            ExprKind::Star => "*".to_string(),
        }
    }

    pub fn make_alias(&self, name: &str) -> Self {
        Self {
            kind: ExprKind::Aliased {
                expr: Box::new(self.clone()),
                alias: name.to_string(),
            },
        }
    }

    fn binary_op(&self, op: &str, other: &Expression) -> Self {
        Self {
            kind: ExprKind::FunctionCall {
                name: op.to_string(),
                args: vec![self.clone(), other.clone()],
            },
        }
    }

    fn unary_op(&self, op: &str) -> Self {
        Self {
            kind: ExprKind::FunctionCall {
                name: op.to_string(),
                args: vec![self.clone()],
            },
        }
    }
}

#[pymethods]
impl Expression {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        lit_from_py(value)
    }

    fn __repr__(&self) -> String {
        format!("Expression({})", self.render_sql())
    }
    fn __str__(&self) -> String {
        self.render_sql()
    }
    fn to_sql(&self) -> String {
        self.render_sql()
    }
    fn alias(&self, name: &str) -> Self {
        self.make_alias(name)
    }
    fn set_alias(&self, name: &str) -> Self {
        self.make_alias(name)
    }

    // Arithmetic operators. Operands may be Expressions or raw Python scalars
    // (coerced to literals), so `col("b") == 42` builds an Expression rather
    // than falling back to Python's default object comparison.
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("+", &coerce_operand(other)?))
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(coerce_operand(other)?.binary_op("+", self))
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("-", &coerce_operand(other)?))
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(coerce_operand(other)?.binary_op("-", self))
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("*", &coerce_operand(other)?))
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(coerce_operand(other)?.binary_op("*", self))
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("/", &coerce_operand(other)?))
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(coerce_operand(other)?.binary_op("/", self))
    }
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("//", &coerce_operand(other)?))
    }
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("%", &coerce_operand(other)?))
    }
    fn __pow__(&self, other: &Bound<'_, PyAny>, _mod: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(self.binary_op("^", &coerce_operand(other)?))
    }
    fn __neg__(&self) -> Self {
        self.unary_op("-")
    }

    // Logical operators
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("AND", &coerce_operand(other)?))
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("OR", &coerce_operand(other)?))
    }
    fn __invert__(&self) -> Self {
        self.unary_op("NOT")
    }

    // Comparison operators
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("=", &coerce_operand(other)?))
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("<>", &coerce_operand(other)?))
    }
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("<", &coerce_operand(other)?))
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op("<=", &coerce_operand(other)?))
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op(">", &coerce_operand(other)?))
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.binary_op(">=", &coerce_operand(other)?))
    }

    // Expression methods (order modifiers render without wrapping parens)
    fn asc(&self) -> Self {
        Self {
            kind: ExprKind::Bare(format!("{} ASC", self.render_sql())),
        }
    }
    fn desc(&self) -> Self {
        Self {
            kind: ExprKind::Bare(format!("{} DESC", self.render_sql())),
        }
    }
    fn nulls_first(&self) -> Self {
        Self {
            kind: ExprKind::Bare(format!("{} NULLS FIRST", self.render_sql())),
        }
    }
    fn nulls_last(&self) -> Self {
        Self {
            kind: ExprKind::Bare(format!("{} NULLS LAST", self.render_sql())),
        }
    }
    fn cast(&self, type_str: &str) -> Self {
        Self {
            kind: ExprKind::RawSql(format!("CAST({} AS {type_str})", self.render_sql())),
        }
    }

    fn isin(&self, values: Vec<Expression>) -> Self {
        let vals: Vec<String> = values.iter().map(|v| v.render_sql()).collect();
        Self {
            kind: ExprKind::RawSql(format!("{} IN ({})", self.render_sql(), vals.join(", "))),
        }
    }

    fn isnotin(&self, values: Vec<Expression>) -> Self {
        let vals: Vec<String> = values.iter().map(|v| v.render_sql()).collect();
        Self {
            kind: ExprKind::RawSql(format!(
                "{} NOT IN ({})",
                self.render_sql(),
                vals.join(", ")
            )),
        }
    }

    fn isnull(&self) -> Self {
        Self {
            kind: ExprKind::RawSql(format!("{} IS NULL", self.render_sql())),
        }
    }
    fn isnotnull(&self) -> Self {
        Self {
            kind: ExprKind::RawSql(format!("{} IS NOT NULL", self.render_sql())),
        }
    }

    fn between(&self, lower: &Expression, upper: &Expression) -> Self {
        Self {
            kind: ExprKind::RawSql(format!(
                "{} BETWEEN {} AND {}",
                self.render_sql(),
                lower.render_sql(),
                upper.render_sql()
            )),
        }
    }

    fn get_name(&self) -> String {
        match &self.kind {
            ExprKind::Column(name) => name.clone(),
            ExprKind::Aliased { alias, .. } => alias.clone(),
            _ => String::new(),
        }
    }
}

pub fn col(name: &str) -> Expression {
    Expression {
        kind: ExprKind::Column(name.to_string()),
    }
}
pub fn lit<T: Into<LiteralValue>>(value: T) -> Expression {
    Expression {
        kind: ExprKind::Literal(value.into()),
    }
}
pub fn sql_expr(sql: &str) -> Expression {
    Expression {
        kind: ExprKind::RawSql(sql.to_string()),
    }
}

/// Coerce an operator operand: an existing Expression passes through, any other
/// Python value becomes a literal Expression. Lets `col("b") == 42` build an
/// Expression instead of falling back to Python's default comparison.
fn coerce_operand(other: &Bound<'_, PyAny>) -> PyResult<Expression> {
    if let Ok(e) = other.extract::<Expression>() {
        Ok(e)
    } else {
        lit_from_py(other)
    }
}

pub fn lit_from_py(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    if value.is_none() {
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Null),
        });
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Boolean(b)),
        });
    }
    if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Boolean(b)),
        });
    }
    if value.is_instance_of::<PyInt>() {
        let i: i64 = value.extract()?;
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Int(i)),
        });
    }
    if value.is_instance_of::<PyFloat>() {
        let f: f64 = value.extract()?;
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Float(f)),
        });
    }
    if value.is_instance_of::<PyString>() {
        let s: String = value.extract()?;
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Str(s)),
        });
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Str(s)),
        });
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Int(i)),
        });
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Expression {
            kind: ExprKind::Literal(LiteralValue::Float(f)),
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot create literal from value of type {}",
        value.get_type().name()?
    )))
}

impl From<bool> for LiteralValue {
    fn from(v: bool) -> Self {
        LiteralValue::Boolean(v)
    }
}
impl From<i64> for LiteralValue {
    fn from(v: i64) -> Self {
        LiteralValue::Int(v)
    }
}
impl From<i32> for LiteralValue {
    fn from(v: i32) -> Self {
        LiteralValue::Int(v as i64)
    }
}
impl From<f64> for LiteralValue {
    fn from(v: f64) -> Self {
        LiteralValue::Float(v)
    }
}
impl From<f32> for LiteralValue {
    fn from(v: f32) -> Self {
        LiteralValue::Float(v as f64)
    }
}
impl From<String> for LiteralValue {
    fn from(v: String) -> Self {
        LiteralValue::Str(v)
    }
}
impl From<&str> for LiteralValue {
    fn from(v: &str) -> Self {
        LiteralValue::Str(v.to_string())
    }
}

fn is_infix_operator(name: &str) -> bool {
    matches!(
        name,
        "+" | "-"
            | "*"
            | "/"
            | "//"
            | "%"
            | "^"
            | "="
            | "<>"
            | "!="
            | "<"
            | "<="
            | ">"
            | ">="
            | "AND"
            | "OR"
            | "LIKE"
            | "ILIKE"
            | "||"
    )
}

fn quote_identifier(name: &str) -> String {
    if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
        name.to_string()
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

fn render_literal(l: &LiteralValue) -> String {
    match l {
        LiteralValue::Null => "NULL".to_string(),
        LiteralValue::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        LiteralValue::Int(i) => i.to_string(),
        LiteralValue::Float(f) => f.to_string(),
        LiteralValue::Str(s) => format!("'{}'", s.replace('\'', "''")),
    }
}
