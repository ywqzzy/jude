//! Logical plan IR for jude relations.
//!
//! A `Relation` is a thin handle over an `Arc<LogicalPlan>` — a real operator
//! tree, not a SQL string. This is what makes the system extensible: the
//! distributed runner walks the tree to find scan/exchange/map boundaries,
//! multimodal operators are first-class nodes, and optimizer passes are tree
//! rewrites. SQL is merely *one lowering* of the IR (`to_sql`) used for local
//! DuckDB execution; nothing in the plan is stored as opaque text.
//!
//! Vane keeps a native `shared_ptr<Relation>` operator DAG for the same reason;
//! this is the equivalent for a stock-DuckDB-backed engine.

use duckdb::arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::connection::{escape_sql_string, quote_ident};

/// A node in the logical plan tree. Children are `Arc<LogicalPlan>` so plans
/// are cheaply cloneable and shareable (a derived relation references its
/// input without copying it).
#[derive(Clone)]
pub enum LogicalPlan {
    /// Scan a base table / view by name.
    Table { name: String },
    /// Scan a data source via a table function, e.g. read_parquet('...').
    ScanFunction { func: String, args: Vec<String> },
    /// A raw SQL query used as a leaf (escape hatch: sql(), values(), etc.).
    RawSql { sql: String },
    /// In-memory Arrow batches (e.g. produced by map_batches or from_arrow).
    /// Lowered by materializing into a TEMP table on demand.
    Materialized { batches: Arc<Vec<RecordBatch>> },
    /// WHERE filter.
    Filter {
        input: Arc<LogicalPlan>,
        predicate: String,
    },
    /// Projection (SELECT list). `exprs` empty means SELECT *.
    Project {
        input: Arc<LogicalPlan>,
        exprs: Vec<String>,
    },
    /// GROUP BY aggregation. `group` empty means global aggregate.
    Aggregate {
        input: Arc<LogicalPlan>,
        group: Vec<String>,
        aggs: Vec<String>,
    },
    /// Join two inputs.
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        condition: String,
        how: JoinType,
    },
    /// Set operation (UNION ALL / UNION / INTERSECT / EXCEPT).
    SetOp {
        op: SetOpKind,
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
    },
    /// ORDER BY.
    Order {
        input: Arc<LogicalPlan>,
        keys: Vec<String>,
    },
    /// LIMIT [OFFSET].
    Limit {
        input: Arc<LogicalPlan>,
        n: usize,
        offset: usize,
    },
    /// DISTINCT.
    Distinct { input: Arc<LogicalPlan> },
    /// Alias the input as a named subquery.
    Alias {
        input: Arc<LogicalPlan>,
        name: String,
    },
    /// Summarize (DESCRIBE / SUMMARIZE).
    Summarize { input: Arc<LogicalPlan> },
    /// WITH cte AS (input) <sql> — run arbitrary SQL over a named CTE.
    Query {
        input: Arc<LogicalPlan>,
        cte: String,
        sql: String,
    },
    /// A partition/exchange boundary — a hint for the distributed runner. Row
    /// set is identical to the input (identity when executed locally).
    Repartition {
        input: Arc<LogicalPlan>,
        num_partitions: usize,
        by: Vec<String>,
    },
    /// Apply a Python batch UDF (map_batches / flat_map). The udf is opaque to
    /// the IR here (held out-of-band by the Relation) — this node marks WHERE a
    /// UDF applies so the distributed planner can place it; local lowering
    /// materializes the input then runs the UDF in Rust.
    MapBatches {
        input: Arc<LogicalPlan>,
        marker: u64,
    },
    /// UNNEST a list/array column into one row per element (explode).
    Unnest {
        input: Arc<LogicalPlan>,
        column: String,
        recursive: bool,
    },
    /// Random sample of rows (percentage in 0..=100, or a fixed row count).
    Sample {
        input: Arc<LogicalPlan>,
        spec: String,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
    Cross,
    Semi,
    Anti,
}

impl JoinType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "left" | "left_outer" => JoinType::Left,
            "right" | "right_outer" => JoinType::Right,
            "outer" | "full" | "full_outer" => JoinType::Outer,
            "cross" => JoinType::Cross,
            "semi" => JoinType::Semi,
            "anti" => JoinType::Anti,
            _ => JoinType::Inner,
        }
    }
    /// Lowercase name ("inner"/"left"/…) for the distributed runner's `how` arg.
    pub fn name(self) -> &'static str {
        match self {
            JoinType::Inner => "inner",
            JoinType::Left => "left",
            JoinType::Right => "right",
            JoinType::Outer => "outer",
            JoinType::Cross => "cross",
            JoinType::Semi => "semi",
            JoinType::Anti => "anti",
        }
    }

    fn sql_kw(self) -> &'static str {
        match self {
            JoinType::Inner => "INNER JOIN",
            JoinType::Left => "LEFT JOIN",
            JoinType::Right => "RIGHT JOIN",
            JoinType::Outer => "FULL OUTER JOIN",
            JoinType::Cross => "CROSS JOIN",
            JoinType::Semi => "SEMI JOIN",
            JoinType::Anti => "ANTI JOIN",
        }
    }
}

#[derive(Clone, Copy)]
pub enum SetOpKind {
    UnionAll,
    Union,
    Intersect,
    Except,
}

impl SetOpKind {
    /// The SQL keyword for this set operation.
    pub fn sql_kw(self) -> &'static str {
        match self {
            SetOpKind::UnionAll => "UNION ALL",
            SetOpKind::Union => "UNION",
            SetOpKind::Intersect => "INTERSECT",
            SetOpKind::Except => "EXCEPT",
        }
    }
}

impl LogicalPlan {
    pub fn arc(self) -> Arc<LogicalPlan> {
        Arc::new(self)
    }

    /// Lower this plan to a SQL string that produces the same result set when
    /// executed against the connection. This is the *local-execution* lowering;
    /// distributed planning walks the tree directly instead.
    ///
    /// A `resolve_materialized` callback turns a Materialized node into a table
    /// name (by registering its batches as a TEMP table). It is only invoked
    /// for Materialized/MapBatches leaves.
    pub fn to_sql(
        &self,
        resolve: &mut dyn FnMut(&LogicalPlan) -> Result<String, crate::error::Error>,
    ) -> Result<String, crate::error::Error> {
        Ok(match self {
            LogicalPlan::Table { name } => format!(
                "SELECT * FROM {}",
                crate::connection::quote_qualified_name(name)
            ),
            LogicalPlan::ScanFunction { func, args } => {
                format!("SELECT * FROM {func}({})", args.join(", "))
            }
            LogicalPlan::RawSql { sql } => sql.clone(),
            LogicalPlan::Materialized { .. } => {
                let table = resolve(self)?;
                format!("SELECT * FROM {table}")
            }
            LogicalPlan::MapBatches { .. } => {
                // MapBatches is resolved by the Relation (it runs the UDF and
                // hands back a Materialized plan); if we reach here it means the
                // caller wants SQL over the UDF output, so resolve to a temp table.
                let table = resolve(self)?;
                format!("SELECT * FROM {table}")
            }
            LogicalPlan::Filter { input, predicate } => {
                format!(
                    "SELECT * FROM ({}) AS _t WHERE {predicate}",
                    input.to_sql(resolve)?
                )
            }
            LogicalPlan::Project { input, exprs } => {
                let proj = if exprs.is_empty() {
                    "*".to_string()
                } else {
                    exprs.join(", ")
                };
                format!("SELECT {proj} FROM ({}) AS _t", input.to_sql(resolve)?)
            }
            LogicalPlan::Aggregate { input, group, aggs } => {
                let inner = input.to_sql(resolve)?;
                if group.is_empty() {
                    // GROUP BY ALL: DuckDB auto-groups by the non-aggregate
                    // SELECT columns, so `aggregate("j, sum(i)")` groups by j,
                    // and a pure-aggregate `aggregate("sum(i)")` collapses to one
                    // row. Matches DuckDB's relational `aggregate` with no groups.
                    format!(
                        "SELECT {} FROM ({inner}) AS _t GROUP BY ALL",
                        aggs.join(", ")
                    )
                } else {
                    format!(
                        "SELECT {}, {} FROM ({inner}) AS _t GROUP BY {}",
                        group.join(", "),
                        aggs.join(", "),
                        group.join(", ")
                    )
                }
            }
            LogicalPlan::Join {
                left,
                right,
                condition,
                how,
            } => {
                let l = left.to_sql(resolve)?;
                let r = right.to_sql(resolve)?;
                if *how == JoinType::Cross {
                    format!("SELECT * FROM ({l}) AS lhs CROSS JOIN ({r}) AS rhs")
                } else if is_key_list(condition) {
                    // A bare column (or comma list) is an equi-join key: USING
                    // dedups the key column (DuckDB relational join semantics).
                    format!(
                        "SELECT * FROM ({l}) AS lhs {} ({r}) AS rhs USING ({condition})",
                        how.sql_kw()
                    )
                } else {
                    format!(
                        "SELECT * FROM ({l}) AS lhs {} ({r}) AS rhs ON {condition}",
                        how.sql_kw()
                    )
                }
            }
            LogicalPlan::SetOp { op, left, right } => {
                let l = left.to_sql(resolve)?;
                let r = right.to_sql(resolve)?;
                format!("({l}) {} ({r})", op.sql_kw())
            }
            LogicalPlan::Order { input, keys } => {
                format!(
                    "SELECT * FROM ({}) AS _t ORDER BY {}",
                    input.to_sql(resolve)?,
                    keys.join(", ")
                )
            }
            LogicalPlan::Limit { input, n, offset } => {
                let inner = input.to_sql(resolve)?;
                if *offset > 0 {
                    format!("SELECT * FROM ({inner}) AS _t LIMIT {n} OFFSET {offset}")
                } else {
                    format!("SELECT * FROM ({inner}) AS _t LIMIT {n}")
                }
            }
            LogicalPlan::Distinct { input } => {
                format!("SELECT DISTINCT * FROM ({}) AS _t", input.to_sql(resolve)?)
            }
            LogicalPlan::Alias { input, name } => {
                format!(
                    "SELECT * FROM ({}) AS {}",
                    input.to_sql(resolve)?,
                    quote_ident(name)
                )
            }
            LogicalPlan::Summarize { input } => {
                format!("SUMMARIZE ({})", input.to_sql(resolve)?)
            }
            LogicalPlan::Query { input, cte, sql } => {
                format!(
                    "WITH {} AS ({}) {sql}",
                    quote_ident(cte),
                    input.to_sql(resolve)?
                )
            }
            LogicalPlan::Repartition { input, .. } => {
                // Identity on the row set for local execution.
                format!("SELECT * FROM ({}) AS _t", input.to_sql(resolve)?)
            }
            LogicalPlan::Unnest {
                input,
                column,
                recursive,
            } => {
                let rec = if *recursive {
                    ", recursive := true"
                } else {
                    ""
                };
                // Explode one list column while keeping the others.
                format!(
                    "SELECT * EXCLUDE ({column}), UNNEST({column}{rec}) AS {column} FROM ({}) AS _t",
                    input.to_sql(resolve)?
                )
            }
            LogicalPlan::Sample { input, spec } => {
                format!(
                    "SELECT * FROM ({}) AS _t USING SAMPLE {spec}",
                    input.to_sql(resolve)?
                )
            }
        })
    }

    /// The requested partition count anywhere along this plan (the nearest
    /// Repartition hint), for the local runner's partitioning.
    pub fn partition_hint(&self) -> Option<usize> {
        match self {
            LogicalPlan::Repartition { num_partitions, .. } => Some(*num_partitions),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Order { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Alias { input, .. }
            | LogicalPlan::Summarize { input }
            | LogicalPlan::Query { input, .. }
            | LogicalPlan::MapBatches { input, .. }
            | LogicalPlan::Unnest { input, .. }
            | LogicalPlan::Sample { input, .. } => input.partition_hint(),
            _ => None,
        }
    }

    /// A short human-readable name for the top operator (for repr/debug).
    pub fn op_name(&self) -> &'static str {
        match self {
            LogicalPlan::Table { .. } => "Table",
            LogicalPlan::ScanFunction { .. } => "ScanFunction",
            LogicalPlan::RawSql { .. } => "Sql",
            LogicalPlan::Materialized { .. } => "Materialized",
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Project { .. } => "Project",
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::Join { .. } => "Join",
            LogicalPlan::SetOp { .. } => "SetOp",
            LogicalPlan::Order { .. } => "Order",
            LogicalPlan::Limit { .. } => "Limit",
            LogicalPlan::Distinct { .. } => "Distinct",
            LogicalPlan::Alias { .. } => "Alias",
            LogicalPlan::Summarize { .. } => "Summarize",
            LogicalPlan::Query { .. } => "Query",
            LogicalPlan::Repartition { .. } => "Repartition",
            LogicalPlan::MapBatches { .. } => "MapBatches",
            LogicalPlan::Unnest { .. } => "Unnest",
            LogicalPlan::Sample { .. } => "Sample",
        }
    }
}

/// True when `condition` is a bare equi-join key list (one or more identifiers
/// separated by commas, no operators) rather than a boolean ON expression — then
/// the join uses `USING (...)`, which dedups the key column(s). Mirrors DuckDB's
/// relational `join(other, "col")` semantics.
fn is_key_list(condition: &str) -> bool {
    let c = condition.trim();
    if c.is_empty() {
        return false;
    }
    c.split(',').all(|part| {
        let p = part.trim();
        !p.is_empty()
            && p.chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
            && p.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
}

/// Build a table-function scan plan with a single quoted-string argument.
pub fn scan_function_str(func: &str, path: &str) -> LogicalPlan {
    LogicalPlan::ScanFunction {
        func: func.to_string(),
        args: vec![format!("'{}'", escape_sql_string(path))],
    }
}
