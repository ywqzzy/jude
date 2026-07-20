//! Physical decomposition — turn a `LogicalPlan` into the pieces a *general*
//! streaming stage-DAG executor consumes, so ANY chain of shuffle boundaries
//! (agg→join→order, multi-join, distinct-over-union, …) runs distributed with
//! intermediate exchanges kept in the object store — never gathered to the
//! driver between stages.
//!
//! This is the runtime `stage.rs` deferred. `stage.rs` produces the *shape* (a
//! DAG of stages); here we produce, for the root of any (sub)plan, exactly what
//! the executor needs to run one step and recurse:
//!
//! - the **partition-wise region at the top** (everything above the nearest
//!   shuffle), rendered as SQL over a placeholder table `part`, plus whether it
//!   is safe to push per-partition (row-wise) or needs a gather (Limit/Sample);
//! - the **shuffle at the boundary** (kind + keys + join-type / set-op), and
//! - the **child sub-plans** feeding that shuffle, to recurse on.
//!
//! No fault tolerance (no spooling/attempts) — this is the streaming executor
//! the design promised, deliberately without Vane's FTE framework.

use std::sync::Arc;

use crate::plan::LogicalPlan;

/// Is this operator a shuffle boundary (needs a global view across partitions)?
pub fn is_shuffle(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Aggregate { .. }
            | LogicalPlan::Join { .. }
            | LogicalPlan::SetOp { .. }
            | LogicalPlan::Order { .. }
            | LogicalPlan::Distinct { .. }
            | LogicalPlan::Repartition { .. }
    )
}

/// Does the subtree contain any shuffle boundary?
pub fn subtree_has_shuffle(plan: &LogicalPlan) -> bool {
    if is_shuffle(plan) {
        return true;
    }
    unary_input(plan)
        .map(|c| subtree_has_shuffle(c))
        .unwrap_or(false)
        || binary_inputs(plan)
            .map(|(l, r)| subtree_has_shuffle(l) || subtree_has_shuffle(r))
            .unwrap_or(false)
}

/// Does the subtree contain a UDF (MapBatches) node? Such regions aren't
/// SQL-expressible, so the executor falls back to local materialization.
pub fn has_map_batches(plan: &LogicalPlan) -> bool {
    if matches!(plan, LogicalPlan::MapBatches { .. }) {
        return true;
    }
    unary_input(plan).map(has_map_batches).unwrap_or(false)
        || binary_inputs(plan)
            .map(|(l, r)| has_map_batches(l) || has_map_batches(r))
            .unwrap_or(false)
}

/// The single input of a unary op (None for leaves and binary ops).
fn unary_input(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Order { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Alias { input, .. }
        | LogicalPlan::Summarize { input }
        | LogicalPlan::Query { input, .. }
        | LogicalPlan::Repartition { input, .. }
        | LogicalPlan::MapBatches { input, .. }
        | LogicalPlan::Unnest { input, .. }
        | LogicalPlan::Sample { input, .. } => Some(input),
        _ => None,
    }
}

fn binary_inputs(plan: &LogicalPlan) -> Option<(&LogicalPlan, &LogicalPlan)> {
    match plan {
        LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
            Some((left, right))
        }
        _ => None,
    }
}

/// Rebuild a unary partition-wise op with a new input plan.
fn rebuild_unary(plan: &LogicalPlan, new_input: Arc<LogicalPlan>) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter { predicate, .. } => LogicalPlan::Filter {
            input: new_input,
            predicate: predicate.clone(),
        },
        LogicalPlan::Project { exprs, .. } => LogicalPlan::Project {
            input: new_input,
            exprs: exprs.clone(),
        },
        LogicalPlan::Limit { n, offset, .. } => LogicalPlan::Limit {
            input: new_input,
            n: *n,
            offset: *offset,
        },
        LogicalPlan::Alias { name, .. } => LogicalPlan::Alias {
            input: new_input,
            name: name.clone(),
        },
        LogicalPlan::Summarize { .. } => LogicalPlan::Summarize { input: new_input },
        LogicalPlan::Query { cte, sql, .. } => LogicalPlan::Query {
            input: new_input,
            cte: cte.clone(),
            sql: sql.clone(),
        },
        LogicalPlan::MapBatches { marker, .. } => LogicalPlan::MapBatches {
            input: new_input,
            marker: *marker,
        },
        LogicalPlan::Unnest {
            column, recursive, ..
        } => LogicalPlan::Unnest {
            input: new_input,
            column: column.clone(),
            recursive: *recursive,
        },
        LogicalPlan::Sample { spec, .. } => LogicalPlan::Sample {
            input: new_input,
            spec: spec.clone(),
        },
        // Not a unary pw op we peel; return as-is (shouldn't happen).
        other => other.clone(),
    }
}

/// Is this pw op safe to push *per-partition* (row-wise), i.e. running it on
/// each output shard then concatenating equals running it on the whole? Filter/
/// Project/Alias/Unnest/Query are; Limit/Sample/Summarize need a global view.
fn is_pushable(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Filter { .. }
            | LogicalPlan::Project { .. }
            | LogicalPlan::Alias { .. }
            | LogicalPlan::Unnest { .. }
            | LogicalPlan::Query { .. }
    )
}

/// The result of peeling the top partition-wise region.
pub struct Peel {
    /// The region rendered over placeholder `part` (a plan whose leaf is
    /// `Table{"part"}`). `None` when the root itself is the boundary (trivial).
    pub local: Option<LogicalPlan>,
    /// Whether the whole peeled region is push-per-partition safe.
    pub pushable: bool,
    /// Whether the peeled region contains a UDF (→ not SQL-expressible).
    pub has_udf: bool,
    /// The boundary node the region sits on (a shuffle, or a leaf base).
    pub boundary: LogicalPlan,
}

/// Peel the partition-wise region at the top of `plan`, stopping at the nearest
/// shuffle boundary or leaf. Returns the region-over-`part`, its pushability,
/// and the boundary node.
pub fn peel(plan: &LogicalPlan) -> Peel {
    if is_shuffle(plan) || unary_input(plan).is_none() && binary_inputs(plan).is_none() {
        // Boundary reached (a shuffle, or a leaf with no children): the local
        // region is just the placeholder — the caller handles the boundary.
        return Peel {
            local: None,
            pushable: true,
            has_udf: false,
            boundary: plan.clone(),
        };
    }
    // A unary pw op: recurse into its input, rebuild the op over the peeled child.
    let input = unary_input(plan).expect("pw op has one input");
    let child = peel(input);
    let child_local = child.local.unwrap_or(LogicalPlan::Table {
        name: "part".into(),
    });
    let local = rebuild_unary(plan, Arc::new(child_local));
    let is_udf = matches!(plan, LogicalPlan::MapBatches { .. });
    Peel {
        local: Some(local),
        pushable: child.pushable && is_pushable(plan),
        has_udf: child.has_udf || is_udf,
        boundary: child.boundary,
    }
}

/// Render a peeled local region to SQL over `part`. Errors only if a Materialized
/// leaf survives (it shouldn't — those are handled as UDF/leaf cases).
pub fn render_local_sql(local: &LogicalPlan) -> Result<String, crate::error::Error> {
    local.to_sql(&mut |_p| Ok("part".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{JoinType, LogicalPlan};

    fn scan(name: &str) -> Arc<LogicalPlan> {
        Arc::new(LogicalPlan::Table {
            name: name.to_string(),
        })
    }

    #[test]
    fn no_shuffle_detected_for_scan_chain() {
        let p = LogicalPlan::Filter {
            input: scan("t"),
            predicate: "a > 1".into(),
        };
        assert!(!subtree_has_shuffle(&p));
    }

    #[test]
    fn shuffle_detected_under_pw() {
        // Filter(Aggregate(scan)) — a shuffle is under the filter.
        let p = LogicalPlan::Filter {
            input: Arc::new(LogicalPlan::Aggregate {
                input: scan("t"),
                group: vec!["g".into()],
                aggs: vec!["SUM(v)".into()],
            }),
            predicate: "s > 0".into(),
        };
        assert!(subtree_has_shuffle(&p));
    }

    #[test]
    fn peel_stops_at_shuffle_and_renders_over_part() {
        // Project(Filter(Aggregate(scan))): peel Project+Filter, boundary=Aggregate.
        let agg = LogicalPlan::Aggregate {
            input: scan("t"),
            group: vec!["g".into()],
            aggs: vec!["SUM(v) AS s".into()],
        };
        let p = LogicalPlan::Project {
            input: Arc::new(LogicalPlan::Filter {
                input: Arc::new(agg),
                predicate: "s > 0".into(),
            }),
            exprs: vec!["g".into(), "s".into()],
        };
        let peel = peel(&p);
        assert!(peel.pushable, "filter+project are row-wise");
        assert!(!peel.has_udf);
        assert!(matches!(peel.boundary, LogicalPlan::Aggregate { .. }));
        let sql = render_local_sql(&peel.local.unwrap()).unwrap();
        assert!(sql.contains("part"), "renders over placeholder: {sql}");
        assert!(sql.contains("g, s") || sql.contains("s > 0"), "{sql}");
    }

    #[test]
    fn root_shuffle_has_no_local_region() {
        let p = LogicalPlan::Aggregate {
            input: scan("t"),
            group: vec!["g".into()],
            aggs: vec!["COUNT(*)".into()],
        };
        let peel = peel(&p);
        assert!(peel.local.is_none(), "root is the boundary → no top region");
        assert!(matches!(peel.boundary, LogicalPlan::Aggregate { .. }));
    }

    #[test]
    fn limit_region_not_pushable() {
        // Limit(Aggregate(scan)) — the LIMIT above the shuffle needs a gather.
        let p = LogicalPlan::Limit {
            input: Arc::new(LogicalPlan::Aggregate {
                input: scan("t"),
                group: vec!["g".into()],
                aggs: vec!["COUNT(*)".into()],
            }),
            n: 10,
            offset: 0,
        };
        let peel = peel(&p);
        assert!(!peel.pushable, "LIMIT must not be pushed per-partition");
        assert!(matches!(peel.boundary, LogicalPlan::Aggregate { .. }));
    }

    #[test]
    fn udf_region_flagged() {
        let p = LogicalPlan::MapBatches {
            input: Arc::new(LogicalPlan::Aggregate {
                input: scan("t"),
                group: vec!["g".into()],
                aggs: vec!["COUNT(*)".into()],
            }),
            marker: 7,
        };
        let peel = peel(&p);
        assert!(peel.has_udf, "UDF above a shuffle is flagged for fallback");
    }

    #[test]
    fn leaf_boundary_is_the_scan() {
        // Pure pw over a scan: boundary is the scan, region renders over part.
        let p = LogicalPlan::Filter {
            input: scan("t"),
            predicate: "a > 1".into(),
        };
        let peel = peel(&p);
        assert!(matches!(peel.boundary, LogicalPlan::Table { .. }));
        assert!(!has_map_batches(&p));
        let sql = render_local_sql(&peel.local.unwrap()).unwrap();
        assert!(sql.contains("part") && sql.contains("a > 1"), "{sql}");
    }

    #[test]
    fn join_boundary_detected() {
        let p = LogicalPlan::Join {
            left: scan("a"),
            right: scan("b"),
            condition: "id".into(),
            how: JoinType::Inner,
        };
        let peel = peel(&p);
        assert!(peel.local.is_none());
        assert!(matches!(peel.boundary, LogicalPlan::Join { .. }));
    }
}
