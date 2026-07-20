//! Stage planning — cut a `LogicalPlan` tree into an ordered DAG of stages at
//! shuffle boundaries (distributed milestone 5).
//!
//! A distributed plan is the operator tree sliced at the operators that need a
//! global view — aggregate, join, distinct, sort, set-ops, explicit repartition.
//! Everything between two boundaries is one *stage* that runs partition-wise
//! (the same local query on each shard); a boundary is where data must be
//! shuffled/exchanged. This generalizes the single-level classifier that used to
//! live in `Relation::plan_json` into a recursive planner over the whole tree.
//!
//! `stage.rs` ships the stage *shape* (the DAG). The general streaming executor
//! that consumes an arbitrary stage DAG now lives in `dist::physical` (the
//! per-node decomposition) + `RayRunner.execute_dag` (the streaming runtime):
//! it runs every shuffle boundary distributed and exchanges intermediate results
//! as object-store shard refs — no gather to the driver between stages, and
//! deliberately no fault-tolerance (spooling/attempts) machinery. The two-phase
//! aggregate and hash join remain as purpose-built fast paths for single-shuffle
//! plans; `execute_dag` composes them for nested shuffles.

use std::sync::Arc;

use crate::plan::LogicalPlan;

/// What kind of boundary produced a stage's *root* operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageKind {
    /// A leaf scan (table / SQL / materialized source) — no upstream stage.
    Scan,
    /// A partition-wise region with no shuffle at its root (e.g. a Project/Filter
    /// chain over a scan). Rare as a *root* kind; mostly the final stage.
    Partitionwise,
    /// The root operator is a shuffle: it consumes upstream stage output
    /// redistributed by `partition_keys`.
    Shuffle,
}

/// One stage in the plan DAG.
#[derive(Clone, Debug)]
pub struct Stage {
    pub id: u32,
    pub kind: StageKind,
    /// The top operator's name (Aggregate / Join / Sql / …).
    pub op: &'static str,
    /// Keys this stage shuffles on (group-by cols, join condition, order keys,
    /// repartition-by). Empty for a scan or a keyless shuffle (distinct/union).
    pub partition_keys: Vec<String>,
    /// Upstream stage ids this stage consumes (0, 1 for a join, …).
    pub inputs: Vec<u32>,
}

/// Is this operator a shuffle boundary (needs a global view across partitions)?
fn is_shuffle(plan: &LogicalPlan) -> bool {
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

/// The shuffle keys for a boundary operator (empty when the op shuffles without
/// a key, e.g. distinct / union / global aggregate).
fn shuffle_keys(plan: &LogicalPlan) -> Vec<String> {
    match plan {
        LogicalPlan::Aggregate { group, .. } => group.clone(),
        LogicalPlan::Join { condition, .. } => vec![condition.clone()],
        LogicalPlan::Order { keys, .. } => keys.clone(),
        LogicalPlan::Repartition { by, .. } => by.clone(),
        _ => Vec::new(),
    }
}

/// The direct child plans of a node (0 for a leaf, 1 for unary, 2 for join/setop).
fn children(plan: &LogicalPlan) -> Vec<&Arc<LogicalPlan>> {
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
        | LogicalPlan::Sample { input, .. } => vec![input],
        LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
            vec![left, right]
        }
        _ => Vec::new(),
    }
}

/// Plan the stage DAG for `root`. Returns stages in dependency order (every
/// stage's `inputs` refer to stages earlier in the list), the last stage being
/// the root of the query.
pub fn plan_stages(root: &LogicalPlan) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    build(root, &mut stages);
    stages
}

/// Recursively emit stages for `plan`, returning the id of the stage that
/// produces `plan`'s output. A shuffle operator becomes its own stage whose
/// inputs are the stages of its children; a partition-wise region rides on the
/// same stage as its (single) child, or becomes a leaf/final stage.
fn build(plan: &LogicalPlan, stages: &mut Vec<Stage>) -> u32 {
    let kids = children(plan);

    // Leaf scan: its own Scan stage.
    if kids.is_empty() {
        let id = stages.len() as u32;
        stages.push(Stage {
            id,
            kind: StageKind::Scan,
            op: plan.op_name(),
            partition_keys: Vec::new(),
            inputs: Vec::new(),
        });
        return id;
    }

    // Recurse into children first (post-order), collecting the stage id each
    // child's output lands in.
    let input_ids: Vec<u32> = kids.iter().map(|c| build(c, stages)).collect();

    if is_shuffle(plan) {
        // A shuffle operator is a new stage consuming its children's stages.
        let id = stages.len() as u32;
        stages.push(Stage {
            id,
            kind: StageKind::Shuffle,
            op: plan.op_name(),
            partition_keys: shuffle_keys(plan),
            inputs: input_ids,
        });
        id
    } else {
        // A partition-wise unary op fuses into its child's stage (no boundary).
        // (All non-shuffle nodes here are unary, so exactly one input id.)
        input_ids[0]
    }
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
    fn scan_only_is_one_stage() {
        let stages = plan_stages(&LogicalPlan::Table { name: "t".into() });
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].kind, StageKind::Scan);
    }

    #[test]
    fn partitionwise_chain_fuses_into_scan_stage() {
        // Filter(Project(Scan)) — no shuffle, so one stage.
        let plan = LogicalPlan::Filter {
            input: Arc::new(LogicalPlan::Project {
                input: scan("t"),
                exprs: vec!["a".into()],
            }),
            predicate: "a > 1".into(),
        };
        let stages = plan_stages(&plan);
        assert_eq!(stages.len(), 1, "no shuffle => single stage");
        assert_eq!(stages[0].kind, StageKind::Scan);
    }

    #[test]
    fn aggregate_is_two_stages() {
        // Aggregate(Filter(Scan)) — scan stage + shuffle(aggregate) stage.
        let plan = LogicalPlan::Aggregate {
            input: Arc::new(LogicalPlan::Filter {
                input: scan("t"),
                predicate: "a > 1".into(),
            }),
            group: vec!["g".into()],
            aggs: vec!["SUM(v)".into()],
        };
        let stages = plan_stages(&plan);
        assert_eq!(stages.len(), 2);
        let agg = stages.last().unwrap();
        assert_eq!(agg.kind, StageKind::Shuffle);
        assert_eq!(agg.op, "Aggregate");
        assert_eq!(agg.partition_keys, vec!["g".to_string()]);
        assert_eq!(agg.inputs, vec![0]); // consumes the scan stage
    }

    #[test]
    fn join_is_three_stages() {
        // Join(Scan(a), Scan(b)) — two scan stages + one shuffle(join) stage.
        let plan = LogicalPlan::Join {
            left: scan("a"),
            right: scan("b"),
            condition: "a.id = b.id".into(),
            how: JoinType::Inner,
        };
        let stages = plan_stages(&plan);
        assert_eq!(stages.len(), 3);
        let join = stages.last().unwrap();
        assert_eq!(join.kind, StageKind::Shuffle);
        assert_eq!(join.op, "Join");
        assert_eq!(join.inputs, vec![0, 1]); // both scan stages
        assert_eq!(join.partition_keys, vec!["a.id = b.id".to_string()]);
    }

    #[test]
    fn stacked_shuffles_chain() {
        // Order(Aggregate(Scan)) — scan + aggregate-shuffle + order-shuffle = 3.
        let plan = LogicalPlan::Order {
            input: Arc::new(LogicalPlan::Aggregate {
                input: scan("t"),
                group: vec!["g".into()],
                aggs: vec!["COUNT(*)".into()],
            }),
            keys: vec!["g".into()],
        };
        let stages = plan_stages(&plan);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].kind, StageKind::Scan);
        assert_eq!(stages[1].op, "Aggregate");
        assert_eq!(stages[2].op, "Order");
        assert_eq!(stages[2].inputs, vec![1]); // order consumes the aggregate stage
    }
}
