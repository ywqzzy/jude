use once_cell::sync::Lazy;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;

pub const RUNNER_VALUES: &[&str] = &["local", "ray"];

static ENV_OVERRIDES: Lazy<RwLock<HashMap<String, String>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

fn get_var(name: &str) -> Option<String> {
    if let Some(v) = ENV_OVERRIDES.read().get(name) {
        return Some(v.clone());
    }
    std::env::var(name).ok()
}

fn set_var(name: &str, value: &str) {
    ENV_OVERRIDES
        .write()
        .insert(name.to_string(), value.to_string());
}

fn parse_bool(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

fn parse_int(s: &str) -> i64 {
    s.parse().unwrap_or(0)
}

#[pyclass(name = "EnvRegistry")]
pub struct EnvRegistry {
    _private: (),
}

impl EnvRegistry {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

#[pymethods]
impl EnvRegistry {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[getter]
    pub fn runner(&self) -> String {
        get_var("JUDE_RUNNER")
            .map(|v| v.to_lowercase())
            .unwrap_or_else(|| "ray".to_string())
    }

    #[setter]
    pub fn set_runner(&self, value: &str) -> PyResult<()> {
        let normalized = value.to_lowercase();
        if !RUNNER_VALUES.contains(&normalized.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "runner must be 'local' or 'ray'",
            ));
        }
        set_var("JUDE_RUNNER", &normalized);
        Ok(())
    }

    #[getter]
    pub fn ray_scan_task_size_grouping(&self) -> bool {
        get_var("JUDE_RAY_SCAN_TASK_SIZE_GROUPING")
            .map(|v| parse_bool(&v))
            .unwrap_or(true)
    }

    #[setter]
    pub fn set_ray_scan_task_size_grouping(&self, value: bool) {
        set_var(
            "JUDE_RAY_SCAN_TASK_SIZE_GROUPING",
            if value { "true" } else { "false" },
        );
    }

    #[getter]
    pub fn ray_max_task_backlog(&self) -> i64 {
        get_var("JUDE_RAY_MAX_TASK_BACKLOG")
            .map(|v| parse_int(&v))
            .unwrap_or(0)
    }

    #[setter]
    pub fn set_ray_max_task_backlog(&self, value: i64) {
        set_var("JUDE_RAY_MAX_TASK_BACKLOG", &value.to_string());
    }

    #[getter]
    pub fn ray_scan_task_open_cost_bytes(&self) -> i64 {
        get_var("JUDE_RAY_SCAN_TASK_OPEN_COST_BYTES")
            .map(|v| parse_int(&v))
            .unwrap_or(4 * 1024 * 1024)
    }

    #[setter]
    pub fn set_ray_scan_task_open_cost_bytes(&self, value: i64) {
        set_var("JUDE_RAY_SCAN_TASK_OPEN_COST_BYTES", &value.to_string());
    }

    #[getter]
    pub fn ray_scan_task_min_partition_num(&self) -> i64 {
        get_var("JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM")
            .map(|v| parse_int(&v))
            .unwrap_or(0)
    }

    #[setter]
    pub fn set_ray_scan_task_min_partition_num(&self, value: i64) {
        set_var("JUDE_RAY_SCAN_TASK_MIN_PARTITION_NUM", &value.to_string());
    }

    #[getter]
    pub fn ray_init_sql(&self) -> String {
        get_var("JUDE_RAY_INIT_SQL").unwrap_or_default()
    }

    #[setter]
    pub fn set_ray_init_sql(&self, value: &str) {
        set_var("JUDE_RAY_INIT_SQL", value);
    }

    #[getter]
    pub fn udf_parallel(&self) -> bool {
        get_var("JUDE_UDF_PARALLEL")
            .map(|v| parse_bool(&v))
            .unwrap_or(false)
    }

    #[setter]
    pub fn set_udf_parallel(&self, value: bool) {
        set_var("JUDE_UDF_PARALLEL", if value { "true" } else { "false" });
    }

    #[getter]
    pub fn udf_arrow_fastpath(&self) -> bool {
        get_var("JUDE_UDF_ARROW_FASTPATH")
            .map(|v| parse_bool(&v))
            .unwrap_or(true)
    }

    #[setter]
    pub fn set_udf_arrow_fastpath(&self, value: bool) {
        set_var(
            "JUDE_UDF_ARROW_FASTPATH",
            if value { "true" } else { "false" },
        );
    }

    #[getter]
    pub fn local_exchange_buffer(&self) -> String {
        get_var("JUDE_LOCAL_EXCHANGE_BUFFER").unwrap_or_else(|| "32MB".to_string())
    }

    #[setter]
    pub fn set_local_exchange_buffer(&self, value: &str) {
        set_var("JUDE_LOCAL_EXCHANGE_BUFFER", value);
    }

    pub fn as_dict(&self) -> HashMap<String, String> {
        let mut d = HashMap::new();
        d.insert("runner".into(), self.runner());
        d.insert(
            "ray_scan_task_size_grouping".into(),
            self.ray_scan_task_size_grouping().to_string(),
        );
        d.insert(
            "ray_max_task_backlog".into(),
            self.ray_max_task_backlog().to_string(),
        );
        d.insert(
            "ray_scan_task_open_cost_bytes".into(),
            self.ray_scan_task_open_cost_bytes().to_string(),
        );
        d.insert(
            "ray_scan_task_min_partition_num".into(),
            self.ray_scan_task_min_partition_num().to_string(),
        );
        d.insert("ray_init_sql".into(), self.ray_init_sql());
        d.insert("udf_parallel".into(), self.udf_parallel().to_string());
        d.insert(
            "udf_arrow_fastpath".into(),
            self.udf_arrow_fastpath().to_string(),
        );
        d.insert("local_exchange_buffer".into(), self.local_exchange_buffer());
        d
    }

    fn __repr__(&self) -> String {
        let d = self.as_dict();
        let items: Vec<String> = d.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
        format!("EnvRegistry({})", items.join(", "))
    }
}
