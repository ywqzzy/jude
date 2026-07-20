use crate::env;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct Config {
    pub runner: String,
    pub ray_scan_task_size_grouping: bool,
    pub ray_max_task_backlog: i64,
    pub ray_scan_task_open_cost_bytes: i64,
    pub ray_scan_task_min_partition_num: i64,
    pub ray_init_sql: String,
    pub udf_parallel: bool,
    pub udf_arrow_fastpath: bool,
    pub local_exchange_buffer: String,
}

impl Config {
    pub fn from_env() -> Self {
        let reg = env::EnvRegistry::new();
        Self {
            runner: reg.runner(),
            ray_scan_task_size_grouping: reg.ray_scan_task_size_grouping(),
            ray_max_task_backlog: reg.ray_max_task_backlog(),
            ray_scan_task_open_cost_bytes: reg.ray_scan_task_open_cost_bytes(),
            ray_scan_task_min_partition_num: reg.ray_scan_task_min_partition_num(),
            ray_init_sql: reg.ray_init_sql(),
            udf_parallel: reg.udf_parallel(),
            udf_arrow_fastpath: reg.udf_arrow_fastpath(),
            local_exchange_buffer: reg.local_exchange_buffer(),
        }
    }
}

#[pymethods]
impl Config {
    #[getter]
    fn runner(&self) -> String {
        self.runner.clone()
    }
    #[getter]
    fn ray_scan_task_size_grouping(&self) -> bool {
        self.ray_scan_task_size_grouping
    }
    #[getter]
    fn ray_max_task_backlog(&self) -> i64 {
        self.ray_max_task_backlog
    }
    #[getter]
    fn ray_scan_task_open_cost_bytes(&self) -> i64 {
        self.ray_scan_task_open_cost_bytes
    }
    #[getter]
    fn ray_scan_task_min_partition_num(&self) -> i64 {
        self.ray_scan_task_min_partition_num
    }
    #[getter]
    fn ray_init_sql(&self) -> String {
        self.ray_init_sql.clone()
    }
    #[getter]
    fn udf_parallel(&self) -> bool {
        self.udf_parallel
    }
    #[getter]
    fn udf_arrow_fastpath(&self) -> bool {
        self.udf_arrow_fastpath
    }
    #[getter]
    fn local_exchange_buffer(&self) -> String {
        self.local_exchange_buffer.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(runner={:?}, ray_scan_task_size_grouping={}, udf_parallel={}, local_exchange_buffer={:?})",
            self.runner, self.ray_scan_task_size_grouping, self.udf_parallel, self.local_exchange_buffer
        )
    }
}

#[pyfunction]
#[pyo3(signature = (**kw))]
pub fn configure(kw: Option<Bound<'_, PyDict>>) -> PyResult<Config> {
    let mut config = Config::from_env();

    if let Some(kw) = kw {
        for (key, value) in kw.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "runner" => {
                    let val: String = value.extract()?;
                    let normalized = val.to_lowercase();
                    if !env::RUNNER_VALUES.contains(&normalized.as_str()) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "runner must be 'local' or 'ray'",
                        ));
                    }
                    config.runner = normalized;
                }
                "ray_scan_task_size_grouping" => {
                    config.ray_scan_task_size_grouping = value.extract()?
                }
                "ray_max_task_backlog" => config.ray_max_task_backlog = value.extract()?,
                "ray_scan_task_open_cost_bytes" => {
                    config.ray_scan_task_open_cost_bytes = value.extract()?
                }
                "ray_scan_task_min_partition_num" => {
                    config.ray_scan_task_min_partition_num = value.extract()?
                }
                "ray_init_sql" => config.ray_init_sql = value.extract()?,
                "udf_parallel" => config.udf_parallel = value.extract()?,
                "udf_arrow_fastpath" => config.udf_arrow_fastpath = value.extract()?,
                "local_exchange_buffer" => config.local_exchange_buffer = value.extract()?,
                _ => {
                    return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                        "Unknown config field: '{key}'"
                    )))
                }
            }
        }
    }

    let reg = env::EnvRegistry::new();
    reg.set_runner(&config.runner)?;
    reg.set_ray_scan_task_size_grouping(config.ray_scan_task_size_grouping);
    reg.set_ray_max_task_backlog(config.ray_max_task_backlog);
    reg.set_ray_scan_task_open_cost_bytes(config.ray_scan_task_open_cost_bytes);
    reg.set_ray_scan_task_min_partition_num(config.ray_scan_task_min_partition_num);
    reg.set_ray_init_sql(&config.ray_init_sql);
    reg.set_udf_parallel(config.udf_parallel);
    reg.set_udf_arrow_fastpath(config.udf_arrow_fastpath);
    reg.set_local_exchange_buffer(&config.local_exchange_buffer);

    Ok(config)
}

#[pyfunction]
pub fn current_config() -> Config {
    Config::from_env()
}
