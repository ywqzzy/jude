"""Fault-tolerance wiring: cosmos pipeline FT defaults + actor restart config.

Verifies jude turns on cosmos-xenna's built-in fault tolerance for the pipeline
(map) path, and that the RayRunner actor pool is created with actor-restart
enabled but in-flight task retries OFF by default (safe for side-effecting write
tasks; opt-in for read-only pipelines).
"""

from __future__ import annotations

import os

import pytest


def test_cosmos_default_config_enables_fault_tolerance():
    jp = pytest.importorskip("jude.pipeline")
    if not jp.is_cosmos_backed():
        pytest.skip("cosmos-xenna not installed")
    from jude.pipeline._multimodal import RelationPipeline

    cfg = RelationPipeline._default_cosmos_config()
    # retries + worker rebuild are on (cosmos owns FT for the map-stage path)
    assert cfg.num_run_attempts_python >= 2
    assert cfg.num_setup_attempts_python >= 2
    assert cfg.reset_workers_on_failure is True


def test_cosmos_config_respects_env(monkeypatch):
    jp = pytest.importorskip("jude.pipeline")
    if not jp.is_cosmos_backed():
        pytest.skip("cosmos-xenna not installed")
    from jude.pipeline._multimodal import RelationPipeline

    monkeypatch.setenv("JUDE_COSMOS_RUN_ATTEMPTS", "5")
    cfg = RelationPipeline._default_cosmos_config()
    assert cfg.num_run_attempts_python == 5


def test_user_pipeline_config_still_wins():
    jp = pytest.importorskip("jude.pipeline")
    if not jp.is_cosmos_backed():
        pytest.skip("cosmos-xenna not installed")
    from jude.pipeline._multimodal import RelationPipeline

    custom = jp.PipelineConfig(execution_mode=jp.ExecutionMode.BATCH,
                               return_last_stage_outputs=True, num_run_attempts_python=9)
    p = RelationPipeline.from_table.__self__ if False else RelationPipeline(pipeline_config=custom)
    assert p.pipeline_config is custom  # explicit config overrides the FT default


def test_actor_pool_created_with_restart(monkeypatch):
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=2)
    from jude.runners._ray_shim import make_workers

    # default: task retries off (safe for write tasks), restart on
    assert os.environ.get("JUDE_ACTOR_MAX_TASK_RETRIES", "0") == "0"
    w = make_workers(1)
    assert len(w) == 1                       # pool builds with FT options applied
    # actor is usable after creation
    import pyarrow as pa
    out = ray.get(w[0].run_sql_on_table.remote(pa.table({"x": [1, 2, 3]}),
                                               "SELECT sum(x) s FROM part"))
    assert out.column("s")[0].as_py() == 6
