"""jude.pipeline — multimodal streaming inference pipelines.

Jude uses **cosmos-xenna** directly as its multimodal pipeline engine (NVIDIA's
Ray-based stage/worker pipeline library). When cosmos-xenna is installed, this
module re-exports its public API so you build pipelines the cosmos way:

    import jude.pipeline as jp

    class Caption(jp.Stage):
        @property
        def required_resources(self):
            return jp.Resources(gpus=1, cpus=2)
        @property
        def stage_batch_size(self):
            return 16
        def setup(self, worker_metadata):
            self.model = load_vlm()            # weights load once per worker
        def process_data(self, samples):
            for s in samples:
                s.caption = self.model.caption(s.image)
            return samples

    spec = jp.PipelineSpec(input_data=samples, stages=[jp.StageSpec(Caption())])
    jp.run_pipeline(spec)

cosmos-xenna handles heterogeneous CPU/GPU resource allocation, saturation-aware
autoscaling, backpressure, and SPMD tensor-parallel model stages. It is a
multimodal *pipeline* engine — not a SQL engine — so pipelines are built over
Python objects / Arrow, not over SQL relations.

If cosmos-xenna is not installed, a minimal local fallback (``Stage`` /
``Pipeline``) runs stages sequentially in-process so simple pipelines still work
single-node without the heavy dependency.
"""

from __future__ import annotations

_COSMOS = False

try:  # Prefer the real engine.
    import cosmos_xenna.pipelines.v1 as _cx

    Stage = _cx.Stage
    StageSpec = _cx.StageSpec
    PipelineSpec = _cx.PipelineSpec
    PipelineConfig = _cx.PipelineConfig
    Resources = _cx.Resources
    ExecutionMode = _cx.ExecutionMode
    WorkerMetadata = _cx.WorkerMetadata
    NodeInfo = _cx.NodeInfo
    run_pipeline = _cx.run_pipeline
    _COSMOS = True

    __all__ = [
        "Stage",
        "StageSpec",
        "PipelineSpec",
        "PipelineConfig",
        "Resources",
        "ExecutionMode",
        "WorkerMetadata",
        "NodeInfo",
        "run_pipeline",
        "is_cosmos_backed",
    ]
except Exception:
    # Minimal local fallback (no cosmos-xenna / Ray). Enough for single-node
    # sequential stage execution; see jude.pipeline._fallback.
    from jude.pipeline._fallback import (  # noqa: F401
        ExecutionMode,
        Pipeline,
        Resources,
        Stage,
        StageSpec,
    )

    __all__ = [
        "Stage",
        "StageSpec",
        "Resources",
        "ExecutionMode",
        "Pipeline",
        "is_cosmos_backed",
    ]


def is_cosmos_backed() -> bool:
    """True if jude.pipeline is backed by the real cosmos-xenna engine."""
    return _COSMOS


# Relation-integrated multi-stage pipeline (source/sink are jude relations).
# Imported after Stage/Resources are bound above so _multimodal can subclass them.
from jude.pipeline._multimodal import (  # noqa: E402,F401
    ArrowStage,
    DecodeStage,
    LoadFilesStage,
    MapBatchesStage,
    RelationPipeline,
    relation_to_shards,
    shards_to_relation,
    shards_to_table,
)

__all__ += [
    "ArrowStage",
    "DecodeStage",
    "LoadFilesStage",
    "MapBatchesStage",
    "RelationPipeline",
    "relation_to_shards",
    "shards_to_relation",
    "shards_to_table",
]
