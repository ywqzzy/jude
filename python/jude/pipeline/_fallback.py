"""Minimal local fallback for jude.pipeline when cosmos-xenna is unavailable.

Runs a linear list of stages sequentially in-process over a list of input items
(cosmos-xenna's programming model, minus Ray / autoscaling / GPU scheduling).
Enough for single-node development without the heavy dependency; production
multimodal pipelines use the real cosmos-xenna engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Resources:
    cpus: float = 1.0
    gpus: float = 0.0
    is_spmd: bool = False


class ExecutionMode:
    STREAMING = "streaming"
    BATCH = "batch"


class Stage:
    """Base stage: implement process_data(list) -> list. setup() loads state."""

    @property
    def required_resources(self) -> Resources:
        return Resources()

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: Any = None) -> None:
        """One-time per-worker init (load models)."""

    def process_data(self, samples: list) -> list:
        raise NotImplementedError

    def destroy(self) -> None:
        pass


@dataclass
class StageSpec:
    stage: Stage
    num_workers: Optional[int] = None


def _as_spec(s: Any) -> StageSpec:
    return s if isinstance(s, StageSpec) else StageSpec(stage=s)


class Pipeline:
    """Local sequential pipeline over a list of input items."""

    def __init__(self, input_data: Sequence[Any], stages: list[Any], *, mode: str = ExecutionMode.BATCH):
        self.input_data = list(input_data)
        self.specs = [_as_spec(s) for s in stages]
        self.mode = mode

    def run(self) -> list:
        for spec in self.specs:
            spec.stage.setup(None)
        data = self.input_data
        for spec in self.specs:
            stage = spec.stage
            bs = max(1, stage.stage_batch_size)
            out: list = []
            for start in range(0, len(data), bs):
                batch = data[start : start + bs]
                res = stage.process_data(batch)
                if res is not None:
                    out.extend(res)
            data = out
        for spec in self.specs:
            spec.stage.destroy()
        return data
