"""Tests for jude.pipeline — direct cosmos-xenna integration.

jude.pipeline re-exports cosmos-xenna's Stage/PipelineSpec/run_pipeline when it
is installed (the real multimodal pipeline engine), else a local fallback.
"""

import pytest

import jude.pipeline as jp


class TestCosmosIntegration:
    def test_is_cosmos_backed(self):
        pytest.importorskip("cosmos_xenna")
        assert jp.is_cosmos_backed() is True

    def test_reexports_cosmos_api(self):
        pytest.importorskip("cosmos_xenna")
        # The public cosmos surface is available through jude.pipeline.
        for name in ("Stage", "StageSpec", "PipelineSpec", "Resources", "run_pipeline"):
            assert hasattr(jp, name), name

    def test_build_a_cosmos_stage_and_spec(self):
        pytest.importorskip("cosmos_xenna")

        class Double(jp.Stage):
            @property
            def required_resources(self):
                return jp.Resources(cpus=1)

            @property
            def stage_batch_size(self):
                return 4

            def process_data(self, samples):
                return [x * 2 for x in samples]

        # Constructing a Stage + PipelineSpec should work without running Ray.
        spec = jp.PipelineSpec(input_data=[1, 2, 3, 4], stages=[jp.StageSpec(Double())])
        assert spec is not None
        assert isinstance(Double().required_resources, jp.Resources)
        assert Double().stage_batch_size == 4

    def test_resources_gpu(self):
        pytest.importorskip("cosmos_xenna")
        r = jp.Resources(cpus=2, gpus=1)
        assert r.cpus == 2
        assert r.gpus == 1


class TestFallback:
    """The local fallback runs stages sequentially without cosmos/Ray."""

    def test_fallback_module_runs_sequential(self):
        # Exercise the fallback directly (independent of whether cosmos is present).
        from jude.pipeline import _fallback as fb

        class Add(fb.Stage):
            @property
            def stage_batch_size(self):
                return 2

            def process_data(self, samples):
                return [x + 10 for x in samples]

        class Str(fb.Stage):
            def process_data(self, samples):
                return [str(x) for x in samples]

        pipe = fb.Pipeline(input_data=[1, 2, 3], stages=[Add(), Str()])
        assert pipe.run() == ["11", "12", "13"]
