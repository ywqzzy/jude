"""Resource admission control (jude.dist.ResourceManager) — GPU/mem-aware
admission in Rust. Unit tests need no Ray; the dispatch test is Ray-gated."""
import pytest

import jude


def test_gpu_admission_and_release():
    rm = jude.dist.ResourceManager(0.0, 4.0, 0, 0, 0)
    assert rm.admit_count(0.0, 1.0, 0, 0) == 4
    assert rm.try_reserve(0.0, 1.0, 0, 0)
    assert rm.try_reserve(0.0, 1.0, 0, 0)
    assert rm.inflight == 2
    assert rm.admit_count(0.0, 1.0, 0, 0) == 2
    rm.release(0.0, 1.0, 0, 0)
    assert rm.admit_count(0.0, 1.0, 0, 0) == 3


def test_oversubscribe_rejected():
    rm = jude.dist.ResourceManager(0.0, 1.0, 0, 0, 0)
    assert rm.try_reserve(0.0, 1.0, 0, 0)
    assert not rm.try_reserve(0.0, 1.0, 0, 0)


def test_memory_dimension_binds():
    rm = jude.dist.ResourceManager(0.0, 0.0, 100, 0, 0)
    assert rm.admit_count(0.0, 0.0, 30, 0) == 3


def test_max_inflight_cap():
    rm = jude.dist.ResourceManager(0.0, 0.0, 0, 0, 2)
    assert rm.admit_count(0.0, 0.0, 0, 0) == 2
    assert rm.try_reserve(0.0, 0.0, 0, 0)
    assert rm.try_reserve(0.0, 0.0, 0, 0)
    assert not rm.try_reserve(0.0, 0.0, 0, 0)


def test_admission_dispatch_bounds_concurrency():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners import _ray_shim as shim

    # 2-GPU budget, each task wants 1 GPU -> never more than 2 concurrent.
    rm = jude.dist.ResourceManager(0.0, 2.0, 0, 0, 0)

    @ray.remote
    def work(x):
        return x * x

    submit = [(lambda i=i: work.remote(i)) for i in range(8)]
    out = shim.run_bounded_admission(submit, rm, 0.0, 1.0, 0, 0)
    assert out == [i * i for i in range(8)]
    # All released at the end.
    assert rm.inflight == 0
