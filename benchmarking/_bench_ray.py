"""Shared bench helper: connect to the resident Ray started by
`python -m jude.observe` (so bench runs show up on the dashboard), falling back
to a local cluster if none is running.
"""

from __future__ import annotations


def connect_ray(num_cpus: int = 8):
    """Attach to an already-running Ray cluster (the one `python -m jude.observe`
    started) via address='auto'; else start a local one. Returns the ray module.
    Prefer the resident cluster so bench executions are recorded by jude.observe
    and visible on the dashboard."""
    import ray

    if ray.is_initialized():
        return ray
    try:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
        print("[bench] attached to resident Ray (python -m jude.observe)")
    except Exception:  # noqa: BLE001 — no resident cluster; start local
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=num_cpus)
        print(f"[bench] started local Ray (num_cpus={num_cpus}); "
              f"run 'python -m jude.observe' first to record on the dashboard")
    return ray
