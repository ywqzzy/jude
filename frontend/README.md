# jude dashboard (React)

A live cluster/query dashboard for jude, served against the `jude.observe`
metrics HTTP endpoint.

## What it shows

- **Cluster nodes** — Ray nodes (id, CPU/GPU, alive) from `ray.nodes()`.
- **Distributed stages** — per-stage progress bars (tasks done/total), rows,
  bytes, and attempt retries.
- **Queries** — recent queries with kind (local/distributed/pipeline), status,
  rows, elapsed, and their stage chain (e.g. `partial → assemble`).
- **UDF pools** — subprocess / ray_actor pool utilization (workers, batches,
  rows, busy time).
- **Activity** — a rolling event feed.

Everything is read from the Rust `MetricsRegistry` (GIL-free), so the dashboard
never perturbs execution.

## Run

**Quickest (resident dev server):** one command brings up a local Ray cluster +
the metrics endpoint + a background cluster poller, and keeps them alive across
your edit/run cycles:

```bash
python -m jude.observe          # Ray (+dashboard :8265) + metrics at :8477, blocks
# in another terminal:
cd frontend && npm install && npm run dev     # http://localhost:5273
```

The dev server prints the Ray dashboard URL, the metrics URL, and the exact
`npm run dev` command to point at it.

**Manual (embed in your own script):**

1. Start the metrics endpoint from Python:

   ```python
   import jude
   from jude import observe
   observe.serve(port=8477)          # serves /api/metrics with open CORS
   observe.start_node_poller()       # optional: keep cluster view live while idle
   # ... run jude workloads; they record into the registry automatically ...
   ```

2. Start the dashboard (dev):

   ```bash
   cd frontend
   npm install
   npm run dev        # http://localhost:5273, proxies /api -> :8477
   ```

   Point at a different metrics server with `JUDE_METRICS_URL=http://host:port npm run dev`.
   Point the header "Ray dashboard" link elsewhere with `VITE_RAY_DASHBOARD=http://host:8265`.

3. Or build static assets (`npm run build` → `dist/`) and serve them from any
   static host; they fetch `/api/metrics` relative to their origin.
