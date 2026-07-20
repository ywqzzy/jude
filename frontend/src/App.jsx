import React, { useEffect, useState, useCallback } from "react";

const EMPTY = {
  queries: [],
  stages: [],
  pools: [],
  nodes: [],
  cluster: {},
  events: [],
  summary: {},
};

function useMetrics(intervalMs = 1500) {
  const [data, setData] = useState(EMPTY);
  const [online, setOnline] = useState(false);
  const [lastError, setLastError] = useState(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetch("/api/metrics", { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setData({ ...EMPTY, ...j });
      setOnline(true);
      setLastError(null);
    } catch (e) {
      setOnline(false);
      setLastError(e.message);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, intervalMs);
    return () => clearInterval(id);
  }, [poll, intervalMs]);

  return { data, online, lastError };
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function fmt(n) {
  if (n == null) return "0";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function fmtBytes(b) {
  if (!b) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(1)} ${u[i]}`;
}

function elapsed(q) {
  const end = q.ended_at || Date.now() / 1000;
  const s = end - q.started_at;
  if (s < 0) return "—";
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  return `${s.toFixed(1)}s`;
}

function QueriesPanel({ queries, stages }) {
  const stagesByQuery = {};
  for (const s of stages) (stagesByQuery[s.query_id] ||= []).push(s);
  const recent = [...queries].reverse().slice(0, 25);
  return (
    <div className="panel">
      <h2>Queries</h2>
      {recent.length === 0 ? (
        <div className="body muted-empty">No queries recorded yet.</div>
      ) : (
        <table>
          <thead>
            <tr><th>Label</th><th>Kind</th><th>Status</th><th className="num">Rows</th><th className="num">Elapsed</th><th>Stages</th></tr>
          </thead>
          <tbody>
            {recent.map((q) => (
              <tr key={q.id}>
                <td>{q.label}</td>
                <td><span className={`badge ${q.kind}`}>{q.kind}</span></td>
                <td><span className={`badge ${q.status}`}>{q.status}</span></td>
                <td className="num">{fmt(q.rows)}</td>
                <td className="num">{elapsed(q)}</td>
                <td>{(stagesByQuery[q.id] || []).map((s) => s.name).join(" → ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StagesPanel({ stages }) {
  const active = stages.filter((s) => s.status !== "done" || s.tasks_total > 0).slice(-20);
  if (active.length === 0) return null;
  return (
    <div className="panel">
      <h2>Distributed Stages</h2>
      <table>
        <thead>
          <tr><th>Stage</th><th>Status</th><th>Progress</th><th className="num">Rows</th><th className="num">Bytes</th><th className="num">Retries</th></tr>
        </thead>
        <tbody>
          {active.map((s) => {
            const pct = s.tasks_total ? Math.round((100 * s.tasks_done) / s.tasks_total) : 0;
            return (
              <tr key={s.id}>
                <td>{s.name}</td>
                <td><span className={`badge ${s.status === "done" ? "done" : "running"}`}>{s.status}</span></td>
                <td>
                  <div className="bar" title={`${s.tasks_done}/${s.tasks_total}`}>
                    <span style={{ width: `${pct}%` }} />
                  </div>
                </td>
                <td className="num">{fmt(s.rows)}</td>
                <td className="num">{fmtBytes(s.bytes)}</td>
                <td className="num">{s.attempts || 0}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ResourceBar({ label, used, total }) {
  const pct = total > 0 ? Math.round((100 * used) / total) : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
        <span className="muted-label">{label}</span>
        <span>{used.toFixed(1)} / {total.toFixed(1)} ({pct}%)</span>
      </div>
      <div className="bar" style={{ height: 8 }}>
        <span style={{ width: `${pct}%`, background: pct > 85 ? "var(--red)" : pct > 60 ? "var(--amber)" : "var(--green)" }} />
      </div>
    </div>
  );
}

function ClusterResourcesPanel({ cluster }) {
  const dims = cluster ? Object.keys(cluster) : [];
  if (dims.length === 0) return null;
  const label = { CPU: "CPU", GPU: "GPU", memory: "Heap memory", object_store_memory: "Object store" };
  const scale = (dim, v) => (dim.includes("memory") ? v / 2 ** 30 : v); // bytes -> GiB
  const unit = (dim) => (dim.includes("memory") ? " GiB" : "");
  return (
    <div className="panel">
      <h2>Cluster Resource Utilization</h2>
      <div className="body">
        {dims.map((d) => (
          <ResourceBar
            key={d}
            label={(label[d] || d) + unit(d)}
            used={scale(d, cluster[d].used)}
            total={scale(d, cluster[d].total)}
          />
        ))}
      </div>
    </div>
  );
}

function NodesPanel({ nodes }) {
  return (
    <div className="panel">
      <h2>Cluster Nodes</h2>
      <div className="body">
        {nodes.length === 0 ? (
          <span className="muted-empty">No Ray cluster connected (single-process mode).</span>
        ) : (
          nodes.map((n) => (
            <div className="node-card" key={n.node_id}>
              <span className="nid">{n.node_id.slice(0, 12)}…</span>
              {n.address ? <span className="nid">{n.address}</span> : null}
              <span className="res">{n.cpus} CPU · {n.gpus} GPU</span>
              <span><span className={`status-dot ${n.alive ? "ok" : "off"}`} />{n.alive ? "alive" : "dead"}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function PoolsPanel({ pools }) {
  if (pools.length === 0) return null;
  return (
    <div className="panel">
      <h2>UDF Pools</h2>
      <table>
        <thead>
          <tr><th>Backend</th><th className="num">Workers</th><th className="num">Batches</th><th className="num">Rows</th><th className="num">Busy</th></tr>
        </thead>
        <tbody>
          {pools.map((p) => (
            <tr key={p.key}>
              <td><span className="badge distributed">{p.backend}</span></td>
              <td className="num">{p.num_workers}</td>
              <td className="num">{fmt(p.batches_in)}</td>
              <td className="num">{fmt(p.rows)}</td>
              <td className="num">{(p.busy_ns / 1e9).toFixed(2)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventsPanel({ events }) {
  const recent = [...events].reverse().slice(0, 40);
  return (
    <div className="panel">
      <h2>Activity</h2>
      <div className="body events">
        {recent.length === 0 ? (
          <div className="empty">No activity yet.</div>
        ) : (
          recent.map((e, i) => (
            <div className="ev" key={i}>
              <span className="kind">{e.kind}</span> — {e.msg}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function useSummary(intervalMs = 2000) {
  const [sum, setSum] = useState({});
  useEffect(() => {
    let live = true;
    const poll = async () => {
      try {
        const r = await fetch("/api/summary", { cache: "no-store" });
        if (r.ok && live) setSum(await r.json());
      } catch {
        /* summary endpoint may be absent; ignore */
      }
    };
    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [intervalMs]);
  return sum;
}

function PerfPanel({ summary }) {
  const q = summary.queries;
  if (!q || !q.latency_ms) return null;
  const L = q.latency_ms;
  return (
    <div className="panel">
      <h2>Query Performance <span className="audit-stats">· end-to-end latency (done queries)</span></h2>
      <div className="summary-grid">
        <Stat label="p50" value={`${L.p50.toFixed(1)}ms`} />
        <Stat label="p95" value={`${L.p95.toFixed(1)}ms`} />
        <Stat label="p99" value={`${L.p99.toFixed(1)}ms`} />
        <Stat label="Rows/sec" value={fmt(Math.round(q.rows_per_sec || 0))} />
      </div>
    </div>
  );
}

function DataQualityPanel({ summary }) {
  const c = summary.curation;
  if (!c || !c.ops) return null;
  const keep = Math.round((c.keep_rate || 0) * 100);
  const color = keep < 40 ? "var(--red)" : keep < 70 ? "var(--amber)" : "var(--green)";
  return (
    <div className="panel">
      <h2>Data Quality <span className="audit-stats">· {c.ops} curation ops</span></h2>
      <div className="body">
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
          <span className="muted-label">Keep-rate (rows surviving curation)</span>
          <span>{fmt(c.rows_out)} / {fmt(c.rows_in)} kept · {fmt(c.removed)} removed ({keep}%)</span>
        </div>
        <div className="bar" style={{ height: 10 }}>
          <span style={{ width: `${keep}%`, background: color }} />
        </div>
      </div>
    </div>
  );
}

function useAudit(intervalMs = 3000) {
  const [audit, setAudit] = useState({ records: [], stats: {} });
  useEffect(() => {
    let live = true;
    const poll = async () => {
      try {
        const r = await fetch("/api/audit?limit=100", { cache: "no-store" });
        if (r.ok && live) setAudit(await r.json());
      } catch {
        /* audit endpoint may be absent; ignore */
      }
    };
    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [intervalMs]);
  return audit;
}

function CurationFunnelPanel() {
  const { records } = useAudit();
  // most recent pipeline run that carries a curation funnel
  const withFunnel = records.filter((r) => r.detail && Array.isArray(r.detail.funnel) && r.detail.funnel.length);
  if (withFunnel.length === 0) return null;
  const rec = withFunnel[0];
  const funnel = rec.detail.funnel;
  const start = rec.detail.input_rows || (funnel[0] && funnel[0].rows_in) || 0;
  return (
    <div className="panel">
      <h2>
        Curation Funnel <span className="audit-stats">· {rec.label} · {fmt(start)} → {fmt(rec.detail.output_rows)} rows</span>
      </h2>
      <table>
        <thead>
          <tr><th>Stage</th><th className="num">Rows in</th><th className="num">Rows out</th><th className="num">Dropped</th><th>Kept</th></tr>
        </thead>
        <tbody>
          {funnel.map((s, i) => (
            <tr key={i}>
              <td>{s.op}</td>
              <td className="num">{fmt(s.rows_in)}</td>
              <td className="num">{fmt(s.rows_out)}</td>
              <td className="num">{fmt(s.dropped)}</td>
              <td>
                <div className="bar" title={`${s.pct_kept}%`}>
                  <span style={{ width: `${s.pct_kept}%`, background: s.pct_kept < 50 ? "var(--amber)" : "var(--green)" }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AuditPanel() {
  const { records, stats } = useAudit();
  const [open, setOpen] = useState(null);
  return (
    <div className="panel">
      <h2>
        Audit / Execution History (persistent)
        {stats && stats.total != null ? (
          <span className="audit-stats">
            {" "}· {stats.total} total · {stats.done} done · {stats.error} error · {fmt(stats.rows_total)} rows
          </span>
        ) : null}
      </h2>
      {records.length === 0 ? (
        <div className="body muted-empty">No persisted executions yet (run some queries).</div>
      ) : (
        <table>
          <thead>
            <tr><th>#</th><th>Label</th><th>Kind</th><th>Status</th><th className="num">Rows</th><th className="num">Duration</th><th>Stages</th></tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <React.Fragment key={r.audit_id}>
                <tr onClick={() => setOpen(open === r.audit_id ? null : r.audit_id)} style={{ cursor: "pointer" }}>
                  <td className="num">{r.audit_id}</td>
                  <td>{r.label}</td>
                  <td><span className={`badge ${r.kind}`}>{r.kind}</span></td>
                  <td><span className={`badge ${r.status}`}>{r.status}</span></td>
                  <td className="num">{fmt(r.rows)}</td>
                  <td className="num">{r.duration_ms != null ? `${r.duration_ms.toFixed(1)}ms` : "—"}</td>
                  <td>{(r.stages || []).join(" → ") || "—"}</td>
                </tr>
                {open === r.audit_id && (
                  <tr>
                    <td colSpan={7} className="audit-detail">
                      <pre>{JSON.stringify(r, null, 2)}</pre>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function App() {
  const { data, online, lastError } = useMetrics();
  const summary = useSummary();
  const s = data.summary || {};
  return (
    <div className="app">
      <header>
        <h1><span className="brand">jude</span> · cluster dashboard</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <a
            href={import.meta.env.VITE_RAY_DASHBOARD || "http://127.0.0.1:8265"}
            target="_blank"
            rel="noreferrer"
            className="ray-link"
            title="Ray's official dashboard (jobs, actors, logs, timeline)"
          >
            Ray dashboard ↗
          </a>
          <span>
            <span className={`status-dot ${online ? "ok" : "off"}`} />
            {online ? "connected" : `offline${lastError ? ` (${lastError})` : ""}`}
          </span>
        </div>
      </header>

      <div className="summary-grid">
        <Stat label="Queries" value={s.queries_total ?? 0} />
        <Stat label="Running" value={s.queries_running ?? 0} />
        <Stat label="UDF Pools" value={s.pools ?? 0} />
        <Stat label="Nodes Alive" value={`${s.nodes_alive ?? 0}/${s.nodes_total ?? 0}`} />
      </div>

      <NodesPanel nodes={data.nodes} />
      <ClusterResourcesPanel cluster={data.cluster} />
      <PerfPanel summary={summary} />
      <DataQualityPanel summary={summary} />
      <StagesPanel stages={data.stages} />
      <QueriesPanel queries={data.queries} stages={data.stages} />
      <PoolsPanel pools={data.pools} />
      <CurationFunnelPanel />
      <AuditPanel />
      <EventsPanel events={data.events} />

      <footer>
        Polling <code>/api/metrics</code> every 1.5s · jude.observe
      </footer>
    </div>
  );
}
