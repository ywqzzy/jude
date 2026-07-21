"""jude.dist_cc — distributed connected-components via label propagation.

The last scale gap in global fuzzy dedup (L2.1 follow-up). The streaming
union-find in ``curate_dist`` bounds driver memory to the label array (n ints) +
one bucket's edges — fine to ~1e9 docs, but the label array itself lives on the
driver. This distributes the LABEL ARRAY too: labels are sharded across Ray
actors by ``rid % W`` and propagated along edges with a per-round message
shuffle, converging to each vertex's component-minimum id. The driver never
holds all n labels — only the transient per-round snapshot of the *active*
(edge-touched) vertices, which for a real corpus (mostly-unique docs, few
near-dups) is a small minority.

Edges are supplied as Ray refs to Arrow tables with ``a``/``b`` (int64) columns —
the same edge tables ``curate_fuzzy_edges_bucket`` already produces — so edges
never fully materialize on the driver either.

    label = connected_components(edge_refs, num_workers=4)   # {rid: component_min}
    # rids not in `label` are singletons (survive); rid with label[rid]==rid is a rep.
"""

from __future__ import annotations

from typing import Any

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None  # type: ignore


if ray is not None:

    @ray.remote
    class _CCShard:
        """Owns the labels for rids with ``rid % num_shards == shard_id`` and a
        slice of the edge set. Labels default to the rid itself (lazy)."""

        def __init__(self, shard_id: int, num_shards: int):
            self.sid = shard_id
            self.w = num_shards
            self.label: dict[int, int] = {}
            self.edges: list[tuple[int, int]] = []
            self._refd: set[int] | None = None

        def add_edges(self, edge_refs: list) -> int:
            for t in ray.get(edge_refs):
                if t is None or t.num_rows == 0:
                    continue
                aa = t.column("a").to_pylist()
                bb = t.column("b").to_pylist()
                self.edges.extend(zip(aa, bb))
            self._refd = None
            return len(self.edges)

        def referenced_rids(self) -> list:
            if self._refd is None:
                s: set[int] = set()
                for a, b in self.edges:
                    s.add(a)
                    s.add(b)
                self._refd = s
            return list(self._refd)

        def labels_for(self, rids: list) -> dict:
            """Labels this shard owns among ``rids`` (default = rid)."""
            return {r: self.label.get(r, r) for r in rids if r % self.w == self.sid}

        def compute_messages(self, snapshot: dict) -> dict:
            """Given current labels of all referenced rids, for each edge (a,b)
            propose ``min(la, lb)`` to both endpoints. Returns messages grouped by
            owning shard: ``{owner_shard: [(rid, label), ...]}``."""
            out: dict[int, list] = {}
            for a, b in self.edges:
                m = snapshot[a] if snapshot[a] < snapshot[b] else snapshot[b]
                for r in (a, b):
                    out.setdefault(r % self.w, []).append((r, m))
            return out

        def apply(self, messages: list) -> int:
            """Take the min of incoming proposals; return how many labels changed."""
            changed = 0
            for r, m in messages:
                if r % self.w == self.sid and m < self.label.get(r, r):
                    self.label[r] = m
                    changed += 1
            return changed

        def label_map(self) -> dict:
            return dict(self.label)


def connected_components(edge_refs: list, *, num_workers: int = 4,
                         max_rounds: int = 100, runner: Any = None) -> dict:
    """Distributed connected components over the graph in ``edge_refs`` (Ray refs
    to Arrow (a,b) edge tables). Returns ``{rid: component_min_rid}`` for every
    rid that appears in an edge (rids absent from the result are singletons).
    Converges when a full round changes no label. Label array is sharded across
    ``num_workers`` actors — never fully on the driver."""
    if ray is None:
        raise ImportError("connected_components needs ray")
    w = max(1, num_workers)
    shards = [_CCShard.remote(s, w) for s in range(w)]
    # distribute edge refs round-robin across shards
    ray.get([shards[i % w].add_edges.remote([ref]) for i, ref in enumerate(edge_refs)])
    refd = ray.get([s.referenced_rids.remote() for s in shards])  # per-shard referenced rids

    for _ in range(max_rounds):
        # gather each shard's needed labels from their owners (targeted, not all n)
        snapshots = []
        for i in range(w):
            need = refd[i]
            if not need:
                snapshots.append({})
                continue
            parts = ray.get([shards[o].labels_for.remote(need) for o in range(w)])
            snap: dict = {}
            for p in parts:
                snap.update(p)
            snapshots.append(snap)
        # each shard proposes min-labels along its edges
        msg_groups = ray.get([shards[i].compute_messages.remote(snapshots[i]) for i in range(w)])
        # route proposals to owning shards and apply
        per_owner: list[list] = [[] for _ in range(w)]
        for i in range(w):
            for owner, msgs in msg_groups[i].items():
                per_owner[owner].extend(msgs)
        changed = sum(ray.get([shards[o].apply.remote(per_owner[o]) for o in range(w)]))
        if changed == 0:
            break

    label: dict = {}
    for m in ray.get([s.label_map.remote() for s in shards]):
        label.update(m)
    return label
