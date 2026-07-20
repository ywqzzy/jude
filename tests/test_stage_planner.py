"""Distributed stage planning — jude.dist StagePlanner exposed via Relation.plan_stages."""

import jude


class TestWorkerManagerLocality:
    def test_locality_prefers_matching_node(self):
        # 4 workers, workers 0,2 on nodeA; 1,3 on nodeB.
        m = jude.dist.WorkerManager(4, 0, True, 0, 4 * 1024 * 1024, 0, ["nodeA", "nodeB", "nodeA", "nodeB"])
        w1 = m.worker_for_locality(0, ["nodeB"])
        w2 = m.worker_for_locality(0, ["nodeB"])
        assert {w1, w2} <= {1, 3}  # a nodeB task lands on a nodeB worker
        assert w1 != w2  # balanced across the node's workers
        # no matching node / no hint -> round-robin fallback
        assert m.worker_for_locality(2, ["nodeC"]) == 2 % 4
        assert m.worker_for_locality(3, []) == 3 % 4

    def test_no_worker_map_is_round_robin(self):
        m = jude.dist.WorkerManager(3)
        assert m.worker_for_locality(4, ["nodeA"]) == 4 % 3


class TestPlanStages:
    def _con(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT n % 3 AS g, n AS v FROM range(100) t(n)")
        return con

    def test_scan_only(self):
        con = self._con()
        stages = con.sql("SELECT * FROM t").plan_stages()
        assert len(stages) == 1
        assert stages[0]["kind"] == "scan"

    def test_partitionwise_fuses(self):
        con = self._con()
        # filter+project over a scan: no shuffle -> a single stage
        stages = con.sql("SELECT * FROM t").filter("v > 1").project("v").plan_stages()
        assert len(stages) == 1
        assert stages[0]["kind"] == "scan"

    def test_aggregate_two_stages(self):
        con = self._con()
        stages = con.sql("SELECT * FROM t").aggregate("sum(v)", "g").plan_stages()
        assert len(stages) == 2
        agg = stages[-1]
        assert agg["kind"] == "shuffle"
        assert agg["op"] == "Aggregate"
        assert agg["partition_keys"] == ["g"]
        assert agg["inputs"] == [0]

    def test_stacked_shuffles(self):
        con = self._con()
        stages = con.sql("SELECT * FROM t").filter("v > 1").aggregate("sum(v)", "g").order("g").plan_stages()
        kinds = [(s["op"], s["kind"]) for s in stages]
        assert kinds == [("Sql", "scan"), ("Aggregate", "shuffle"), ("Order", "shuffle")]
        # dependency chain: order <- aggregate <- scan
        assert stages[1]["inputs"] == [0]
        assert stages[2]["inputs"] == [1]
