"""Tests for unnest/explode (multimodal fan-out) and sample."""

import jude


class TestUnnest:
    def test_unnest_list_column(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT 1 AS id, [10, 20, 30] AS vals")
        out = con.sql("SELECT * FROM t").unnest("vals")
        assert out.num_rows == 3
        assert out.fetchall() == [(1, 10), (1, 20), (1, 30)]

    def test_explode_alias(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT 'doc' AS name, ['a', 'b'] AS chunks")
        out = con.sql("SELECT * FROM t").explode("chunks")
        assert out.num_rows == 2
        assert out.order("chunks").fetchall() == [("doc", "a"), ("doc", "b")]

    def test_unnest_multimodal_fanout(self):
        # 1 "video" -> N "frames" — the multimodal fan-out pattern.
        con = jude.connect()
        con.execute(
            "CREATE TABLE videos AS SELECT * FROM (VALUES "
            "(1, [100, 101, 102]), (2, [200, 201])) t(vid, frames)"
        )
        out = con.sql("SELECT * FROM videos").unnest("frames")
        assert out.num_rows == 5  # 3 + 2 frames
        by_vid = {}
        for vid, frame in out.fetchall():
            by_vid.setdefault(vid, []).append(frame)
        assert sorted(by_vid[1]) == [100, 101, 102]
        assert sorted(by_vid[2]) == [200, 201]

    def test_unnest_then_map(self):
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT 1 AS id, [5, 6, 7] AS vals")
        out = con.sql("SELECT * FROM t").unnest("vals").map(lambda x: x * 2, "vals", output_column="d")
        assert out.num_rows == 3
        assert sorted(r[-1] for r in out.fetchall()) == [10, 12, 14]


class TestSample:
    def test_sample_row_count(self):
        con = jude.connect()
        con.execute("CREATE TABLE big AS SELECT * FROM range(1000) t(n)")
        assert con.sql("SELECT * FROM big").sample("100 ROWS").num_rows == 100

    def test_reservoir_sample(self):
        con = jude.connect()
        con.execute("CREATE TABLE big AS SELECT * FROM range(1000) t(n)")
        assert con.sql("SELECT * FROM big").sample("reservoir(50 ROWS)").num_rows == 50
