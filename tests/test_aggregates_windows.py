"""Extended aggregate + window function surface (Vane's DuckDBPyRelation surface)."""

import jude


def _grouped():
    con = jude.connect()
    con.execute("CREATE TABLE t AS SELECT n%3 AS g, n AS v FROM range(12) t(n)")
    return con


class TestExtendedAggregates:
    def test_arg_max(self):
        con = _grouped()
        # Vane signature: (arg, value, groups, window_spec, projected_columns).
        # The group column is only in the output when projected_columns names it.
        assert sorted(
            con.sql("SELECT * FROM t").arg_max("v", "v", "g", projected_columns="g").fetchall()
        ) == [
            (0, 9),
            (1, 10),
            (2, 11),
        ]

    def test_arg_min(self):
        con = _grouped()
        assert sorted(
            con.sql("SELECT * FROM t").arg_min("v", "v", "g", projected_columns="g").fetchall()
        ) == [
            (0, 0),
            (1, 1),
            (2, 2),
        ]

    def test_string_agg(self):
        con = _grouped()
        out = dict(
            con.sql("SELECT * FROM t").string_agg("CAST(v AS VARCHAR)", "|", "g", projected_columns="g").fetchall()
        )
        assert out[0] == "0|3|6|9"

    def test_quantile_cont(self):
        con = _grouped()
        out = dict(
            con.sql("SELECT * FROM t").quantile_cont("v", 0.5, "g", projected_columns="g").fetchall()
        )
        assert out[0] == 4.5

    def test_list(self):
        con = _grouped()
        out = {
            r[0]: sorted(r[1])
            for r in con.sql("SELECT * FROM t").list("v", "g", projected_columns="g").fetchall()
        }
        assert out[1] == [1, 4, 7, 10]

    def test_std_pop_var_samp(self):
        con = _grouped()
        assert con.sql("SELECT * FROM t").std_pop("v").num_rows == 1
        assert con.sql("SELECT * FROM t").var_samp("v").num_rows == 1

    def test_product(self):
        con = jude.connect()
        con.execute("CREATE TABLE p AS SELECT * FROM range(1,6) t(n)")
        assert con.sql("SELECT * FROM p").product("n").fetchone()[0] == 120.0

    def test_bool_and_or(self):
        con = jude.connect()
        con.execute("CREATE TABLE b AS SELECT n, n > 0 AS pos FROM range(5) t(n)")
        assert con.sql("SELECT * FROM b").bool_and("pos").fetchone()[0] is False
        assert con.sql("SELECT * FROM b").bool_or("pos").fetchone()[0] is True

    def test_value_counts(self):
        con = jude.connect()
        con.execute("CREATE TABLE vc AS SELECT n%2 AS k FROM range(10) t(n)")
        out = dict(con.sql("SELECT * FROM vc").value_counts("k").fetchall())
        assert out == {0: 5, 1: 5}


class TestWindowFunctions:
    def test_row_number(self):
        con = _grouped()
        out = con.sql("SELECT * FROM t").row_number("ORDER BY v DESC", "v").limit(3).fetchall()
        assert out == [(11, 1), (10, 2), (9, 3)]

    def test_rank_partitioned(self):
        con = _grouped()
        out = con.sql("SELECT * FROM t").rank("PARTITION BY g ORDER BY v", "g, v").fetchall()
        # within each group, ranks restart at 1
        assert (1, 1, 1) in out

    def test_lag_with_default(self):
        con = _grouped()
        out = con.sql("SELECT * FROM t").lag("v", "ORDER BY v", 1, "-1", "v").limit(3).fetchall()
        assert out == [(0, -1), (1, 0), (2, 1)]

    def test_lead(self):
        con = _grouped()
        out = con.sql("SELECT * FROM t").lead("v", "ORDER BY v", 1, None, "v").order("v").limit(2).fetchall()
        assert out[0] == (0, 1)

    def test_dense_rank_and_ntile(self):
        con = _grouped()
        assert con.sql("SELECT * FROM t").dense_rank("ORDER BY g", "g").num_rows == 12
        assert con.sql("SELECT * FROM t").n_tile("ORDER BY v", 4, "v").num_rows == 12

    def test_generic_window_function(self):
        con = _grouped()
        out = con.sql("SELECT * FROM t").generic_window_function(
            "sum", "v", "ORDER BY v", "v"
        )
        assert out.num_rows == 12
        assert "sum" in out.columns
