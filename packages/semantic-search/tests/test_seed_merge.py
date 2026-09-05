"""Tests for vectorization.seed_merge: plan-building (pure), phase ordering,
COLUMN_NOT_FOUND retry fallback, and cleanup-on-exception."""
import types

import sqlglot

from semantic_search.config.models import SeedStageTimingConfig
from semantic_search.vectorization import seed_merge
from semantic_search.vectorization.db_seed_source import _build_query
from semantic_search.vectorization.seed_merge import (
    SeedMergeStatement,
    _create_stg_base_with_fallback,
    build_seed_merge_plan,
    cleanup_seed_merge,
    run_seed_merge,
)
from semantic_search.vectorization.stage_timing import StageTimingSession


def _disabled_timing() -> StageTimingSession:
    return StageTimingSession(
        SeedStageTimingConfig(
            enabled=False,
            log_page_stages=False,
            log_merge_phases=False,
            log_indexer_substages=False,
            include_in_response=False,
            elapsed_ms_decimals=1,
        )
    )


def _table_cfg(**overrides):
    defaults = dict(
        table_name="auction_audit_cln",
        strategy="datewise",
        lookback_days=7,
        max_records=1000,
        active_only=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _db_cfg(**overrides):
    defaults = dict(
        name="signals_platform_cln",
        tables=[_table_cfg()],
        timeout_seconds=60.0,
        majestic=types.SimpleNamespace(
            database="domain_majestic", table_name="domain_majestic_metric_snap", timeout_seconds=30.0
        ),
        search_rollup=None,
        merge_database="_tmp_auc_semsearch",
        bid_source_database="the_resale_place",
        bid_source_table="item_bids_cln",
        bid_winning_table="item_winning_bids_cln",
        merge_page_size=50000,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _build_plan(db_cfg, run_token="run1", s3_root="s3://bucket/athena-query/"):
    base_queries = {tc.table_name: _build_query(db_cfg.name, tc) for tc in db_cfg.tables}
    return build_seed_merge_plan(db_cfg, run_token, s3_root, base_queries)


class TestBuildSeedMergePlanSyntax:
    """sqlglot (athena dialect) parse check for every statement the plan
    generates, across strategy/active_only/rollup-configured combinations."""

    def test_every_statement_parses_across_strategy_and_active_only_combos(self):
        for strategy in ("datewise", "count"):
            for active_only in (False, True):
                db_cfg = _db_cfg(tables=[_table_cfg(strategy=strategy, active_only=active_only)])
                plan = _build_plan(db_cfg)
                assert plan.database == "_tmp_auc_semsearch"
                sqlglot.parse(plan.create_database_sql, dialect="athena")
                for stmt in plan.all_statements:
                    sqlglot.parse(stmt.drop_sql, dialect="athena")
                    sqlglot.parse(stmt.create_sql, dialect="athena")
                for stmt in plan.phase2:
                    assert stmt.fallback_create_sql is not None
                    assert stmt.source_label is not None
                    sqlglot.parse(stmt.fallback_create_sql, dialect="athena")

    def test_every_statement_parses_with_search_rollup_configured(self):
        db_cfg = _db_cfg(
            search_rollup=types.SimpleNamespace(
                database="signals_platform_cln", table_name="domain_search_rollup", lookback_days=30
            )
        )
        plan = _build_plan(db_cfg)
        for stmt in plan.all_statements:
            sqlglot.parse(stmt.drop_sql, dialect="athena")
            sqlglot.parse(stmt.create_sql, dialect="athena")
        for stmt in plan.phase2:
            sqlglot.parse(stmt.fallback_create_sql, dialect="athena")
        # Rollup configured -> a shared stg_search_rollup_* statement exists in phase2.
        assert any(s.table.startswith("stg_search_rollup_") for s in plan.phase2)

    def test_no_search_rollup_statement_when_not_configured(self):
        db_cfg = _db_cfg(search_rollup=None)
        plan = _build_plan(db_cfg)
        assert not any(s.table.startswith("stg_search_rollup_") for s in plan.phase2)

    def test_every_statement_parses_with_semrush_and_estibot_configured(self):
        db_cfg = _db_cfg(
            semrush=types.SimpleNamespace(database="domain_semrush", table_name="domain_semrush_metric_snap"),
            estibot=types.SimpleNamespace(database="domain_auction_mart", table_name="estibot_domain_enrichments"),
        )
        plan = _build_plan(db_cfg)
        for stmt in plan.all_statements:
            sqlglot.parse(stmt.drop_sql, dialect="athena")
            sqlglot.parse(stmt.create_sql, dialect="athena")
        for stmt in plan.phase2:
            sqlglot.parse(stmt.fallback_create_sql, dialect="athena")
        assert any(s.table.startswith("stg_semrush_") for s in plan.phase2)
        assert any(s.table.startswith("stg_estibot_") for s in plan.phase2)

    def test_every_statement_parses_with_aftermarket_boost_configured(self):
        db_cfg = _db_cfg(
            aftermarket_boost=types.SimpleNamespace(
                database="domain_aftermarket_mart", table_name="ims_listing",
                domain_name_column="domain_name", tier_column="domain_boost_tier",
                boosted_tier_values=["deluxe"],
            ),
        )
        plan = _build_plan(db_cfg)
        for stmt in plan.all_statements:
            sqlglot.parse(stmt.drop_sql, dialect="athena")
            sqlglot.parse(stmt.create_sql, dialect="athena")
        for stmt in plan.phase2:
            sqlglot.parse(stmt.fallback_create_sql, dialect="athena")
        assert any(s.table.startswith("stg_aftermarket_") for s in plan.phase2)

    def test_no_aftermarket_statement_when_not_configured(self):
        plan = _build_plan(_db_cfg())
        assert not any(s.table.startswith("stg_aftermarket_") for s in plan.phase2)

    def test_multiple_tables_produce_one_stg_base_and_final_per_table(self):
        db_cfg = _db_cfg(tables=[_table_cfg(table_name="auction_audit_cln"), _table_cfg(table_name="other_tbl")])
        plan = _build_plan(db_cfg)
        assert set(plan.stg_base_tables.keys()) == {"auction_audit_cln", "other_tbl"}
        assert set(plan.final_tables.keys()) == {"auction_audit_cln", "other_tbl"}
        # One shared stg_majestic, no rollup, one stg_bid_offer per table.
        bid_offer_stmts = [s for s in plan.phase2 if s.table.startswith("stg_bid_offer_")]
        assert len(bid_offer_stmts) == 2
        majestic_stmts = [s for s in plan.phase2 if s.table.startswith("stg_majestic_")]
        assert len(majestic_stmts) == 1


class _FakeAthenaClient:
    def __init__(self, fail_first_create_for_table=None, database_exists=False):
        self.ddl_calls = []
        self.database_exists_calls = []
        self._fail_first_create_for_table = fail_first_create_for_table
        self._database_exists = database_exists
        self._failed_once = set()

    async def database_exists(self, database):
        self.database_exists_calls.append(database)
        return self._database_exists

    async def execute_ddl(self, query, timeout_seconds):
        self.ddl_calls.append(query)
        target = self._fail_first_create_for_table
        if (
            target is not None
            and query.startswith(f"CREATE TABLE _tmp_auc_semsearch.{target}")
            and target not in self._failed_once
        ):
            self._failed_once.add(target)
            raise RuntimeError("COLUMN_NOT_FOUND: Column 'bid_cnt' cannot be resolved")


class TestRunSeedMergePhaseOrdering:
    async def test_phase1_runs_before_phase2_and_phase3(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        client = _FakeAthenaClient()

        missing_by_table = await run_seed_merge(client, plan, 60.0, _disabled_timing())

        phase1_tables = {s.table for s in plan.phase1}
        phase2_tables = {s.table for s in plan.phase2}
        phase3_tables = {s.table for s in plan.phase3}

        create_calls = [q for q in client.ddl_calls if q.startswith("CREATE TABLE")]
        create_index = {}
        for idx, q in enumerate(create_calls):
            for table_set_name, tables in (("phase1", phase1_tables), ("phase2", phase2_tables), ("phase3", phase3_tables)):
                for t in tables:
                    if q.startswith(f"CREATE TABLE _tmp_auc_semsearch.{t}"):
                        create_index[t] = (table_set_name, idx)

        last_phase1_idx = max(idx for name, idx in create_index.values() if name == "phase1")
        first_phase2_idx = min(idx for name, idx in create_index.values() if name == "phase2")
        last_phase2_idx = max(idx for name, idx in create_index.values() if name == "phase2")
        first_phase3_idx = min(idx for name, idx in create_index.values() if name == "phase3")

        assert last_phase1_idx < first_phase2_idx
        assert last_phase2_idx < first_phase3_idx
        assert missing_by_table == {tc.table_name: [] for tc in db_cfg.tables}

    async def test_create_database_runs_first_when_missing(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        client = _FakeAthenaClient(database_exists=False)
        await run_seed_merge(client, plan, 60.0, _disabled_timing())
        assert client.database_exists_calls == [plan.database]
        assert client.ddl_calls[0] == plan.create_database_sql

    async def test_skips_create_database_when_already_exists(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        client = _FakeAthenaClient(database_exists=True)
        await run_seed_merge(client, plan, 60.0, _disabled_timing())
        assert client.database_exists_calls == [plan.database]
        assert plan.create_database_sql not in client.ddl_calls
        assert client.ddl_calls[0].startswith("DROP TABLE IF EXISTS")


class _Phase2FailingClient:
    """Fails a phase-2 statement's real create_sql exactly once per table in
    ``fail_tables``; every other DDL (including fallback_create_sql) succeeds."""

    def __init__(self, fail_tables=(), error_message="boom"):
        self.ddl_calls = []
        self._fail_tables = set(fail_tables)
        self._failed_once = set()
        self._error_message = error_message

    async def database_exists(self, database):
        return False

    async def execute_ddl(self, query, timeout_seconds):
        self.ddl_calls.append(query)
        for table in self._fail_tables:
            if (
                query.startswith(f"CREATE TABLE _tmp_auc_semsearch.{table}")
                and "VALUES" not in query
                and table not in self._failed_once
            ):
                self._failed_once.add(table)
                raise RuntimeError(self._error_message)


class TestRunSeedMergePhase2Resilience:
    async def test_one_bad_source_degrades_to_stub_without_aborting_run(self):
        db_cfg = _db_cfg(
            estibot=types.SimpleNamespace(database="domain_auction_mart", table_name="estibot_domain_enrichments"),
        )
        plan = _build_plan(db_cfg)
        estibot_stmt = next(s for s in plan.phase2 if s.source_label == "estibot")
        client = _Phase2FailingClient(
            fail_tables=[estibot_stmt.table],
            error_message=(
                "HIVE_BAD_DATA: Malformed ORC file. Invalid postscript "
                "[s3://gd-dridata-prod-aftermarket/.../estibot_domain_enrichments/...]"
            ),
        )

        missing_by_table = await run_seed_merge(client, plan, 60.0, _disabled_timing())

        # Run completed (no raise) and phase3 still ran.
        phase3_tables = {s.table for s in plan.phase3}
        assert any(
            q.startswith(f"CREATE TABLE _tmp_auc_semsearch.{t}") for q in client.ddl_calls for t in phase3_tables
        )
        # The estibot statement's real create_sql was attempted once, then its
        # fallback stub ran instead - no exception propagated out of run_seed_merge.
        assert estibot_stmt.create_sql in client.ddl_calls
        assert estibot_stmt.fallback_create_sql in client.ddl_calls
        assert missing_by_table == {tc.table_name: [] for tc in db_cfg.tables}

    async def test_aftermarket_source_degrades_to_stub_without_aborting_run(self):
        db_cfg = _db_cfg(
            aftermarket_boost=types.SimpleNamespace(
                database="domain_aftermarket_mart", table_name="ims_listing",
                domain_name_column="domain_name", tier_column="domain_boost_tier",
                boosted_tier_values=["deluxe"],
            ),
        )
        plan = _build_plan(db_cfg)
        aftermarket_stmt = next(s for s in plan.phase2 if s.source_label == "aftermarket_boost")
        client = _Phase2FailingClient(fail_tables=[aftermarket_stmt.table], error_message="boom")

        missing_by_table = await run_seed_merge(client, plan, 60.0, _disabled_timing())

        phase3_tables = {s.table for s in plan.phase3}
        assert any(
            q.startswith(f"CREATE TABLE _tmp_auc_semsearch.{t}") for q in client.ddl_calls for t in phase3_tables
        )
        assert aftermarket_stmt.create_sql in client.ddl_calls
        assert aftermarket_stmt.fallback_create_sql in client.ddl_calls
        assert missing_by_table == {tc.table_name: [] for tc in db_cfg.tables}

    async def test_two_independent_source_failures_each_degrade_independently(self):
        db_cfg = _db_cfg(
            semrush=types.SimpleNamespace(database="domain_semrush", table_name="domain_semrush_metric_snap"),
            estibot=types.SimpleNamespace(database="domain_auction_mart", table_name="estibot_domain_enrichments"),
        )
        plan = _build_plan(db_cfg)
        estibot_stmt = next(s for s in plan.phase2 if s.source_label == "estibot")
        semrush_stmt = next(s for s in plan.phase2 if s.source_label == "semrush")
        majestic_stmt = next(s for s in plan.phase2 if s.source_label == "majestic")
        client = _Phase2FailingClient(fail_tables=[estibot_stmt.table, semrush_stmt.table])

        await run_seed_merge(client, plan, 60.0, _disabled_timing())

        assert estibot_stmt.fallback_create_sql in client.ddl_calls
        assert semrush_stmt.fallback_create_sql in client.ddl_calls
        # Majestic was never told to fail - it should use its real create_sql, not a stub.
        assert majestic_stmt.create_sql in client.ddl_calls
        assert majestic_stmt.fallback_create_sql not in client.ddl_calls

    async def test_phase1_failure_still_propagates_unchanged(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        stg_base_table = plan.phase1[0].table

        class _Phase1FailingClient:
            def __init__(self):
                self.ddl_calls = []

            async def database_exists(self, database):
                return False

            async def execute_ddl(self, query, timeout_seconds):
                self.ddl_calls.append(query)
                if query.startswith(f"CREATE TABLE _tmp_auc_semsearch.{stg_base_table}"):
                    raise RuntimeError("some unrelated fatal Athena error")

        try:
            await run_seed_merge(_Phase1FailingClient(), plan, 60.0, _disabled_timing())
            raised = False
        except RuntimeError:
            raised = True
        assert raised is True

    async def test_phase3_failure_still_propagates_unchanged(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        final_table = plan.phase3[0].table

        class _Phase3FailingClient:
            def __init__(self):
                self.ddl_calls = []

            async def database_exists(self, database):
                return False

            async def execute_ddl(self, query, timeout_seconds):
                self.ddl_calls.append(query)
                if query.startswith(f"CREATE TABLE _tmp_auc_semsearch.{final_table}"):
                    raise RuntimeError("some unrelated fatal Athena error")

        try:
            await run_seed_merge(_Phase3FailingClient(), plan, 60.0, _disabled_timing())
            raised = False
        except RuntimeError:
            raised = True
        assert raised is True


class TestCreateStgBaseWithFallback:
    async def test_column_not_found_retries_with_null_fallback_and_reports_missing(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)
        stmt = plan.phase1[0]
        client = _FakeAthenaClient(fail_first_create_for_table=stmt.table)

        missing_cols = await _create_stg_base_with_fallback(client, plan.database, stmt, ddl_timeout_seconds=60.0)

        assert "bid_cnt" in missing_cols
        create_calls = [q for q in client.ddl_calls if q.startswith("CREATE TABLE")]
        assert len(create_calls) == 2

    async def test_unrelated_error_propagates_without_retry(self):
        plan_stmt = SeedMergeStatement(
            table="stg_base_x_run1",
            drop_sql="DROP TABLE IF EXISTS _tmp_auc_semsearch.stg_base_x_run1",
            create_sql="CREATE TABLE _tmp_auc_semsearch.stg_base_x_run1 WITH (format = 'PARQUET') AS SELECT 1",
            base_select="SELECT 1",
        )

        class _AlwaysFailClient:
            async def execute_ddl(self, query, timeout_seconds):
                raise RuntimeError("some other Athena error, not COLUMN_NOT_FOUND")

        try:
            await _create_stg_base_with_fallback(_AlwaysFailClient(), "_tmp_auc_semsearch", plan_stmt, 60.0)
            raised = False
        except RuntimeError:
            raised = True
        assert raised is True


class TestCleanupSeedMerge:
    async def test_cleanup_drops_every_statement_table_even_after_phase3_failure(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)

        class _FailPhase3Client(_FakeAthenaClient):
            async def execute_ddl(self, query, timeout_seconds):
                self.ddl_calls.append(query)
                if any(query.startswith(f"CREATE TABLE _tmp_auc_semsearch.{s.table}") for s in plan.phase3):
                    raise RuntimeError("boom in phase 3")

        client = _FailPhase3Client()
        try:
            await run_seed_merge(client, plan, 60.0, _disabled_timing())
            failed = False
        except RuntimeError:
            failed = True
        assert failed is True

        cleanup_client = _FakeAthenaClient()
        await cleanup_seed_merge(cleanup_client, plan, ddl_timeout_seconds=60.0)

        expected_drops = {stmt.drop_sql for stmt in plan.all_statements}
        actual_drops = set(cleanup_client.ddl_calls)
        assert expected_drops == actual_drops

    async def test_cleanup_is_best_effort_one_drop_failure_does_not_block_others(self):
        db_cfg = _db_cfg()
        plan = _build_plan(db_cfg)

        class _OneDropFailsClient:
            def __init__(self):
                self.ddl_calls = []

            async def execute_ddl(self, query, timeout_seconds):
                self.ddl_calls.append(query)
                if query == plan.all_statements[0].drop_sql:
                    raise RuntimeError("drop failed for this one table")

        client = _OneDropFailsClient()
        await cleanup_seed_merge(client, plan, ddl_timeout_seconds=60.0)

        expected_drops = {stmt.drop_sql for stmt in plan.all_statements}
        actual_drops = set(client.ddl_calls)
        assert expected_drops == actual_drops
