"""Tests for db_seed_source._apply_find_aliases (payload dual-write)
and fetch_seed_pages_from_db's delegation to vectorization.seed_merge."""
import types
from contextlib import aclosing

from semantic_search.config.models import SeedStageTimingConfig
from semantic_search.core.exceptions import DataIngestInterruptedError
from semantic_search.vectorization import db_seed_source, seed_merge
from semantic_search.vectorization.db_seed_source import (
    _apply_find_aliases,
    fetch_seed_from_db,
    fetch_seed_pages_from_db,
)
from semantic_search.vectorization.stage_timing import StageTimingSession


def _disabled_timing() -> StageTimingSession:
    """Stage timer with all logs off — tests assert merge/page behavior, not timing."""
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
        find_payload_aliases={},
        find_bool_aliases={},
        majestic=types.SimpleNamespace(
            database="domain_majestic", table_name="domain_majestic_metric_snap", timeout_seconds=30.0
        ),
        search_rollup=None,
        aftermarket_boost=None,
        merge_database="_tmp_auc_semsearch",
        merge_database_location="s3://bucket/tmp_auc_semsearch",
        bid_source_database="the_resale_place",
        bid_source_table="item_bids_cln",
        bid_winning_table="item_winning_bids_cln",
        merge_page_size=50000,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestApplyFindAliases:
    def test_govalue_score_dual_written_to_appraised_value(self):
        doc = {'govalue_score': 4200.0}
        payload_aliases = {'govalue_score': ['valuation_price', 'valuation_price_usd', 'appraised_value']}
        _apply_find_aliases(doc, payload_aliases, {})
        assert doc['valuation_price'] == 4200.0
        assert doc['valuation_price_usd'] == 4200.0
        assert doc['appraised_value'] == 4200.0

    def test_appraised_value_absent_when_source_missing(self):
        doc = {'tld': 'com'}
        payload_aliases = {'govalue_score': ['valuation_price', 'valuation_price_usd', 'appraised_value']}
        _apply_find_aliases(doc, payload_aliases, {})
        assert 'appraised_value' not in doc

    def test_govalue_score_none_dual_writes_none(self):
        doc = {'govalue_score': None}
        payload_aliases = {'govalue_score': ['valuation_price', 'valuation_price_usd', 'appraised_value']}
        _apply_find_aliases(doc, payload_aliases, {})
        assert doc['appraised_value'] is None


class _FakeAthenaClient:
    """Fakes the merge DDL + paged final-table reads for one table_cfg."""

    def __init__(self, total_pages=1, row_per_page=True):
        self.credentials_available = True
        self.s3_output_location = "s3://bucket/athena-query/"
        self.ddl_calls = []
        self._total_pages = total_pages
        self._row_per_page = row_per_page

    async def database_exists(self, database):
        return False

    async def execute_ddl(self, query, timeout_seconds):
        self.ddl_calls.append(query)

    async def fetch_sql_async(self, query, timeout_seconds):
        if "MAX(_page_num)" in query:
            return [{"max_page": str(self._total_pages - 1)}], ["max_page"], 1.0
        if "_page_num = " in query:
            if not self._row_per_page:
                return [], [], 0.0
            page_num = int(query.rsplit("=", 1)[1].strip())
            return (
                [{"domain_name": f"page{page_num}.com", "auction_id": str(page_num + 1), "tld": "com"}],
                ["domain_name"],
                1.0,
            )
        return [], [], 0.0


class TestFetchSeedPagesFromDb:
    """fetch_seed_pages_from_db delegates the merge to seed_merge and pages
    each table_cfg's final table, cleaning up on both success and failure."""

    async def test_credentials_unavailable_yields_nothing_and_skips_merge(self):
        class _NoCredsClient:
            credentials_available = False

            async def execute_ddl(self, query, timeout_seconds):
                raise AssertionError("execute_ddl must not be called without credentials")

        db_cfg = _db_cfg()
        pages = []
        async with aclosing(fetch_seed_pages_from_db(_NoCredsClient(), db_cfg, "test", run_token="run1", timing=_disabled_timing())) as gen:
            async for docs, missing_cols, table_name in gen:
                pages.append((docs, missing_cols, table_name))

        assert pages == []

    async def test_happy_path_pages_final_table_and_cleans_up(self, monkeypatch):
        plan_holder = {}

        real_build = seed_merge.build_seed_merge_plan

        def _capture_build(db_cfg, run_token, s3_root, base_queries):
            plan = real_build(db_cfg, run_token, s3_root, base_queries)
            plan_holder["plan"] = plan
            return plan

        monkeypatch.setattr(db_seed_source.seed_merge, "build_seed_merge_plan", _capture_build)

        client = _FakeAthenaClient(total_pages=3)
        db_cfg = _db_cfg()
        pages = []
        async with aclosing(fetch_seed_pages_from_db(client, db_cfg, "test", run_token="run1", timing=_disabled_timing())) as gen:
            async for docs, missing_cols, table_name in gen:
                pages.append((docs, missing_cols, table_name))

        assert len(pages) == 3
        assert [len(docs) for docs, _, _ in pages] == [1, 1, 1]
        assert all(table_name == "auction_audit_cln" for _, _, table_name in pages)
        assert all(missing_cols == [] for _, missing_cols, _ in pages)

        plan = plan_holder["plan"]
        expected_drops = {stmt.drop_sql for stmt in plan.all_statements}
        actual_drops = {q for q in client.ddl_calls if q.startswith("DROP TABLE IF EXISTS")}
        assert expected_drops <= actual_drops

    async def test_zero_rows_still_yields_one_empty_page(self):
        client = _FakeAthenaClient(total_pages=1, row_per_page=False)
        db_cfg = _db_cfg()
        pages = []
        async with aclosing(fetch_seed_pages_from_db(client, db_cfg, "test", run_token="run1", timing=_disabled_timing())) as gen:
            async for docs, missing_cols, table_name in gen:
                pages.append((docs, missing_cols, table_name))

        assert len(pages) == 1
        docs, missing_cols, table_name = pages[0]
        assert docs == []
        assert table_name == "auction_audit_cln"

    async def test_merge_failure_cleans_up_and_raises_data_ingest_interrupted(self, monkeypatch):
        async def _boom(athena_client, plan, ddl_timeout_seconds, timing):
            raise RuntimeError("COLUMN_NOT_FOUND: Column 'bogus' cannot be resolved")

        monkeypatch.setattr(db_seed_source.seed_merge, "run_seed_merge", _boom)

        cleanup_calls = []

        async def _record_cleanup(athena_client, plan, ddl_timeout_seconds):
            cleanup_calls.append(plan)

        monkeypatch.setattr(db_seed_source.seed_merge, "cleanup_seed_merge", _record_cleanup)

        client = _FakeAthenaClient()
        db_cfg = _db_cfg()
        pages = []
        raised = None
        try:
            async with aclosing(fetch_seed_pages_from_db(client, db_cfg, "test", run_token="run1", timing=_disabled_timing())) as gen:
                async for docs, missing_cols, table_name in gen:
                    pages.append(docs)
        except DataIngestInterruptedError as exc:
            raised = exc

        assert raised is not None
        assert pages == []
        assert len(cleanup_calls) == 1

    async def test_exception_mid_page_loop_still_triggers_cleanup(self, monkeypatch):
        cleanup_calls = []

        real_cleanup = seed_merge.cleanup_seed_merge

        async def _record_cleanup(athena_client, plan, ddl_timeout_seconds):
            cleanup_calls.append(plan)
            await real_cleanup(athena_client, plan, ddl_timeout_seconds)

        monkeypatch.setattr(db_seed_source.seed_merge, "cleanup_seed_merge", _record_cleanup)

        class _FlakyClient(_FakeAthenaClient):
            def __init__(self):
                super().__init__(total_pages=3)
                self.page_select_calls = 0

            async def fetch_sql_async(self, query, timeout_seconds):
                if "_page_num = " in query:
                    self.page_select_calls += 1
                    if self.page_select_calls == 2:
                        raise RuntimeError("boom page 2")
                return await super().fetch_sql_async(query, timeout_seconds)

        client = _FlakyClient()
        db_cfg = _db_cfg()
        pages = []
        raised = None
        try:
            async with aclosing(fetch_seed_pages_from_db(client, db_cfg, "test", run_token="run1", timing=_disabled_timing())) as gen:
                async for docs, missing_cols, table_name in gen:
                    pages.append(docs)
        except RuntimeError as exc:
            raised = exc

        assert raised is not None and "boom page 2" in str(raised)
        assert len(pages) == 1
        assert len(cleanup_calls) == 1


class TestFetchSeedFromDb:
    """fetch_seed_from_db drains fetch_seed_pages_from_db into one summary."""

    async def test_drains_pages_into_summary(self):
        client = _FakeAthenaClient(total_pages=2)
        db_cfg = _db_cfg()
        all_docs, summary = await fetch_seed_from_db(
            client, db_cfg, "test", _disabled_timing(),
        )

        assert len(all_docs) == 2
        assert summary.documents == 2
        assert summary.tables_queried == 1
        assert summary.tables_skipped == 0
        assert summary.missing_columns == []
