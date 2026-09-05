"""Tests for the 16 new SQL template functions in query_templates."""
import pytest
from semantic_search.analytics.query_templates import (
    TimeGrain, FilterSpec, DateRangeFilter,
    price_distribution_sql, window_funnel_sql, top_k_sql, moving_average_sql,
    bid_velocity_acceleration_sql, watch_bid_conversion_sql, hold_time_distribution_sql,
    counter_offer_sql, price_realization_sql, buyer_retention_cohorts_sql,
    auction_timing_patterns_sql, name_structure_sql, cross_tld_spread_sql,
    search_attribution_sql, registrar_hhi_sql, comparable_sale_price_sql,
)

_INJECT = "'; DROP TABLE foo; --"


class TestPriceDistributionSql:
    def test_returns_nonempty_string(self):
        sql = price_distribution_sql(table='auctions', metric_column='price', group_column='tld', time_column='ends_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = price_distribution_sql(table='auctions', metric_column='price', group_column='tld', time_column='ends_at')
        assert 'auctions' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql
        assert 'quantileExact' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            price_distribution_sql(table=_INJECT, metric_column='price', group_column='tld', time_column='ends_at')

    def test_injection_rejected_on_metric(self):
        with pytest.raises(ValueError):
            price_distribution_sql(table='auctions', metric_column=_INJECT, group_column='tld', time_column='ends_at')


class TestWindowFunnelSql:
    def test_returns_nonempty_string(self):
        sql = window_funnel_sql(
            table='events', entity_column='user_id', time_column='created_at',
            event_column='event_type', funnel_values=['search', 'bid', 'sold'], window_seconds=3600,
        )
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = window_funnel_sql(
            table='events', entity_column='user_id', time_column='created_at',
            event_column='event_type', funnel_values=['search', 'bid', 'sold'], window_seconds=3600,
        )
        assert 'events' in sql
        assert 'windowFunnel' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'reached_step_1' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            window_funnel_sql(
                table=_INJECT, entity_column='user_id', time_column='created_at',
                event_column='event_type', funnel_values=['search'], window_seconds=3600,
            )


class TestTopKSql:
    def test_returns_nonempty_string(self):
        sql = top_k_sql(table='auctions', keyword_column='tld', time_column='ends_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = top_k_sql(table='auctions', keyword_column='tld', time_column='ends_at', k=25)
        assert 'auctions' in sql
        assert 'topK(25)' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            top_k_sql(table=_INJECT, keyword_column='tld', time_column='ends_at')


class TestMovingAverageSql:
    def test_returns_nonempty_string(self):
        sql = moving_average_sql(table='auctions', metric_column='price', entity_column='tld', time_column='ends_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = moving_average_sql(table='auctions', metric_column='price', entity_column='tld', time_column='ends_at', window_size=7)
        assert 'auctions' in sql
        assert 'moving_avg' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql
        assert 'PARTITION BY' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            moving_average_sql(table=_INJECT, metric_column='price', entity_column='tld', time_column='ends_at')


class TestBidVelocityAccelerationSql:
    def test_returns_nonempty_string(self):
        sql = bid_velocity_acceleration_sql(bid_velocity_mv='analytics.mv_bid_velocity')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = bid_velocity_acceleration_sql(bid_velocity_mv='analytics.mv_bid_velocity', window_hours=2, limit=50)
        assert 'analytics.mv_bid_velocity' in sql
        assert 'velocity_delta' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT 50' in sql

    def test_injection_rejected_on_mv(self):
        with pytest.raises(ValueError):
            bid_velocity_acceleration_sql(bid_velocity_mv=_INJECT)


class TestWatchBidConversionSql:
    def test_returns_nonempty_string(self):
        sql = watch_bid_conversion_sql(watch_density_mv='analytics.mv_watch', bid_velocity_item_mv='analytics.mv_bids')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = watch_bid_conversion_sql(watch_density_mv='analytics.mv_watch', bid_velocity_item_mv='analytics.mv_bids')
        assert 'analytics.mv_watch' in sql
        assert 'analytics.mv_bids' in sql
        assert 'watch_to_bid_rate' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_watch_mv(self):
        with pytest.raises(ValueError):
            watch_bid_conversion_sql(watch_density_mv=_INJECT, bid_velocity_item_mv='analytics.mv_bids')


class TestHoldTimeDistributionSql:
    def test_returns_nonempty_string(self):
        sql = hold_time_distribution_sql(hold_time_mv='analytics.mv_hold_time', group_column='auction_type_name')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = hold_time_distribution_sql(hold_time_mv='analytics.mv_hold_time', group_column='auction_type_name')
        assert 'analytics.mv_hold_time' in sql
        assert 'auction_type_name' in sql
        assert 'avg_hold_days' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_mv(self):
        with pytest.raises(ValueError):
            hold_time_distribution_sql(hold_time_mv=_INJECT, group_column='auction_type_name')

    def test_injection_rejected_on_group_column(self):
        with pytest.raises(ValueError):
            hold_time_distribution_sql(hold_time_mv='analytics.mv_hold_time', group_column=_INJECT)


class TestCounterOfferSql:
    def test_returns_nonempty_string(self):
        sql = counter_offer_sql(bid_events_table='analytics.bid_events', time_column='event_utc_ts')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = counter_offer_sql(bid_events_table='analytics.bid_events', time_column='event_utc_ts')
        assert 'analytics.bid_events' in sql
        assert 'counter_offer_rate' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            counter_offer_sql(bid_events_table=_INJECT, time_column='event_utc_ts')


class TestPriceRealizationSql:
    def test_returns_nonempty_string(self):
        sql = price_realization_sql(transactions_table='analytics.transactions', group_column='tld', time_column='sold_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = price_realization_sql(transactions_table='analytics.transactions', group_column='tld', time_column='sold_at')
        assert 'analytics.transactions' in sql
        assert 'avg_realization_rate' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            price_realization_sql(transactions_table=_INJECT, group_column='tld', time_column='sold_at')


class TestBuyerRetentionCohortsSql:
    def test_returns_nonempty_string(self):
        sql = buyer_retention_cohorts_sql(transactions_table='analytics.transactions', buyer_column='buyer_user_id', time_column='sold_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = buyer_retention_cohorts_sql(transactions_table='analytics.transactions', buyer_column='buyer_user_id', time_column='sold_at')
        assert 'analytics.transactions' in sql
        assert 'buyer_user_id' in sql
        assert 'retention_rate' in sql
        assert 'cohort_month' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            buyer_retention_cohorts_sql(transactions_table=_INJECT, buyer_column='buyer_user_id', time_column='sold_at')


class TestAuctionTimingPatternsSql:
    def test_returns_nonempty_string(self):
        sql = auction_timing_patterns_sql(table='auctions', time_column='ends_at', metric_column='bid_count', sold_flag_column='sold_flag')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = auction_timing_patterns_sql(table='auctions', time_column='ends_at', metric_column='bid_count', sold_flag_column='sold_flag')
        assert 'auctions' in sql
        assert 'close_hour' in sql
        assert 'close_dow' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            auction_timing_patterns_sql(table=_INJECT, time_column='ends_at', metric_column='bid_count', sold_flag_column='sold_flag')


class TestNameStructureSql:
    def test_returns_nonempty_string(self):
        sql = name_structure_sql(table='auctions', domain_column='domain_name', price_column='price', sold_flag_column='sold_flag', time_column='ends_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = name_structure_sql(table='auctions', domain_column='domain_name', price_column='price', sold_flag_column='sold_flag', time_column='ends_at')
        assert 'auctions' in sql
        assert 'name_length_bucket' in sql
        assert 'sell_through_rate' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            name_structure_sql(table=_INJECT, domain_column='domain_name', price_column='price', sold_flag_column='sold_flag', time_column='ends_at')


class TestCrossTldSpreadSql:
    def test_returns_nonempty_string(self):
        sql = cross_tld_spread_sql(table='auctions', domain_column='domain_name', tld_column='tld', price_column='price', sold_flag_column='sold_flag', time_column='ends_at', reference_tld='com')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = cross_tld_spread_sql(table='auctions', domain_column='domain_name', tld_column='tld', price_column='price', sold_flag_column='sold_flag', time_column='ends_at', reference_tld='com')
        assert 'auctions' in sql
        assert 'price_spread_ratio' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql
        assert 'com' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            cross_tld_spread_sql(table=_INJECT, domain_column='domain_name', tld_column='tld', price_column='price', sold_flag_column='sold_flag', time_column='ends_at', reference_tld='com')


class TestSearchAttributionSql:
    def test_returns_nonempty_string(self):
        sql = search_attribution_sql(signals_table='analytics.signals', transactions_table='analytics.txns', signal_time_column='created_at', transaction_time_column='sold_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = search_attribution_sql(signals_table='analytics.signals', transactions_table='analytics.txns', signal_time_column='created_at', transaction_time_column='sold_at')
        assert 'analytics.signals' in sql
        assert 'analytics.txns' in sql
        assert 'attribution_rate' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql

    def test_injection_rejected_on_signals_table(self):
        with pytest.raises(ValueError):
            search_attribution_sql(signals_table=_INJECT, transactions_table='analytics.txns', signal_time_column='created_at', transaction_time_column='sold_at')


class TestRegistrarHhiSql:
    def test_returns_nonempty_string(self):
        sql = registrar_hhi_sql(table='auctions', registrar_column='registrar_name', sold_flag_column='sold_flag', time_column='ends_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = registrar_hhi_sql(table='auctions', registrar_column='registrar_name', sold_flag_column='sold_flag', time_column='ends_at')
        assert 'auctions' in sql
        assert 'hhi_contribution' in sql
        assert 'market_share' in sql
        assert 'ORDER BY' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            registrar_hhi_sql(table=_INJECT, registrar_column='registrar_name', sold_flag_column='sold_flag', time_column='ends_at')


class TestComparableSalePriceSql:
    def test_returns_nonempty_string(self):
        sql = comparable_sale_price_sql(transactions_table='analytics.txns', tld_column='tld', price_column='sale_price', domain_column='domain_name', time_column='sold_at')
        assert isinstance(sql, str) and len(sql) > 0

    def test_contains_expected_substrings(self):
        sql = comparable_sale_price_sql(transactions_table='analytics.txns', tld_column='tld', price_column='sale_price', domain_column='domain_name', time_column='sold_at')
        assert 'analytics.txns' in sql
        assert 'name_length_bucket' in sql
        assert 'GROUP BY' in sql
        assert 'ORDER BY' in sql
        assert 'LIMIT' in sql
        assert 'p50_price' in sql

    def test_injection_rejected_on_table(self):
        with pytest.raises(ValueError):
            comparable_sale_price_sql(transactions_table=_INJECT, tld_column='tld', price_column='sale_price', domain_column='domain_name', time_column='sold_at')
