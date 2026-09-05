"""Tests for the 7 new enrichment step functions in enrichment_pipeline."""
import datetime
from typing import Any, Dict, List

import pytest

from semantic_search.analytics.enrichment_pipeline import (
    EnrichmentContext,
    enrich_heat_score,
    enrich_watcher_count,
    enrich_unique_bidders,
    enrich_fair_value_band,
    enrich_tld_momentum,
    enrich_sell_through_prob,
    enrich_time_urgency,
)


def _ctx() -> EnrichmentContext:
    return EnrichmentContext(request_id='test-req', entity_column='tld', time_column='period')


class TestEnrichHeatScore:
    def test_empty_input_returns_empty(self) -> None:
        result = enrich_heat_score([], _ctx())
        assert result == []

    def test_high_tier(self) -> None:
        rows = [{'velocity_delta': 10.0}]
        result = enrich_heat_score(rows, _ctx(), high_threshold=5.0, medium_threshold=1.0)
        assert result[0]['heat_score'] == 'high'

    def test_medium_tier(self) -> None:
        rows = [{'velocity_delta': 3.0}]
        result = enrich_heat_score(rows, _ctx(), high_threshold=5.0, medium_threshold=1.0)
        assert result[0]['heat_score'] == 'medium'

    def test_low_tier(self) -> None:
        rows = [{'velocity_delta': 0.5}]
        result = enrich_heat_score(rows, _ctx(), high_threshold=5.0, medium_threshold=1.0)
        assert result[0]['heat_score'] == 'low'

    def test_missing_velocity_delta_gives_unknown(self) -> None:
        rows = [{'other_col': 1}]
        result = enrich_heat_score(rows, _ctx())
        assert result[0]['heat_score'] == 'unknown'

    def test_returns_new_list(self) -> None:
        rows = [{'velocity_delta': 10.0}]
        result = enrich_heat_score(rows, _ctx())
        assert result is not rows


class TestEnrichWatcherCount:
    def test_empty_input_returns_same(self) -> None:
        result = enrich_watcher_count([], _ctx(), watcher_data={1: 5})
        assert result == []

    def test_none_watcher_data_returns_same(self) -> None:
        rows = [{'member_item_id': 1}]
        result = enrich_watcher_count(rows, _ctx(), watcher_data=None)
        assert result is rows

    def test_lookup_populates_count(self) -> None:
        rows = [{'member_item_id': 42}]
        result = enrich_watcher_count(rows, _ctx(), watcher_data={42: 7})
        assert result[0]['watcher_count'] == 7

    def test_missing_key_gives_zero(self) -> None:
        rows = [{'member_item_id': 99}]
        result = enrich_watcher_count(rows, _ctx(), watcher_data={1: 5})
        assert result[0]['watcher_count'] == 0


class TestEnrichUniqueBidders:
    def test_empty_input_returns_same(self) -> None:
        result = enrich_unique_bidders([], _ctx(), bidder_data={1: 3})
        assert result == []

    def test_none_bidder_data_returns_same(self) -> None:
        rows = [{'auction_id': 1}]
        result = enrich_unique_bidders(rows, _ctx(), bidder_data=None)
        assert result is rows

    def test_lookup_populates_count(self) -> None:
        rows = [{'auction_id': 10}]
        result = enrich_unique_bidders(rows, _ctx(), bidder_data={10: 4})
        assert result[0]['unique_bidder_count'] == 4

    def test_missing_key_gives_zero(self) -> None:
        rows = [{'auction_id': 999}]
        result = enrich_unique_bidders(rows, _ctx(), bidder_data={1: 3})
        assert result[0]['unique_bidder_count'] == 0


class TestEnrichFairValueBand:
    def test_empty_input_returns_same(self) -> None:
        result = enrich_fair_value_band([], _ctx(), comparable_data={'com': {'p25': 100.0, 'p75': 500.0}})
        assert result == []

    def test_none_comparable_data_returns_same(self) -> None:
        rows = [{'tld': 'com'}]
        result = enrich_fair_value_band(rows, _ctx(), comparable_data=None)
        assert result is rows

    def test_band_annotated(self) -> None:
        rows = [{'tld': 'com', 'current_price': 300.0}]
        result = enrich_fair_value_band(rows, _ctx(), comparable_data={'com': {'p25': 100.0, 'p75': 500.0}})
        assert result[0]['fair_value_p25'] == 100.0
        assert result[0]['fair_value_p75'] == 500.0

    def test_missing_tld_gives_none(self) -> None:
        rows = [{'tld': 'xyz', 'current_price': 50.0}]
        result = enrich_fair_value_band(rows, _ctx(), comparable_data={'com': {'p25': 100.0, 'p75': 500.0}})
        assert result[0]['fair_value_p25'] is None
        assert result[0]['fair_value_p75'] is None


class TestEnrichTldMomentum:
    def test_empty_input_returns_same(self) -> None:
        result = enrich_tld_momentum([], _ctx(), momentum_data={'com': 20.0})
        assert result == []

    def test_none_momentum_data_returns_same(self) -> None:
        rows = [{'tld': 'com'}]
        result = enrich_tld_momentum(rows, _ctx(), momentum_data=None)
        assert result is rows

    def test_rising_label(self) -> None:
        rows = [{'tld': 'com'}]
        result = enrich_tld_momentum(rows, _ctx(), momentum_data={'com': 15.0}, positive_threshold=10.0)
        assert result[0]['tld_momentum_label'] == 'rising'

    def test_falling_label(self) -> None:
        rows = [{'tld': 'net'}]
        result = enrich_tld_momentum(rows, _ctx(), momentum_data={'net': -15.0}, negative_threshold=-10.0)
        assert result[0]['tld_momentum_label'] == 'falling'

    def test_stable_label(self) -> None:
        rows = [{'tld': 'org'}]
        result = enrich_tld_momentum(rows, _ctx(), momentum_data={'org': 2.0}, positive_threshold=10.0, negative_threshold=-10.0)
        assert result[0]['tld_momentum_label'] == 'stable'

    def test_unknown_label_when_tld_missing(self) -> None:
        rows = [{'tld': 'io'}]
        result = enrich_tld_momentum(rows, _ctx(), momentum_data={'com': 15.0})
        assert result[0]['tld_momentum_label'] == 'unknown'


class TestEnrichSellThroughProb:
    def test_empty_input_returns_same(self) -> None:
        result = enrich_sell_through_prob([], _ctx(), sell_through_data={'com': 0.7})
        assert result == []

    def test_none_sell_through_data_returns_same(self) -> None:
        rows = [{'tld': 'com'}]
        result = enrich_sell_through_prob(rows, _ctx(), sell_through_data=None)
        assert result is rows

    def test_high_tier(self) -> None:
        rows = [{'tld': 'com'}]
        result = enrich_sell_through_prob(rows, _ctx(), sell_through_data={'com': 0.75}, high_threshold=0.6)
        assert result[0]['sell_through_tier'] == 'high'

    def test_medium_tier(self) -> None:
        rows = [{'tld': 'net'}]
        result = enrich_sell_through_prob(rows, _ctx(), sell_through_data={'net': 0.45}, high_threshold=0.6, low_threshold=0.3)
        assert result[0]['sell_through_tier'] == 'medium'

    def test_low_tier(self) -> None:
        rows = [{'tld': 'org'}]
        result = enrich_sell_through_prob(rows, _ctx(), sell_through_data={'org': 0.1}, low_threshold=0.3)
        assert result[0]['sell_through_tier'] == 'low'

    def test_unknown_when_tld_missing(self) -> None:
        rows = [{'tld': 'io'}]
        result = enrich_sell_through_prob(rows, _ctx(), sell_through_data={'com': 0.7})
        assert result[0]['sell_through_tier'] == 'unknown'


class TestEnrichTimeUrgency:
    def test_empty_input_returns_empty(self) -> None:
        result = enrich_time_urgency([], _ctx())
        assert result == []

    def test_critical_label(self) -> None:
        ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)
        rows = [{'ends_at': ends.isoformat()}]
        result = enrich_time_urgency(rows, _ctx(), critical_hours=1.0, high_hours=6.0, medium_hours=24.0)
        assert result[0]['time_urgency_label'] == 'critical'

    def test_high_label(self) -> None:
        ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=3)
        rows = [{'ends_at': ends.isoformat()}]
        result = enrich_time_urgency(rows, _ctx(), critical_hours=1.0, high_hours=6.0, medium_hours=24.0)
        assert result[0]['time_urgency_label'] == 'high'

    def test_medium_label(self) -> None:
        ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=12)
        rows = [{'ends_at': ends.isoformat()}]
        result = enrich_time_urgency(rows, _ctx(), critical_hours=1.0, high_hours=6.0, medium_hours=24.0)
        assert result[0]['time_urgency_label'] == 'medium'

    def test_low_label(self) -> None:
        ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=48)
        rows = [{'ends_at': ends.isoformat()}]
        result = enrich_time_urgency(rows, _ctx(), critical_hours=1.0, high_hours=6.0, medium_hours=24.0)
        assert result[0]['time_urgency_label'] == 'low'

    def test_unparseable_gives_unknown(self) -> None:
        rows = [{'ends_at': 'not-a-date'}]
        result = enrich_time_urgency(rows, _ctx())
        assert result[0]['time_urgency_label'] == 'unknown'

    def test_none_ends_at_gives_unknown(self) -> None:
        rows = [{'ends_at': None}]
        result = enrich_time_urgency(rows, _ctx())
        assert result[0]['time_urgency_label'] == 'unknown'

    def test_datetime_object_input(self) -> None:
        ends = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=48)
        rows = [{'ends_at': ends}]
        result = enrich_time_urgency(rows, _ctx(), critical_hours=1.0, high_hours=6.0, medium_hours=24.0)
        assert result[0]['time_urgency_label'] == 'low'
