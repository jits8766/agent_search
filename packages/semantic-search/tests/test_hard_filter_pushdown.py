"""Config-driven explore CH hard-filter SQL pushdown."""
from __future__ import annotations

import pytest

from semantic_search.config.models import HardFilterPushdownConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.explore.ch_clickhouse_sources import apply_hard_filter_pushdown


def _pushdown(**slots) -> HardFilterPushdownConfig:
    return HardFilterPushdownConfig.from_dict({
        'enabled': True,
        'slots': {
            name: {'column': spec['column'], 'op': spec['op']}
            for name, spec in slots.items()
        },
    })


def test_pushdown_requires_enabled_and_slots():
    with pytest.raises(ConfigurationError):
        HardFilterPushdownConfig.from_dict({'enabled': True})
    with pytest.raises(ConfigurationError):
        HardFilterPushdownConfig.from_dict({'enabled': True, 'slots': {}})


def test_apply_hard_filter_pushdown_builds_where():
    cfg = _pushdown(
        tld={'column': 'tld', 'op': 'in'},
        price_min={'column': 'price', 'op': 'gte'},
        price_max={'column': 'price', 'op': 'lte'},
    )
    sql = 'SELECT item_id, tld, price FROM auctions'
    out = apply_hard_filter_pushdown(
        sql,
        {'tld': ['com', 'net'], 'price_min': 10, 'price_max': 500},
        cfg,
    )
    assert 'AS _explore_hf_push WHERE' in out
    assert "tld IN ('com', 'net')" in out
    assert 'price >= 10' in out
    assert 'price <= 500' in out


def test_apply_hard_filter_pushdown_disabled_noop():
    cfg = HardFilterPushdownConfig.from_dict({
        'enabled': False,
        'slots': {'tld': {'column': 'tld', 'op': 'in'}},
    })
    sql = 'SELECT 1'
    assert apply_hard_filter_pushdown(sql, {'tld': ['com']}, cfg) == sql


def test_apply_hard_filter_pushdown_skips_unsafe_values():
    cfg = _pushdown(tld={'column': 'tld', 'op': 'in'})
    sql = 'SELECT tld FROM t'
    out = apply_hard_filter_pushdown(sql, {'tld': ["com'; DROP TABLE x--"]}, cfg)
    assert out == sql  # unsafe → no clause
