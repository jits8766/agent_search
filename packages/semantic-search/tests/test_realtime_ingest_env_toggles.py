"""All four realtime-ingest env toggles share the same string true/false semantics.

Katana injects strings. ``_coerce_bool`` must treat ``"false"`` as False
(not Python ``bool("false")`` which is True).
"""
from __future__ import annotations

import os
from typing import Tuple

import pytest

from semantic_search.config.analytics_models import _coerce_bool
from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig

_ENV_KEYS = (
    "DELTA_REFRESH_ENABLED",
    "EVENT_INGEST_BID_ENABLED",
    "EVENT_INGEST_WATCH_ENABLED",
    "EVENT_INGEST_QDRANT_ENRICH_ENABLED",
)

_TRUE_STRINGS = ("true", "TRUE", "True", "1", "yes", "on", " YES ")
_FALSE_STRINGS = ("false", "FALSE", "False", "0", "no", "off", "", "maybe", "2")


def _clear_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _flags(cfg: AgentSearchConfig) -> Tuple[bool, bool, bool, bool]:
    ei = cfg.vectorization.event_ingest
    assert ei is not None and ei.bid_events and ei.watch_events and ei.qdrant_enrich
    return (
        cfg.vectorization.delta_refresh.enabled,
        ei.bid_events.enabled,
        ei.watch_events.enabled,
        ei.qdrant_enrich.enabled,
    )


@pytest.mark.parametrize("raw", _TRUE_STRINGS)
def test_coerce_bool_true_forms(raw: str):
    assert _coerce_bool(raw) is True


@pytest.mark.parametrize("raw", _FALSE_STRINGS)
def test_coerce_bool_false_forms(raw: str):
    assert _coerce_bool(raw) is False


def test_coerce_bool_native_bools_passthrough():
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False


def test_all_four_env_unset_default_false():
    _clear_env()
    assert _flags(AgentSearchConfig.from_dict(load_config())) == (False, False, False, False)


@pytest.mark.parametrize("env_key,index", list(zip(_ENV_KEYS, range(4))))
def test_each_env_string_false_disables_only_that_flag(env_key: str, index: int):
    _clear_env()
    for k in _ENV_KEYS:
        os.environ[k] = "true"
    os.environ[env_key] = "false"
    flags = _flags(AgentSearchConfig.from_dict(load_config()))
    for i, v in enumerate(flags):
        assert v is (i != index), (env_key, flags)


@pytest.mark.parametrize("env_key,index", list(zip(_ENV_KEYS, range(4))))
def test_each_env_string_true_enables_only_that_flag(env_key: str, index: int):
    _clear_env()
    os.environ[env_key] = "true"
    flags = _flags(AgentSearchConfig.from_dict(load_config()))
    for i, v in enumerate(flags):
        assert v is (i == index), (env_key, flags)


def test_event_ingest_driver_string_false_not_truthy():
    """Driver must use _coerce_bool — bare bool('false') would wrongly enable."""
    from unittest.mock import MagicMock

    from semantic_search.nl_to_sql.athena_client import AthenaClient
    from semantic_search.vectorization.event_ingest_driver import EventIngestDriver

    class _FakeAthena(AthenaClient):
        def __init__(self) -> None:
            self.credentials_available = True

    driver = EventIngestDriver(
        bid_config={"enabled": "false"},
        watch_config={"enabled": "false"},
        qdrant_enrich_config={"enabled": "false", "enrich_limit": 100},
        athena_client=_FakeAthena(),
        qdrant_factory=MagicMock(),
    )
    assert driver._bid_enabled is False
    assert driver._watch_enabled is False
    assert driver._enrich_enabled is False
