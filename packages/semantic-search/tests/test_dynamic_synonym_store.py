"""Tests for DynamicSynonymStore and DynamicSynonymConfig."""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from semantic_search.config.models import DynamicSynonymConfig
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.retrieval.dynamic_synonym_store import DynamicSynonymStore, _build_bidir


def _make_config(db_path: str, enabled: bool = True, min_freq: int = 1, queue_max: int = 50) -> DynamicSynonymConfig:
    return DynamicSynonymConfig(enabled=enabled, db_path=db_path, miss_queue_max=queue_max, min_freq_to_expand=min_freq, expansion_concurrency=2, expansion_weight=0.4, max_synonyms_per_token=3, max_db_entries=10000, min_confidence=0.7)


# --- _build_bidir unit tests ---

def test_build_bidir_basic() -> None:
    result = _build_bidir({"ai": ["artificial", "intelligence"]})
    assert "artificial" in result["ai"]
    assert "intelligence" in result["ai"]
    assert "ai" in result["artificial"]
    assert "ai" in result["intelligence"]


def test_build_bidir_self_edge_dropped() -> None:
    result = _build_bidir({"shop": ["shop", "store"]})
    assert "shop" not in result.get("shop", set())
    assert "store" in result["shop"]


def test_build_bidir_empty_input() -> None:
    assert _build_bidir({}) == {}


def test_build_bidir_invalid_key_raises() -> None:
    with pytest.raises(ValidationError):
        _build_bidir({123: ["foo"]})  # type: ignore[dict-item]


# --- DynamicSynonymConfig validation ---

def test_dynamic_synonym_config_invalid_weight() -> None:
    with pytest.raises(Exception):
        DynamicSynonymConfig(enabled=True, db_path="/tmp/x.db", miss_queue_max=10, min_freq_to_expand=1, expansion_concurrency=1, expansion_weight=0.0, max_synonyms_per_token=3)


def test_dynamic_synonym_config_from_dict() -> None:
    d = {"enabled": True, "db_path": "/tmp/x.db", "miss_queue_max": 100, "min_freq_to_expand": 2, "expansion_concurrency": 3, "expansion_weight": 0.5, "max_synonyms_per_token": 4, "max_db_entries": 5000, "min_confidence": 0.7}
    cfg = DynamicSynonymConfig.from_dict(d)
    assert cfg.enabled is True
    assert cfg.expansion_weight == 0.5
    assert cfg.min_confidence == 0.7


# --- DynamicSynonymStore functional tests ---

def test_store_disabled_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"), enabled=False)
        store = DynamicSynonymStore(cfg)
        assert store.get("ai") == []
        store.record_miss("ai")
        assert store.drain_miss_queue(10) == []


def test_store_put_and_get_bidirectional() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        store.put("cloud", ["hosting", "aws"])
        assert "hosting" in store.get("cloud")
        assert "aws" in store.get("cloud")
        assert "cloud" in store.get("hosting")
        assert "cloud" in store.get("aws")
        store.close()


def test_store_persists_across_restart() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = str(Path(tmpdir) / "s.db")
        cfg = _make_config(db)
        store = DynamicSynonymStore(cfg)
        store.put("shop", ["store", "retail"])
        store.close()
        store2 = DynamicSynonymStore(cfg)
        assert "store" in store2.get("shop")
        assert "shop" in store2.get("retail")
        store2.close()


def test_store_miss_queue_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"), min_freq=3)
        store = DynamicSynonymStore(cfg)
        store.record_miss("tools")
        store.record_miss("tools")
        assert store.drain_miss_queue(10) == []
        store.record_miss("tools")
        batch = store.drain_miss_queue(10)
        assert "tools" in batch
        store.close()


def test_store_already_known_token_not_queued() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        store.put("ai", ["artificial"])
        store.record_miss("ai")
        assert store.drain_miss_queue(10) == []
        store.close()


def test_store_put_invalid_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        with pytest.raises(ValidationError):
            store.put("", ["foo"])
        with pytest.raises(ValidationError):
            store.put("tok", "not-a-list")  # type: ignore[arg-type]
        store.close()


def test_store_drain_max_count() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        for i in range(10):
            store.record_miss(f"tok{i}")
        batch = store.drain_miss_queue(3)
        assert len(batch) == 3
        store.close()


def test_store_get_unknown_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        assert store.get("nonexistent") == []
        store.close()


def test_store_map_size_grows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(str(Path(tmpdir) / "s.db"))
        store = DynamicSynonymStore(cfg)
        assert store.map_size == 0
        store.put("tech", ["technology"])
        assert store.map_size >= 2
        store.close()
