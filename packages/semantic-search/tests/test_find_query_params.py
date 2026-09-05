"""FIND wire format from qie_only identified_filters and keywords."""

from datetime import datetime, timezone
from urllib.parse import parse_qs

from semantic_search.config.models import FindWireConfig
from semantic_search.qi.find_query_params import build_find_wire_payload

_FIXED_NOW = datetime(2026, 8, 2, 8, 30, 0, tzinfo=timezone.utc)
# Sole keyword gate: qi.l0_llm_entity.keyword_min_probability percent/100.
_MIN_KW = 0.7


def _wire(**overrides) -> FindWireConfig:
    """Build FindWireConfig with explicit values aligned to base.yaml find_wire."""
    base = dict(
        prefer_keywords_for_query=True,
        empty_query_fallback="*",
        max_keyword_terms=3,
        keyword_term_separator=" ",
        set_use_semantic_search_when_keywords=True,
        use_semantic_search_param="useSemanticSearch",
        use_semantic_search_value="true",
    )
    base.update(overrides)
    return FindWireConfig(**base)


def _payload(identified, *, wire=None, keywords=None, min_kw=_MIN_KW, now=_FIXED_NOW):
    return build_find_wire_payload(
        identified,
        wire=wire or _wire(),
        keywords=keywords,
        min_keyword_probability=min_kw,
        now=now,
    )


def test_relative_end_time_before_converts_to_find_iso():
    identified = [
        {
            "name": "endTimeBefore",
            "value": "-1d",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(identified)
    assert out["find_query_params"]["endTimeBefore"] == "2026-08-03T08:30:00Z"
    assert out["find_query_params"]["query"] == "*"
    assert "endTimeBefore=2026-08-03T08%3A30%3A00Z" in out["find_query_string"] or (
        parse_qs(out["find_query_string"])["endTimeBefore"] == ["2026-08-03T08:30:00Z"]
    )


def test_start_time_after_relative_looks_back():
    identified = [
        {
            "name": "startTimeAfter",
            "value": "-1d",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(identified)
    assert out["find_query_params"]["startTimeAfter"] == "2026-08-01T08:30:00Z"


def test_type_include_label_expands_to_ids():
    identified = [
        {
            "name": "typeIncludeList",
            "value": "godaddy",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(identified)
    assert "typeIncludeList" in out["find_query_params"]
    ids = set(out["find_query_params"]["typeIncludeList"].split(","))
    assert ids  # non-empty expanded ids


def test_soft_chip_not_in_find_params():
    identified = [
        {
            "name": "topic_include",
            "value": "fintech",
            "chip_kind": "soft",
            "source": "L0_llm",
            "confidence": 0.9,
        },
        {
            "name": "maxPrice",
            "value": 500,
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
    ]
    out = _payload(identified)
    assert "topic_include" not in out["find_query_params"]
    assert out["find_query_params"]["maxPrice"] == "500"
    assert any(c.get("name") == "topic_include" for c in out["soft_chips"])


def test_unknown_hard_name_skipped():
    identified = [
        {
            "name": "notARealFindParam",
            "value": "x",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(identified)
    assert "notARealFindParam" not in out["find_query_params"]
    assert any(s.get("name") == "notARealFindParam" for s in out["find_skipped"])
    # Hard locals must not be dumped into soft_chips (qie_only UX clarity).
    assert not any(c.get("name") == "notARealFindParam" for c in out["soft_chips"])


def test_hard_keyword_exclude_not_soft_chip():
    """keyword_contains_exclude is hard + non-FIND: skip FIND, never soft_chips."""
    identified = [
        {
            "name": "keyword_contains_exclude",
            "value": ["crypto"],
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
        {
            "name": "tldExcludeList",
            "value": "ai",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
    ]
    out = _payload(identified)
    assert "keyword_contains_exclude" not in out["find_query_params"]
    assert any(
        s.get("name") == "keyword_contains_exclude" and s.get("reason") == "not_find_filterable"
        for s in out["find_skipped"]
    )
    assert not any(c.get("name") == "keyword_contains_exclude" for c in out["soft_chips"])
    # FIND-native exclude stays on the wire.
    assert out["find_query_params"].get("tldExcludeList") == "ai"


def test_empty_identified_uses_star_query():
    out = _payload([])
    assert out["find_query_params"]["query"] == "*"


def test_keywords_drive_query_with_hard_filters():
    identified = [
        {
            "name": "maxPrice",
            "value": 99,
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
        {
            "name": "filterPriceCurrency",
            "value": "USD",
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
    ]
    out = _payload(
        identified,
        keywords=[
            {"term": "cofee", "probability": 0.96},
            {"term": "pizza", "probability": 0.94},
        ],
    )
    assert out["find_query_params"]["query"] == "cofee pizza"
    assert out["find_query_params"]["maxPrice"] == "99"
    assert out["find_query_params"]["filterPriceCurrency"] == "USD"
    assert out["find_query_params"]["useSemanticSearch"] == "true"
    parsed = parse_qs(out["find_query_string"])
    assert parsed["query"] == ["cofee pizza"]
    assert parsed["useSemanticSearch"] == ["true"]


def test_keyword_fallback_when_no_hard_filters():
    out = _payload(
        [],
        keywords=[{"term": "brandable", "probability": 0.8}],
    )
    assert out["find_query_params"]["query"] == "brandable"
    assert out["find_query_params"]["useSemanticSearch"] == "true"


def test_low_probability_keywords_dropped_to_empty_fallback():
    identified = [
        {
            "name": "maxPrice",
            "value": 100,
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(
        identified,
        keywords=[{"term": "weak", "probability": 0.5}],
        min_kw=0.9,
    )
    assert out["find_query_params"]["query"] == "*"
    assert "useSemanticSearch" not in out["find_query_params"]


def test_max_keyword_terms_caps_and_orders_by_probability():
    out = _payload(
        [],
        wire=_wire(max_keyword_terms=2),
        keywords=[
            {"term": "c", "probability": 0.75},
            {"term": "a", "probability": 0.99},
            {"term": "b", "probability": 0.85},
        ],
    )
    assert out["find_query_params"]["query"] == "a b"


def test_prefer_keywords_false_keeps_star_with_hard_filters():
    identified = [
        {
            "name": "maxPrice",
            "value": 50,
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        }
    ]
    out = _payload(
        identified,
        wire=_wire(prefer_keywords_for_query=False),
        keywords=[{"term": "coffee", "probability": 0.95}],
    )
    assert out["find_query_params"]["query"] == "*"
    assert "useSemanticSearch" not in out["find_query_params"]


def test_boolean_and_list_params_stringify():
    identified = [
        {
            "name": "excludeDigits",
            "value": True,
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
        {
            "name": "tldIncludeList",
            "value": ["com", "io"],
            "chip_kind": "hard",
            "source": "L0_llm",
            "confidence": 0.9,
        },
    ]
    out = _payload(identified)
    assert out["find_query_params"]["excludeDigits"] == "true"
    assert "com" in out["find_query_params"]["tldIncludeList"]
