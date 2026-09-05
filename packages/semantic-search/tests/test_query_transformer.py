"""Unit tests for QueryTransformer: config contract, token gate, LLM tier, local fallback tier, echo guard, soft-fail."""
from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from semantic_search.config.models import QueryTransformerConfig
from semantic_search.core.exceptions import ConfigurationError, LLMError, ValidationError
from semantic_search.qi.query_transformer import QueryRewriteResponse, QueryTransformResult, QueryTransformer

from ._contract_helpers import matrix_param


# ---------------------------------------------------------------------------
# Minimal valid config factory
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> QueryTransformerConfig:
    base = dict(
        enabled=True,
        rewrite_enabled=True,
        llm_tier_enabled=True,
        rewrite_threshold=7,
        rewrite_echo_max_start_offset=1,
        task_type="query_rewrite",
        prompt_tag="query_rewrite_v1",
        timeout_seconds=5.0,
        llm_system_prompt_template="Condense to keywords. Respond with JSON.",
        llm_user_prompt_template="Query: {query}",
        model_path="/models/google/flan-t5-base",
        max_tokens=128,
        rewrite_max_new_tokens=10,
        rewrite_prompt_template="Rewrite as a concise search query: {query}",
        signal_preservation_patterns=[r'\d+', r'\.[a-z]{2,}', r'[,;]|\b(?:and|or)\b'],
        encode_from_rewrite=True,
        classify_on_transformed_query=True,
        combine_rewrite_with_l0_extract=True,
        on_rewrite_reject_reextract=True,
    )
    base.update(overrides)
    return QueryTransformerConfig(**base)


# ---------------------------------------------------------------------------
# Fake torch/transformers objects so the local tier never hits disk
# ---------------------------------------------------------------------------

def _make_fake_model() -> MagicMock:
    """Build a mock that quacks like AutoModelForSeq2SeqLM (exposes generate)."""
    ids = torch.tensor([[0, 5, 6, 7, 1]])
    m = MagicMock()
    m.eval.return_value = m
    m.generate.side_effect = lambda **kwargs: ids
    return m


def _make_fake_tokenizer(decoded: str) -> MagicMock:
    """Build a mock tokenizer aligned to the fake seq2seq model."""
    tok = MagicMock()

    def _call(text: str, return_tensors: str | None = None, truncation: bool | None = None, max_length: int | None = None) -> MagicMock:
        ids = torch.tensor([[5, 6, 7, 1]])
        out = MagicMock()
        out.items = lambda: {'input_ids': ids, 'attention_mask': torch.ones(1, 4, dtype=torch.long)}.items()
        return out

    tok.side_effect = _call
    tok.decode.return_value = decoded
    return tok


def _make_router(rewritten: str | None = None, exc: BaseException | None = None, delay: float = 0.0) -> MagicMock:
    """Build a mock LLMCallRouter whose call_structured returns a rewrite, raises, or stalls."""
    router = MagicMock()

    async def _cs(**kwargs) -> tuple:
        if delay > 0.0:
            await asyncio.sleep(delay)
        if exc is not None:
            raise exc
        return QueryRewriteResponse(rewritten_query=rewritten), {'model': 'test-model'}

    router.call_structured = AsyncMock(side_effect=_cs)
    return router


def _build_qt(cfg: QueryTransformerConfig | None = None, router: MagicMock | None = None, local_decoded: str = "tech startup domain", with_local: bool = True) -> QueryTransformer:
    """Build a QueryTransformer with injected router + local model, bypassing model load."""
    cfg = cfg or _cfg()
    qt = QueryTransformer.__new__(QueryTransformer)
    qt._config = cfg
    qt._call_router = router
    qt._device = torch.device("cpu")
    qt._signal_res = [re.compile(p, re.IGNORECASE) for p in cfg.signal_preservation_patterns]
    if with_local:
        qt._local_tokenizer = _make_fake_tokenizer(decoded=local_decoded)
        qt._local_model = _make_fake_model()
    else:
        qt._local_tokenizer = None
        qt._local_model = None
    return qt


_LONG_QUERY = "i am really looking for a short brandable tech startup domain under fifty dollars"


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------

class TestQueryTransformerConfigContract:

    VALID = dict(
        enabled=True,
        rewrite_enabled=True,
        llm_tier_enabled=True,
        rewrite_threshold=7,
        rewrite_echo_max_start_offset=1,
        task_type="query_rewrite",
        prompt_tag="query_rewrite_v1",
        timeout_seconds=0.8,
        llm_system_prompt_template="Condense to keywords.",
        llm_user_prompt_template="Query: {query}",
        model_path="/models/google/flan-t5-base",
        max_tokens=128,
        rewrite_max_new_tokens=10,
        rewrite_prompt_template="Rewrite as a concise search query: {query}",
        signal_preservation_patterns=[r'\d+', r'\.[a-z]{2,}'],
        encode_from_rewrite=True,
        classify_on_transformed_query=True,
        combine_rewrite_with_l0_extract=True,
        on_rewrite_reject_reextract=True,
    )

    def test_happy_path(self) -> None:
        cfg = QueryTransformerConfig(**self.VALID)
        assert cfg.task_type == "query_rewrite"
        assert cfg.timeout_seconds == 0.8
        assert "{query}" in cfg.llm_user_prompt_template
        assert "{query}" in cfg.rewrite_prompt_template

    @pytest.mark.parametrize("field,bad,match", [
        matrix_param("empty_task_type",             "task_type",                 "", "task_type"),
        matrix_param("empty_prompt_tag",            "prompt_tag",                "", "prompt_tag"),
        matrix_param("zero_timeout",                "timeout_seconds",           0.0, "timeout_seconds"),
        matrix_param("neg_timeout",                 "timeout_seconds",           -1.0, "timeout_seconds"),
        matrix_param("empty_llm_system",            "llm_system_prompt_template", "", "llm_system_prompt_template"),
        matrix_param("llm_user_missing_placeholder","llm_user_prompt_template",  "Query text", "placeholder"),
        matrix_param("empty_model_path",            "model_path",                "", "model_path"),
        matrix_param("zero_max_tokens",             "max_tokens",                0,  "max_tokens"),
        matrix_param("zero_max_new_tokens",         "rewrite_max_new_tokens",    0,  "rewrite_max_new_tokens"),
        matrix_param("neg_echo_offset",             "rewrite_echo_max_start_offset", -1, "rewrite_echo_max_start_offset"),
        matrix_param("local_missing_placeholder",   "rewrite_prompt_template",   "Rewrite this query", "placeholder"),
        matrix_param("zero_threshold",              "rewrite_threshold",         0,  "rewrite_threshold"),
        matrix_param("signal_patterns_not_list",    "signal_preservation_patterns", "abc", "signal_preservation_patterns"),
        matrix_param("signal_pattern_empty_entry",  "signal_preservation_patterns", [""], "signal_preservation_patterns"),
    ])
    def test_invalid_fields(self, field: str, bad: object, match: str) -> None:
        kw = dict(self.VALID)
        kw[field] = bad
        with pytest.raises((ConfigurationError, ValidationError), match=match):
            QueryTransformerConfig(**kw)

    def test_from_dict_requires_all_fields(self) -> None:
        d = dict(self.VALID)
        del d['task_type']
        with pytest.raises((ConfigurationError, ValidationError, KeyError)):
            QueryTransformerConfig.from_dict(d)

    def test_from_dict_requires_classify_on_transformed_query(self) -> None:
        d = dict(self.VALID)
        del d['classify_on_transformed_query']
        with pytest.raises(ConfigurationError, match='classify_on_transformed_query'):
            QueryTransformerConfig.from_dict(d)

    def test_from_dict_requires_encode_from_rewrite(self) -> None:
        d = dict(self.VALID)
        del d['encode_from_rewrite']
        with pytest.raises(ConfigurationError, match='encode_from_rewrite'):
            QueryTransformerConfig.from_dict(d)

    def test_from_dict_roundtrip(self) -> None:
        cfg = QueryTransformerConfig.from_dict(dict(self.VALID))
        assert cfg.task_type == self.VALID['task_type']
        assert cfg.timeout_seconds == self.VALID['timeout_seconds']
        assert cfg.llm_user_prompt_template == self.VALID['llm_user_prompt_template']
        assert cfg.classify_on_transformed_query is True
        assert cfg.encode_from_rewrite is True


# ---------------------------------------------------------------------------
# Token-count gate
# ---------------------------------------------------------------------------

class TestTokenGate:

    def test_passthrough_at_threshold(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="x y z"))
        r = asyncio.run(qt.transform("a b c d e f g"))
        assert r.mode == 'passthrough'
        assert r.transformed is False
        assert r.query == r.original_query

    def test_passthrough_short_query(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="x y z"))
        r = asyncio.run(qt.transform("cheap io domain"))
        assert r.mode == 'passthrough'
        assert r.transformed is False

    def test_rewrite_disabled_passthrough(self) -> None:
        qt = _build_qt(_cfg(rewrite_enabled=False), router=_make_router(rewritten="x y z"))
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'

    def test_empty_input_passthrough(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="x y z"))
        r = asyncio.run(qt.transform(""))
        assert r.mode == 'passthrough'
        assert r.transformed is False


# ---------------------------------------------------------------------------
# Tier 1: LLM primary
# ---------------------------------------------------------------------------

class TestLLMTier:

    def test_llm_rewrite_selected(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="tech startup domain"))
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'llm_rewrite'
        assert r.query == "tech startup domain"
        assert r.transformed is True
        assert r.engine == "test-model"

    def test_llm_user_prompt_substituted(self) -> None:
        router = _make_router(rewritten="tech startup domain")
        qt = _build_qt(router=router)
        asyncio.run(qt.transform(_LONG_QUERY))
        kwargs = router.call_structured.call_args.kwargs
        assert kwargs['user_prompt'] == f"Query: {_LONG_QUERY}"
        assert kwargs['task_type'] == "query_rewrite"
        assert kwargs['response_schema'] is QueryRewriteResponse

    def test_llm_result_stripped(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="  short io names  "))
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.query == "short io names"


# ---------------------------------------------------------------------------
# Tier 2: local fallback (LLM timeout / error / absence)
# ---------------------------------------------------------------------------

class TestLocalFallbackTier:

    def test_llm_error_falls_back_to_local(self) -> None:
        qt = _build_qt(router=_make_router(exc=LLMError("chain exhausted")), local_decoded="local keywords out")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "local keywords out"
        assert r.transformed is True
        assert r.engine == "/models/google/flan-t5-base"

    def test_llm_timeout_falls_back_to_local(self) -> None:
        cfg = _cfg(timeout_seconds=0.05)
        qt = _build_qt(cfg, router=_make_router(rewritten="never returned", delay=0.5), local_decoded="local keywords out")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "local keywords out"

    def test_no_router_uses_local(self) -> None:
        qt = _build_qt(router=None, local_decoded="local keywords out")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "local keywords out"

    def test_llm_tier_disabled_uses_local_despite_router(self) -> None:
        # llm_tier_enabled=False -> skip LLM entirely, flan-t5 local tier serves even with a working router.
        qt = _build_qt(_cfg(llm_tier_enabled=False), router=_make_router(rewritten="llm output ignored"), local_decoded="local keywords out")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "local keywords out"

    def test_local_max_new_tokens_forwarded(self) -> None:
        qt = _build_qt(_cfg(rewrite_max_new_tokens=10), router=None, local_decoded="local keywords out")
        asyncio.run(qt.transform(_LONG_QUERY))
        assert qt._local_model.generate.call_args.kwargs['max_new_tokens'] == 10


# ---------------------------------------------------------------------------
# Both tiers unavailable / failing -> passthrough
# ---------------------------------------------------------------------------

class TestPassthroughWhenBothFail:

    def test_no_router_no_local_passthrough(self) -> None:
        qt = _build_qt(router=None, with_local=False)
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'
        assert r.query == _LONG_QUERY
        assert r.transformed is False

    def test_llm_error_and_no_local_passthrough(self) -> None:
        qt = _build_qt(router=_make_router(exc=LLMError("down")), with_local=False)
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'
        assert r.transformed is False

    def test_llm_error_and_local_generate_raises_passthrough(self) -> None:
        qt = _build_qt(router=_make_router(exc=LLMError("down")))
        qt._local_model.generate.side_effect = RuntimeError("kernel error")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'
        assert r.query == r.original_query


# ---------------------------------------------------------------------------
# Echo guard (shared by both tiers)
# ---------------------------------------------------------------------------

class TestEchoGuard:

    def test_llm_echo_falls_back_to_local(self) -> None:
        # LLM copies a leading run (index 0) -> echo rejected -> local tier serves.
        qt = _build_qt(router=_make_router(rewritten="i am really looking for a short brand"), local_decoded="short brandable tech")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "short brandable tech"

    def test_llm_echo_and_local_echo_passthrough(self) -> None:
        qt = _build_qt(router=_make_router(rewritten="i am really looking"), local_decoded="i am really looking for a")
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'
        assert r.query == _LONG_QUERY

    def test_interior_drop_accepted(self) -> None:
        # Drops an interior word -> not a contiguous span of the original -> accepted by LLM tier.
        query = "hey i would like to buy a cheap dot com domain for my blog"
        qt = _build_qt(router=_make_router(rewritten="cheap dot com domain for blog"))
        r = asyncio.run(qt.transform(query))
        assert r.mode == 'llm_rewrite'
        assert r.query == "cheap dot com domain for blog"

    def test_offset_zero_allows_dropped_leading_word(self) -> None:
        query = "hey i would like to buy a cheap dot com domain for my blog"
        qt = _build_qt(_cfg(rewrite_echo_max_start_offset=0), router=_make_router(rewritten="i would like to buy a cheap"))
        r = asyncio.run(qt.transform(query))
        assert r.mode == 'llm_rewrite'
        assert r.query == "i would like to buy a cheap"


# ---------------------------------------------------------------------------
# Guarded replace: reject a rewrite that drops a hard signal present in original
# ---------------------------------------------------------------------------

class TestSignalPreservationGuard:

    PRICE_QUERY = "i am looking for good brandable io domains priced under 50 dollars for my startup"

    def test_llm_drops_price_falls_back_to_local(self) -> None:
        # LLM rewrite loses the "50" digit -> guard rejects -> local tier (keeps 50) serves.
        qt = _build_qt(router=_make_router(rewritten="brandable io domains startup"), local_decoded="io domains under 50")
        r = asyncio.run(qt.transform(self.PRICE_QUERY))
        assert r.mode == 'local_fallback'
        assert r.query == "io domains under 50"

    def test_both_drop_price_passthrough(self) -> None:
        # Neither tier keeps the digit -> both rejected -> passthrough preserves original.
        qt = _build_qt(router=_make_router(rewritten="brandable io domains startup"), local_decoded="io domains startup")
        r = asyncio.run(qt.transform(self.PRICE_QUERY))
        assert r.mode == 'passthrough'
        assert r.query == self.PRICE_QUERY
        assert r.transformed is False

    def test_preserved_signal_accepted(self) -> None:
        # Rewrite keeps the digit -> guard passes -> llm_rewrite accepted.
        qt = _build_qt(router=_make_router(rewritten="io domains under 50 startup"))
        r = asyncio.run(qt.transform(self.PRICE_QUERY))
        assert r.mode == 'llm_rewrite'
        assert r.query == "io domains under 50 startup"

    def test_empty_patterns_disables_guard(self) -> None:
        # No configured patterns -> a digit-dropping rewrite is accepted.
        qt = _build_qt(_cfg(signal_preservation_patterns=[]), router=_make_router(rewritten="brandable io domains startup"))
        r = asyncio.run(qt.transform(self.PRICE_QUERY))
        assert r.mode == 'llm_rewrite'
        assert r.query == "brandable io domains startup"


# ---------------------------------------------------------------------------
# Soft-fail contract
# ---------------------------------------------------------------------------

class TestSoftFail:

    def test_passthrough_on_unexpected_error(self) -> None:
        qt = _build_qt(router=None)
        qt._config = None  # force an attribute error inside _transform_inner
        r = asyncio.run(qt.transform(_LONG_QUERY))
        assert r.mode == 'passthrough'
        assert r.query == _LONG_QUERY
        assert r.transformed is False


# ---------------------------------------------------------------------------
# QueryTransformResult dataclass
# ---------------------------------------------------------------------------

class TestQueryTransformResult:

    def test_fields_accessible(self) -> None:
        r = QueryTransformResult(query="tech", original_query="cheap tech", mode="llm_rewrite", transformed=True, engine="test-model")
        assert r.query == "tech"
        assert r.original_query == "cheap tech"
        assert r.mode == "llm_rewrite"
        assert r.transformed is True
        assert r.engine == "test-model"

    def test_passthrough_equality(self) -> None:
        r = QueryTransformResult(query="x", original_query="x", mode="passthrough", transformed=False, engine="")
        assert r.transformed is False
        assert r.query == r.original_query
        assert r.engine == ""
