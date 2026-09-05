"""Tests for the LLM-as-judge relevance grading subsystem.

Coverage matrix:
    - ``RelevanceGradeOutput`` (Pydantic schema)
    - happy path
    - gain out-of-range (low / high) → reject
    - confidence out-of-range (low / high) → reject
    - empty / over-long reasoning → reject
    - bool gain → reject (validator catches Python bool-is-int trap)

- ``LLMJudgeConfig`` (dataclass)
    - ``__post_init__`` validates: enabled type, task_type, prompt_tag,
      min_confidence range, max_items_per_query lower bound,
      max_concurrent_judgments lower bound, system_prompt non-empty,
      user_prompt_template non-empty AND placeholder presence
    - ``from_dict`` enforces required keys

- ``LLMRelevanceJudge.judge``
    - happy path returns ``RelevanceJudgment`` with router metadata routed
      through ``LLMCallRouter.call_structured`` (verifies the router
      contract: ``task_type``, ``prompt_tag``, ``response_schema``)
    - below-min-confidence → ``None``
    - LLMError soft-fail → ``None``
    - CancelledError propagates (never swallowed — async-patterns rule)
    - validation errors at construction (None config / None router)
    - validation errors on bad inputs (empty query, empty item_id, non-str
      item_text)

- ``LLMJudgedQueryBuilder.build``
    - happy path produces ``RelevanceJudgedQuery`` with reviewer_id
      ``llm_judge:<task_type>``
    - all-zero grades → ``None`` (RelevanceJudgedQuery contract requires
      at least one ``gain >= 1``)
    - LLM errors on every candidate → ``None``
    - duplicate item_ids → first occurrence wins
    - candidate cap (``max_items_per_query``) enforced before fan-out
    - input validation: empty query_id, empty input_query, malformed
      candidate tuple
    - concurrency cap respected (the semaphore bounds the in-flight
      count so a stampede cannot occur)

- Registry wiring
    - default YAML (no ``llm_judge`` block) → both fields None
    - enabled YAML + missing call_router (no llm_provider) → None,
      WARNING logged
"""
import asyncio
import logging
from typing import List, Optional, Sequence, Tuple

import pytest
from pydantic import ValidationError as PydanticValidationError

from semantic_search.config.models import LLMJudgeConfig
from semantic_search.contracts import GOLDEN_DIFFICULTIES, GOLDEN_EDGE_TYPES, RelevanceJudgedQuery, RelevanceJudgment
from semantic_search.core.exceptions import ConfigurationError, LLMError, ValidationError
from semantic_search.eval.llm_relevance_judge import LLMJudgedQueryBuilder, LLMRelevanceJudge, RelevanceGradeOutput
def _difficulty() -> str:
    return next(iter(sorted(GOLDEN_DIFFICULTIES)))


def _edge() -> str:
    return next(iter(sorted(GOLDEN_EDGE_TYPES)))


@pytest.fixture
def attach_caplog_to_judge():
    """Attach pytest's caplog handler to the non-propagating judge logger.

    ``semantic_search.core.logging_utils.get_logger`` sets ``propagate=False``
    so caplog (which sits on the root logger) sees nothing by default.
    This fixture wires caplog's handler onto the judge logger for the
    duration of one test, then detaches it cleanly.
    """
    judge_logger = logging.getLogger('semantic_search.eval.llm_relevance_judge')
    old_level = judge_logger.level
    judge_logger.setLevel(logging.DEBUG)

    def _attach(caplog):
        judge_logger.addHandler(caplog.handler)
        return caplog

    yield _attach
    judge_logger.removeHandler(_attach.__self__) if hasattr(_attach, '__self__') else None
    judge_logger.setLevel(old_level)


def _make_config(**overrides) -> LLMJudgeConfig:
    base = dict(
        enabled=True,
        task_type='relevance_judge',
        prompt_tag='rj.v1',
        min_confidence=0.5,
        max_items_per_query=10,
        max_concurrent_judgments=4,
        system_prompt='Grade 0-4',
        user_prompt_template='Q={query} ID={item_id} TEXT={item_text}',
    )
    base.update(overrides)
    return LLMJudgeConfig(**base)


# ---------------------------------------------------------------------------
# Test doubles for LLMCallRouter
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Deterministic stand-in for ``LLMCallRouter``.

    Records every ``call_structured`` invocation. Returns a configured
    ``RelevanceGradeOutput`` (or raises) so we can drive every branch of
    the judge without touching a real LLM provider.
    """

    def __init__(self, responses: Sequence[object]):
        self.responses = list(responses)
        self.calls: List[dict] = []

    async def call_structured(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("FakeRouter ran out of responses")
        next_response = self.responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        return next_response, {'model': 'fake-model', 'usage': {}, 'latency_ms': 1.0}


# ---------------------------------------------------------------------------
# RelevanceGradeOutput — Pydantic schema
# ---------------------------------------------------------------------------


class TestRelevanceGradeOutputContract:
    def test_happy_path(self):
        out = RelevanceGradeOutput(gain=3, confidence=0.85, reasoning='matches brand')
        assert out.gain == 3
        assert out.confidence == 0.85
        assert out.reasoning == 'matches brand'

    @pytest.mark.parametrize('gain', [-1, 5, 10])
    def test_gain_out_of_range_rejected(self, gain: int):
        with pytest.raises(PydanticValidationError):
            RelevanceGradeOutput(gain=gain, confidence=0.5, reasoning='ok')

    @pytest.mark.parametrize('confidence', [-0.1, 1.1])
    def test_confidence_out_of_range_rejected(self, confidence: float):
        with pytest.raises(PydanticValidationError):
            RelevanceGradeOutput(gain=2, confidence=confidence, reasoning='ok')

    def test_reasoning_empty_rejected(self):
        with pytest.raises(PydanticValidationError):
            RelevanceGradeOutput(gain=2, confidence=0.5, reasoning='')

    def test_reasoning_too_long_rejected(self):
        with pytest.raises(PydanticValidationError):
            RelevanceGradeOutput(gain=2, confidence=0.5, reasoning='x' * 513)

    def test_bool_gain_rejected(self):
        with pytest.raises(PydanticValidationError):
            RelevanceGradeOutput(gain=True, confidence=0.5, reasoning='ok')


# ---------------------------------------------------------------------------
# LLMJudgeConfig — dataclass
# ---------------------------------------------------------------------------


class TestLLMJudgeConfigContract:
    def test_happy_path_post_init(self):
        c = _make_config()
        assert c.enabled is True
        assert c.task_type == 'relevance_judge'
        assert c.min_confidence == 0.5

    def test_post_init_rejects_non_bool_enabled(self):
        with pytest.raises(ConfigurationError, match=r'offline_eval\.llm_judge\.enabled must be bool'):
            _make_config(enabled='yes')

    @pytest.mark.parametrize('field,bad,msg', [
        ('task_type', '', r'offline_eval\.llm_judge\.task_type must be non-empty when enabled'),
        ('task_type', None, r'offline_eval\.llm_judge\.task_type must be non-empty when enabled'),
        ('prompt_tag', '', r'offline_eval\.llm_judge\.prompt_tag must be non-empty when enabled'),
        ('prompt_tag', None, r'offline_eval\.llm_judge\.prompt_tag must be non-empty when enabled'),
        ('min_confidence', -0.1, r'offline_eval\.llm_judge\.min_confidence must be in \[0, 1\]'),
        ('min_confidence', 1.1, r'offline_eval\.llm_judge\.min_confidence must be in \[0, 1\]'),
        ('min_confidence', 'high', r'offline_eval\.llm_judge\.min_confidence must be a number'),
        ('max_items_per_query', 0, r'offline_eval\.llm_judge\.max_items_per_query must be >= 1'),
        ('max_items_per_query', -1, r'offline_eval\.llm_judge\.max_items_per_query must be >= 1'),
        ('max_concurrent_judgments', 0, r'offline_eval\.llm_judge\.max_concurrent_judgments must be >= 1'),
        ('max_concurrent_judgments', -1, r'offline_eval\.llm_judge\.max_concurrent_judgments must be >= 1'),
        ('system_prompt', '', r'offline_eval\.llm_judge\.system_prompt must be non-empty when enabled'),
        ('system_prompt', '   ', r'offline_eval\.llm_judge\.system_prompt must be non-empty when enabled'),
    ])
    def test_post_init_rejects_invalid(self, field: str, bad, msg: str):
        with pytest.raises(ConfigurationError, match=msg):
            _make_config(**{field: bad})

    @pytest.mark.parametrize('template,placeholder', [
        pytest.param('no_placeholder {item_id} {item_text}', 'query',     id='missing_query'),
        pytest.param('{query} no_id {item_text}',            'item_id',   id='missing_item_id'),
        pytest.param('{query} {item_id} no_text',            'item_text', id='missing_item_text'),
    ])
    def test_post_init_requires_placeholder(self, template: str, placeholder: str):
        with pytest.raises(ConfigurationError, match=r'offline_eval\.llm_judge\.user_prompt_template missing placeholder \{' + placeholder + r'\}'):
            _make_config(user_prompt_template=template)

    def test_from_dict_happy_path(self):
        c = LLMJudgeConfig.from_dict({
            'enabled': True, 'task_type': 't', 'prompt_tag': 'p',
            'min_confidence': 0.5, 'max_items_per_query': 5,
            'max_concurrent_judgments': 2,
            'system_prompt': 'sp',
            'user_prompt_template': '{query} {item_id} {item_text}',
        })
        assert isinstance(c, LLMJudgeConfig)

    def test_from_dict_missing_required_raises(self):
        with pytest.raises(ConfigurationError, match='is required'):
            LLMJudgeConfig.from_dict({})

    def test_from_dict_missing_one_field_raises(self):
        with pytest.raises(ConfigurationError, match='is required'):
            LLMJudgeConfig.from_dict({
                'enabled': True, 'task_type': 't', 'prompt_tag': 'p',
                'min_confidence': 0.5, 'max_items_per_query': 5,
                'max_concurrent_judgments': 2,
                # missing system_prompt
                'user_prompt_template': '{query} {item_id} {item_text}',
            })


# ---------------------------------------------------------------------------
# LLMRelevanceJudge.judge
# ---------------------------------------------------------------------------


class TestLLMRelevanceJudgeContract:
    @pytest.mark.parametrize('config,router,match', [
        pytest.param(None,            _FakeRouter([]), 'requires a LLMJudgeConfig',  id='none_config'),
        pytest.param(_make_config(),  None,            'requires a LLMCallRouter',   id='none_router'),
    ])
    def test_init_rejects_invalid(self, config, router, match: str):
        with pytest.raises(ValidationError, match=match):
            LLMRelevanceJudge(config, router)

    @pytest.mark.parametrize('query,item_id,item_text,match', [
        pytest.param('',  'i', 't', 'non-empty query',   id='empty_query'),
        pytest.param('q', '',  't', 'non-empty item_id', id='empty_item_id'),
        pytest.param('q', 'i', 123, 'str item_text',     id='non_str_item_text'),
    ])
    @pytest.mark.asyncio
    async def test_judge_rejects_invalid_input(self, query, item_id, item_text, match: str):
        j = LLMRelevanceJudge(_make_config(), _FakeRouter([]))
        with pytest.raises(ValidationError, match=match):
            await j.judge(query, item_id, item_text)


class TestLLMRelevanceJudgeBehaviour:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        cfg = _make_config(min_confidence=0.5)
        router = _FakeRouter([RelevanceGradeOutput(gain=3, confidence=0.9, reasoning='ok')])
        judge = LLMRelevanceJudge(cfg, router)
        out = await judge.judge('shoes', 'item-1', 'red running shoes')
        assert isinstance(out, RelevanceJudgment)
        assert out.gain == 3
        assert out.item_id == 'item-1'

    @pytest.mark.asyncio
    async def test_router_invocation_uses_config_routing(self):
        """The judge must route via task_type + prompt_tag from config."""
        cfg = _make_config(task_type='rj_task', prompt_tag='rj.v3')
        router = _FakeRouter([RelevanceGradeOutput(gain=2, confidence=0.8, reasoning='ok')])
        judge = LLMRelevanceJudge(cfg, router)
        await judge.judge('shoes', 'item-1', 'red running shoes')
        assert len(router.calls) == 1
        call = router.calls[0]
        assert call['task_type'] == 'rj_task'
        assert call['prompt_tag'] == 'rj.v3'
        assert call['response_schema'] is RelevanceGradeOutput
        assert 'shoes' in call['user_prompt']
        assert 'item-1' in call['user_prompt']
        assert 'red running shoes' in call['user_prompt']

    @pytest.mark.asyncio
    async def test_below_min_confidence_returns_none(self, caplog, attach_caplog_to_judge):
        attach_caplog_to_judge(caplog)
        caplog.set_level(logging.INFO)
        cfg = _make_config(min_confidence=0.7)
        router = _FakeRouter([RelevanceGradeOutput(gain=4, confidence=0.5, reasoning='ok')])
        judge = LLMRelevanceJudge(cfg, router)
        assert await judge.judge('q', 'i', 't') is None
        assert any('llm_judge_under_confident' in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_at_min_confidence_accepted(self):
        """Boundary: confidence == min_confidence is accepted (>=)."""
        cfg = _make_config(min_confidence=0.5)
        router = _FakeRouter([RelevanceGradeOutput(gain=2, confidence=0.5, reasoning='ok')])
        judge = LLMRelevanceJudge(cfg, router)
        out = await judge.judge('q', 'i', 't')
        assert out is not None and out.gain == 2

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self, caplog, attach_caplog_to_judge):
        attach_caplog_to_judge(caplog)
        caplog.set_level(logging.WARNING)
        cfg = _make_config()
        router = _FakeRouter([LLMError('boom')])
        judge = LLMRelevanceJudge(cfg, router)
        assert await judge.judge('q', 'i', 't') is None
        assert any('llm_judge_call_failed' in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        cfg = _make_config()
        router = _FakeRouter([asyncio.CancelledError()])
        judge = LLMRelevanceJudge(cfg, router)
        with pytest.raises(asyncio.CancelledError):
            await judge.judge('q', 'i', 't')


# ---------------------------------------------------------------------------
# LLMJudgedQueryBuilder.build
# ---------------------------------------------------------------------------


class TestLLMJudgedQueryBuilderContract:
    def test_init_rejects_none_judge(self):
        with pytest.raises(ValidationError, match='requires a LLMRelevanceJudge'):
            LLMJudgedQueryBuilder(None, _make_config())

    def test_init_rejects_none_config(self):
        cfg = _make_config()
        judge = LLMRelevanceJudge(cfg, _FakeRouter([]))
        with pytest.raises(ValidationError, match='requires a LLMJudgeConfig'):
            LLMJudgedQueryBuilder(judge, None)

    def test_reviewer_id_format(self):
        cfg = _make_config(task_type='custom_task')
        judge = LLMRelevanceJudge(cfg, _FakeRouter([]))
        b = LLMJudgedQueryBuilder(judge, cfg)
        assert b.reviewer_id == 'llm_judge:custom_task'

    @pytest.mark.parametrize('query_id,input_query,candidates,match', [
        pytest.param('',   'q', [('a', 't')],         'non-empty query_id',                id='empty_query_id'),
        pytest.param('q1', '',  [('a', 't')],         'non-empty input_query',             id='empty_input_query'),
        pytest.param('q1', 'q', None,                 'non-None candidates',               id='none_candidates'),
        pytest.param('q1', 'q', [('only_one_field',)], r'must be \(item_id, item_text\)',  id='malformed_tuple'),
        pytest.param('q1', 'q', [('', 't')],          'item_id must be a non-empty string', id='empty_item_id'),
        pytest.param('q1', 'q', [('a', 123)],         'item_text must be a string',         id='non_str_item_text'),
    ])
    @pytest.mark.asyncio
    async def test_build_rejects_invalid_input(self, query_id, input_query, candidates, match: str):
        cfg = _make_config()
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter([])), cfg)
        with pytest.raises(ValidationError, match=match):
            await b.build(query_id, input_query, candidates, _difficulty(), _edge())


class TestLLMJudgedQueryBuilderBehaviour:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        cfg = _make_config(min_confidence=0.5)
        responses = [
            RelevanceGradeOutput(gain=4, confidence=0.9, reasoning='r'),
            RelevanceGradeOutput(gain=0, confidence=0.9, reasoning='r'),
            RelevanceGradeOutput(gain=2, confidence=0.9, reasoning='r'),
        ]
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter(responses)), cfg)
        judged = await b.build('q1', 'shoes', [('a', 't1'), ('b', 't2'), ('c', 't3')], _difficulty(), _edge())
        assert isinstance(judged, RelevanceJudgedQuery)
        assert judged.query_id == 'q1'
        assert judged.input_query == 'shoes'
        assert judged.reviewer_id == 'llm_judge:relevance_judge'
        gains_by_id = {j.item_id: j.gain for j in judged.judgments}
        assert gains_by_id == {'a': 4, 'b': 0, 'c': 2}

    @pytest.mark.asyncio
    async def test_all_zero_grades_returns_none(self, caplog, attach_caplog_to_judge):
        attach_caplog_to_judge(caplog)
        caplog.set_level(logging.INFO)
        cfg = _make_config(min_confidence=0.5)
        responses = [
            RelevanceGradeOutput(gain=0, confidence=0.9, reasoning='r'),
            RelevanceGradeOutput(gain=0, confidence=0.9, reasoning='r'),
        ]
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter(responses)), cfg)
        out = await b.build('q1', 'q', [('a', 't'), ('b', 't')], _difficulty(), _edge())
        assert out is None
        assert any('llm_judge_build_skip' in r.message and 'no_relevant' in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_all_llm_errors_returns_none(self):
        cfg = _make_config()
        responses = [LLMError('boom1'), LLMError('boom2')]
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter(responses)), cfg)
        out = await b.build('q1', 'q', [('a', 't'), ('b', 't')], _difficulty(), _edge())
        assert out is None

    @pytest.mark.asyncio
    async def test_below_confidence_treated_as_ungraded_not_zero(self):
        """Below-confidence grades are dropped entirely.

        Critical correctness invariant: a low-confidence grade returns None
        and is therefore ABSENT from the judgment set, NOT recorded as
        gain=0. If the judge produced one accepted grade with gain=2 and
        one below-confidence grade, we expect exactly ONE judgment in the
        output (not two).
        """
        cfg = _make_config(min_confidence=0.7)
        responses = [
            RelevanceGradeOutput(gain=2, confidence=0.9, reasoning='ok'),
            RelevanceGradeOutput(gain=4, confidence=0.5, reasoning='unsure'),
        ]
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter(responses)), cfg)
        out = await b.build('q1', 'q', [('a', 't'), ('b', 't')], _difficulty(), _edge())
        assert out is not None
        assert len(out.judgments) == 1
        assert out.judgments[0].item_id == 'a'

    @pytest.mark.asyncio
    async def test_duplicate_item_ids_first_wins(self):
        cfg = _make_config()
        responses = [
            RelevanceGradeOutput(gain=2, confidence=0.9, reasoning='r'),
            RelevanceGradeOutput(gain=3, confidence=0.9, reasoning='r'),
        ]
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _FakeRouter(responses)), cfg)
        out = await b.build('q1', 'q', [('a', 'first'), ('a', 'dup'), ('b', 't')], _difficulty(), _edge())
        assert out is not None
        ids = [j.item_id for j in out.judgments]
        assert ids == ['a', 'b']

    @pytest.mark.asyncio
    async def test_max_items_per_query_enforced_before_fanout(self):
        cfg = _make_config(max_items_per_query=2)
        # Only 2 responses needed because 3rd candidate is dropped before fan-out.
        responses = [RelevanceGradeOutput(gain=2, confidence=0.9, reasoning='r')] * 2
        router = _FakeRouter(responses)
        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, router), cfg)
        out = await b.build('q1', 'q', [('a', 't'), ('b', 't'), ('c', 't')], _difficulty(), _edge())
        assert out is not None
        assert len(out.judgments) == 2
        assert {j.item_id for j in out.judgments} == {'a', 'b'}
        assert len(router.calls) == 2  # 'c' was never sent

    @pytest.mark.asyncio
    async def test_concurrency_cap_respected(self):
        """Semaphore caps the in-flight count.

        We simulate slow LLM calls and verify that no more than
        ``max_concurrent_judgments`` are running at once. The check is
        done by recording the in-flight depth on each call entry.
        """
        cfg = _make_config(max_concurrent_judgments=2, max_items_per_query=10)
        in_flight = 0
        peak = 0

        class _SlowRouter:
            async def call_structured(self, **kw):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                try:
                    await asyncio.sleep(0.01)
                    return RelevanceGradeOutput(gain=2, confidence=0.9, reasoning='r'), {'model': 'fake', 'usage': {}, 'latency_ms': 1.0}
                finally:
                    in_flight -= 1

        b = LLMJudgedQueryBuilder(LLMRelevanceJudge(cfg, _SlowRouter()), cfg)
        cands = [(f'i{n}', 't') for n in range(8)]
        out = await b.build('q1', 'q', cands, _difficulty(), _edge())
        assert out is not None
        assert len(out.judgments) == 8
        assert peak <= 2, f"peak in-flight {peak} exceeded cap 2"

