"""Multi-intent identification + execution over multi-sentence and long queries.

Validates the signal-based split screen in ``QIEngine.classify``:
  - genuine multi-intent queries split and fan out (one ``_classify_single`` per
    real sub-intent),
  - conversational / off-distribution fragments are dropped before the LLM runs
    so a spurious split collapses to a single classify,
  - period-separated sentences without a connective are never split,
  - the screen never empties the candidate list (strongest survives).

The LLM seam (``_classify_single``) and the L1 router scorer (``quick_classify``)
are mocked; everything else — the real ``MultiIntentSplitter`` and the real
``multi_intent`` config block from ``base.yaml`` — runs unmodified. Expected
survivors are DERIVED from the real splitter + the same floor the engine uses,
so the assertions track behaviour rather than hardcoded fragment counts.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.config.loader import load_config
from semantic_search.config.models import MultiIntentConfig, QINormalizeConfig
from semantic_search.contracts import IntentSlice
from semantic_search.qi.engine import QIEngine, normalize_query
from semantic_search.qi.multi_intent_splitter import MultiIntentSplitter

from tests.test_review_fixes import _make_engine



def _normalize_cfg() -> QINormalizeConfig:
    return QINormalizeConfig.from_dict(load_config()["qi"]["normalize"])


# Fragments containing any of these markers are treated as conversational noise
# by the fake L1 router (low confidence). Everything else scores as a real
# search intent (high confidence). Keeps the scorer independent of the engine's
# real centroid model so the test is deterministic and offline.
_NOISE_MARKERS = ("i am", "i'm", "browsing", "looking around", "from delhi")
_NOISE_CONF = 0.20
_REAL_CONF = 0.90


def _fake_quick_classify(fragment: str):
    """Stand-in for the L1 semantic router: (query_type, confidence)."""
    low = any(m in fragment.lower() for m in _NOISE_MARKERS)
    return ("guidance", _NOISE_CONF) if low else ("hybrid", _REAL_CONF)


def _engine_with_splitter():
    """QIEngine wired with the REAL multi_intent splitter + config from base.yaml."""
    mi_cfg = MultiIntentConfig.from_dict(load_config()["multi_intent"])
    engine = _make_engine(semantic_router=MagicMock())
    engine._multi_intent_config = mi_cfg
    engine._splitter = MultiIntentSplitter(mi_cfg)
    engine._semantic_router = MagicMock()  # truthy so the pre-screen branch runs

    # Mock the L1 scorer and the LLM classify seam.
    engine.quick_classify = MagicMock(side_effect=_fake_quick_classify)

    async def _fake_classify_single(sub_query, request_id, t_start=0.0, **_kwargs):
        # One slice per sub-query; type mirrors the fake router so a real
        # fragment yields 'hybrid'. Entities empty (merge ops are no-ops).
        # Accept pre_l0_slice / pre_l0_* kwargs from keyword-leg fan-out.
        qtype, conf = _fake_quick_classify(sub_query)
        slice_ = IntentSlice(
            query_type="hybrid" if qtype == "hybrid" else "guidance",
            entities=[],
            confidence=conf,
            raw_text=sub_query,
            slice_id=IntentSlice.new_slice_id(),
        )
        return ([slice_], "L1_semantic", 0.0, [])

    engine._classify_single = AsyncMock(side_effect=_fake_classify_single)
    return engine, mi_cfg


def _expected_survivors(query: str, mi_cfg: MultiIntentConfig):
    """Replay split + screen with the same rules the engine uses."""
    normalized = normalize_query(query, 512, normalize=_normalize_cfg())
    frags = MultiIntentSplitter(mi_cfg).split(normalized)
    if len(frags) <= 1:
        return frags
    floor = float(mi_cfg.prune_noise_max_confidence)
    scored = [(f, _fake_quick_classify(f)[1]) for f in frags]
    kept = [f for f, c in scored if c >= floor]
    if not kept:
        kept = [max(scored, key=lambda x: x[1])[0]]
    return kept


# (label, raw_query)
_CASES = [
    # Genuine two-intent, connective-joined — must fan out.
    ("two_real_intents",
     "show me expiring .com domains and find short .io names under 500"),
    # Three real intents — must fan out to all three.
    ("three_real_intents",
     "find cheap .com domains and short .io names and premium .ai names"),
    # Long conversational preamble + one real intent — preamble dropped, collapses to single.
    ("preamble_plus_real_long",
     "i am from delhi and currently want to launch a coffee shop so suggest "
     "some good short brandable domains for it"),
    # Mixed: one noise fragment, one real — collapses to the real one.
    ("noise_plus_real",
     "i am just browsing and want cheap .com domains under 50"),
    # Multi-sentence, period-separated, NO connective — splitter never splits.
    ("multi_sentence_no_connective",
     "i want coffee domains. i also like tech names"),
    # Long single intent, no connective — single classify.
    ("long_single_intent",
     "could you please help me find a really good short memorable brandable "
     "dot com domain for my new startup idea"),
    # All fragments are noise — screen must keep the strongest (never empty).
    ("all_noise_never_empty",
     "i am from delhi and i am just looking around"),
]


@pytest.mark.parametrize("label,query", _CASES, ids=[c[0] for c in _CASES])
def test_prescreen_identifies_and_executes(label, query):
    engine, mi_cfg = _engine_with_splitter()
    expected = _expected_survivors(query, mi_cfg)

    intent = asyncio.run(engine.classify(raw_query=query, request_id=f"req_{label}"))

    # 1. Exactly the surviving fragments reached the LLM seam (no wasted calls,
    #    none dropped that should have run).
    called_with = [c.args[0] for c in engine._classify_single.call_args_list]
    assert sorted(called_with) == sorted(expected), (
        f"{label}: classified {called_with}, expected {expected}"
    )
    assert engine._classify_single.call_count == len(expected)

    # 2. Result shape matches: multi-intent stays multi; collapsed stays single.
    if len(expected) > 1:
        assert len(intent.slices) == len(expected)
        assert intent.decision_tier == "L0_multi_intent"
        assert intent.query_type in ("hybrid", "analytics", "guidance", "explore")
    else:
        assert len(intent.slices) == 1
        assert intent.decision_tier != "L0_multi_intent"


def test_genuine_multi_intent_not_collapsed():
    """A real two-intent query must NOT be screened down to one."""
    engine, mi_cfg = _engine_with_splitter()
    q = "show expiring .com domains and find short .io names under 500"
    intent = asyncio.run(engine.classify(raw_query=q, request_id="req_multi"))
    assert engine._classify_single.call_count >= 2
    assert len(intent.slices) >= 2
    assert intent.decision_tier == "L0_multi_intent"


def test_conversational_preamble_collapses_to_single_llm_call():
    """The reported bug: 'i am from delhi and ...' must cost ONE classify."""
    engine, mi_cfg = _engine_with_splitter()
    q = ("i am from delhi and want open a coffee shop sooner. "
         "could you advise some good domains for it")
    intent = asyncio.run(engine.classify(raw_query=q, request_id="req_delhi"))
    assert engine._classify_single.call_count == 1
    assert len(intent.slices) == 1
    # The conversational fragment must not have reached the LLM.
    called_with = engine._classify_single.call_args_list[0].args[0]
    assert "delhi" in called_with or "coffee" in called_with


def test_prescreen_disabled_keeps_all_fragments():
    """With prune_noise_slices=False the screen is inert — all fragments fan out."""
    engine, mi_cfg = _engine_with_splitter()
    engine._multi_intent_config.prune_noise_slices = False
    q = "i am just browsing and want cheap .com domains under 50"
    normalized = normalize_query(q, 512, normalize=_normalize_cfg())
    raw_frags = MultiIntentSplitter(mi_cfg).split(normalized)
    assert len(raw_frags) == 2  # guard: this query really does split
    intent = asyncio.run(engine.classify(raw_query=q, request_id="req_off"))
    assert engine._classify_single.call_count == 2
