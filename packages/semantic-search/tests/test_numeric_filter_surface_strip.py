"""Regression tests: operator/number/unit surface must not reach the dense leg,
and a coordinated numeric list bound to a measure unit must stay single-intent.

Covers the "under 3 or 5 chars" defect where the comparator leaked into the vector
encode text and the splitter fragmented the coordinated length list into two intents.

- numeric_filter_surfaces (`semantic_search/qi/regex_entity_extractor.py`)
- build_semantic_encode_text (`semantic_search/qi/residual_extractor.py`)
- MultiIntentSplitter.split no_split_patterns guard (`config/base.yaml`)
"""
from typing import List

from semantic_search.config.loader import load_config
from semantic_search.config.models import MultiIntentConfig
from semantic_search.contracts import Entity
from semantic_search.qi.multi_intent_splitter import MultiIntentSplitter
from semantic_search.qi.regex_entity_extractor import numeric_filter_surfaces
from semantic_search.qi.residual_extractor import build_semantic_encode_text


def _ent(name: str, value: object) -> Entity:
    return Entity(name=name, value=value, confidence=0.95, source="L0_regex", chip_kind="hard")


def _splitter() -> MultiIntentSplitter:
    cfg = MultiIntentConfig.from_dict(load_config()["multi_intent"])
    return MultiIntentSplitter(cfg)


class TestNumericFilterSurfaces:
    def test_comparator_numbers_and_unit_collected(self) -> None:
        assert numeric_filter_surfaces("under 3 or 5 chars") == {"under", "3", "5", "or", "chars"}

    def test_range_measure_collected(self) -> None:
        toks = numeric_filter_surfaces("5 to 7 letters")
        assert {"5", "7", "to", "letters"} <= toks

    def test_no_numeric_returns_empty(self) -> None:
        assert numeric_filter_surfaces("brandable coffee names") == set()


class TestBuildSemanticEncodeTextSurfaceStrip:
    def test_pure_numeric_filter_yields_empty_encode_text(self) -> None:
        # Entity value carries only the collapsed bound (5); comparator/sibling
        # number/coordinator/unit must not survive into the dense leg.
        out = build_semantic_encode_text("under 3 or 5 chars", [_ent("name_length_max", 5)], None)
        assert out == ""

    def test_concept_kept_numeric_surface_stripped(self) -> None:
        ents = [_ent("name_length_min", 5), _ent("name_length_max", 7)]
        out = build_semantic_encode_text("5 to 7 letter brandable coffee names", ents, None)
        toks = out.split()
        assert "brandable" in toks and "coffee" in toks
        for t in ("5", "7", "to", "letter"):
            assert t not in toks

    def test_tld_only_query_still_returns_verbatim(self) -> None:
        # No numeric surface -> preserve the pre-existing degenerate fallback.
        tld = Entity(name="tld", value=["io"], confidence=0.95, source="L0_entity", chip_kind="hard")
        assert build_semantic_encode_text("io", [tld], None) == "io"


class TestSplitterCoordinatedNumericList:
    def test_coordinated_length_list_not_split(self) -> None:
        assert _splitter().split("under 3 or 5 chars") == ["under 3 or 5 chars"]

    def test_range_measure_not_split(self) -> None:
        assert _splitter().split("5 to 7 letter domains") == ["5 to 7 letter domains"]

    def test_genuine_multi_intent_still_splits(self) -> None:
        frags: List[str] = _splitter().split("coffee shop or bakery domains")
        assert len(frags) == 2
