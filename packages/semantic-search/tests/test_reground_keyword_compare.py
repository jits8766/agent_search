"""Keyword three-arm compare helpers in reground_filters_four_way harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_search.config.offline_harness_defaults import HARNESS_GROUNDING_MODEL_ENV
from semantic_search.offline_harness import reground_filters_four_way as harness


ARM_LLMJ = harness.ARM_LLMJ
ARM_QIE = harness.ARM_QIE
ARM_FULL = harness.ARM_FULL
ARM_REGEX = harness.ARM_REGEX
ARMS = harness.ARMS


@pytest.fixture(autouse=True)
def _stub_harness_grounding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_row_update`` stamps ``grounding_model`` via resolve; unit tests have no LLM keys."""
    monkeypatch.setenv(HARNESS_GROUNDING_MODEL_ENV, "test-harness-model")
    monkeypatch.setattr(harness, "_GROUNDING_MODEL_CACHE", None)


def _empty_sets() -> dict:
    return {a: [] for a in ARMS}


def _empty_latencies() -> dict:
    return {a: harness._latency_parts() for a in ARMS}


def test_keywords_to_set_casefolds_and_drops_probability() -> None:
    terms = harness._keywords_to_set(
        [
            {"term": "Coffee", "probability": 0.9},
            {"term": "coffee", "probability": 0.1},
            {"term": "  Tea ", "probability": 0.8},
            {"term": "", "probability": 1.0},
            "not-a-dict",
        ]
    )
    assert terms == ["coffee", "tea"]


def test_keyword_three_arm_status_ok_and_diff() -> None:
    ok = harness._keyword_three_arm_status(
        {
            ARM_LLMJ: ["a", "b"],
            ARM_QIE: ["b", "a"],
            ARM_FULL: ["a", "b"],
        }
    )
    assert ok == "OK"
    diff = harness._keyword_three_arm_status(
        {
            ARM_LLMJ: ["a"],
            ARM_QIE: ["a"],
            ARM_FULL: ["a", "b"],
        }
    )
    assert diff == "DIFF"


def test_row_update_sets_keyword_status_and_detail() -> None:
    row = {"query": "coffee shops", "query_index": 1}
    keywords = {
        ARM_LLMJ: [{"term": "coffee", "probability": 0.91}],
        ARM_QIE: [{"term": "Coffee", "probability": 0.7}],
        ARM_FULL: [{"term": "coffee", "probability": 0.85}],
        ARM_REGEX: [{"term": "shops", "probability": 0.5}],
    }
    out = harness._row_update(
        row,
        sets=_empty_sets(),
        latencies=_empty_latencies(),
        keywords=keywords,
    )
    assert out["keyword_status"] == "OK"
    assert out["status"] == "OK"
    assert out[f"{ARM_LLMJ}_kw_set"] == ["coffee"]
    assert out[f"{ARM_QIE}_kw_set"] == ["coffee"]
    assert out[f"{ARM_FULL}_kw_set"] == ["coffee"]
    detail = out[harness._arm_kw_detail_key(ARM_LLMJ)]
    assert detail[0]["term"] == "coffee"
    assert detail[0]["probability"] == pytest.approx(0.91)


def test_row_update_keyword_status_diff_when_full_differs() -> None:
    keywords = {
        ARM_LLMJ: [{"term": "a"}],
        ARM_QIE: [{"term": "a"}],
        ARM_FULL: [{"term": "b"}],
        ARM_REGEX: [],
    }
    out = harness._row_update(
        {"query": "x"},
        sets=_empty_sets(),
        latencies=_empty_latencies(),
        keywords=keywords,
    )
    assert out["keyword_status"] == "DIFF"
    assert out[f"{ARM_LLMJ}_kw_minus_{ARM_FULL}"] == ["a"]
    assert out[f"{ARM_FULL}_kw_minus_{ARM_LLMJ}"] == ["b"]


def test_pairwise_keyword_rows_and_three_arm_aggregate(tmp_path: Path) -> None:
    def _row(status: str, kw_llmj, kw_qie, kw_full) -> dict:
        return harness._row_update(
            {"query": f"{kw_llmj}-{kw_qie}-{kw_full}", "status": status},
            sets=_empty_sets(),
            latencies=_empty_latencies(),
            keywords={
                ARM_LLMJ: [{"term": t} for t in kw_llmj],
                ARM_QIE: [{"term": t} for t in kw_qie],
                ARM_FULL: [{"term": t} for t in kw_full],
                ARM_REGEX: [],
            },
        )

    rows = [
        _row("OK", ["a"], ["a"], ["a"]),
        _row("DIFF", ["a"], ["a"], ["b"]),  # filter status OK from empty sets; force DIFF label
        _row("ERROR", ["z"], ["z"], ["z"]),  # excluded from pairwise denom
    ]
    # Force first row status OK / second DIFF for filter status independently of sets.
    rows[0]["status"] = "OK"
    rows[1]["status"] = "DIFF"
    rows[2]["status"] = "ERROR"

    xlsx = harness.write_analysis(rows, tmp_path, holdout_frac=0.0)
    assert xlsx.is_file()

    pairwise_paths = sorted(tmp_path.glob("pairwise_summary_*.json"))
    assert pairwise_paths, "pairwise_summary JSON missing"
    payload = json.loads(pairwise_paths[-1].read_text())
    kw = payload["keywords"]
    assert kw["arms"] == [ARM_LLMJ, ARM_QIE, ARM_FULL]
    assert kw["three_arm_n"] == 2
    assert kw["three_arm_exact"] == 1
    assert kw["three_arm_diff"] == 1
    assert kw["three_arm_exact_pct"] == 50.0

    by_pair = {p["pair"]: p for p in kw["pairs"]}
    llmj_qie = by_pair[f"{ARM_LLMJ}={ARM_QIE}"]
    assert llmj_qie["exact"] == 2
    assert llmj_qie["diff"] == 0
    llmj_full = by_pair[f"{ARM_LLMJ}={ARM_FULL}"]
    assert llmj_full["exact"] == 1
    assert llmj_full["diff"] == 1

    kw_sheet = sorted(tmp_path.glob("keyword_overlap_*.json"))
    assert kw_sheet, "keyword_overlap sheet JSON missing"
    sheet = json.loads(kw_sheet[-1].read_text())
    assert sheet["three_arm_exact"] == 1
    assert sheet["three_arm_n"] == 2

    # analysis-only backfill: keyword_status stamped on completed rows
    results = sorted(tmp_path.glob("results_*.json"))
    stamped = json.loads(results[-1].read_text())
    assert stamped[0]["keyword_status"] == "OK"
    assert stamped[1]["keyword_status"] == "DIFF"


def test_collect_keywords_llmj_unions_chips() -> None:
    raw = [
        {"name": "keyword_contains", "value": "chip-term"},
        {"name": "tldIncludeList", "value": "com"},
    ]
    body_kw = [{"term": "body-term", "probability": 0.8}]
    merged = harness._collect_keywords_llmj(raw, body_kw)
    terms = {str(k.get("term") or "").lower() for k in merged}
    assert "chip-term" in terms
    assert "body-term" in terms


def test_collect_and_format_qie_find_wire() -> None:
    body = {
        "find_query_params": {
            "query": "coffee",
            "tldIncludeList": "com",
            "useSemanticSearch": "true",
        },
        "find_query_string": "query=coffee&tldIncludeList=com&useSemanticSearch=true",
    }
    params, qs = harness._collect_qie_find_wire(body)
    assert params["query"] == "coffee"
    assert params["tldIncludeList"] == "com"
    assert qs.startswith("query=coffee")
    disp = harness._format_find_query_params(params)
    assert "query=coffee" in disp
    assert "tldIncludeList=com" in disp
    assert "," in disp  # single column, not one col per param


def test_row_update_stores_qie_find_wire_info() -> None:
    out = harness._row_update(
        {"query": "coffee"},
        sets=_empty_sets(),
        latencies=_empty_latencies(),
        find_query_params={"query": "coffee", "tldIncludeList": "com"},
        find_query_string="query=coffee&tldIncludeList=com",
    )
    assert out["find_query_params"] == {"query": "coffee", "tldIncludeList": "com"}
    assert out["find_query_string"] == "query=coffee&tldIncludeList=com"
    # Info-only: does not alter filter/keyword agreement fields.
    assert out["status"] == "OK"
    assert out["keyword_status"] == "OK"
