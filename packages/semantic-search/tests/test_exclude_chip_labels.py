"""Exclude chip labels for qie_only / multi-intent strip clarity."""
from semantic_search.qi.chip_format import format_chip_label


def test_tld_exclude_label() -> None:
    assert format_chip_label("tldExcludeList", ["ai", "io"]) == "exclude .ai,.io"


def test_type_exclude_label() -> None:
    assert format_chip_label("typeExcludeList", ["auction"]) == "exclude type=auction"


def test_keyword_exclude_label() -> None:
    assert format_chip_label("keyword_contains_exclude", ["crypto"]) == "does not contain crypto"
    assert format_chip_label("keyword_contains_exclude", "spam") == "does not contain spam"
