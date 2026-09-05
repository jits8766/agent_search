"""Tests for ``semantic_search.core.text_distance``.

Hand-calculated Damerau-Levenshtein distances and normalized similarity ratios.
"""
import pytest

from semantic_search.core.text_distance import damerau_levenshtein, similarity_ratio


@pytest.mark.parametrize(
    "a,b,max_distance,expected",
    [
        ("hello", "hello", 2, 0),
        ("cat", "bat", 2, 1),
        ("cat", "cats", 2, 1),
        ("cats", "cat", 2, 1),
        ("ab", "ba", 2, 1),
        ("expring", "expirng", 2, 1),
    ],
)
def test_damerau_levenshtein_correctness(a, b, max_distance, expected):
    """True distance when within cap."""
    assert damerau_levenshtein(a, b, max_distance) == expected


@pytest.mark.parametrize(
    "a,b,max_distance,expected",
    [
        ("kitten", "sitting", 2, 3),
        ("a", "abcdefghij", 2, 3),
    ],
)
def test_damerau_levenshtein_early_exit_sentinel(a, b, max_distance, expected):
    """Beyond cap or length short-circuit returns max_distance + 1."""
    assert damerau_levenshtein(a, b, max_distance) == expected


@pytest.mark.parametrize(
    "a,b,max_distance,expected",
    [
        ("", "abc", 3, 3),
        ("abc", "", 3, 3),
        ("", "", 2, 0),
    ],
)
def test_damerau_levenshtein_empty_strings(a, b, max_distance, expected):
    """Empty-string edge cases."""
    assert damerau_levenshtein(a, b, max_distance) == expected


@pytest.mark.parametrize(
    "a,b,max_distance,expected",
    [
        ("hello", "hello", 2, 1.0),
        ("", "abc", 2, 0.0),
        ("abc", "", 2, 0.0),
        ("rentals", "rentls", 2, 1.0 - 1.0 / 7.0),
        ("kitten", "sitting", 1, 0.0),
    ],
)
def test_similarity_ratio_values(a, b, max_distance, expected):
    """Identical, empty, in-cap typo, and beyond-cap pairs."""
    assert similarity_ratio(a, b, max_distance) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a,b,max_distance",
    [
        ("a", "b", 0),
        ("cat", "dog", 2),
        ("rentals", "rentls", 2),
        ("", "x", 1),
        ("xy", "xy", 3),
    ],
)
def test_similarity_ratio_bounded_zero_one(a, b, max_distance):
    """Every ratio lies in [0.0, 1.0]."""
    ratio = similarity_ratio(a, b, max_distance)
    assert 0.0 <= ratio <= 1.0
