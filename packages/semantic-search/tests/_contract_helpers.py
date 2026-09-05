"""Contract-test helpers (test-only, not shipped).

Drives the standard contract-validation matrix from a spec dict so that
``testing.mdc`` Category 8 (Input Contract Validation) is satisfied without
hand-writing one Python function per parameter rejection.

Every helper raises ``AssertionError`` directly so failures point at the
calling test, not the helper. Helpers do not catch the production exception —
they propagate any non-matching exception up so the test surface stays
strict.

Usage::

    from ._contract_helpers import assert_dataclass_contract

    def test_token_correction_contract():
        valid = dict(original='expring', corrected='expiring', edit_distance=1)
        assert_dataclass_contract(
            TokenCorrection,
            valid_kwargs=valid,
            type_violations={
                'original':      ('',  'original'),
                'corrected':     ('',  'corrected'),
                'edit_distance': (0,   'positive'),
            },
            equality_violations={
                ('original', 'corrected'): ('same', 'same', 'differ'),
            },
            exception_cls=ValidationError,
        )

This single call exercises 4 rejection paths in 8 lines of test source.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type

import pytest


def assert_raises_with_message( callable_, *args, exception_cls: Type[BaseException], match: str, **kwargs, ) -> BaseException:
    """Invoke ``callable_`` and assert it raises ``exception_cls`` with a
    message matching the ``match`` regex. Returns the captured exception so
    callers can inspect ``__cause__``, ``args``, etc.
    """
    with pytest.raises(exception_cls, match=match) as excinfo:
        callable_(*args, **kwargs)
    return excinfo.value


def assert_dataclass_contract(
    cls: Type,
    *,
    valid_kwargs: Mapping[str, Any],
    type_violations: Optional[Mapping[str, Tuple[Any, str]]] = None,
    missing_field_violations: Optional[Iterable[str]] = None,
    equality_violations: Optional[Mapping[Tuple[str, str], Tuple[Any, Any, str]]] = None,
    extra_violations: Optional[Mapping[str, Tuple[Mapping[str, Any], str]]] = None,
    exception_cls: Type[BaseException] = ValueError,
) -> None:
    """Drive standard contract-validation matrix for a dataclass.

    :param cls: Dataclass under test.
    :param valid_kwargs: Minimum-viable valid kwargs that must succeed.
    :param type_violations: ``{field_name: (bad_value, error_match_regex)}`` -
        for each entry, instantiate ``cls(**valid_kwargs, field=bad_value)`` and
        assert it raises ``exception_cls`` with the given match.
    :param missing_field_violations: Fields whose omission must raise
        ``exception_cls`` with the field name in the message.
    :param equality_violations: ``{(field_a, field_b): (value_a, value_b, match)}``
        - assert that setting both fields to the given values raises
        ``exception_cls`` matching the regex.
    :param extra_violations: ``{label: (kwargs_override, match)}`` - bespoke
        per-case overrides for cross-field invariants the matrix above can't
        express.
    :param exception_cls: Exception class expected for every violation case.

    First, asserts the happy path (``cls(**valid_kwargs)``) succeeds.
    """
    instance = cls(**valid_kwargs)
    assert instance is not None, f'{cls.__name__}(**valid_kwargs) returned None'

    if type_violations:
        for field, (bad_value, match) in type_violations.items():
            kwargs = dict(valid_kwargs)
            kwargs[field] = bad_value
            with pytest.raises(exception_cls, match=match):
                cls(**kwargs)

    if missing_field_violations:
        for field in missing_field_violations:
            kwargs = {k: v for k, v in valid_kwargs.items() if k != field}
            with pytest.raises(exception_cls, match=field):
                cls(**kwargs)

    if equality_violations:
        for (field_a, field_b), (val_a, val_b, match) in equality_violations.items():
            kwargs = dict(valid_kwargs)
            kwargs[field_a] = val_a
            kwargs[field_b] = val_b
            with pytest.raises(exception_cls, match=match):
                cls(**kwargs)

    if extra_violations:
        for _label, (overrides, match) in extra_violations.items():
            kwargs = dict(valid_kwargs)
            kwargs.update(overrides)
            with pytest.raises(exception_cls, match=match):
                cls(**kwargs)


def assert_frozen(instance: Any, field: str) -> None:
    """Assert that ``setattr(instance, field, ...)`` raises (i.e. the
    dataclass is frozen). Uses a sentinel value so it is never confused with
    the original.
    """
    sentinel = object()
    with pytest.raises(Exception):
        setattr(instance, field, sentinel)


def matrix_param(case_id: str, *args, marks: Sequence = ()) -> "pytest.param":
    """Wrapper around ``pytest.param`` that places ``case_id`` first so the
    coverage-matrix row reads naturally at the call site::

        matrix_param('empty_original_raises', '', 'expiring', 1, 'original')

    is equivalent to::

        pytest.param('', 'expiring', 1, 'original', id='empty_original_raises')
    """
    return pytest.param(*args, id=case_id, marks=marks)


__all__ = [
    'assert_raises_with_message',
    'assert_dataclass_contract',
    'assert_frozen',
    'matrix_param',
]
