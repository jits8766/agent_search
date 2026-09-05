"""Structural fingerprint for QueryIntent — excludes raw wording so distinct
queries that classify to the same structured intent share one Tier-3 key.

Uses sorted slices + sorted entities + deterministic JSON so the digest is
consistent across Python versions for identical logical intents.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, List

from semantic_search.contracts import Entity, IntentSlice, QueryIntent


def _stable_dumps(value: Any) -> str:
    """Deterministic JSON serialization — sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _stable_external_value(value: Any) -> Any:
    """Normalize entity values for deterministic hashing."""
    if isinstance(value, dict):
        return {str(k): _stable_external_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_external_value(v) for v in value]
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value)


def _entity_sig(ent: Entity) -> dict:
    return {'name': ent.name, 'value': _stable_external_value(ent.value), 'chip_kind': ent.chip_kind}


def _slice_sig(slc: IntentSlice) -> dict:
    ents = sorted(slc.entities, key=lambda e: e.name)
    return {'query_type': slc.query_type, 'entities': [_entity_sig(e) for e in ents]}


def intent_plan_fingerprint(intent: QueryIntent) -> str:
    """Return sha256 hex digest of the structural intent shape.

    Omits raw_query, normalized_query, request_id, intent_record_id, reasoning_trace,
    spell-correction payloads, alternatives, per-slice raw_text, slice_id, and
    monetary calibration fields so \"same intent / different words\" collides
    by design.

    :param intent: QueryIntent - Post-classification intent
    :return: str - 64-character hex digest
    """
    slices_sig: List[dict] = sorted((_slice_sig(s) for s in intent.slices), key=lambda x: _stable_dumps(x))
    payload = {'primary_query_type': intent.query_type, 'slices': slices_sig}
    return hashlib.sha256(_stable_dumps(payload).encode('utf-8')).hexdigest()


__all__ = ['intent_plan_fingerprint']
