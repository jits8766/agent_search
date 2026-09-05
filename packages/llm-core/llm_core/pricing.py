"""Pricing reference and capability inference for the LLM model registry.

This module is the single source of truth for:
  * Per-model token pricing (loaded once from the bundled `pricing.yaml`).
  * `selection_exclusions` plus optional `selection_max_output_usd_per_million`:
    priced models with output rate strictly above that cap are dropped from
    `LLMProvider` ranking (merged into `load_selection_exclusions`).
  * Capability tier inference from model name (no hardcoded service-side maps).

Why pricing lives here, not in service `base.yaml`:
    Vendors (OpenAI, Anthropic, Google) do not expose per-token rates via API.
    Every framework that needs them maintains a manual reference. Co-locating
    that reference with `llm_core` means consumer services declare zero models
    and zero pricing — they only declare the env var holding the API key.

Why capability is inferred:
    Vendors do not expose a "capability tier" either. Naming convention is the
    only signal available without a benchmark suite. Heuristics here mirror the
    common vendor convention:
        nano / mini / haiku / flash-lite      -> tier 2 (fast, cheap, smaller)
        flash / lite                          -> tier 3 (balanced fast)
        sonnet / standard / "4o"              -> tier 4 (balanced)
        opus / pro / max / o1 / o3 / gpt-5    -> tier 5 (top capability)
    Anything we cannot classify lands at tier 3 (neutral).

Both pricing entries and inferred capability are overridable per service via
`model_capability_overrides` in the service config — used only for the rare
case where the heuristic is wrong.
"""
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import yaml

from llm_core.logging_utils import get_logger, mask_path

logger = get_logger(__name__)

_PRICING_PATH = Path(__file__).resolve().parent / 'pricing.yaml'
_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None
_SELECTION_EXCLUSIONS_CACHE: Optional[frozenset[str]] = None
_PRICING_LOCK = Lock()

_SELECTION_EXCLUSIONS_YAML_KEY = 'selection_exclusions'
_MAX_OUTPUT_CAP_YAML_KEY = 'selection_max_output_usd_per_million'
_NON_MODEL_PRICE_IDS = frozenset({'default', _SELECTION_EXCLUSIONS_YAML_KEY, _MAX_OUTPUT_CAP_YAML_KEY})
_FALLBACK_PRICING: Dict[str, float] = {'input': 5.0, 'output': 15.0}


def _parse_max_output_cap(raw: Dict[str, Any]) -> Optional[float]:
    raw_cap = raw.get(_MAX_OUTPUT_CAP_YAML_KEY)
    if isinstance(raw_cap, (int, float)):
        return float(raw_cap)
    if isinstance(raw_cap, str) and raw_cap.strip():
        try:
            return float(raw_cap.strip())
        except ValueError:
            logger.warning(f"llm_pricing_invalid_max_output_cap raw={raw_cap!r}")
            return None
    return None


def load_pricing() -> Dict[str, Dict[str, float]]:
    """Return the bundled per-model pricing reference.

    Loaded once and memoised; subsequent calls return the same dict instance.
    The YAML keys `selection_exclusions` and `selection_max_output_usd_per_million`
    are not model rows — they drive `load_selection_exclusions()` and are omitted
    from this dict.
    Raises if the YAML is missing or malformed because pricing is a required
    runtime dependency for cost-aware ranking and cost analysis.

    :return: Dict[str, Dict[str, float]] - {model_id: {'input': float, 'output': float}}
    :raises FileNotFoundError: pricing.yaml not shipped with the package
    :raises ValueError: pricing.yaml exists but is empty or not a mapping
    """
    global _PRICING_CACHE, _SELECTION_EXCLUSIONS_CACHE
    with _PRICING_LOCK:
        if _PRICING_CACHE is not None:
            return _PRICING_CACHE
        if not _PRICING_PATH.exists():
            raise FileNotFoundError(f"llm_core pricing reference missing: {_PRICING_PATH}")
        with _PRICING_PATH.open('r', encoding='utf-8') as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"llm_core pricing reference is empty or malformed: {_PRICING_PATH}")
        exclusions: frozenset[str] = frozenset()
        raw_excl = raw.get(_SELECTION_EXCLUSIONS_YAML_KEY)
        if isinstance(raw_excl, list):
            exclusions = frozenset(str(x).strip() for x in raw_excl if isinstance(x, str) and len(str(x).strip()) > 0)
        normalised: Dict[str, Dict[str, float]] = {}
        for model_id, entry in raw.items():
            sid = str(model_id)
            if sid == _SELECTION_EXCLUSIONS_YAML_KEY or sid == _MAX_OUTPUT_CAP_YAML_KEY:
                continue
            if not isinstance(entry, dict):
                continue
            normalised[str(model_id)] = {'input': float(entry.get('input', 0.0)), 'output': float(entry.get('output', 0.0))}
        max_out = _parse_max_output_cap(raw)
        auto_excl: set[str] = set()
        if max_out is not None and max_out >= 0.0:
            for mid, pe in normalised.items():
                if mid in _NON_MODEL_PRICE_IDS:
                    continue
                if float(pe['output']) > max_out:
                    auto_excl.add(mid)
        combined = frozenset(set(exclusions) | auto_excl)
        _PRICING_CACHE = normalised
        _SELECTION_EXCLUSIONS_CACHE = combined
        logger.info(
            f"llm_pricing_loaded entries={len(normalised)} selection_exclusions_explicit={len(exclusions)} "
            f"selection_exclusions_output_cap={len(auto_excl)} max_output_cap={max_out} path={mask_path(_PRICING_PATH)}"
        )
        return _PRICING_CACHE


def load_selection_exclusions() -> frozenset[str]:
    """Return model ids excluded from `LLMProvider` discovery ranking and rank_candidates.

    Union of `selection_exclusions` in `pricing.yaml` and any priced model whose
    output USD/M is strictly greater than `selection_max_output_usd_per_million`
    when that key is set.

    :return: frozenset[str] - vendor model identifiers to omit from selection lists
    """
    load_pricing()
    with _PRICING_LOCK:
        assert _SELECTION_EXCLUSIONS_CACHE is not None
        return _SELECTION_EXCLUSIONS_CACHE


def compute_call_cost_usd(model: str, usage: Optional[Dict[str, Any]]) -> float:
    """Compute USD cost for a single LLM call from model name and provider usage payload.

    Looks up per-1M-token rates from the bundled pricing reference and multiplies
    by the actual token counts in `usage`. Returns 0.0 when `usage` is absent or
    contains no token counts.

    :param model: str - Vendor model identifier (e.g. 'claude-sonnet-4-6')
    :param usage: Dict[str, Any] | None - Provider usage dict with keys
        ``prompt_tokens`` (input) and ``completion_tokens`` (output)
    :return: float - Cost in USD (>= 0.0)
    """
    if not usage:
        return 0.0
    pricing = load_pricing()
    rates = pricing.get(model) or pricing.get('default') or _FALLBACK_PRICING
    pt = int(usage.get('prompt_tokens', 0) or 0)
    ct = int(usage.get('completion_tokens', 0) or 0)
    return (pt * rates['input'] + ct * rates['output']) / 1_000_000.0


def reset_pricing_cache() -> None:
    """Drop the in-memory pricing and selection-exclusion caches. Test-only."""
    global _PRICING_CACHE, _SELECTION_EXCLUSIONS_CACHE
    with _PRICING_LOCK:
        _PRICING_CACHE = None
        _SELECTION_EXCLUSIONS_CACHE = None


# Capability inference — name patterns, evaluated top-to-bottom (first match wins).
# Tier 5 (top) listed first so e.g. "opus-mini" (hypothetical) matches tier 5 not tier 2.
_CAPABILITY_RULES = [
    (5, ('opus', '-pro', '-max', '-ultra')),
    (5, ('o1', 'o3', 'o4', 'gpt-5')),
    (5, ('grok-4', 'glm-5')),
    (4, ('sonnet', 'gpt-4o', 'gpt-4.1', 'gpt-4-turbo')),
    (4, ('glm-4', 'grok-build', 'grok-code', 'grok-3', 'deepseek', '-plus')),
    (3, ('flash', '-lite', 'gpt-3.5', '-air')),
    (2, ('haiku', 'mini', 'nano')),
]

# Token substrings that downgrade a high-tier base match. Catches e.g. `gpt-5-mini`,
# `grok-4.1-fast`, `glm-4.7-flash`: family is strong but SKU is the cheap variant.
_DOWNGRADE_TOKENS = ('-mini', '-nano', '-lite', '-haiku', '-fast', '-air', 'flash')


def infer_capability_tier(model_name: str) -> int:
    """Heuristically classify a model into capability tier 1..5 from its name.

    Used by `ModelRegistry` for any model without an explicit override in
    `model_capability_overrides`. Returns 3 (neutral) when nothing matches so
    unknown models remain rankable rather than being filtered out.

    :param model_name: str - vendor model identifier (e.g. 'gpt-5.1', 'claude-opus-4-5-20251101')
    :return: int - capability tier in [1, 5]
    """
    if not model_name:
        return 3
    name = model_name.lower()
    for tier, tokens in _CAPABILITY_RULES:
        if any(tok in name for tok in tokens):
            if tier >= 4 and any(dt in name for dt in _DOWNGRADE_TOKENS):
                return 3
            return tier
    return 3


# Family keys for same-lineage recency tiebreaks only. Cross-family pairs never
# compare recency (latest grok must not jump a Claude model on version alone).
_FAMILY_PREFIX_RULES = (
    ('claude-opus', lambda n: 'claude' in n and 'opus' in n),
    ('claude-sonnet', lambda n: 'claude' in n and 'sonnet' in n),
    ('claude-haiku', lambda n: 'claude' in n and 'haiku' in n),
    ('claude', lambda n: n.startswith('claude')),
    ('grok', lambda n: n.startswith('grok')),
    ('glm', lambda n: n.startswith('glm')),
    ('deepseek', lambda n: n.startswith('deepseek')),
    ('qwen', lambda n: 'qwen' in n),
    ('gemini', lambda n: n.startswith('gemini')),
    ('openai-o', lambda n: n.startswith(('o1', 'o3', 'o4')) or bool(re.match(r'^o\d', n))),
    ('gpt', lambda n: n.startswith('gpt')),
    ('kimi', lambda n: n.startswith('kimi')),
)

_DOTTED_VERSION_RE = re.compile(r'(?:^|[^0-9])(\d{1,2}\.\d{1,3})')
_HYPHEN_VERSION_RE = re.compile(r'(?:^|-)(\d{1,2})-(\d{1,2})(?:-|$)')
_DATE_VERSION_RE = re.compile(r'(20\d{6})')


def infer_model_family(model_name: str) -> str:
    """Return product-family key used only for same-lineage recency tiebreaks.

    :param model_name: str - vendor model identifier
    :return: str - family key; unknown ids use the full name (singleton family)
    """
    if not model_name:
        return ''
    name = model_name.lower()
    for family, predicate in _FAMILY_PREFIX_RULES:
        if predicate(name):
            return family
    return name


def infer_model_recency(model_name: str) -> Tuple[float, int, int, int]:
    """Extract a sortable recency key from a model id (higher = newer).

    Dotted versions use float semantics so ``4.5 > 4.3 > 4.20``. Hyphen pairs
    cover Claude-style ``opus-4-8``. Date suffixes (YYYYMMDD) break remaining ties.

    :param model_name: str - vendor model identifier
    :return: Tuple[float, int, int, int] - (dotted_ver, major, minor, yyyymmdd)
    """
    if not model_name:
        return (0.0, 0, 0, 0)
    name = model_name.lower()
    dotted = [float(m) for m in _DOTTED_VERSION_RE.findall(name)]
    dotted_ver = max(dotted) if dotted else 0.0
    hy = [(int(a), int(b)) for a, b in _HYPHEN_VERSION_RE.findall(name)]
    major, minor = max(hy) if hy else (0, 0)
    dates = [int(d) for d in _DATE_VERSION_RE.findall(name)]
    date_key = max(dates) if dates else 0
    return (dotted_ver, major, minor, date_key)
