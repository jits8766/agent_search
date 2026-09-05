"""Layer-0 sanitizer for LLM ingress and retrieved-content paths.
Unicode NFKC normalization, length cap, substring blocklist, and regex PII
patterns. Read-only on the original input; callers receive a verdict and
``masked_text`` (NFKC-normalized with PII replaced by ``[REDACTED]``).
"""
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern

from semantic_search.config.models import SanitizerConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SanitizerVerdict:
    """Sanitizer result: passed, reasons, masked_text (PII → [REDACTED])."""
    passed: bool
    reasons: List[str] = field(default_factory=list)
    masked_text: str = ''


class LayerZeroSanitizer:
    """Text-safety gate (NFKC normalisation, length cap, blocklist, PII patterns)."""

    def __init__(self, config: SanitizerConfig):
        self._config = config
        self._blocked_lower = [p.lower() for p in config.blocked_patterns]
        self._pii_patterns: List[Pattern[str]] = [re.compile(p) for p in config.pii_patterns]
        self._counter_lock = threading.Lock()
        self._total_evaluated = 0
        self._total_rejected = 0

    @property
    def applies_to_llm_ingress(self) -> bool:
        """Gate applied to LLMCallRouter.call_structured ingress."""
        return bool(self._config.applies_to_llm_ingress)

    @property
    def system_max_chars(self) -> int:
        """System prompt length cap (higher than user-query cap)."""
        return self._config.system_max_chars

    def stats(self) -> Dict[str, int]:
        """Lifetime counters (used by sanitizer-rejection-rate signal)."""
        with self._counter_lock:
            return {'total_evaluated': self._total_evaluated, 'total_rejected': self._total_rejected}

    def sanitize(self, text: str, *, max_chars: Optional[int] = None) -> SanitizerVerdict:
        """Run Layer-0 checks, return verdict + NFKC-masked text (PII redacted)."""
        if not self._config.enabled:
            return SanitizerVerdict(passed=True, reasons=[], masked_text=text if isinstance(text, str) else '')
        if text is None or not isinstance(text, str):
            return SanitizerVerdict(passed=False, reasons=['non_string_input'], masked_text='')
        normalized = unicodedata.normalize('NFKC', text) if self._config.encoding_normalize else text
        effective_max = max_chars if max_chars is not None else self._config.max_chars
        reasons: List[str] = []
        if len(normalized) == 0:
            reasons.append('empty_input')
        if len(normalized) > effective_max:
            reasons.append(f'length_exceeds_{effective_max}')
        lowered = normalized.lower()
        for pat in self._blocked_lower:
            if pat and pat in lowered:
                reasons.append('blocked_pattern_matched')
                break
        masked = normalized
        pii_hit_count = 0
        for pattern in self._pii_patterns:
            new_masked, n = pattern.subn('[REDACTED]', masked)
            masked = new_masked
            pii_hit_count += n
        if pii_hit_count:
            reasons.append('pii_detected')
        passed = len(reasons) == 0
        if not passed:
            logger.warning(f"sanitizer_rejected reasons={reasons} input_length={len(text)} pii_hits={pii_hit_count}")
        with self._counter_lock:
            self._total_evaluated += 1
            if not passed:
                self._total_rejected += 1
        return SanitizerVerdict(passed=passed, reasons=reasons, masked_text=masked)
