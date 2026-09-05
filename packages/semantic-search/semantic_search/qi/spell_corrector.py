"""Tier-0 spell-correct and 'did you mean' surface.

Runs BEFORE the QI cascade and BEFORE the cache lookup. SymSpell
frequency-dictionary-backed lookup, offline-safe, ~1–2 ms per query.

Soft-fail contract: any internal exception is caught and produces a
"no correction" result so search never blocks on a corrector glitch.
"""
import importlib.util
import pathlib
import re
import string
from typing import Dict, List, Optional, Tuple

from symspellpy import SymSpell, Verbosity

from semantic_search.config.models import SpellCorrectConfig
from semantic_search.contracts import SpellCorrection, TokenCorrection
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Word tokenizer — matches contiguous ASCII lowercase letters. Punctuation,
# digits, domain extensions (.com) and currency tokens ($100) are not
# word-tokens so the corrector never proposes a rewrite for them.
_WORD_TOKEN = re.compile(r"[a-z]+")

# Allowed dictionary alphabet — defensive filter so noisy tokens cannot
# enter the corrector path. Matches the word tokenizer.
_DICT_ALPHABET = frozenset(string.ascii_lowercase)

_VERBOSITY_MAP: Dict[str, Verbosity] = {
    "TOP": Verbosity.TOP,
    "CLOSEST": Verbosity.CLOSEST,
    "ALL": Verbosity.ALL,
}


def _resolve_dict_path(frequency_dict_path: str) -> str:
    """Resolve 'symspellpy:<filename>' sentinel to the bundled dictionary path.

    Any other string is returned as-is (treated as an absolute or relative
    filesystem path that the caller is responsible for).
    """
    sentinel = "symspellpy:"
    if not frequency_dict_path.startswith(sentinel):
        return frequency_dict_path
    filename = frequency_dict_path[len(sentinel):]
    spec = importlib.util.find_spec("symspellpy")
    if spec is None or spec.origin is None:
        raise ValidationError("symspellpy package not found; cannot resolve bundled dictionary")
    resolved = pathlib.Path(spec.origin).parent / filename
    if not resolved.is_file():
        raise ValidationError(f"symspellpy bundled dictionary not found: {resolved}")
    return str(resolved)


class SymSpellCorrector:
    """Tier-0 spell corrector backed by a SymSpell frequency dictionary.

    :param config: SpellCorrectConfig — all parameters loaded from YAML.
    """

    def __init__(self, config: SpellCorrectConfig) -> None:
        if not isinstance(config, SpellCorrectConfig):
            raise ValidationError("SymSpellCorrector requires a SpellCorrectConfig instance")
        self._config = config
        self._verbosity = _VERBOSITY_MAP[config.verbosity.upper()]
        self._protected_tokens = frozenset(t.lower() for t in config.protected_tokens)
        self._protected_patterns = [
            re.compile(p, re.IGNORECASE) for p in config.protected_phrase_patterns
        ]
        dict_path = _resolve_dict_path(config.frequency_dict_path)
        self._sym = SymSpell(
            max_dictionary_edit_distance=config.max_edit_distance,
            prefix_length=config.prefix_length,
        )
        loaded = self._sym.load_dictionary(dict_path, term_index=0, count_index=1)
        if not loaded:
            raise ValidationError(f"SymSpell failed to load dictionary at: {dict_path}")
        logger.info("SymSpellCorrector loaded dictionary: %s", dict_path)

    def _protected_phrase_tokens(self, normalized_query: str) -> frozenset:
        tokens: set = set()
        for pattern in self._protected_patterns:
            for m in pattern.finditer(normalized_query):
                for group_val in m.groups():
                    if group_val:
                        tokens.update(_WORD_TOKEN.findall(group_val.lower()))
        return frozenset(tokens)

    def correct(self, normalized_query: str) -> Optional[SpellCorrection]:
        """Return SpellCorrection if any token was rewritten, else None.

        Soft-fail: catches all internal exceptions and returns None so
        the orchestrator always gets a usable result.
        """
        try:
            return self._correct_inner(normalized_query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SymSpellCorrector soft-fail on %r: %s", normalized_query, exc)
            return None

    def _correct_inner(self, normalized_query: str) -> Optional[SpellCorrection]:
        phrase_protected = self._protected_phrase_tokens(normalized_query)
        spans: List[Tuple[int, int, str]] = [
            (m.start(), m.end(), m.group()) for m in _WORD_TOKEN.finditer(normalized_query)
        ]

        corrections: List[TokenCorrection] = []
        correction_map: Dict[str, str] = {}

        for _start, _end, token in spans:
            if len(token) < self._config.min_token_length:
                continue
            if token in self._protected_tokens or token in phrase_protected:
                continue
            if len(corrections) >= self._config.max_tokens_to_correct:
                break
            if token in correction_map:
                continue
            suggestions = self._sym.lookup(
                token,
                self._verbosity,
                max_edit_distance=self._config.max_edit_distance,
            )
            if not suggestions:
                continue
            best = suggestions[0]
            if best.term == token:
                continue
            tc = TokenCorrection(original=token, corrected=best.term, edit_distance=best.distance)
            corrections.append(tc)
            correction_map[token] = best.term

        if not corrections:
            return None

        corrected_query = _rewrite(normalized_query, spans, correction_map)

        return SpellCorrection(
            original_query=normalized_query,
            corrected_query=corrected_query,
            corrections=corrections,
            applied=self._config.auto_apply,
        )


def _rewrite(
    normalized_query: str,
    spans: List[Tuple[int, int, str]],
    correction_map: Dict[str, str],
) -> str:
    """Reconstruct query string applying all token corrections.

    Iterates right-to-left over spans so earlier character offsets remain
    valid after each substitution.
    """
    chars = list(normalized_query)
    for start, end, token in reversed(spans):
        replacement = correction_map.get(token)
        if replacement is not None:
            chars[start:end] = list(replacement)
    return "".join(chars)
