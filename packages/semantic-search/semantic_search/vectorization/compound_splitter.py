"""Dictionary-driven compound-word splitter for unspaced domain labels.

Domain registrable labels often concatenate multiple words without a
delimiter — ``techstartup``, ``cloudstack``, ``brandable``. The
TLD-aware splitter in :mod:`semantic_search.vectorization.segmenter` only
breaks on hyphens and digit boundaries, so labels like ``techstartup``
emit as a single token and the embedding stage sees ``techstartup`` as
one opaque string. This module adds a stage that produces ``["tech", "startup"]``
so semantically-similar
queries match.

This module solves that with a deterministic Viterbi pass over an
injected unigram dictionary. There is no hardcoded vocabulary — every
word and every unigram cost flows from a caller-supplied
``Mapping[str, float]`` (typically loaded from a CSV / Parquet
frequency dump). The splitter is stateless once constructed; safe to
share across threads or asyncio tasks.

Layer rules: stdlib + ``core`` + sibling vectorization primitives only.
The splitter never imports orchestration, retrieval, or QI modules.

Algorithmic notes:

- Cost model: each candidate word ``w`` carries cost
  ``-log(p(w)) + length_penalty``. The dictionary entries supply the
  raw frequencies; the splitter normalises to probabilities and applies
  the penalty per character, so two short equally-frequent words tie
  with one long equally-frequent word — the longest word wins on
  ``min_segment_length`` >= 2.
- Out-of-dictionary fallback: a single character or run that no
  dictionary word covers receives a per-character ``oov_char_cost``
  (also config-supplied). This guarantees a segmentation always
  exists; a label of pure noise yields one segment of itself.
- Determinism: ties broken by (1) fewer segments, (2) longer leading
  segment, (3) earlier in the input. No randomness; identical input +
  dictionary always yields identical output.
"""
import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitResult:
    """Split output: original, segments, total_cost, had_oov flag."""
    original: str
    segments: Tuple[str, ...]
    total_cost: float
    had_oov: bool


class CompoundWordSplitter:
    """Viterbi splitter for unspaced compound words (dictionary-driven costs).
        English (filters out single-letter noise).
    :param max_segments: int - Hard cap on the number of segments the
        splitter will emit. Labels that would split into more segments
        than this are truncated to the single-token baseline. Must be
        ``>= 1``.
    :param oov_char_cost: float - Per-character cost charged when a
        run of characters has no dictionary coverage. Must be ``> 0``;
        higher values discourage OOV segmentation. Typical: ``20.0``
        (roughly equivalent to a unigram probability of ``2e-9``).
    :param length_penalty: float - Per-character bonus subtracted from
        each in-dictionary segment's cost. Must be ``>= 0``; ``0.0``
        disables length preference. Typical: ``0.0`` for raw Viterbi,
        ``0.5`` to gently bias toward longer dictionary matches.
    :raises ValidationError: When the dictionary is empty / malformed,
        or any numeric parameter is out of its declared range.
    """

    def __init__(self, dictionary: Mapping[str, float], min_segment_length: int, max_segments: int, oov_char_cost: float, length_penalty: float):
        if dictionary is None or not isinstance(dictionary, Mapping):
            raise ValidationError("CompoundWordSplitter.dictionary must be a non-None Mapping")
        if not dictionary:
            raise ValidationError("CompoundWordSplitter.dictionary must be non-empty")
        if not isinstance(min_segment_length, int) or min_segment_length < 1:
            raise ValidationError("CompoundWordSplitter.min_segment_length must be int >= 1")
        if not isinstance(max_segments, int) or max_segments < 1:
            raise ValidationError("CompoundWordSplitter.max_segments must be int >= 1")
        if not isinstance(oov_char_cost, (int, float)) or float(oov_char_cost) <= 0.0:
            raise ValidationError("CompoundWordSplitter.oov_char_cost must be a number > 0")
        if not isinstance(length_penalty, (int, float)) or float(length_penalty) < 0.0:
            raise ValidationError("CompoundWordSplitter.length_penalty must be a number >= 0")

        normalised: dict = {}
        total_weight = 0.0
        for word, weight in dictionary.items():
            if not isinstance(word, str) or not word:
                raise ValidationError(
                    "CompoundWordSplitter.dictionary keys must be non-empty strings"
                )
            if not isinstance(weight, (int, float)):
                raise ValidationError(
                    "CompoundWordSplitter.dictionary values must be numeric weights"
                )
            w = float(weight)
            if w < 0.0:
                raise ValidationError(
                    "CompoundWordSplitter.dictionary weights must be >= 0"
                )
            if w == 0.0:
                continue
            key = word.casefold()
            if len(key) < min_segment_length:
                continue
            normalised[key] = normalised.get(key, 0.0) + w
            total_weight += w
        if not normalised or total_weight <= 0.0:
            raise ValidationError(
                "CompoundWordSplitter.dictionary must contain at least one entry "
                f"with weight > 0 and length >= min_segment_length={min_segment_length}"
            )

        # Convert weights -> -log(probability) for use as additive cost
        # in the Viterbi pass. Lower cost = more likely word. Pre-computing
        # avoids the log call in the hot path.
        self._costs: dict = {
            word: -math.log(weight / total_weight) for word, weight in normalised.items()
        }
        self._min_segment_length = min_segment_length
        self._max_segments = max_segments
        self._oov_char_cost = float(oov_char_cost)
        self._length_penalty = float(length_penalty)
        # Pre-compute the maximum word length seen so the Viterbi inner
        # loop bounds j-i below max_word_length and avoids quadratic blow-up
        # on long labels.
        self._max_word_length = max(len(w) for w in self._costs)

    @property
    def dictionary_size(self) -> int:
        """Effective dictionary size after normalisation (diagnostics)."""
        return len(self._costs)

    def split(self, label: str) -> SplitResult:
        """Segment ``label`` into dictionary-aware compound words.

        :param label: str - Lowercase ASCII alphabetic label
            (typical input from ``DomainNameSegmenter``). Empty / None /
            non-string raises ``ValidationError``.
        :return: SplitResult - Segmentation + cost diagnostics. When the
            label is itself a dictionary word OR no split improves on
            the single-token baseline, ``segments == (label,)``.
        :raises ValidationError: When ``label`` is None / empty / non-string.
        """
        if label is None:
            raise ValidationError("CompoundWordSplitter.split requires a non-None label")
        if not isinstance(label, str):
            raise ValidationError(
                f"CompoundWordSplitter.split requires a string, got {type(label).__name__}"
            )
        if not label:
            raise ValidationError("CompoundWordSplitter.split requires a non-empty label")

        normalised = label.casefold()
        n = len(normalised)

        # Edge case: label shorter than min_segment_length cannot be split
        # against the dictionary; emit as-is.
        if n < self._min_segment_length:
            return SplitResult(original=label, segments=(label,), total_cost=0.0, had_oov=True)

        # Viterbi: best[j] = (cost, prev_index, word_or_None) for the
        # optimal segmentation of normalised[:j]. word_or_None is the
        # segment that ends at j; None marks an OOV character run.
        # Empty prefix has zero cost.
        best: List[Optional[Tuple[float, int, Optional[str]]]] = [None] * (n + 1)
        best[0] = (0.0, 0, None)

        for j in range(1, n + 1):
            # Try all segments ending at j with length in [min_segment_length, max_word_length].
            # Also allow length=1 OOV character extension.
            best_cost = math.inf
            best_prev = -1
            best_word: Optional[str] = None

            # In-dictionary candidates.
            i_lower = max(0, j - self._max_word_length)
            i_upper_inclusive = j - self._min_segment_length
            for i in range(i_lower, i_upper_inclusive + 1):
                if best[i] is None:
                    continue
                candidate = normalised[i:j]
                cost = self._costs.get(candidate)
                if cost is None:
                    continue
                # Length penalty: subtract per-character bonus.
                effective_cost = cost - self._length_penalty * (j - i)
                if effective_cost < 0.0:
                    effective_cost = 0.0
                total = best[i][0] + effective_cost
                if total < best_cost:
                    best_cost = total
                    best_prev = i
                    best_word = candidate

            # OOV single-character extension (always available — guarantees
            # a path exists even when no dictionary word covers any prefix).
            if best[j - 1] is not None:
                oov_total = best[j - 1][0] + self._oov_char_cost
                if oov_total < best_cost:
                    best_cost = oov_total
                    best_prev = j - 1
                    best_word = None

            if best_cost < math.inf:
                best[j] = (best_cost, best_prev, best_word)

        if best[n] is None:
            # Should never happen because the OOV extension always
            # provides a fallback; defensive guard for malformed input.
            logger.warning(
                f"compound_splitter_no_path label={label!r} fallback=single_token"
            )
            return SplitResult(original=label, segments=(label,), total_cost=0.0, had_oov=True)

        # Backtrack to recover segments. Consecutive OOV characters merge
        # into a single segment so noise like ``zxq`` emits as one segment
        # rather than three.
        segments_rev: List[str] = []
        j = n
        had_oov = False
        oov_buffer = ""
        while j > 0:
            entry = best[j]
            if entry is None:
                break
            cost, prev, word = entry
            if word is None:
                had_oov = True
                oov_buffer = normalised[prev:j] + oov_buffer
            else:
                if oov_buffer:
                    segments_rev.append(oov_buffer)
                    oov_buffer = ""
                segments_rev.append(word)
            j = prev
        if oov_buffer:
            segments_rev.append(oov_buffer)
        segments = tuple(reversed(segments_rev))

        # Single-token guard: if the split result equals the input as one
        # segment OR exceeds max_segments, return the baseline. Both keep
        # downstream consumers from receiving over-fragmented output.
        if not segments:
            return SplitResult(original=label, segments=(label,), total_cost=0.0, had_oov=True)
        if len(segments) > self._max_segments:
            return SplitResult(
                original=label,
                segments=(label,),
                total_cost=best[n][0],
                had_oov=had_oov,
            )

        return SplitResult(
            original=label,
            segments=segments,
            total_cost=best[n][0],
            had_oov=had_oov,
        )
