"""Domain-name segmenter — splits a registered domain into TLD-aware tokens.

Operates on the canonical ``label[.label...].tld`` form a domain name
takes when it reaches the ingest pipeline. Produces three artefacts per
input:

- ``registrable_label`` — the second-level label immediately to the left
  of the public-suffix TLD (e.g. ``foo`` in ``foo.com``, ``foo-bar`` in
  ``shop.foo-bar.io``). This is the primary surface the embedding stage
  vectorises.
- ``subdomain_labels`` — labels to the left of the registrable label
  (rare in the auctions corpus but preserved when present so the
  retrieval layer can opt in to subdomain matching).
- ``tokens`` — TLD-aware sequence of ASCII alphabetic runs, numeric
  runs, and the TLD itself as a single first-class token. Hyphens act as
  segment delimiters (``foo-bar`` -> ``[foo, bar]``); embedded numerals
  produce their own token (``cloud9`` -> ``[cloud, 9]``).

Unicode + IDN handling:

- Inputs are first lower-cased via ``str.casefold`` (full-Unicode lower
  case; preserves CJK, Cyrillic, etc.).
- An IDN A-label (``xn--``-prefixed) is decoded to its U-label via
  ``codecs.decode(..., 'idna')`` so the segmenter sees the human-readable
  form. Decoding failures fall back to the raw A-label so the encoder
  never crashes the indexer.
- For non-ASCII labels the segmenter emits the casefolded label as a
  single token; the embedding stage handles tokenisation downstream.
  ASCII-only labels go through the alphanumeric/hyphen splitter.
- TLDs are emitted verbatim (the U-label form). Callers that need the
  punycoded form can derive it via ``codecs.encode(tld, 'idna')``.

Stdlib-only. No public-suffix-list dependency: the segmenter treats the
final dot-separated component as the TLD. For multi-component TLDs
(``co.uk``, ``com.br``) callers can pass the full TLD via
``known_compound_tlds`` to override the trailing-component default.
"""
import codecs
import re
import unicodedata
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.vectorization.compound_splitter import CompoundWordSplitter

logger = get_logger(__name__)

# Matches contiguous runs of ASCII letters OR ASCII digits. The two run
# types are kept separate so ``cloud9`` segments cleanly into ``cloud``
# (alpha) and ``9`` (numeric). Hyphens, dots, and any other punctuation
# act as delimiters by virtue of NOT being matched.
_ASCII_ALPHA_RUN = re.compile(r"[a-z]+")
_ASCII_DIGIT_RUN = re.compile(r"[0-9]+")
_ASCII_ALNUM_SEGMENT = re.compile(r"[a-z]+|[0-9]+")

# Matches a label that contains *only* ASCII letters/digits/hyphens —
# the safe subset where the alphanumeric segmenter is meaningful.
_ASCII_LABEL = re.compile(r"^[a-z0-9-]+$")

# IDN A-label prefix per RFC 5891 (lowercased; matches after casefold).
_IDN_PREFIX = "xn--"


@dataclass(frozen=True)
class SegmentedDomain:
    """Output of :meth:`DomainNameSegmenter.segment`.

    All string fields are casefolded U-label form (post-IDN decode).
    ``tokens`` is the TLD-aware token stream the embedding + BM25 stages
    consume. ``original`` carries the input verbatim for traceability.

    :param original: str - The input domain string verbatim
    :param registrable_label: str - Second-level label (left of TLD)
    :param subdomain_labels: Tuple[str, ...] - Labels left of the
        registrable label (empty tuple when there are none)
    :param tld: str - TLD as a U-label (decoded from punycode when input
        was an A-label; never empty)
    :param tokens: Tuple[str, ...] - Ordered token stream: subdomain
        segments, then registrable-label segments, then the TLD
    :param had_idn_decode_failure: bool - True iff IDN decoding raised
        and the segmenter fell back to the raw A-label form
    """

    original: str
    registrable_label: str
    subdomain_labels: Tuple[str, ...]
    tld: str
    tokens: Tuple[str, ...]
    had_idn_decode_failure: bool = False


class DomainNameSegmenter:
    """TLD-aware splitter for registered domain names.

    :param known_compound_tlds: Optional[FrozenSet[str]] - Multi-component
        TLDs to recognise (e.g. ``frozenset({"co.uk", "com.br"})``).
        When None or empty, the segmenter treats the trailing
        dot-separated component as the TLD.
    :param compound_splitter: Optional[CompoundWordSplitter] - When
        provided, ASCII-alpha runs that resolve to a single token via
        the alphanumeric segmenter are passed through the splitter so
        unspaced compounds like ``techstartup`` emit
        ``[tech, startup]``. ``None`` (default) preserves the legacy
        single-token behaviour for ASCII-alpha runs — required for
        callers that ship without a unigram dictionary. The splitter
        is NEVER applied to digit runs, hyphenated multi-segment
        labels (those already split into components), non-ASCII labels
        (no Latin-script unigram coverage), or the TLD itself.
    """

    def __init__(self, known_compound_tlds: Optional[FrozenSet[str]] = None, compound_splitter: Optional[CompoundWordSplitter] = None):
        if known_compound_tlds is None:
            self._compound_tlds: FrozenSet[str] = frozenset()
        else:
            if not isinstance(known_compound_tlds, (frozenset, set, list, tuple)):
                raise ValidationError(
                    "DomainNameSegmenter.known_compound_tlds must be a frozenset/set/list/tuple"
                )
            normalised = []
            for tld in known_compound_tlds:
                if not isinstance(tld, str) or not tld:
                    raise ValidationError(
                        "DomainNameSegmenter.known_compound_tlds entries must be non-empty strings"
                    )
                normalised.append(tld.casefold().lstrip("."))
            self._compound_tlds = frozenset(normalised)
        if compound_splitter is not None and not isinstance(compound_splitter, CompoundWordSplitter):
            raise ValidationError(
                "DomainNameSegmenter.compound_splitter must be a CompoundWordSplitter or None"
            )
        self._compound_splitter = compound_splitter

    @property
    def compound_tld_count(self) -> int:
        """Count of multi-component TLDs the segmenter recognises (diagnostics)."""
        return len(self._compound_tlds)

    @property
    def has_compound_splitter(self) -> bool:
        """True iff a compound-word splitter is wired (diagnostics)."""
        return self._compound_splitter is not None

    def _maybe_split_alpha_run(self, run: str) -> List[str]:
        """Apply the optional compound splitter to a single ASCII-alpha run.

        Splitter applies only when (a) injected, (b) the run is at least
        two characters, and (c) the run is pure ASCII-alpha (already
        guaranteed by the caller — ``_segment_ascii_label`` only emits
        ``[a-z]+`` runs through this path). Returns the original run as a
        single-element list when the splitter is absent OR when the
        splitter declines to split (returns its input as one segment).
        """
        if self._compound_splitter is None or len(run) < 2:
            return [run]
        result = self._compound_splitter.split(run)
        if len(result.segments) <= 1:
            return [run]
        return list(result.segments)

    @staticmethod
    def _decode_idn_label(label: str) -> Tuple[str, bool]:
        """Decode an IDN A-label to U-label form. Returns (label, decode_failed).

        Non-A-label inputs pass through unchanged with ``decode_failed=False``.
        Decode failures (malformed punycode, codec errors) fall back to the
        raw input with ``decode_failed=True`` and a WARNING log so operators
        notice corpus quality issues without losing the document.
        """
        if not label.startswith(_IDN_PREFIX):
            return label, False
        try:
            decoded = codecs.decode(label.encode("ascii"), "idna").casefold()
            return decoded, False
        except (UnicodeError, UnicodeDecodeError, ValueError) as e:
            logger.warning(
                f"domain_segmenter_idn_decode_failed label={label!r} error={str(e)}"
            )
            return label, True

    def _split_tld(self, casefolded: str) -> Tuple[List[str], str]:
        """Return (left_labels, tld) using compound-TLD recognition.

        Compound-TLD match is greedy: the longest matching trailing
        suffix wins. When no compound TLD matches, the trailing
        single-label is the TLD.
        """
        labels = casefolded.split(".")
        if len(labels) < 2:
            raise ValidationError(
                f"DomainNameSegmenter requires at least one dot (label.tld); got {casefolded!r}"
            )
        if self._compound_tlds:
            for span in (3, 2):
                if len(labels) >= span + 1:
                    candidate = ".".join(labels[-span:])
                    if candidate in self._compound_tlds:
                        return labels[:-span], candidate
        return labels[:-1], labels[-1]

    def _segment_ascii_label(self, label: str) -> List[str]:
        """Split an ASCII label into alpha runs and numeric runs.

        Hyphens (``foo-bar``) act as natural delimiters because the
        regex matches only ``[a-z]+`` or ``[0-9]+``. The segmenter
        preserves the order of segments as they appear in the label.
        Each alpha run is then optionally passed through the compound
        word splitter when one is wired; digit runs pass through
        unchanged.
        """
        out: List[str] = []
        for match in _ASCII_ALNUM_SEGMENT.finditer(label):
            run = match.group(0)
            if run and run[0].isalpha():
                out.extend(self._maybe_split_alpha_run(run))
            else:
                out.append(run)
        return out

    def _segment_label(self, label: str) -> List[str]:
        """Split a single label (post-IDN-decode, casefolded).

        ASCII labels go through the alphanumeric/hyphen segmenter.
        Non-ASCII labels (CJK, Cyrillic, etc.) are emitted as a single
        token because full Unicode word segmentation is out of scope
        for the runtime segmenter; the embedding stage handles those.
        """
        if not label:
            return []
        if _ASCII_LABEL.match(label):
            return self._segment_ascii_label(label)
        # Strip combining marks (NFKD) so accented forms collapse to their
        # base letters before being emitted as a single token.
        normalised = unicodedata.normalize("NFKD", label)
        normalised = "".join(c for c in normalised if not unicodedata.combining(c))
        return [normalised]

    def segment(self, domain: str) -> SegmentedDomain:
        """Segment a registered domain name into TLD-aware tokens.

        :param domain: str - Registered domain name (``label[.label...].tld``)
        :return: SegmentedDomain - Structured split with token stream
        :raises ValidationError: When ``domain`` is None / empty / lacks a TLD
        """
        if domain is None:
            raise ValidationError("DomainNameSegmenter.segment requires a non-None domain")
        if not isinstance(domain, str):
            raise ValidationError(
                f"DomainNameSegmenter.segment requires a string, got {type(domain).__name__}"
            )
        stripped = domain.strip().rstrip(".")
        if not stripped:
            raise ValidationError("DomainNameSegmenter.segment requires a non-empty domain")
        casefolded = stripped.casefold()
        left_labels, tld_raw = self._split_tld(casefolded)
        if not tld_raw:
            raise ValidationError(f"DomainNameSegmenter could not extract TLD from {domain!r}")

        # Decode every label individually so a subdomain in punycode form
        # also surfaces in U-label form. TLD decoding failures bubble up
        # via ``had_idn_decode_failure`` so the indexer can record them.
        idn_failed = False
        decoded_left: List[str] = []
        for label in left_labels:
            decoded, failed = self._decode_idn_label(label)
            decoded_left.append(decoded)
            idn_failed = idn_failed or failed
        # For compound TLDs the components are decoded individually then re-joined.
        tld_components = tld_raw.split(".")
        decoded_tld_parts: List[str] = []
        for part in tld_components:
            decoded, failed = self._decode_idn_label(part)
            decoded_tld_parts.append(decoded)
            idn_failed = idn_failed or failed
        tld = ".".join(decoded_tld_parts)

        if not decoded_left:
            raise ValidationError(
                f"DomainNameSegmenter requires a registrable label left of the TLD; got {domain!r}"
            )
        registrable_label = decoded_left[-1]
        subdomain_labels = tuple(decoded_left[:-1])

        # Build the token stream: subdomains (in order), registrable-label
        # segments (in order), then the TLD as a single token. Subdomain
        # labels themselves are segmented so ``shop.foo.bar.com`` produces
        # tokens for every sub-component.
        token_stream: List[str] = []
        for sub_label in subdomain_labels:
            token_stream.extend(self._segment_label(sub_label))
        token_stream.extend(self._segment_label(registrable_label))
        if tld:
            token_stream.append(tld)

        return SegmentedDomain(
            original=domain,
            registrable_label=registrable_label,
            subdomain_labels=subdomain_labels,
            tld=tld,
            tokens=tuple(token_stream),
            had_idn_decode_failure=idn_failed,
        )
