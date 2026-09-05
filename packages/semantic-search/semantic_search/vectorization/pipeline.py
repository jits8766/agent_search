"""Compose-once doc-vectorization pipeline for the offline indexer.

Wires the three stateless primitives that produce a document's sparse + dense
vector pair for Qdrant ingest:

  1. ``DomainNameSegmenter.segment(domain)`` — TLD-aware token stream
  2. (optional) ``SynonymExpander.expand(tokens)`` — bidirectional synonym
     expansion that mirrors what runs on the query side; weights are
     dropped here because the doc-side BM25 encoder already handles term
     frequency saturation, so synonyms enter the bag exactly once
  3. ``BM25DocEncoder.encode(tokens)`` — BM25 doc-side TF saturation +
     length normalisation, yielding sorted ``(indices, values)`` lists
     that match the query-side hashing scheme bucket-for-bucket

The pipeline is intentionally pure compute: no I/O, no Qdrant import, no
async surface. The offline indexer composes it with a Qdrant client and a
dense encoder; ad-hoc backfills can use it standalone (e.g. when emitting
Parquet sparse vectors).

Layer rules: imports stdlib + ``core`` + ``retrieval.synonym_expander`` +
sibling vectorization primitives only. Per ``architecture.mdc``, the
vectorization layer never imports orchestration code (registry,
orchestrator, surface).
"""
from typing import List, Optional, Sequence, Tuple, Union

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.bm42_sparse_encoder import BM42SparseEncoder
from semantic_search.retrieval.synonym_expander import SynonymExpander
from semantic_search.vectorization.bm25_doc_encoder import BM25DocEncoder
from semantic_search.vectorization.segmenter import DomainNameSegmenter, SegmentedDomain

logger = get_logger(__name__)


class DocVectorizationPipeline:
    """Per-document segment -> expand -> BM25-encode pipeline.

    Stateless across calls — the three injected components hold all state
    (vocab size, synonym map, BM25 parameters). One pipeline instance is
    safe to share across threads or asyncio tasks.

    :param segmenter: DomainNameSegmenter - Required.
    :param doc_encoder: BM25DocEncoder | BM42SparseEncoder - Required. Must
        use the same encoder as the runtime query-side sparse encoder or
        sparse retrieval silently misses matches.
    :param expander: Optional[SynonymExpander] - Optional. ``None``
        disables synonym expansion entirely (zero overhead). When the
        expander's own ``enabled=False`` flag is set, the pipeline still
        calls ``expand`` but the expander short-circuits and returns the
        originals only — that is the contract documented in
        ``SynonymExpander.expand``.
    :raises ValidationError: When ``segmenter`` or ``doc_encoder`` is None
        or wrong-typed.
    """

    def __init__(self, segmenter: DomainNameSegmenter, doc_encoder: "Union[BM25DocEncoder, BM42SparseEncoder]", expander: Optional[SynonymExpander] = None):
        if segmenter is None or not isinstance(segmenter, DomainNameSegmenter):
            raise ValidationError("DocVectorizationPipeline requires a DomainNameSegmenter")
        if doc_encoder is None or not isinstance(doc_encoder, (BM25DocEncoder, BM42SparseEncoder)):
            raise ValidationError(
                "DocVectorizationPipeline requires a BM25DocEncoder or BM42SparseEncoder"
            )
        if expander is not None and not isinstance(expander, SynonymExpander):
            raise ValidationError(
                "DocVectorizationPipeline.expander must be a SynonymExpander or None"
            )
        self._segmenter = segmenter
        self._doc_encoder = doc_encoder
        self._expander = expander

    @property
    def has_expander(self) -> bool:
        """True iff a synonym expander is wired (independent of its enabled flag)."""
        return self._expander is not None

    def segment(self, domain: str) -> SegmentedDomain:
        """Run only the segmentation stage (test + diagnostics surface)."""
        return self._segmenter.segment(domain)

    def tokens_for(self, domain: str) -> List[str]:
        """Produce the post-expansion token sequence for ``domain``.

        Useful for the offline indexer's debug + golden-set inspection
        paths without committing to a sparse-vector representation.

        :param domain: str - Raw domain name (``foo.com`` / ``xn--fiqs8s``)
        :return: List[str] - Lowercased tokens after segment + (optional) expand
        :raises ValidationError: When ``domain`` is None
        """
        if domain is None:
            raise ValidationError("DocVectorizationPipeline.tokens_for requires a non-None domain")
        segmented = self._segmenter.segment(domain)
        tokens: Sequence[str] = segmented.tokens
        if self._expander is None:
            return list(tokens)
        # The expander emits ``(token, weight)`` pairs; on the doc side we
        # discard the weight because BM25 doc-side TF saturation already
        # handles repetition, and we want the same surface form to enter
        # the doc bag exactly once whether or not it was a synonym. The
        # expander's bidirectional + dedupe contract ensures we don't
        # double-count.
        expanded_pairs = self._expander.expand(list(tokens))
        return [tok for tok, _weight in expanded_pairs]

    def dense_text_for(self, domain: str) -> str:
        """Produce the dense-encoder input text for ``domain``.
        Joins the segmented token stream into a space-separated phrase so the
        dense encoder embeds real words (``techstartup.com`` -> ``tech startup com``)
        instead of the raw concatenated label. Synonyms are NOT applied: dense
        embeddings stay free of the sparse-leg synonym terms. Falls back to the
        raw ``domain`` when segmentation yields no tokens.
        :param domain: str - Raw domain name
        :return: str - Space-joined segmented tokens, or the raw domain on empty
        :raises ValidationError: When ``domain`` is None
        """
        if domain is None:
            raise ValidationError("DocVectorizationPipeline.dense_text_for requires a non-None domain")
        segmented = self._segmenter.segment(domain)
        tokens = [tok for tok in segmented.tokens if tok]
        if not tokens:
            return domain
        return " ".join(tokens)

    def encode(self, domain: str) -> Tuple[List[int], List[float]]:
        """Produce the sparse ``(indices, values)`` for ``domain``.

        Returns an empty pair when the segmentation yields no tokens
        (e.g. an all-numeric label past the encoder's filters). The
        offline indexer treats the empty pair as "no sparse leg" — the
        dense vector still indexes; the document is simply not retrievable
        via the BM25 leg.

        :param domain: str - Raw domain name
        :return: Tuple[List[int], List[float]] - Sorted-ascending parallel lists
        :raises ValidationError: When ``domain`` is None
        """
        tokens = self.tokens_for(domain)
        return self._doc_encoder.encode(tokens)

    def encode_batch(
        self, domains: Sequence[str], *, sparse_embed_batch_size: int
    ) -> List[Tuple[List[int], List[float]]]:
        """Produce sparse ``(indices, values)`` for each domain in ``domains``.

        BM42 path batches FastEmbed ``embed`` at ``sparse_embed_batch_size``.
        Hash-BM25 path encodes sequentially (no model batch API).

        :param domains: Sequence[str] - Raw domain names (order preserved)
        :param sparse_embed_batch_size: int - BM42 embed micro-batch width (>= 1)
        :return: List[Tuple[List[int], List[float]]] - One pair per domain
        :raises ValidationError: When ``domains`` is None or batch size invalid
        """
        if domains is None:
            raise ValidationError("DocVectorizationPipeline.encode_batch requires a non-None domains sequence")
        if (
            not isinstance(sparse_embed_batch_size, int)
            or isinstance(sparse_embed_batch_size, bool)
            or sparse_embed_batch_size < 1
        ):
            raise ValidationError(
                "DocVectorizationPipeline.encode_batch sparse_embed_batch_size must be int >= 1"
            )
        materialised = list(domains)
        if not materialised:
            return []
        if isinstance(self._doc_encoder, BM42SparseEncoder):
            texts: List[str] = []
            for domain in materialised:
                tokens = self.tokens_for(domain)
                texts.append(" ".join(t for t in tokens if t and str(t).strip()))
            return self._doc_encoder.encode_texts(texts, batch_size=sparse_embed_batch_size)
        return [self.encode(domain) for domain in materialised]
