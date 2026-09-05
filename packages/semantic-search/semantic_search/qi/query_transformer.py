"""Two-tier query rewriter: LLM primary via LLMCallRouter, local seq2seq fallback on timeout or router absence.

Tier 1 (primary): the LLM selected by ``config.task_type`` (provider fallback chain,
cost-capped, no model name in code) rewrites the query. The call is bounded by
``config.timeout_seconds`` through ``asyncio.wait_for``.

Tier 2 (fallback): when tier 1 times out, errors, or no router is wired, a local
seq2seq model at ``config.model_path`` (AutoModelForSeq2SeqLM) runs on CPU as a
best-effort rewrite.

Both tiers share the echo guard and the token-count gate. Any failure in both tiers
returns the original query unchanged (passthrough). The rewriter never raises into
the request path.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Tuple

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from semantic_search.config.models import QueryTransformerConfig
from semantic_search.core.exceptions import ConfigurationError, LLMError
from semantic_search.core.llm_client import LLMCallRouter
from semantic_search.core.logging_utils import get_logger

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

logger = get_logger(__name__)

_LLM_TIER_ERRORS = (LLMError, asyncio.TimeoutError, TimeoutError, ConnectionError, OSError, RuntimeError, ValueError)


@dataclass
class QueryTransformResult:
    """Outcome of a query-transform step: the query text, the tier and engine that produced it, and whether it changed."""
    query: str
    original_query: str
    mode: str
    transformed: bool
    engine: str


class QueryRewriteResponse(BaseModel):
    """Structured LLM rewrite response carrying the condensed search query."""
    rewritten_query: str = Field(min_length=1, max_length=512)


class QueryTransformer:
    """Rewrite verbose queries to concise search queries via an LLM, with a local seq2seq fallback."""

    def __init__(self, config: QueryTransformerConfig, call_router: Optional[LLMCallRouter]) -> None:
        """Store config and router; load the local fallback model when its files are present."""
        if not isinstance(config, QueryTransformerConfig):
            raise ConfigurationError("QueryTransformer requires a QueryTransformerConfig instance")
        self._config = config
        self._call_router = call_router
        self._device = torch.device("cpu")
        self._signal_res: List[Pattern[str]] = []
        for _pat in config.signal_preservation_patterns:
            try:
                self._signal_res.append(re.compile(_pat, re.IGNORECASE))
            except re.error as exc:
                raise ConfigurationError(f"qi.query_transformer.signal_preservation_patterns invalid regex pattern={_pat!r} error={exc}") from exc
        self._local_tokenizer: Optional[AutoTokenizer] = None
        self._local_model: Optional[AutoModelForSeq2SeqLM] = None
        try:
            self._local_tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)  # nosec B615 - local_files_only=True blocks any Hub network fetch; no revision-pinning risk
            self._local_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, local_files_only=True, dtype=torch.float32).to(self._device).eval()  # nosec B615 - local_files_only=True blocks any Hub network fetch; no revision-pinning risk
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            logger.warning(f"query_transformer_local_fallback_unavailable model_path={config.model_path} error_type={type(exc).__name__} error={exc}")
        _has_router = 'yes' if call_router is not None else 'no'
        _has_local = 'yes' if self._local_model is not None else 'no'
        logger.info(f"query_transformer_built task_type={config.task_type} timeout_seconds={config.timeout_seconds} llm_router={_has_router} local_fallback={_has_local}")

    def needs_rewrite(self, normalized_query: str) -> bool:
        """True when rewrite is enabled and whitespace token count exceeds ``rewrite_threshold``."""
        if not self._config.rewrite_enabled:
            return False
        if not isinstance(normalized_query, str) or not normalized_query:
            return False
        return len(normalized_query.split()) > self._config.rewrite_threshold

    def accept_rewrite(self, original: str, candidate: Optional[str]) -> Optional[str]:
        """Public echo/signal accept gate used by combined rewrite+extract path."""
        return self._accept(original, candidate)

    def passthrough_result(self, query: str) -> QueryTransformResult:
        """Build a passthrough ``QueryTransformResult`` (unchanged query)."""
        return self._passthrough(query if isinstance(query, str) else '')

    async def transform(self, normalized_query: str) -> QueryTransformResult:
        """Select the rewrite tier and apply it; return passthrough on any failure (soft-fail)."""
        try:
            return await self._transform_inner(normalized_query)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — soft-fail: transform must never raise into the request path
            logger.warning(f"query_transformer_soft_fail query_len={len(normalized_query) if isinstance(normalized_query, str) else 0} error_type={type(exc).__name__} error={exc}")
            return self._passthrough(normalized_query if isinstance(normalized_query, str) else '')

    async def _transform_inner(self, normalized_query: str) -> QueryTransformResult:
        """Run the token-count gate, then tier 1 (LLM), then tier 2 (local), then passthrough."""
        if not isinstance(normalized_query, str) or not normalized_query:
            return self._passthrough(normalized_query if isinstance(normalized_query, str) else '')
        if not self.needs_rewrite(normalized_query):
            return self._passthrough(normalized_query)
        llm_candidate = await self._llm_rewrite(normalized_query)
        if llm_candidate is not None:
            accepted = self._accept(normalized_query, llm_candidate[0])
            if accepted is not None:
                logger.debug(f"query_rewrite tier=llm engine={llm_candidate[1]} original_len={len(normalized_query)} rewritten_len={len(accepted)}")
                return QueryTransformResult(query=accepted, original_query=normalized_query, mode='llm_rewrite', transformed=True, engine=llm_candidate[1])
        local_candidate = await self._local_rewrite(normalized_query)
        accepted = self._accept(normalized_query, local_candidate)
        if accepted is not None:
            logger.debug(f"query_rewrite tier=local_fallback engine={self._config.model_path} original_len={len(normalized_query)} rewritten_len={len(accepted)}")
            return QueryTransformResult(query=accepted, original_query=normalized_query, mode='local_fallback', transformed=True, engine=self._config.model_path)
        return self._passthrough(normalized_query)

    async def _llm_rewrite(self, query: str) -> Optional[Tuple[str, str]]:
        """Rewrite via the router-selected model under a hard timeout; return (query, model) or None when the tier is disabled, unrouted, timed out, or errored."""
        if not self._config.llm_tier_enabled or self._call_router is None:
            return None
        system_prompt = self._config.llm_system_prompt_template
        user_prompt = self._config.llm_user_prompt_template.format(query=query)
        try:
            parsed, metadata = await asyncio.wait_for(
                self._call_router.call_structured(
                    task_type=self._config.task_type,
                    prompt_tag=self._config.prompt_tag,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=QueryRewriteResponse,
                ),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except _LLM_TIER_ERRORS as exc:
            logger.warning(f"query_rewrite_llm_tier_failed prompt_tag={self._config.prompt_tag} timeout_seconds={self._config.timeout_seconds} error_type={type(exc).__name__} error={exc}")
            return None
        return parsed.rewritten_query.strip(), str(metadata.get('model', ''))

    async def _local_rewrite(self, query: str) -> Optional[str]:
        """Rewrite via the local seq2seq model off the event loop; return None when unavailable or on error."""
        if self._local_model is None or self._local_tokenizer is None:
            return None
        try:
            return await asyncio.to_thread(self._local_generate, query)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"query_rewrite_local_tier_failed model_path={self._config.model_path} error_type={type(exc).__name__} error={exc}")
            return None

    def _local_generate(self, query: str) -> str:
        """Greedy-decode the local seq2seq model for the condensed query (synchronous CPU inference)."""
        prompt = self._config.rewrite_prompt_template.format(query=query)
        enc = self._local_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self._config.max_tokens)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self._local_model.generate(**enc, max_new_tokens=self._config.rewrite_max_new_tokens, num_beams=1, do_sample=False)
        return self._local_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _accept(self, original: str, candidate: Optional[str]) -> Optional[str]:
        """Return the candidate when it is non-empty, not an echo, and drops no configured hard signal; else None."""
        if not candidate:
            return None
        if self._is_echo(original, candidate):
            logger.debug(f"query_rewrite_echo_rejected original_len={len(original)} rewritten_len={len(candidate)}")
            return None
        dropped = self._dropped_signal(original, candidate)
        if dropped is not None:
            logger.warning(f"query_rewrite_signal_dropped pattern={dropped!r} original_len={len(original)} rewritten_len={len(candidate)} action=reject")
            return None
        return candidate

    def _dropped_signal(self, original: str, candidate: str) -> Optional[str]:
        """Return the first configured pattern the candidate matches fewer times than the original, or None when all preserved."""
        for pat in self._signal_res:
            if len(pat.findall(candidate)) < len(pat.findall(original)):
                return pat.pattern
        return None

    def _is_echo(self, original: str, rewrite: str) -> bool:
        """Return True when the rewrite is a contiguous span of the original anchored within the configured start offset."""
        rt = rewrite.lower().split()
        ot = original.lower().split()
        if not rt or len(rt) > len(ot):
            return False
        max_start = min(self._config.rewrite_echo_max_start_offset, len(ot) - len(rt))
        last = len(rt) - 1
        for start in range(max_start + 1):
            span = ot[start:start + len(rt)]
            if span[:last] == rt[:last] and span[last].startswith(rt[last]):
                return True
        return False

    def _passthrough(self, query: str) -> QueryTransformResult:
        """Build a passthrough result that leaves the query unchanged."""
        return QueryTransformResult(query=query, original_query=query, mode='passthrough', transformed=False, engine='')
