"""Feedback + offline-eval + identity subsystem config dataclasses.

Loaded by ``semantic_search.config.models`` so ``FeedbackConfig``, ``IdentityConfig``,
and offline-eval nested types resolve without circular imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from semantic_search.config.analytics_models import _coerce_bool
from semantic_search.core.exceptions import ConfigurationError


def _sb_require(d: Dict[str, Any], keys: List[str], context: str) -> None:
    """Raise ConfigurationError if any key is missing from ``d``."""
    if not isinstance(d, dict):
        raise ConfigurationError(f"{context}: expected dict, got {type(d).__name__}")
    for key in keys:
        if key not in d:
            raise ConfigurationError(f"{context}.{key} is required")


_FEEDBACK_SEARCH_ID_MODES = frozenset({"required", "soft_generate"})


@dataclass
class IdentityConfig:
    """Trace vs durable search identity (request_id vs search_id).

    All fields are required in YAML.
    """

    request_id_prefix: str
    search_id_prefix: str
    request_id_header: str
    search_id_header: str
    accept_client_request_id: bool
    accept_client_search_id: bool
    id_hex_length: int
    max_id_chars: int
    id_value_pattern: str
    feedback_search_id_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id_prefix, str) or not self.request_id_prefix.strip():
            raise ConfigurationError("identity.request_id_prefix must be non-empty string")
        if not isinstance(self.search_id_prefix, str) or not self.search_id_prefix.strip():
            raise ConfigurationError("identity.search_id_prefix must be non-empty string")
        if self.request_id_prefix.strip() == self.search_id_prefix.strip():
            raise ConfigurationError(
                "identity.request_id_prefix and identity.search_id_prefix must differ"
            )
        if not isinstance(self.request_id_header, str) or not self.request_id_header.strip():
            raise ConfigurationError("identity.request_id_header must be non-empty string")
        if not isinstance(self.search_id_header, str) or not self.search_id_header.strip():
            raise ConfigurationError("identity.search_id_header must be non-empty string")
        if not isinstance(self.accept_client_request_id, bool):
            raise ConfigurationError("identity.accept_client_request_id must be bool")
        if not isinstance(self.accept_client_search_id, bool):
            raise ConfigurationError("identity.accept_client_search_id must be bool")
        if (
            not isinstance(self.id_hex_length, int)
            or isinstance(self.id_hex_length, bool)
            or self.id_hex_length < 1
        ):
            raise ConfigurationError("identity.id_hex_length must be int >= 1")
        if (
            not isinstance(self.max_id_chars, int)
            or isinstance(self.max_id_chars, bool)
            or self.max_id_chars < 8
        ):
            raise ConfigurationError("identity.max_id_chars must be int >= 8")
        if not isinstance(self.id_value_pattern, str) or not self.id_value_pattern.strip():
            raise ConfigurationError("identity.id_value_pattern must be non-empty string")
        try:
            re.compile(self.id_value_pattern)
        except re.error as exc:
            raise ConfigurationError(
                f"identity.id_value_pattern must be a valid regex: {exc}"
            ) from exc
        mode = str(self.feedback_search_id_mode).strip()
        if mode not in _FEEDBACK_SEARCH_ID_MODES:
            raise ConfigurationError(
                f"identity.feedback_search_id_mode must be one of {sorted(_FEEDBACK_SEARCH_ID_MODES)}"
            )
        self.feedback_search_id_mode = mode
        self.request_id_prefix = self.request_id_prefix.strip()
        self.search_id_prefix = self.search_id_prefix.strip()
        self.request_id_header = self.request_id_header.strip()
        self.search_id_header = self.search_id_header.strip()
        self.id_value_pattern = self.id_value_pattern.strip()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdentityConfig":
        _sb_require(
            d,
            [
                "request_id_prefix",
                "search_id_prefix",
                "request_id_header",
                "search_id_header",
                "accept_client_request_id",
                "accept_client_search_id",
                "id_hex_length",
                "max_id_chars",
                "id_value_pattern",
                "feedback_search_id_mode",
            ],
            "identity",
        )
        return cls(
            request_id_prefix=str(d["request_id_prefix"]),
            search_id_prefix=str(d["search_id_prefix"]),
            request_id_header=str(d["request_id_header"]),
            search_id_header=str(d["search_id_header"]),
            accept_client_request_id=_coerce_bool(d["accept_client_request_id"]),
            accept_client_search_id=_coerce_bool(d["accept_client_search_id"]),
            id_hex_length=int(d["id_hex_length"]),
            max_id_chars=int(d["max_id_chars"]),
            id_value_pattern=str(d["id_value_pattern"]),
            feedback_search_id_mode=str(d["feedback_search_id_mode"]),
        )


@dataclass
class FeedbackConfig:
    """Append-only feedback signal store config."""
    enabled: bool
    signal_log_path: str
    max_in_memory_signals: int
    allowed_signal_types: List[str]
    uat_max_comment_chars: int
    uat_min_rating: int
    uat_max_rating: int
    uat_api_key_env_var: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("feedback.enabled must be bool")
        if not isinstance(self.signal_log_path, str) or not str(self.signal_log_path).strip():
            raise ConfigurationError("feedback.signal_log_path must be non-empty string")
        if int(self.max_in_memory_signals) < 1:
            raise ConfigurationError("feedback.max_in_memory_signals must be >= 1")
        if not isinstance(self.allowed_signal_types, list) or len(self.allowed_signal_types) == 0:
            raise ConfigurationError("feedback.allowed_signal_types must be non-empty list")
        if int(self.uat_max_comment_chars) < 1:
            raise ConfigurationError("feedback.uat_max_comment_chars must be >= 1")
        if int(self.uat_min_rating) < 1:
            raise ConfigurationError("feedback.uat_min_rating must be >= 1")
        if int(self.uat_max_rating) < int(self.uat_min_rating):
            raise ConfigurationError("feedback.uat_max_rating must be >= uat_min_rating")
        if self.uat_api_key_env_var is not None and (not isinstance(self.uat_api_key_env_var, str) or not self.uat_api_key_env_var.strip()):
            raise ConfigurationError("feedback.uat_api_key_env_var must be a non-empty string when provided")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeedbackConfig':
        _sb_require(d, ['enabled', 'signal_log_path', 'max_in_memory_signals', 'allowed_signal_types', 'uat_max_comment_chars', 'uat_min_rating', 'uat_max_rating'], 'feedback')
        return cls(
            enabled=bool(d['enabled']),
            signal_log_path=str(d['signal_log_path']).strip(),
            max_in_memory_signals=int(d['max_in_memory_signals']),
            allowed_signal_types=[str(t) for t in d['allowed_signal_types']],
            uat_max_comment_chars=int(d['uat_max_comment_chars']),
            uat_min_rating=int(d['uat_min_rating']),
            uat_max_rating=int(d['uat_max_rating']),
            uat_api_key_env_var=str(d['uat_api_key_env_var']) if d.get('uat_api_key_env_var') is not None else None,
        )


@dataclass
class RetrievalEvalConfig:
    """Retrieval quality evaluator config."""
    enabled: bool
    top_k: int
    max_concurrent_queries: int
    request_id_prefix: str
    min_mean_ndcg: float
    min_mean_recall: float
    min_mean_mrr: float
    max_error_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("offline_eval.retrieval_eval.enabled must be bool")
        if int(self.top_k) < 1:
            raise ConfigurationError("offline_eval.retrieval_eval.top_k must be >= 1")
        if int(self.max_concurrent_queries) < 1:
            raise ConfigurationError("offline_eval.retrieval_eval.max_concurrent_queries must be >= 1")
        if not isinstance(self.request_id_prefix, str) or not str(self.request_id_prefix).strip():
            raise ConfigurationError("offline_eval.retrieval_eval.request_id_prefix must be non-empty string")
        for name, val in (('min_mean_ndcg', self.min_mean_ndcg), ('min_mean_recall', self.min_mean_recall), ('min_mean_mrr', self.min_mean_mrr)):
            if not 0.0 <= float(val) <= 1.0:
                raise ConfigurationError(f"offline_eval.retrieval_eval.{name} must be in [0, 1]")
        if not 0.0 <= float(self.max_error_rate) <= 1.0:
            raise ConfigurationError("offline_eval.retrieval_eval.max_error_rate must be in [0, 1]")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RetrievalEvalConfig':
        _sb_require(d, ['enabled', 'top_k', 'max_concurrent_queries', 'request_id_prefix', 'min_mean_ndcg', 'min_mean_recall', 'min_mean_mrr', 'max_error_rate'], 'offline_eval.retrieval_eval')
        return cls(
            enabled=bool(d['enabled']),
            top_k=int(d['top_k']),
            max_concurrent_queries=int(d['max_concurrent_queries']),
            request_id_prefix=str(d['request_id_prefix']).strip(),
            min_mean_ndcg=float(d['min_mean_ndcg']),
            min_mean_recall=float(d['min_mean_recall']),
            min_mean_mrr=float(d['min_mean_mrr']),
            max_error_rate=float(d['max_error_rate']),
        )


@dataclass
class LLMJudgeConfig:
    """LLM relevance judge config."""
    enabled: bool
    task_type: str
    prompt_tag: str
    system_prompt: str
    user_prompt_template: str
    min_confidence: float
    max_items_per_query: int
    max_concurrent_judgments: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("offline_eval.llm_judge.enabled must be bool")
        if self.enabled:
            if not isinstance(self.task_type, str) or not str(self.task_type).strip():
                raise ConfigurationError("offline_eval.llm_judge.task_type must be non-empty when enabled")
            if not isinstance(self.prompt_tag, str) or not str(self.prompt_tag).strip():
                raise ConfigurationError("offline_eval.llm_judge.prompt_tag must be non-empty when enabled")
            if not isinstance(self.system_prompt, str) or not str(self.system_prompt).strip():
                raise ConfigurationError("offline_eval.llm_judge.system_prompt must be non-empty when enabled")
            ut = self.user_prompt_template
            if not isinstance(ut, str) or '{' not in str(ut):
                raise ConfigurationError("offline_eval.llm_judge.user_prompt_template must be a format string when enabled")
            tpl = str(ut)
            for ph in ('{query}', '{item_id}', '{item_text}'):
                if ph not in tpl:
                    raise ConfigurationError(f"offline_eval.llm_judge.user_prompt_template missing placeholder {ph}")
        try:
            mc = float(self.min_confidence)
        except (TypeError, ValueError) as e:
            raise ConfigurationError("offline_eval.llm_judge.min_confidence must be a number") from e
        if not 0.0 <= mc <= 1.0:
            raise ConfigurationError("offline_eval.llm_judge.min_confidence must be in [0, 1]")
        if int(self.max_items_per_query) < 1:
            raise ConfigurationError("offline_eval.llm_judge.max_items_per_query must be >= 1")
        if int(self.max_concurrent_judgments) < 1:
            raise ConfigurationError("offline_eval.llm_judge.max_concurrent_judgments must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LLMJudgeConfig':
        _sb_require(d, ['enabled', 'task_type', 'prompt_tag', 'system_prompt', 'user_prompt_template', 'min_confidence', 'max_items_per_query', 'max_concurrent_judgments'], 'offline_eval.llm_judge')
        return cls(
            enabled=bool(d['enabled']),
            task_type=str(d['task_type']),
            prompt_tag=str(d['prompt_tag']),
            system_prompt=str(d['system_prompt']),
            user_prompt_template=str(d['user_prompt_template']),
            min_confidence=float(d['min_confidence']),
            max_items_per_query=int(d['max_items_per_query']),
            max_concurrent_judgments=int(d['max_concurrent_judgments']),
        )
