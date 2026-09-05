"""Master ``clickhouse.enabled`` lever — force nested CH feature flags at boot.

Downstream already gates on analytics / explore rails / price-band adapter
``enabled`` flags. This module is the single place that maps the master lever
onto those flags so call sites stay unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from semantic_search.core.logging_utils import get_logger

if TYPE_CHECKING:
    from semantic_search.config.models import AgentSearchConfig

logger = get_logger(__name__)


def apply_clickhouse_lever(config: 'AgentSearchConfig') -> 'AgentSearchConfig':
    """Return config with CH feature surfaces forced off when the master lever is False.

    When ``clickhouse.enabled`` is True, returns ``config`` unchanged.
    When False, sets:
      - ``nl_to_sql.analytics.enabled`` → False (when analytics block present)
      - ``explore.clickhouse_rails.enabled`` → False
      - ``retrieval.sql.clickhouse_adapter.enabled`` → False (when adapter present)
    """
    if config.clickhouse.enabled:
        logger.info("clickhouse_lever enabled=true")
        return config

    forced: list[str] = []
    nl = config.nl_to_sql
    if nl.analytics is not None and nl.analytics.enabled:
        nl = replace(nl, analytics=replace(nl.analytics, enabled=False))
        forced.append('nl_to_sql.analytics')

    explore = config.explore
    if explore.clickhouse_rails.enabled:
        explore = replace(
            explore,
            clickhouse_rails=replace(explore.clickhouse_rails, enabled=False),
        )
        forced.append('explore.clickhouse_rails')

    retrieval = config.retrieval
    sql = retrieval.sql
    adapter = sql.clickhouse_adapter
    if adapter is not None and adapter.enabled:
        retrieval = replace(
            retrieval,
            sql=replace(sql, clickhouse_adapter=replace(adapter, enabled=False)),
        )
        forced.append('retrieval.sql.clickhouse_adapter')

    logger.warning(
        f"clickhouse_lever enabled=false forced_off={forced or ['(already_off)']}"
    )
    return replace(config, nl_to_sql=nl, explore=explore, retrieval=retrieval)


__all__ = ['apply_clickhouse_lever']
