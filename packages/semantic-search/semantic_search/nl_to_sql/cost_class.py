"""Cost-class classifier for analytics queries.

Maps an EXPLAIN-time estimate of bytes-scanned and/or rows-returned onto a
discrete cost class so the analytics router can mandate a verifier on
expensive plans, hard-reject oversized ones, and prefer cheaper substrates
(DuckDB+Iceberg over Athena) on the upper bands.

The classifier is intentionally tiny and side-effect-free so it can be
wired into ``LogicValidator`` without dragging extra dependencies.
"""
from dataclasses import dataclass
from typing import Optional

from semantic_search.core.exceptions import ValidationError


@dataclass(frozen=True)
class CostClassConfig:
    """Bands for the four cost classes.

    A query maps to the smallest band whose ``max_*`` ceilings it does not
    exceed. ``oversized`` is the implicit fall-through when even the
    expensive band's ceilings are exceeded.

    :param cheap_max_bytes: int - Upper byte ceiling for ``cheap`` band
    :param medium_max_bytes: int - Upper byte ceiling for ``medium`` band
    :param expensive_max_bytes: int - Upper byte ceiling for ``expensive`` band
        (anything above is ``oversized``)
    :param cheap_max_rows: int - Upper row ceiling for ``cheap`` band
    :param medium_max_rows: int - Upper row ceiling for ``medium`` band
    :param expensive_max_rows: int - Upper row ceiling for ``expensive`` band
    :param unknown_default: str - Class returned when both signals are None
        (fallback while EXPLAIN probes are warming up); must be one of the
        four COST_CLASS_LABELS values.
    """
    cheap_max_bytes: int
    medium_max_bytes: int
    expensive_max_bytes: int
    cheap_max_rows: int
    medium_max_rows: int
    expensive_max_rows: int
    unknown_default: str = 'medium'

    def __post_init__(self) -> None:
        for name, value in (
            ('cheap_max_bytes', self.cheap_max_bytes),
            ('medium_max_bytes', self.medium_max_bytes),
            ('expensive_max_bytes', self.expensive_max_bytes),
            ('cheap_max_rows', self.cheap_max_rows),
            ('medium_max_rows', self.medium_max_rows),
            ('expensive_max_rows', self.expensive_max_rows),
        ):
            if int(value) <= 0:
                raise ValidationError(f"CostClassConfig.{name} must be > 0")
        if not (self.cheap_max_bytes < self.medium_max_bytes < self.expensive_max_bytes):
            raise ValidationError(
                "CostClassConfig byte ceilings must be strictly increasing: cheap < medium < expensive"
            )
        if not (self.cheap_max_rows < self.medium_max_rows < self.expensive_max_rows):
            raise ValidationError(
                "CostClassConfig row ceilings must be strictly increasing: cheap < medium < expensive"
            )
        if self.unknown_default not in {'cheap', 'medium', 'expensive', 'oversized'}:
            raise ValidationError(
                "CostClassConfig.unknown_default must be one of cheap|medium|expensive|oversized"
            )

    @classmethod
    def from_dict(cls, raw: dict) -> 'CostClassConfig':
        if not isinstance(raw, dict):
            raise ValidationError("CostClassConfig.from_dict requires a dict")
        return cls(
            cheap_max_bytes=int(raw['cheap_max_bytes']),
            medium_max_bytes=int(raw['medium_max_bytes']),
            expensive_max_bytes=int(raw['expensive_max_bytes']),
            cheap_max_rows=int(raw['cheap_max_rows']),
            medium_max_rows=int(raw['medium_max_rows']),
            expensive_max_rows=int(raw['expensive_max_rows']),
            unknown_default=str(raw.get('unknown_default', 'medium')),
        )


class CostClassifier:
    """Map (estimated_bytes, estimated_rows) to a cost-class label.

    :param config: CostClassConfig - Band ceilings.
    """

    def __init__(self, config: CostClassConfig) -> None:
        if not isinstance(config, CostClassConfig):
            raise ValidationError("CostClassifier requires a CostClassConfig")
        self._config = config

    def classify(self, estimated_bytes: Optional[int] = None, estimated_rows: Optional[int] = None) -> str:
        """Return one of 'cheap' | 'medium' | 'expensive' | 'oversized'.

        Uses the *worst* (highest) class implied by either signal so a query
        that scans little data but produces enormous result sets still trips
        the row-based ceilings. Returns ``unknown_default`` when both
        signals are absent.
        """
        if estimated_bytes is None and estimated_rows is None:
            return self._config.unknown_default

        byte_class = self._classify_one(
            value=estimated_bytes,
            cheap=self._config.cheap_max_bytes,
            medium=self._config.medium_max_bytes,
            expensive=self._config.expensive_max_bytes,
        )
        row_class = self._classify_one(
            value=estimated_rows,
            cheap=self._config.cheap_max_rows,
            medium=self._config.medium_max_rows,
            expensive=self._config.expensive_max_rows,
        )
        return self._max_class(byte_class, row_class)

    @staticmethod
    def _classify_one(value: Optional[int], cheap: int, medium: int, expensive: int) -> Optional[str]:
        if value is None:
            return None
        v = int(value)
        if v <= cheap:
            return 'cheap'
        if v <= medium:
            return 'medium'
        if v <= expensive:
            return 'expensive'
        return 'oversized'

    @staticmethod
    def _max_class(a: Optional[str], b: Optional[str]) -> str:
        order = {'cheap': 0, 'medium': 1, 'expensive': 2, 'oversized': 3}
        if a is None and b is None:
            # Unreachable — caller already short-circuits when both are None
            return 'medium'
        if a is None:
            return b  # type: ignore[return-value]
        if b is None:
            return a
        return a if order[a] >= order[b] else b


__all__ = ['CostClassConfig', 'CostClassifier']
