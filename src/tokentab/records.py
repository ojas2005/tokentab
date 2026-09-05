"""The per-request record that every integration surface produces."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import uuid
from typing import Any, Dict, Optional

from .pricing import CostBreakdown
from .usage import TokenUsage

__all__ = ["UsageRecord", "utcnow"]


def utcnow() -> _dt.datetime:
    """Timezone-aware UTC now (``datetime.utcnow`` is deprecated and naive)."""
    return _dt.datetime.now(_dt.timezone.utc)


@dataclasses.dataclass
class UsageRecord:
    """One LLM call: what it used, what it cost, when, and under what label."""

    model: str
    usage: TokenUsage
    breakdown: CostBreakdown
    timestamp: _dt.datetime = dataclasses.field(default_factory=utcnow)
    tag: Optional[str] = None
    provider: Optional[str] = None
    estimated: bool = False
    """True when the numbers came from a pre-call estimate rather than the
    provider's usage object -- for example a streamed response that never
    reported usage."""
    duration_s: Optional[float] = None
    request_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def cost(self) -> float:
        """Total USD cost of this call."""
        return self.breakdown.total

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Flat, JSON-friendly mapping suitable for a DataFrame row."""
        data: Dict[str, Any] = {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "provider": self.provider,
            "tag": self.tag,
            "estimated": self.estimated,
            "duration_s": self.duration_s,
        }
        data.update(self.usage.to_dict())
        data["total_tokens"] = self.usage.total_tokens
        data.update(self.breakdown.to_dict())
        data["cost"] = self.cost
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data
