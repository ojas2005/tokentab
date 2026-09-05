"""Aggregation and export of usage records."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .budget import BudgetLimits
from .records import UsageRecord

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import pandas

__all__ = ["GroupSummary", "Report"]


@dataclasses.dataclass(frozen=True)
class GroupSummary:
    """Totals for one model, tag or provider."""

    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: float

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


def _summarize(key: str, records: Sequence[UsageRecord]) -> GroupSummary:
    return GroupSummary(
        key=key,
        calls=len(records),
        input_tokens=sum(r.usage.input_tokens for r in records),
        output_tokens=sum(r.usage.output_tokens for r in records),
        cache_read_tokens=sum(r.usage.cache_read_tokens for r in records),
        cache_write_tokens=sum(r.usage.cache_write_tokens for r in records),
        cost=sum(r.cost for r in records),
    )


class Report:
    """Read-only aggregate over a list of :class:`UsageRecord`."""

    def __init__(
        self,
        records: Iterable[UsageRecord],
        *,
        name: Optional[str] = None,
        limits: Optional[BudgetLimits] = None,
    ) -> None:
        self.records: List[UsageRecord] = list(records)
        self.name = name
        self.limits = limits if limits is not None else BudgetLimits()

    # -- aggregates -------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.records)

    @property
    def total_cost(self) -> float:
        return sum(record.cost for record in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(record.total_tokens for record in self.records)

    @property
    def input_tokens(self) -> int:
        return sum(record.usage.input_tokens for record in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(record.usage.output_tokens for record in self.records)

    @property
    def cache_read_tokens(self) -> int:
        return sum(record.usage.cache_read_tokens for record in self.records)

    @property
    def cache_write_tokens(self) -> int:
        return sum(record.usage.cache_write_tokens for record in self.records)

    @property
    def estimated_cost_share(self) -> float:
        """Fraction of spend that came from estimates rather than reported usage."""
        total = self.total_cost
        if total <= 0:
            return 0.0
        return sum(r.cost for r in self.records if r.estimated) / total

    def _group(self, attr: str, default: str) -> Dict[str, GroupSummary]:
        buckets: Dict[str, List[UsageRecord]] = {}
        for record in self.records:
            key = getattr(record, attr, None) or default
            buckets.setdefault(str(key), []).append(record)
        summaries = {key: _summarize(key, items) for key, items in buckets.items()}
        return dict(sorted(summaries.items(), key=lambda kv: -kv[1].cost))

    @property
    def by_model(self) -> Dict[str, GroupSummary]:
        return self._group("model", "unknown")

    @property
    def by_tag(self) -> Dict[str, GroupSummary]:
        return self._group("tag", "untagged")

    @property
    def by_provider(self) -> Dict[str, GroupSummary]:
        return self._group("provider", "unknown")

    def cost_by_model(self) -> Dict[str, float]:
        return {key: group.cost for key, group in self.by_model.items()}

    def cost_by_tag(self) -> Dict[str, float]:
        return {key: group.cost for key, group in self.by_tag.items()}

    def filter(
        self,
        *,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Report:
        """A new report narrowed to matching records."""
        selected = [
            record
            for record in self.records
            if (model is None or record.model == model)
            and (tag is None or record.tag == tag)
            and (provider is None or record.provider == provider)
        ]
        return Report(selected, name=self.name, limits=self.limits)

    # -- export -----------------------------------------------------------

    def to_dict(self, *, include_records: bool = True) -> Dict[str, Any]:
        """Plain nested dict of totals, groupings and (optionally) every record."""
        data: Dict[str, Any] = {
            "name": self.name,
            "call_count": self.call_count,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_share": self.estimated_cost_share,
            "limits": self.limits.to_dict(),
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "by_tag": {k: v.to_dict() for k, v in self.by_tag.items()},
            "by_provider": {k: v.to_dict() for k, v in self.by_provider.items()},
        }
        if self.records:
            data["started_at"] = min(r.timestamp for r in self.records).isoformat()
            data["ended_at"] = max(r.timestamp for r in self.records).isoformat()
        if include_records:
            data["records"] = self.to_records()
        return data

    def to_records(self) -> List[Dict[str, Any]]:
        """One flat dict per call."""
        return [record.to_dict() for record in self.records]

    def to_json(
        self, *, indent: Optional[int] = 2, include_records: bool = True, **kwargs: Any
    ) -> str:
        """JSON string of :meth:`to_dict`."""
        return json.dumps(
            self.to_dict(include_records=include_records), indent=indent, default=str, **kwargs
        )

    def to_dataframe(self) -> pandas.DataFrame:
        """One row per call as a pandas DataFrame.

        pandas is an optional extra: ``pip install 'tokentab[pandas]'``.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() needs pandas: pip install 'tokentab[pandas]'"
            ) from exc
        columns = [
            "request_id", "timestamp", "model", "provider", "tag", "estimated",
            "duration_s", "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "reasoning_tokens", "total_tokens", "input_cost",
            "output_cost", "cache_read_cost", "cache_write_cost", "cost",
        ]
        frame = pd.DataFrame(self.to_records(), columns=columns)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="ISO8601", utc=True)
        return frame

    def summary(self, *, currency: str = "$") -> str:
        """Human-readable text summary, for logs and CLI output."""
        title = f"tokentab report{f' [{self.name}]' if self.name else ''}"
        lines = [
            title,
            "=" * len(title),
            f"calls           {self.call_count}",
            f"total cost      {currency}{self.total_cost:,.6f}",
            f"total tokens    {self.total_tokens:,}"
            f"  (in {self.input_tokens:,} / out {self.output_tokens:,}"
            f" / cache r {self.cache_read_tokens:,} w {self.cache_write_tokens:,})",
        ]
        if self.limits.per_session is not None:
            used = self.total_cost / self.limits.per_session * 100 if self.limits.per_session else 0
            lines.append(
                f"session budget  {currency}{self.total_cost:,.6f}"
                f" / {currency}{self.limits.per_session:,.6f} ({used:.1f}%)"
            )
        for label, groups in (("by model", self.by_model), ("by tag", self.by_tag)):
            if not groups:
                continue
            lines.append("")
            lines.append(label)
            width = max(len(key) for key in groups)
            for key, group in groups.items():
                lines.append(
                    f"  {key.ljust(width)}  {currency}{group.cost:>12,.6f}"
                    f"  {group.calls:>4} calls  {group.total_tokens:>10,} tok"
                )
        return "\n".join(lines)

    def __iter__(self) -> Iterator[UsageRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __bool__(self) -> bool:
        """Always true; an empty report is still a report (see CostTracker.__bool__)."""
        return True

    def __repr__(self) -> str:
        return f"<Report calls={self.call_count} total_cost=${self.total_cost:.6f}>"
