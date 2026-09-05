"""Model pricing registry: bundled data, user overrides, and cost arithmetic."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from .exceptions import PricingError, UnknownModelError
from .usage import TokenUsage

__all__ = [
    "ModelPricing",
    "PricingRegistry",
    "CostBreakdown",
    "default_registry",
    "detect_provider",
    "normalize_model",
]

PathLike = Union[str, "os.PathLike[str]"]

_BUNDLED = Path(__file__).with_name("data") / "pricing.json"
_ENV_VAR = "TOKENTAB_PRICING_FILE"
_MILLION = 1_000_000.0

# Vendor prefixes that appear in gateway / cloud model ids but not in price lists.
_PREFIXES = (
    "anthropic.",
    "anthropic/",
    "openai/",
    "openai.",
    "google/",
    "google.",
    "gemini/",
    "models/",
    "publishers/google/models/",
    "us.",
    "eu.",
    "apac.",
    "azure/",
    "bedrock/",
    "vertex_ai/",
    "vertex/",
)

# Trailing version/date decorations: -20250219, @20250219, -2024-08-06, -v1:0, -latest.
_SUFFIX_RE = re.compile(
    r"(?:[-@_](?:\d{8}|\d{4}-\d{2}-\d{2}|latest|preview|exp|v\d+(?::\d+)?))+$"
)


def normalize_model(model: str) -> str:
    """Reduce a provider model id to the key used in the pricing table.

    ``"us.anthropic.claude-sonnet-4-5-20250929-v1:0"`` becomes
    ``"claude-sonnet-4-5"``; ``"gpt-4o-2024-08-06"`` becomes ``"gpt-4o"``.
    """
    name = model.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    while True:
        stripped = _SUFFIX_RE.sub("", name)
        if stripped == name:
            break
        name = stripped
    return name


def detect_provider(model: str) -> str:
    """Best-effort provider guess from a model id: anthropic, openai, google, or unknown."""
    name = normalize_model(model)
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(
        ("gpt", "o1", "o3", "o4", "chatgpt", "text-embedding", "davinci", "babbage")
    ):
        return "openai"
    if name.startswith(("gemini", "text-bison", "chat-bison", "gemma")):
        return "google"
    return "unknown"


@dataclasses.dataclass(frozen=True)
class ModelPricing:
    """Prices for one model, in USD per million tokens."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("input", "output", "cache_read", "cache_write"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PricingError(f"{name} price must be a number, got {value!r}")
            if value < 0:
                raise PricingError(f"{name} price must be non-negative, got {value}")
            object.__setattr__(self, name, float(value))

    @property
    def effective_cache_read(self) -> float:
        """Cache-read rate, falling back to the full input rate when unpriced."""
        return self.input if self.cache_read is None else self.cache_read

    @property
    def effective_cache_write(self) -> float:
        """Cache-write rate, falling back to the full input rate when unpriced."""
        return self.input if self.cache_write is None else self.cache_write

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], model: Optional[str] = None) -> ModelPricing:
        if not isinstance(data, Mapping):
            raise PricingError(f"pricing entry for {model!r} must be an object, got {data!r}")
        missing = [key for key in ("input", "output") if key not in data]
        if missing:
            raise PricingError(
                f"pricing entry for {model!r} is missing required key(s): {', '.join(missing)}"
            )
        return cls(
            input=data["input"],
            output=data["output"],
            cache_read=data.get("cache_read"),
            cache_write=data.get("cache_write"),
            provider=data.get("provider"),
            model=model or data.get("model"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    def cost(self, usage: TokenUsage) -> CostBreakdown:
        """Price a usage record with this model's rates."""
        return CostBreakdown(
            input_cost=usage.input_tokens * self.input / _MILLION,
            output_cost=usage.output_tokens * self.output / _MILLION,
            cache_read_cost=usage.cache_read_tokens * self.effective_cache_read / _MILLION,
            cache_write_cost=usage.cache_write_tokens * self.effective_cache_write / _MILLION,
        )


@dataclasses.dataclass(frozen=True)
class CostBreakdown:
    """Cost of one request in USD, split by what each token was billed as."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        if not isinstance(other, CostBreakdown):
            return NotImplemented
        return CostBreakdown(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
            cache_write_cost=self.cache_write_cost + other.cache_write_cost,
        )

    def to_dict(self) -> Dict[str, float]:
        data = dataclasses.asdict(self)
        data["total_cost"] = self.total
        return data


class PricingRegistry:
    """Thread-safe lookup from model id to :class:`ModelPricing`.

    Resolution order for a model id: exact match, alias, normalized id,
    normalized alias, then the longest known key that the normalized id starts
    with. That last step means a brand-new dated snapshot of a known model
    (``claude-sonnet-4-5-20991231``) still prices correctly.
    """

    def __init__(
        self,
        pricing: Optional[Mapping[str, Any]] = None,
        *,
        aliases: Optional[Mapping[str, str]] = None,
        fallback: Optional[ModelPricing] = None,
        load_bundled: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, ModelPricing] = {}
        self._aliases: Dict[str, str] = {}
        self._meta: Dict[str, Any] = {}
        self.fallback = fallback
        if load_bundled:
            self._load_mapping(_load_json(_BUNDLED), source=str(_BUNDLED))
            env_path = os.environ.get(_ENV_VAR)
            if env_path:
                self._load_mapping(_load_json(Path(env_path)), source=env_path)
        if pricing:
            self._load_mapping(pricing, source="<argument>")
        if aliases:
            with self._lock:
                self._aliases.update({k.lower(): v.lower() for k, v in aliases.items()})

    # -- construction -----------------------------------------------------

    def _load_mapping(self, data: Mapping[str, Any], *, source: str) -> None:
        if not isinstance(data, Mapping):
            raise PricingError(f"pricing data from {source} must be a JSON object")
        models = data.get("models", data)
        if not isinstance(models, Mapping):
            raise PricingError(f"'models' in {source} must be an object")
        parsed: Dict[str, ModelPricing] = {}
        for name, entry in models.items():
            if name.startswith("_"):
                continue
            parsed[name.lower()] = ModelPricing.from_dict(entry, model=name)
        alias_data = data.get("aliases") or {}
        if not isinstance(alias_data, Mapping):
            raise PricingError(f"'aliases' in {source} must be an object")
        with self._lock:
            self._models.update(parsed)
            self._aliases.update({k.lower(): str(v).lower() for k, v in alias_data.items()})
            meta = data.get("_meta")
            if isinstance(meta, Mapping):
                self._meta.update(meta)

    def load_file(self, path: PathLike) -> PricingRegistry:
        """Merge a JSON pricing file over the current table. Returns ``self``."""
        p = Path(path)
        self._load_mapping(_load_json(p), source=str(p))
        return self

    def load_dict(self, data: Mapping[str, Any]) -> PricingRegistry:
        """Merge an in-memory pricing mapping over the current table."""
        self._load_mapping(data, source="<dict>")
        return self

    def register(
        self,
        model: str,
        pricing: Optional[Union[ModelPricing, Mapping[str, Any]]] = None,
        *,
        input: Optional[float] = None,  # noqa: A002 - deliberately mirrors the JSON key
        output: Optional[float] = None,
        cache_read: Optional[float] = None,
        cache_write: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> ModelPricing:
        """Add or replace one model's pricing."""
        if pricing is None:
            if input is None or output is None:
                raise PricingError(
                    "register() needs either a pricing object or both input= and output="
                )
            entry = ModelPricing(
                input=input,
                output=output,
                cache_read=cache_read,
                cache_write=cache_write,
                provider=provider or detect_provider(model),
                model=model,
            )
        elif isinstance(pricing, ModelPricing):
            entry = dataclasses.replace(pricing, model=pricing.model or model)
        else:
            entry = ModelPricing.from_dict(pricing, model=model)
        with self._lock:
            self._models[model.lower()] = entry
        return entry

    def alias(self, alias: str, target: str) -> None:
        """Point one model id at another model's pricing."""
        with self._lock:
            self._aliases[alias.lower()] = target.lower()

    def copy(self) -> PricingRegistry:
        """Independent copy, so a caller can override prices without global effects."""
        clone = PricingRegistry(load_bundled=False, fallback=self.fallback)
        with self._lock:
            clone._models = dict(self._models)
            clone._aliases = dict(self._aliases)
            clone._meta = copy.deepcopy(self._meta)
        return clone

    # -- lookup -----------------------------------------------------------

    def _resolve(self, model: str) -> Optional[ModelPricing]:
        name = model.strip().lower()
        seen = set()
        # Follow alias chains on the raw id, then repeat on the normalized id.
        for candidate in (name, normalize_model(name)):
            current = candidate
            while current in self._aliases and current not in seen:
                seen.add(current)
                current = self._aliases[current]
            if current in self._models:
                return self._models[current]
            normalized = normalize_model(current)
            if normalized in self._models:
                return self._models[normalized]
        # Longest-prefix match handles unseen dated snapshots of known models.
        target = normalize_model(name)
        best: Optional[str] = None
        for key in self._models:
            if target.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        return self._models[best] if best else None

    def get(self, model: str, default: Optional[ModelPricing] = None) -> Optional[ModelPricing]:
        """Look up pricing, returning ``default`` (not raising) when unknown."""
        with self._lock:
            found = self._resolve(model)
        if found is not None:
            return found
        return default if default is not None else self.fallback

    def __getitem__(self, model: str) -> ModelPricing:
        found = self.get(model)
        if found is None:
            raise UnknownModelError(model)
        return found

    def __contains__(self, model: str) -> bool:
        return self.get(model) is not None

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(sorted(self._models))

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    def __bool__(self) -> bool:
        """Always true, so an empty registry is never mistaken for "no registry"."""
        return True

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata from the loaded pricing files (``as_of`` date, currency, notes)."""
        with self._lock:
            return copy.deepcopy(self._meta)

    def models(self) -> Dict[str, ModelPricing]:
        """Snapshot of every known model."""
        with self._lock:
            return dict(self._models)

    # -- costing ----------------------------------------------------------

    def breakdown(self, model: str, usage: TokenUsage) -> CostBreakdown:
        """Price a usage record. Raises :class:`UnknownModelError` if the model is unknown."""
        return self[model].cost(usage)

    def cost(self, model: str, usage: TokenUsage) -> float:
        """Total USD cost of a usage record."""
        return self.breakdown(model, usage).total

    def estimate(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Total USD cost for raw token counts."""
        return self.cost(
            model,
            TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise PricingError(f"pricing file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PricingError(f"pricing file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PricingError(f"pricing file {path} must contain a JSON object")
    return data


_default_registry: Optional[PricingRegistry] = None
_default_lock = threading.Lock()


def default_registry() -> PricingRegistry:
    """The process-wide registry used when no explicit one is passed."""
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = PricingRegistry()
    return _default_registry
