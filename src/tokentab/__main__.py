"""``python -m tokentab`` / ``tokentab`` CLI: price a call from the shell."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .exceptions import UnknownModelError
from .pricing import PricingRegistry, default_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokentab", description="Inspect model pricing and estimate call costs."
    )
    parser.add_argument("--version", action="version", version=f"tokentab {__version__}")
    parser.add_argument(
        "--pricing-file", metavar="PATH", help="merge a custom pricing JSON file first"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    price = sub.add_parser("price", help="estimate the cost of a call")
    price.add_argument("model")
    price.add_argument("-i", "--input-tokens", type=int, default=0)
    price.add_argument("-o", "--output-tokens", type=int, default=0)
    price.add_argument("--cache-read-tokens", type=int, default=0)
    price.add_argument("--cache-write-tokens", type=int, default=0)
    price.add_argument("--json", action="store_true", help="emit JSON instead of text")

    listing = sub.add_parser("models", help="list models with known pricing")
    listing.add_argument("filter", nargs="?", help="substring to match")
    listing.add_argument("--json", action="store_true")

    sub.add_parser("meta", help="show pricing-data metadata (as-of date, source note)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    registry: PricingRegistry = default_registry()
    if args.pricing_file:
        registry = registry.copy().load_file(args.pricing_file)

    if args.command == "price":
        try:
            pricing = registry[args.model]
        except UnknownModelError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        cost = registry.estimate(
            args.model,
            args.input_tokens,
            args.output_tokens,
            cache_read_tokens=args.cache_read_tokens,
            cache_write_tokens=args.cache_write_tokens,
        )
        if args.json:
            print(json.dumps({"model": args.model, "resolved": pricing.model,
                              "cost": cost, "pricing": pricing.to_dict()}, indent=2))
        else:
            print(f"model     {args.model}  ->  {pricing.model} ({pricing.provider})")
            print(f"rates     ${pricing.input}/Mtok in, ${pricing.output}/Mtok out")
            print(f"cost      ${cost:.6f}")
        return 0

    if args.command == "models":
        needle = (args.filter or "").lower()
        matches = {
            name: entry.to_dict()
            for name, entry in sorted(registry.models().items())
            if needle in name
        }
        if args.json:
            print(json.dumps(matches, indent=2))
        else:
            for name, entry in matches.items():
                print(f"{name:32s} ${entry['input']:>8.4f} in  ${entry['output']:>8.4f} out")
        return 0 if matches else 1

    print(json.dumps(registry.meta, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
