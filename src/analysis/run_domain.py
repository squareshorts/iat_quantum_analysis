from __future__ import annotations

import argparse

from .domain_pipeline import analyze_domain, configure_logging, refresh_public_domain_aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the geometric analysis for a single IAT domain.")
    parser.add_argument("--domain", required=True, help="Domain name, for example Age.")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not refresh the aggregate cross-domain outputs after the single-domain run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    result = analyze_domain(args.domain)
    if not args.skip_aggregate:
        aggregate = refresh_public_domain_aggregate()
        result["aggregate_summary_path"] = aggregate["summary_path"]
    return result


if __name__ == "__main__":
    main()
