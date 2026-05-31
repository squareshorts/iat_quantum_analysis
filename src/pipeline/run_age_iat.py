from __future__ import annotations

import argparse

from src.analysis.domain_pipeline import analyze_domain, configure_logging, refresh_public_domain_aggregate
from src.data.download_age_iat import discover_and_download_age_iat
from src.data.prepare_age_iat import prepare_age_iat
import subprocess
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Age IAT download → prep → analysis pipeline.")
    parser.add_argument("--osf-node", default="cv7iq", help="Root OSF node to recurse from.")
    parser.add_argument("--year", default="2019", help="Age raw-data year to download, or 'all'.")
    parser.add_argument(
        "--critical-blocks",
        nargs="+",
        type=int,
        default=[3, 4, 6, 7],
        help="Critical IAT blocks to retain during preprocessing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    download_result = discover_and_download_age_iat(osf_node=args.osf_node, year=str(args.year))
    prep_result = prepare_age_iat(critical_blocks=list(args.critical_blocks))
    
    # Generate RT distribution figures
    import logging
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Generating RT distribution figures...")
    subprocess.run([sys.executable, "-m", "src.analysis.plot_rt_distributions"], check=True)
    
    analysis_result = analyze_domain("Age")
    aggregate_result = refresh_public_domain_aggregate()

    LOGGER.info("Generating manuscript outputs...")
    subprocess.run([sys.executable, "-m", "src.analysis.generate_manuscript_outputs"], check=True)

    return {
        "download": download_result,
        "prepare": prep_result,
        "analysis": analysis_result,
        "aggregate": aggregate_result,
    }


if __name__ == "__main__":
    main()
