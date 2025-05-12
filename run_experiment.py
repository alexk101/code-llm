#!/usr/bin/env python3
"""
Runner script for code translation experiments.
"""

import argparse
import logging
import os
import time

from experiment import Experiment

logger = logging.getLogger("experiment_runner")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run code translation experiments")

    parser.add_argument(
        "--name",
        type=str,
        default=f"experiment_{int(time.time())}",
        help="Experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Number of problems to include in the experiment",
    )
    parser.add_argument(
        "--min-implementations",
        type=int,
        default=5,
        help="Minimum number of language implementations required for a problem",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        default="Python",
        help="Source language to translate from",
    )
    parser.add_argument(
        "--target-languages",
        type=str,
        nargs="+",
        help="Target languages to translate to (defaults to all supported languages)",
    )
    parser.add_argument(
        "--skip-pseudocode", action="store_true", help="Skip pseudocode generation step"
    )
    parser.add_argument(
        "--use-graphrag",
        action="store_true",
        help="Use GraphRAG to enhance translation with documentation context",
    )
    parser.add_argument(
        "--top-n-languages",
        type=int,
        default=20,
        help="Maximum number of top languages to include from TIOBE index",
    )
    parser.add_argument(
        "--llm-api",
        type=str,
        default="http://localhost:1234/v1/chat/completions",
        help="URL for the LLM API server (default: http://localhost:1234/v1/chat/completions)",
    )

    return parser.parse_args()


def main():
    """Main entry point for experiment runner."""
    args = parse_args()

    # Initialize experiment
    experiment = Experiment(
        experiment_name=args.name,
        output_dir=args.output_dir,
        use_top_n_languages=args.top_n_languages,
        llm_api_url=args.llm_api,
    )

    # Run full experiment (will run both regular and GraphRAG versions)
    experiment.run_full_experiment(
        num_problems=args.num_problems,
        source_language=args.source_language,
        target_languages=args.target_languages,
        use_pseudocode=not args.skip_pseudocode,
        require_all_languages=False,
        use_cleaned_code=True,
    )

    logger.info(f"Experiment '{args.name}' completed successfully")
    logger.info(f"Results saved to: {os.path.join(args.output_dir, args.name)}")
    logger.info(
        (
            "A comparison report between regular and GraphRAG versions is available in "
            "the experiment directory"
        )
    )


if __name__ == "__main__":
    main()
