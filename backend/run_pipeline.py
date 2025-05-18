#!/usr/bin/env python3
"""
TrialGPT+MCP Pipeline - Run the full pipeline from end to end

This script will:
1. Start the MCP server (if --start-server is specified)
2. Run keyword generation for TrialGPT-Retrieval
3. Run hybrid fusion retrieval to get the most relevant trials
4. Run TrialGPT-Matching for detailed criterion-by-criterion analysis
5. Run TrialGPT-Ranking to aggregate and rank the clinical trials
"""

import argparse
import os
import sys
import subprocess
import time
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete TrialGPT+MCP pipeline"
    )
    parser.add_argument(
        "--corpus", default="sigir", help="Corpus to use (sigir, trec_2021, trec_2022)"
    )
    parser.add_argument(
        "--model", default="claude-3-7-sonnet-latest", help="Claude model to use"
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the MCP server before running the pipeline",
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8080/sse", help="MCP server URL"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Parameter k for retrieval score calculation"
    )
    parser.add_argument(
        "--bm25-weight", type=int, default=1, help="Weight for BM25 retriever"
    )
    parser.add_argument(
        "--medcpt-weight", type=int, default=1, help="Weight for MedCPT retriever"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top trials to display per patient",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval steps (use existing results)",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip matching step (use existing results)",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip ranking step (use existing results)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Custom results directory (default: ./results)",
    )

    args = parser.parse_args()

    # Setup paths
    backend_dir = Path(__file__).resolve().parent
    dataset_dir = backend_dir / "dataset"
    retrieval_dir = backend_dir / "trial_mcp_retrieval"
    matching_dir = backend_dir / "trial_mcp_matching"
    ranking_dir = backend_dir / "trial_mcp_ranking"

    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = backend_dir / "results"

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Validate corpus
    if args.corpus not in ["sigir", "trec_2021", "trec_2022"]:
        print(
            f"Error: Invalid corpus '{args.corpus}'. Must be one of: sigir, trec_2021, trec_2022"
        )
        sys.exit(1)

    # Define sample suffix
    sample_suffix = f"_sample{args.sample_size}" if args.sample_size else ""

    # Step 1: Start the MCP server if requested
    server_process = None
    if args.start_server:
        print("\n\n==== Starting MCP Server ====")
        server_process = subprocess.Popen(
            [sys.executable, os.path.join(backend_dir, "main.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(5)  # Give the server some time to start

    try:
        # Step 2: Run keyword generation for TrialGPT-Retrieval
        if not args.skip_retrieval:
            print("\n\n==== Running Keyword Generation ====")
            keyword_cmd = [
                sys.executable,
                os.path.join(retrieval_dir, "keyword_generation.py"),
                args.corpus,
                "--model_name",
                args.model,
                "--mcp_server_url",
                args.server_url,
            ]
            if args.sample_size:
                keyword_cmd.extend(["--sample_size", str(args.sample_size)])

            subprocess.run(keyword_cmd, check=True)

            # Step 3: Run hybrid fusion retrieval
            print("\n\n==== Running Hybrid Fusion Retrieval ====")

            # Construct keyword file path
            keyword_file = (
                results_dir
                / f"retrieval_keywords_mcp_{args.model}_{args.corpus}{sample_suffix}.json"
            )

            # Check if keyword file exists
            if not keyword_file.exists():
                print(f"Error: Keyword file not found at {keyword_file}")
                sys.exit(1)

            retrieval_cmd = [
                sys.executable,
                os.path.join(retrieval_dir, "hybrid_fusion_retrieval.py"),
                args.corpus,
                str(keyword_file),
                str(args.k),
                str(args.bm25_weight),
                str(args.medcpt_weight),
                "--model_name",
                args.model,
            ]

            subprocess.run(retrieval_cmd, check=True)

        # Step 4: Run TrialGPT-Matching
        if not args.skip_matching:
            print("\n\n==== Running TrialGPT-Matching ====")

            # Construct retrieved NCTIDs file path
            retrieved_file = (
                results_dir
                / f"retrieved_nctids_{args.model}_{args.corpus}{sample_suffix}.json"
            )

            # Check if retrieved file exists
            if not retrieved_file.exists():
                # Try with cached file
                cached_file = dataset_dir / args.corpus / "retrieved_trials.json"
                if cached_file.exists():
                    print(f"Retrieved NCTIDs file not found at {retrieved_file}")
                    print(f"Using cached file at {cached_file}")
                    retrieved_file = cached_file
                else:
                    print(f"Error: No retrieved NCTIDs file found")
                    sys.exit(1)

            matching_cmd = [
                sys.executable,
                os.path.join(matching_dir, "run_matching.py"),
                args.corpus,
                str(retrieved_file),
                "--model_name",
                args.model,
                "--mcp_server_url",
                args.server_url,
            ]

            subprocess.run(matching_cmd, check=True)

        # Step 5: Run TrialGPT-Ranking aggregation
        if not args.skip_ranking:
            print("\n\n==== Running TrialGPT-Ranking Aggregation ====")

            # Construct matching results file path
            matching_file = (
                results_dir
                / f"matching_results_mcp_{args.model}_{args.corpus}{sample_suffix}.json"
            )

            # Check if matching file exists
            if not matching_file.exists():
                print(f"Error: Matching results file not found at {matching_file}")
                sys.exit(1)

            aggregation_cmd = [
                sys.executable,
                os.path.join(ranking_dir, "run_aggregation.py"),
                args.corpus,
                str(matching_file),
                "--model_name",
                args.model,
                "--mcp_server_url",
                args.server_url,
            ]

            subprocess.run(aggregation_cmd, check=True)

            # Run ranking to get final scores
            print("\n\n==== Running TrialGPT-Ranking Final Scoring ====")

            # Construct aggregation results file path
            aggregation_file = (
                results_dir
                / f"aggregation_results_mcp_{args.model}_{args.corpus}{sample_suffix}.json"
            )

            # Check if aggregation file exists
            if not aggregation_file.exists():
                print(
                    f"Error: Aggregation results file not found at {aggregation_file}"
                )
                sys.exit(1)

            ranking_cmd = [
                sys.executable,
                os.path.join(ranking_dir, "rank_results.py"),
                str(matching_file),
                str(aggregation_file),
                "--top_k",
                str(args.top_k),
            ]

            subprocess.run(ranking_cmd, check=True)

        print("\n\n==== Pipeline Complete ====")
        print(f"Results are available in the {results_dir} directory")

    finally:
        # Stop the server if we started it
        if server_process:
            print("Stopping MCP server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
