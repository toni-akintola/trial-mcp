__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

from beir.datasets.data_loader import GenericDataLoader
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import time
import asyncio
import argparse

# Adjust sys.path to find client.py in the project root
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from client import MCPClient
from trial_mcp_ranking.TrialMCP import trialgpt_aggregation

# --- Configuration for Parallelization ---
# MAX_AGGREGATION_WORKERS is now handled by asyncio.gather limits
# -----------------------------------------


async def process_aggregation_for_trial_wrapper(
    mcp_client: MCPClient,
    patient_id,
    trial_id,
    patient_note_for_aggregation,
    current_trial_matching_result,
    trial_detail,
    model_name_for_aggregation,
):
    """Wrapper function to call trialgpt_aggregation (now async) and handle its result/errors."""
    try:
        aggregation_data = await trialgpt_aggregation(
            mcp_client,
            patient_note_for_aggregation,
            current_trial_matching_result,
            trial_detail,
            model_name_for_aggregation,
        )
        return patient_id, trial_id, aggregation_data
    except Exception as e:
        print(
            f"Error during trialgpt_aggregation call for patient {patient_id}, trial {trial_id}: {e}"
        )
        return (
            patient_id,
            trial_id,
            {
                "error_aggregation_wrapper": f"Exception in trialgpt_aggregation: {str(e)}"
            },
        )


async def main_async():
    parser = argparse.ArgumentParser(
        description="Run TrialGPT aggregation using MCPClient."
    )
    parser.add_argument("corpus", help="The corpus to use (e.g., sigir, trec2021).")
    parser.add_argument(
        "matching_results_path",
        help="Path to the JSON file containing matching results.",
    )
    parser.add_argument(
        "--mcp_server_url",
        default=os.environ.get("MCP_SERVER_URL", "http://localhost:8080/sse"),
        help="URL of the MCP SSE server (default: http://localhost:8080/sse or MCP_SERVER_URL env var).",
    )
    parser.add_argument(
        "--model_name",
        default="claude-3-opus-20240229",
        help="Model name to be noted (actual model used is via MCPClient config).",
    )
    args = parser.parse_args()

    corpus = args.corpus
    model_name_for_aggregation = args.model_name
    matching_results_path = args.matching_results_path
    mcp_server_url = args.mcp_server_url

    mcp_client = MCPClient()
    try:
        await mcp_client.connect_to_sse_server(server_url=mcp_server_url)
        print(f"Successfully connected to MCP Server at {mcp_server_url}")

        results = json.load(open(matching_results_path))
        trial2info = json.load(open("dataset/trial_info.json"))
        _, queries, _ = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(
            split="test"
        )

        base_matching_filename = os.path.basename(matching_results_path)
        # Try to derive a sensible suffix, assuming input might be like "matching_results_mcp_claude-3-opus-20240229_sigir_sample10.json"
        # or "matching_results_claude-3-7-sonnet-latest_sigir.json"

        # Remove known prefixes to isolate the suffix part
        suffix_part = base_matching_filename
        suffix_part = suffix_part.replace("matching_results_mcp_", "")
        suffix_part = suffix_part.replace("matching_results_", "")
        suffix_part = suffix_part.replace(f"{model_name_for_aggregation}_", "")
        suffix_part = suffix_part.replace(f"{corpus}_", "")
        suffix_part = suffix_part.replace(".json", "")
        if suffix_part.startswith("_"):
            suffix_part = suffix_part[1:]

        output_filename_parts = [
            "aggregation_results_mcp",
            model_name_for_aggregation,
            corpus,
        ]
        if suffix_part:  # Add suffix if it exists and is not just the model/corpus
            output_filename_parts.append(suffix_part)

        output_filename = (
            "_".join(part for part in output_filename_parts if part) + ".json"
        )
        output_path = os.path.join("results", output_filename)

        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    output = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse existing output file {output_path}. Starting fresh."
                )
                output = {}
        else:
            output = {}

        tasks = []
        aggregation_task_identifiers = []  # To map results back

        for patient_id, patient_trials_results in results.items():
            if patient_id not in queries:
                print(
                    f"Warning: Patient query not found for patient_id {patient_id}. Skipping."
                )
                continue
            patient_note_text = queries[patient_id]
            sents = sent_tokenize(patient_note_text)
            sents.append(
                "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
            )
            patient_note_for_aggregation = "\n".join(
                [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
            )

            if patient_id not in output:
                output[patient_id] = {}

            for trial_id, trial_matching_result in patient_trials_results.items():
                # Skip if already processed successfully (i.e., not an error string/dict)
                if trial_id in output[patient_id] and not (
                    isinstance(output[patient_id][trial_id], str)
                    or output[patient_id][trial_id].get("error")
                ):
                    # print(f"Skipping already aggregated trial {trial_id} for patient {patient_id}")
                    continue

                if (
                    not isinstance(trial_matching_result, dict)
                    or not trial_matching_result
                    or trial_matching_result.get("error")
                ):
                    error_val = (
                        trial_matching_result.get(
                            "error", "matching result error or empty"
                        )
                        if isinstance(trial_matching_result, dict)
                        else "matching result error or empty"
                    )
                    print(
                        f"Matching result error or empty for patient {patient_id}, trial {trial_id}. Storing error: {error_val}"
                    )
                    output[patient_id][trial_id] = {
                        "error_upstream_matching": error_val
                    }
                    continue

                if trial_id not in trial2info:
                    print(f"Warning: Trial info not found for {trial_id}. Skipping.")
                    output[patient_id][trial_id] = {"error_trial_info_not_found": True}
                    continue

                current_trial_info = trial2info[trial_id]
                tasks.append(
                    process_aggregation_for_trial_wrapper(
                        mcp_client,
                        patient_id,
                        trial_id,
                        patient_note_for_aggregation,
                        trial_matching_result,
                        current_trial_info,
                        model_name_for_aggregation,
                    )
                )
                aggregation_task_identifiers.append((patient_id, trial_id))

        if not tasks:
            print("No new aggregation tasks to process.")
        else:
            print(f"Submitting {len(tasks)} aggregation tasks asynchronously...")
            # Use asyncio.gather for concurrent execution
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            print("All aggregation tasks completed.")

            for i, agg_result_item in enumerate(all_results):
                p_id, t_id = aggregation_task_identifiers[i]

                if isinstance(agg_result_item, Exception):
                    print(
                        f"Task for trial {t_id}, patient {p_id} generated an exception: {agg_result_item}"
                    )
                    if p_id not in output:
                        output[p_id] = {}
                    output[p_id][t_id] = {
                        "error_async_task_exception": str(agg_result_item)
                    }
                else:
                    # agg_result_item is (patient_id, trial_id, aggregation_data)
                    _, _, aggregation_data = agg_result_item
                    if p_id not in output:
                        output[p_id] = {}
                    output[p_id][t_id] = aggregation_data

                # Incremental save
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                print(
                    f"({i+1}/{len(all_results)}) Aggregated and saved: patient {p_id}, trial {t_id}. Output: {output_path}"
                )

        print(
            f"All aggregation processing finished. Final results saved to {output_path}"
        )

    except ConnectionRefusedError:
        print(
            f"Error: Connection refused. Is the MCP server running at {mcp_server_url}?"
        )
    except Exception as e:
        print(f"An unexpected error occurred in main_async: {e}")
    finally:
        if "mcp_client" in locals() and mcp_client.session:
            print("Cleaning up MCPClient session...")
            await mcp_client.cleanup()
            print("MCPClient session cleaned up.")


if __name__ == "__main__":
    asyncio.run(main_async())
