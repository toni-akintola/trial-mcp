__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""

import json
from nltk.tokenize import sent_tokenize
import os
import sys
import asyncio
import argparse
import time

# Adjust sys.path to find client.py in the project root
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from client import MCPClient
from trial_mcp_matching.TrialMCP import trialgpt_matching

# --- Configuration for Parallelization ---
# MAX_WORKERS is effectively controlled by asyncio.gather and system limits now
# MAX_CONCURRENT_TASKS = 20 # Example: Limit concurrent asyncio tasks if needed, can be adjusted
# -----------------------------------------


def load_patient_queries(queries_file_path):
    """Loads patient queries from a JSONL file into a dict keyed by patient_id."""
    patients = {}
    with open(queries_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            patients[entry["_id"]] = entry["text"]
    return patients


async def process_trial_for_patient_wrapper(
    mcp_client: MCPClient,
    patient_id,
    nctid,
    trial_details,
    patient_text_with_sent_ids,
    model_name,
):
    """Wrapper function to call trialgpt_matching (now async) and handle its result/errors."""
    print(f"Starting matching for patient {patient_id}, trial {nctid}...")
    try:
        # trialgpt_matching is now async and expects mcp_client
        matching_results = await trialgpt_matching(
            mcp_client, trial_details, patient_text_with_sent_ids, model_name
        )
        print(f"Finished matching for patient {patient_id}, trial {nctid}.")
        return patient_id, nctid, matching_results
    except Exception as e:
        print(
            f"Error during trialgpt_matching for patient {patient_id}, trial {nctid}: {e}"
        )
        return patient_id, nctid, {"error": f"Exception in trialgpt_matching: {str(e)}"}


async def main_async():
    parser = argparse.ArgumentParser(
        description="Run TrialGPT matching using MCPClient."
    )
    parser.add_argument("corpus", help="The corpus to use (e.g., sigir, trec2021).")
    parser.add_argument(
        "retrieved_nctids_file_path",
        help="Path to the JSON file containing retrieved NCTIDs for patients.",
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
    model_name = args.model_name
    retrieved_nctids_file_path = args.retrieved_nctids_file_path
    mcp_server_url = args.mcp_server_url

    mcp_client = MCPClient()
    try:
        await mcp_client.connect_to_sse_server(server_url=mcp_server_url)
        print(f"Successfully connected to MCP Server at {mcp_server_url}")

        # Load patient notes
        patient_queries_path = f"dataset/{corpus}/queries.jsonl"
        if not os.path.exists(patient_queries_path):
            print(f"Error: Patient queries file not found at {patient_queries_path}")
            sys.exit(1)
        patient_id_to_text = load_patient_queries(patient_queries_path)

        # Load retrieved NCTIDs
        if not os.path.exists(retrieved_nctids_file_path):
            print(
                f"Error: Retrieved NCTIDs file not found at {retrieved_nctids_file_path}"
            )
            sys.exit(1)
        retrieved_data = json.load(open(retrieved_nctids_file_path))

        # Load master trial information
        trial_info_path = "dataset/trial_info.json"
        if not os.path.exists(trial_info_path):
            print(f"Error: Trial info file not found at {trial_info_path}")
            sys.exit(1)
        trial_info_master = json.load(open(trial_info_path))

        output_filename = os.path.basename(retrieved_nctids_file_path).replace(
            "retrieved_nctids", "matching_results_mcp"
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
        patient_trial_pairs_for_tasks = []

        for patient_id, nctids in retrieved_data.items():
            if patient_id not in patient_id_to_text:
                print(
                    f"Warning: Patient text not found for patient_id {patient_id}. Skipping."
                )
                continue

            patient_full_text = patient_id_to_text[patient_id]
            sents = sent_tokenize(patient_full_text)
            sents.append(
                "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
            )
            patient_text_with_sent_ids = "\n".join(
                [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
            )

            if patient_id not in output:
                output[patient_id] = {}

            for nctid in nctids:
                if nctid not in trial_info_master:
                    print(
                        f"Warning: Trial information not found for NCTID {nctid} (patient {patient_id}). Skipping."
                    )
                    output[patient_id][nctid] = {
                        "error": "Trial info not found in master list"
                    }
                    continue

                if (
                    nctid in output[patient_id]
                    and output[patient_id][nctid].get("error") is None
                ):
                    print(
                        f"Skipping task for already processed trial {nctid} for patient {patient_id}"
                    )
                    continue

                trial_details = trial_info_master[nctid]
                trial_details["NCTID"] = nctid

                tasks.append(
                    process_trial_for_patient_wrapper(
                        mcp_client,
                        patient_id,
                        nctid,
                        trial_details,
                        patient_text_with_sent_ids,
                        model_name,
                    )
                )
                patient_trial_pairs_for_tasks.append((patient_id, nctid))

        if not tasks:
            print("No new tasks to process.")
        else:
            print(f"Submitting {len(tasks)} matching tasks asynchronously...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("All tasks completed.")

            for i, result_data_item in enumerate(results):
                p_id, trial_nctid = patient_trial_pairs_for_tasks[i]
                if isinstance(result_data_item, Exception):
                    print(
                        f"Task for trial {trial_nctid} for patient {p_id} generated an exception: {result_data_item}"
                    )
                    if p_id not in output:
                        output[p_id] = {}
                    output[p_id][trial_nctid] = {
                        "error": f"Async task generated exception: {str(result_data_item)}"
                    }
                else:
                    _, _, matching_data = result_data_item
                    if p_id not in output:
                        output[p_id] = {}
                    output[p_id][trial_nctid] = matching_data

                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                print(
                    f"Incrementally saved results to {output_path} after processing trial {trial_nctid} for patient {p_id}"
                )

        print(f"All processing finished. Final results saved to {output_path}")

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
