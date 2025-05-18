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
import concurrent.futures

from trial_mcp_ranking.TrialMCP import trialgpt_aggregation

# --- Configuration for Parallelization ---
MAX_AGGREGATION_WORKERS = 5  # Number of concurrent API calls for aggregation
# -----------------------------------------


def process_aggregation_for_trial_wrapper(
    patient_id,
    trial_id,
    patient_note_for_aggregation,
    current_trial_matching_result,
    trial_detail,
    model_name_for_aggregation,
):
    """Wrapper function to call trialgpt_aggregation and handle its result/errors for a single trial."""
    # print(f"Starting aggregation for patient {patient_id}, trial {trial_id}...") # Verbose, can be enabled if needed
    try:
        aggregation_data = trialgpt_aggregation(
            patient_note_for_aggregation,
            current_trial_matching_result,
            trial_detail,
            model_name_for_aggregation,
        )
        # print(f"Finished aggregation for patient {patient_id}, trial {trial_id}.") # Verbose
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


if __name__ == "__main__":
    if len(sys.argv) < 3:  # Adjusted for one less argument
        print("Usage: python run_aggregation.py <corpus> <matching_results_path>")
        sys.exit(1)

    corpus = sys.argv[1]
    model_name_for_aggregation = "claude-3-7-sonnet-latest"  # Hardcoded model name
    matching_results_path = sys.argv[2]  # Adjusted index

    # the path of the matching results
    results = json.load(open(matching_results_path))

    # loading the trial2info dict
    trial2info = json.load(open("dataset/trial_info.json"))

    # loading the patient info
    _, queries, _ = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(
        split="test"
    )

    # output file path
    # Derive output path from input matching_results_path
    base_matching_filename = os.path.basename(matching_results_path)
    # Input: matching_results_gpt-4-turbo_sigir_sample10.json
    # Output: aggregation_results_gpt-4-turbo_sigir_sample10.json

    # corpus and model name are already script arguments, can use them for consistency if preferred
    # For example, if base_matching_filename was matching_results_custom_suffix.json, this would make
    # aggregation_results_sigir_gpt-4-turbo_custom_suffix.json.
    # Let's ensure model and corpus from arguments are used, and append only unique suffix from input.

    input_suffix = base_matching_filename.replace(
        f"matching_results_{model_name_for_aggregation}_{corpus}",
        "",  # Use hardcoded model name
    )
    if input_suffix.startswith("_"):
        input_suffix = input_suffix[
            1:
        ]  # remove leading _ if any, like _sample10.json -> sample10.json
    input_suffix = input_suffix.replace(".json", "")  # sample10

    if input_suffix:
        output_path = f"results/aggregation_results_{model_name_for_aggregation}_{corpus}_{input_suffix}.json"  # Use hardcoded model name
    else:
        output_path = f"results/aggregation_results_{model_name_for_aggregation}_{corpus}.json"  # Use hardcoded model name

    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {}

    tasks_to_submit = []

    # First, prepare all tasks
    for patient_id, patient_trials_results in results.items():
        if patient_id not in queries:
            print(
                f"Warning: Patient query not found for patient_id {patient_id}. Skipping aggregation for this patient."
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
            if trial_id in output[patient_id]:
                # print(f"Skipping already aggregated trial {trial_id} for patient {patient_id}") # Verbose
                continue

            if not isinstance(trial_matching_result, dict) or not trial_matching_result:
                print(
                    f"Matching result error or empty for patient {patient_id}, trial {trial_id}. Storing error."
                )
                output[patient_id][trial_id] = "matching result error or empty"
                # No immediate save here, will be saved after all processing or after each future completion
                continue

            if trial_id not in trial2info:
                print(
                    f"Warning: Trial info not found for {trial_id}. Skipping aggregation for this trial."
                )
                output[patient_id][trial_id] = "trial_info not found"
                continue

            current_trial_info = trial2info[trial_id]
            tasks_to_submit.append(
                {
                    "patient_id": patient_id,
                    "trial_id": trial_id,
                    "patient_note": patient_note_for_aggregation,
                    "matching_result": trial_matching_result,
                    "trial_info": current_trial_info,
                    "model": model_name_for_aggregation,  # Use the hardcoded variable
                }
            )

    print(f"Prepared {len(tasks_to_submit)} new aggregation tasks to submit.")

    # Using ThreadPoolExecutor for I/O bound tasks (API calls)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_AGGREGATION_WORKERS
    ) as executor:
        future_to_agg_info = {}

        for task in tasks_to_submit:
            # print(f"Submitting aggregation for patient {task['patient_id']}, trial {task['trial_id']}...") # Verbose
            future = executor.submit(
                process_aggregation_for_trial_wrapper,
                task["patient_id"],
                task["trial_id"],
                task["patient_note"],
                task["matching_result"],
                task["trial_info"],
                task["model"],
            )
            future_to_agg_info[future] = (task["patient_id"], task["trial_id"])

        processed_count = 0
        total_futures = len(future_to_agg_info)
        for future in concurrent.futures.as_completed(future_to_agg_info):
            p_id, t_id = future_to_agg_info[future]
            try:
                _, _, result_agg_data = future.result()
                if p_id not in output:  # Should have been created already
                    output[p_id] = {}
                output[p_id][t_id] = result_agg_data
            except Exception as exc:
                print(
                    f"Aggregation for trial {t_id}, patient {p_id} generated an exception in future: {exc}"
                )
                if p_id not in output:
                    output[p_id] = {}
                output[p_id][t_id] = {"error_future_exception": str(exc)}
            finally:
                processed_count += 1
                # Incremental save after each result is processed
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                print(
                    f"({processed_count}/{total_futures}) Aggregated and saved: patient {p_id}, trial {t_id}. Output: {output_path}"
                )

    print(f"All aggregation processing finished. Final results saved to {output_path}")
