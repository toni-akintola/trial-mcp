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

from TrialGPT import trialgpt_aggregation

if __name__ == "__main__":
    corpus = sys.argv[1]
    model = sys.argv[2]

    # the path of the matching results
    matching_results_path = sys.argv[3]
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
        f"matching_results_{model}_{corpus}", ""
    )
    if input_suffix.startswith("_"):
        input_suffix = input_suffix[
            1:
        ]  # remove leading _ if any, like _sample10.json -> sample10.json
    input_suffix = input_suffix.replace(".json", "")  # sample10

    if input_suffix:
        output_path = (
            f"results/aggregation_results_{model}_{corpus}_{input_suffix}.json"
        )
    else:
        output_path = f"results/aggregation_results_{model}_{corpus}.json"

    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {}

    # patient-level
    for patient_id, patient_trials_results in results.items():
        # get the patient note
        if patient_id not in queries:
            print(
                f"Warning: Patient query not found for patient_id {patient_id}. Skipping aggregation."
            )
            continue
        patient_note_text = queries[patient_id]
        sents = sent_tokenize(patient_note_text)
        sents.append(
            "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
        )
        sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        patient = "\n".join(sents)

        if patient_id not in output:
            output[patient_id] = {}

        # trial-level - patient_trials_results is Dict{trial_id: trial_matching_result}
        for trial_id, trial_matching_result in patient_trials_results.items():
            # already cached results
            if trial_id in output[patient_id]:
                print(
                    f"Skipping already aggregated trial {trial_id} for patient {patient_id}"
                )
                continue

            if type(trial_matching_result) is not dict or not trial_matching_result:
                output[patient_id][trial_id] = "matching result error or empty"
                print(
                    f"Matching result error for patient {patient_id}, trial {trial_id}"
                )
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                continue

            if trial_id not in trial2info:
                print(
                    f"Warning: Trial info not found for {trial_id}. Skipping aggregation for this trial."
                )
                output[patient_id][trial_id] = "trial_info not found"
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                continue

            # specific trial information
            current_trial_info = trial2info[trial_id]

            print(f"Aggregating patient {patient_id}, trial {trial_id}...")
            try:
                # Pass patient_note_text (original text before sentence IDs), trial_matching_result, current_trial_info, model
                aggregation_result = trialgpt_aggregation(
                    patient, trial_matching_result, current_trial_info, model
                )
                output[patient_id][trial_id] = aggregation_result

                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)

            except Exception as e:
                print(
                    f"Error during aggregation for patient {patient_id}, trial {trial_id}: {e}"
                )
                output[patient_id][trial_id] = {"error_aggregation": str(e)}
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
                continue
        print(f"Finished aggregation for patient {patient_id}.")
    print(f"All aggregation finished. Results saved to {output_path}")
