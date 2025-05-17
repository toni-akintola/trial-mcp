__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""

import json
from nltk.tokenize import sent_tokenize
import os
import sys

from TrialGPT import trialgpt_matching


def load_patient_queries(queries_file_path):
    """Loads patient queries from a JSONL file into a dict keyed by patient_id."""
    patients = {}
    with open(queries_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            patients[entry["_id"]] = entry["text"]
    return patients


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python run_matching.py <corpus> <model> <retrieved_nctids_file_path>"
        )
        sys.exit(1)

    corpus = sys.argv[1]
    model_name = sys.argv[2]  # Renamed from 'model' to avoid conflict with module
    retrieved_nctids_file_path = sys.argv[3]

    # Load patient notes
    patient_queries_path = f"dataset/{corpus}/queries.jsonl"
    if not os.path.exists(patient_queries_path):
        print(f"Error: Patient queries file not found at {patient_queries_path}")
        sys.exit(1)
    patient_id_to_text = load_patient_queries(patient_queries_path)

    # Load retrieved NCTIDs
    if not os.path.exists(retrieved_nctids_file_path):
        print(f"Error: Retrieved NCTIDs file not found at {retrieved_nctids_file_path}")
        sys.exit(1)
    retrieved_data = json.load(
        open(retrieved_nctids_file_path)
    )  # Expected: Dict{patient_id: List[NCTID]}

    # Load master trial information
    trial_info_path = "dataset/trial_info.json"
    if not os.path.exists(trial_info_path):
        print(f"Error: Trial info file not found at {trial_info_path}")
        sys.exit(1)
    trial_info_master = json.load(
        open(trial_info_path)
    )  # Expected: Dict{NCTID: trial_details}

    # Derive output path from input retrieved_nctids_file_path to include sample suffix if present
    base_retrieved_filename = os.path.basename(retrieved_nctids_file_path)
    # Input: retrieved_nctids_gpt-4-turbo_sigir_sample10.json
    # Output: matching_results_gpt-4-turbo_sigir_sample10.json
    output_filename = base_retrieved_filename.replace(
        "retrieved_nctids", "matching_results"
    )
    output_path = os.path.join("results", output_filename)

    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {}

    for patient_id, nctids in retrieved_data.items():
        if patient_id not in patient_id_to_text:
            print(
                f"Warning: Patient text not found for patient_id {patient_id}. Skipping."
            )
            continue

        patient_full_text = patient_id_to_text[patient_id]
        sents = sent_tokenize(patient_full_text)
        # Append consent and compliance statement as in original script
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
                continue

            # Already processed
            if nctid in output[patient_id]:
                print(
                    f"Skipping already processed trial {nctid} for patient {patient_id}"
                )
                continue

            trial_details = trial_info_master[nctid]
            # Ensure 'NCTID' is in trial_details if not already, for consistency, though trial_info_master is keyed by it.
            trial_details["NCTID"] = nctid

            print(f"Processing patient {patient_id}, trial {nctid}...")
            try:
                # trialgpt_matching expects the full trial dict and patient text
                matching_results = trialgpt_matching(
                    trial_details, patient_text_with_sent_ids, model_name
                )
                output[patient_id][nctid] = matching_results

                # Save incrementally
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)
            except Exception as e:
                print(f"Error processing patient {patient_id}, trial {nctid}: {e}")
                output[patient_id][nctid] = {"error": str(e)}  # Log error in output
                with open(output_path, "w") as f:  # Save error state
                    json.dump(output, f, indent=4)
                continue
        print(f"Finished processing patient {patient_id}.")
    print(f"All processing finished. Results saved to {output_path}")
