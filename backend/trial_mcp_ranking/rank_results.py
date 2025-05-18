__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import json
import sys
import os
import argparse

eps = 1e-9


def get_matching_score(matching):
    # count only the valid ones
    included = 0
    not_inc = 0
    na_inc = 0
    no_info_inc = 0

    excluded = 0
    not_exc = 0
    na_exc = 0
    no_info_exc = 0

    # first count inclusions
    if "inclusion" in matching and isinstance(matching["inclusion"], dict):
        for criteria, info in matching["inclusion"].items():
            if len(info) != 3:
                continue
            if info[2] == "included":
                included += 1
            elif info[2] == "not included":
                not_inc += 1
            elif info[2] == "not applicable":
                na_inc += 1
            elif info[2] == "not enough information":
                no_info_inc += 1
    else:
        print(
            f"Warning: 'inclusion' data in matching result is not a dictionary or is missing. Skipping inclusion count. Data: {str(matching.get('inclusion'))[:200]}"
        )

    # then count exclusions
    if "exclusion" in matching and isinstance(matching["exclusion"], dict):
        for criteria, info in matching["exclusion"].items():
            if len(info) != 3:
                continue
            if info[2] == "excluded":
                excluded += 1
            elif info[2] == "not excluded":
                not_exc += 1
            elif info[2] == "not applicable":
                na_exc += 1
            elif info[2] == "not enough information":
                no_info_exc += 1
    else:
        print(
            f"Warning: 'exclusion' data in matching result is not a dictionary or is missing. Skipping exclusion count. Data: {str(matching.get('exclusion'))[:200]}"
        )

    # get the matching score
    score = 0

    score += included / (included + not_inc + no_info_inc + eps)

    if not_inc > 0:
        score -= 1

    if excluded > 0:
        score -= 1

    return score


def get_agg_score(assessment):
    try:
        rel_score = float(assessment["relevance_score_R"])
        eli_score = float(assessment["eligibility_score_E"])
    except:
        rel_score = 0
        eli_score = 0

    score = (rel_score + eli_score) / 100

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Rank clinical trials based on patient-trial matching and aggregation results."
    )
    parser.add_argument(
        "matching_results_path",
        help="Path to the JSON file containing matching results.",
    )
    parser.add_argument(
        "aggregation_results_path",
        help="Path to the JSON file containing aggregation results.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top trials to display per patient.",
    )
    args = parser.parse_args()

    matching_results_path = args.matching_results_path
    aggregation_results_path = args.aggregation_results_path
    top_k = args.top_k

    if not os.path.exists(matching_results_path):
        print(f"Error: Matching results file not found at {matching_results_path}")
        sys.exit(1)

    if not os.path.exists(aggregation_results_path):
        print(
            f"Error: Aggregation results file not found at {aggregation_results_path}"
        )
        sys.exit(1)

    matching_results = json.load(open(matching_results_path))
    aggregation_results = json.load(open(aggregation_results_path))

    # Extract corpus and model information from filenames for output naming
    matching_basename = os.path.basename(matching_results_path)
    aggregation_basename = os.path.basename(aggregation_results_path)

    # Define naming components for the output file
    components = []

    # Try to extract the model name from either filename
    model_name = None
    for filename in [matching_basename, aggregation_basename]:
        # Looking for patterns like matching_results_claude-3-opus-20240229_sigir.json
        # or aggregation_results_mcp_claude-3-opus-20240229_sigir.json
        for prefix in [
            "matching_results_",
            "aggregation_results_",
            "matching_results_mcp_",
            "aggregation_results_mcp_",
        ]:
            if filename.startswith(prefix):
                parts = filename[len(prefix) :].split("_")
                if len(parts) >= 1:
                    potential_model = parts[0]
                    if "claude" in potential_model or "gpt" in potential_model:
                        model_name = potential_model
                        break
        if model_name:
            break

    if model_name:
        components.append(model_name)

    # Try to extract the corpus from filenames
    corpus = None
    for filename in [matching_basename, aggregation_basename]:
        for potential_corpus in ["sigir", "trec_2021", "trec_2022"]:
            if (
                f"_{potential_corpus}" in filename
                or f"_{potential_corpus.replace('_', '')}" in filename
            ):
                corpus = potential_corpus
                break
        if corpus:
            break

    if corpus:
        components.append(corpus)

    # Try to extract sample information
    sample_info = None
    for filename in [matching_basename, aggregation_basename]:
        if "sample" in filename:
            parts = filename.split("_")
            for part in parts:
                if part.startswith("sample"):
                    sample_info = part
                    break
        if sample_info:
            break

    if sample_info:
        components.append(sample_info)

    # Construct the output filename
    output_filename = "trial_rankings_" + "_".join(components) + ".json"
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_filepath = os.path.join(results_dir, output_filename)

    all_patients_ranked_scores = {}

    # loop over the patients
    for (
        patient_id,
        patient_matching_data,
    ) in (
        matching_results.items()
    ):  # patient_matching_data is Dict{trial_id: matching_output}

        trial2score = {}

        # for _, trial2results in label2trial2results.items(): # This loop is removed
        for trial_id, single_trial_match_output in patient_matching_data.items():

            if (
                isinstance(single_trial_match_output, str)
                or "error" in single_trial_match_output
            ):
                print(
                    f"Skipping trial {trial_id} for patient {patient_id} due to matching error: {single_trial_match_output}"
                )
                continue

            matching_score = get_matching_score(single_trial_match_output)

            if (
                patient_id not in aggregation_results
                or trial_id not in aggregation_results[patient_id]
            ):
                print(
                    f"Patient {patient_id} Trial {trial_id} not in the aggregation results. Setting agg_score to 0."
                )
                agg_score = 0
            else:
                aggregation_output = aggregation_results[patient_id][trial_id]
                if (
                    isinstance(aggregation_output, str)
                    or "error_aggregation" in aggregation_output
                    or "relevance_score_R" not in aggregation_output
                ):
                    print(
                        f"Skipping trial {trial_id} for patient {patient_id} due to aggregation error or missing scores: {aggregation_output}"
                    )
                    agg_score = 0  # Or handle as error / skip trial score calculation
                else:
                    agg_score = get_agg_score(aggregation_output)

            trial_score = matching_score + agg_score
            trial2score[trial_id] = trial_score

        sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
        all_patients_ranked_scores[patient_id] = dict(sorted_trial2score)

        print()
        print(f"Patient ID: {patient_id}")
        print("Clinical trial ranking:")

        for trial, score in sorted_trial2score[:top_k]:
            print(trial, score)

        print("===")
        print()

    # Save the final ranked scores to a file
    with open(output_filepath, "w") as f:
        json.dump(all_patients_ranked_scores, f, indent=4)
    print(f"Final ranked scores saved to {output_filepath}")


if __name__ == "__main__":
    main()
