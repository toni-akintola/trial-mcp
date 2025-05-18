__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import json
import sys
import os

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


if __name__ == "__main__":
    # args are the results paths
    matching_results_path = sys.argv[1]
    agg_results_path = sys.argv[2]

    # loading the results
    matching_results = json.load(open(matching_results_path))
    agg_results = json.load(open(agg_results_path))

    # Determine output file path from input paths (e.g., based on matching_results_path)
    # base_name = os.path.basename(matching_results_path).replace("matching_results_", "final_ranking_") # Old logic

    # Derive output path from agg_results_path as it's the most processed input for this script
    base_agg_filename = os.path.basename(agg_results_path)
    # Input: aggregation_results_gpt-4-turbo_sigir_sample10.json
    # Output: final_ranking_gpt-4-turbo_sigir_sample10.json
    output_filename = base_agg_filename.replace("aggregation_results", "final_ranking")

    output_dir = "results/"  # Assuming results directory
    os.makedirs(output_dir, exist_ok=True)
    final_ranking_output_path = os.path.join(output_dir, output_filename)

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

            if patient_id not in agg_results or trial_id not in agg_results[patient_id]:
                print(
                    f"Patient {patient_id} Trial {trial_id} not in the aggregation results. Setting agg_score to 0."
                )
                agg_score = 0
            else:
                aggregation_output = agg_results[patient_id][trial_id]
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

        for trial, score in sorted_trial2score:
            print(trial, score)

        print("===")
        print()

    # Save the final ranked scores to a file
    with open(final_ranking_output_path, "w") as f:
        json.dump(all_patients_ranked_scores, f, indent=4)
    print(f"Final ranked scores saved to {final_ranking_output_path}")
