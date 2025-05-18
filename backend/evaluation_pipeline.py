# evaluation_pipeline.py

import json
from collections import defaultdict
import re  # For robust sentence parsing
from datasets import load_dataset, Dataset  # MODIFIED


# --- Utility Functions ---
def load_json_file(filepath):
    """Loads a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_sentence_indices(sentence_input) -> set:
    """
    Parses a string like '[0, 1, 2]', '[]', or even malformed ones,
    or a list of numbers into a set of integers.
    Returns an empty set if input is invalid or represents an empty list.
    """
    # Handle if input is already a list (e.g., from model output)
    if isinstance(sentence_input, list):
        try:
            return set(
                int(i) for i in sentence_input if str(i).isdigit()
            )  # Ensure elements are digits before int conversion
        except (ValueError, TypeError) as e:
            print(
                f"Warning: Could not parse list of sentence indices: {sentence_input}. Error: {e}"
            )
            return set()

    # Existing string parsing logic, now applied to sentence_input if it's a string
    if not isinstance(sentence_input, str):
        print(
            f"Warning: Unexpected type for sentence_input: {type(sentence_input)}. Value: {sentence_input}"
        )
        return set()  # Not a list and not a string, return empty set

    sentence_str = sentence_input  # Now we know it's a string
    if (
        not sentence_str
        or sentence_str.lower() == "none"
        or sentence_str.strip() == "[]"
    ):
        return set()

    # Attempt to use json.loads for well-formed JSON arrays
    try:
        # Normalize: replace single quotes if necessary, ensure it's a list
        processed_str = sentence_str.replace("'", '"')
        if not processed_str.startswith("["):
            processed_str = "[" + processed_str + "]"  # Handle cases like "1, 2"

        indices = json.loads(processed_str)
        if isinstance(indices, list):
            return set(int(i) for i in indices)
        return set()  # Should not happen if json.loads works
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback for strings like "0, 1, 2" or just numbers
        try:
            # Remove brackets and split by comma
            cleaned_str = sentence_str.strip("[] ")
            if not cleaned_str:
                return set()
            return set(
                int(x.strip()) for x in cleaned_str.split(",") if x.strip().isdigit()
            )
        except ValueError:
            print(f"Warning: Could not parse sentence string: {sentence_str}")
            return set()


def calculate_sentence_metrics(pred_indices: set, expert_indices: set) -> dict:
    """Calculates precision, recall, and F1 for sentence indices."""
    tp = len(pred_indices.intersection(expert_indices))
    fp = len(pred_indices.difference(expert_indices))
    fn = len(expert_indices.difference(pred_indices))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# --- Core Evaluation Logic ---
def evaluate_matching_component(gt_data: list, model_results: dict):
    """
    Evaluates the model's matching component against ground truth.
    gt_data: List of dictionaries (Hugging Face dataset format).
    model_results: Dictionary (your model's output format).
    """

    eligibility_correct_count = 0
    total_criteria_evaluated = 0

    sentence_precisions = []
    sentence_recalls = []
    sentence_f1s = []

    # For confusion matrix (optional)
    eligibility_labels_gt = []
    eligibility_labels_pred = []

    # Group ground truth by (patient_id, trial_id) to easily access all criteria for a trial
    # and then by criterion_type to get the implicit index
    grouped_gt = defaultdict(lambda: defaultdict(list))
    for record in gt_data:
        grouped_gt[(record["patient_id"], record["trial_id"])][
            record["criterion_type"]
        ].append(record)

    for (patient_id, trial_id), type_map in grouped_gt.items():
        if patient_id not in model_results:
            print(f"Warning: Patient ID {patient_id} not found in model results.")
            continue
        if trial_id not in model_results[patient_id]:
            print(
                f"Warning: Trial ID {trial_id} for patient {patient_id} not found in model results."
            )
            continue

        trial_model_predictions = model_results[patient_id][trial_id]

        for criterion_type, gt_criteria_list in type_map.items():
            if criterion_type not in trial_model_predictions:
                # This could happen if your model sometimes doesn't output a type (e.g. no exclusions)
                # Or if the ground truth has criteria of a type your model didn't process.
                print(
                    f"Warning: Criterion type '{criterion_type}' not found in model predictions for {patient_id}, {trial_id}."
                )
                total_criteria_evaluated += len(
                    gt_criteria_list
                )  # Count them as missed
                continue

            model_criteria_for_type = trial_model_predictions[criterion_type]

            # ---- defensive check ----
            if not isinstance(model_criteria_for_type, dict):
                print(
                    f"Warning: Expected a dictionary for model_criteria_for_type['{criterion_type}']"
                    f" for patient {patient_id}, trial {trial_id}, but got {type(model_criteria_for_type)}."
                    f" Value: {model_criteria_for_type}"
                )
                total_criteria_evaluated += len(gt_criteria_list)  # Count GT as missed
                continue
            # ---- end defensive check ----

            for i, gt_criterion_record in enumerate(gt_criteria_list):
                total_criteria_evaluated += 1
                criterion_idx_str = str(i)

                if criterion_idx_str not in model_criteria_for_type:
                    print(
                        f"Warning: Criterion index '{criterion_idx_str}' (type: {criterion_type}) "
                        f"not found in model predictions for {patient_id}, {trial_id}. "
                        f"GT Text: {gt_criterion_record['criterion_text']}"
                    )
                    continue  # Or handle as a miss

                model_prediction_list = model_criteria_for_type[criterion_idx_str]

                # Assuming model_prediction_list is [explanation, sentences_str, eligibility_label]
                if len(model_prediction_list) < 3:
                    print(
                        f"Warning: Malformed model prediction for {patient_id}, {trial_id}, type {criterion_type}, idx {criterion_idx_str}: {model_prediction_list}"
                    )
                    continue

                model_eligibility = model_prediction_list[2]
                model_sentences_str = model_prediction_list[1]

                expert_eligibility = gt_criterion_record["expert_eligibility"]
                expert_sentences_str = gt_criterion_record["expert_sentences"]

                # Eligibility Evaluation
                if model_eligibility == expert_eligibility:
                    eligibility_correct_count += 1

                eligibility_labels_gt.append(expert_eligibility)
                eligibility_labels_pred.append(model_eligibility)

                # Sentence Indices Evaluation
                model_indices = parse_sentence_indices(model_sentences_str)
                expert_indices = parse_sentence_indices(expert_sentences_str)

                sentence_eval = calculate_sentence_metrics(
                    model_indices, expert_indices
                )
                sentence_precisions.append(sentence_eval["precision"])
                sentence_recalls.append(sentence_eval["recall"])
                sentence_f1s.append(sentence_eval["f1"])

    # Calculate aggregate metrics
    eligibility_accuracy = (
        eligibility_correct_count / total_criteria_evaluated
        if total_criteria_evaluated > 0
        else 0.0
    )
    avg_sentence_precision = (
        sum(sentence_precisions) / len(sentence_precisions)
        if sentence_precisions
        else 0.0
    )
    avg_sentence_recall = (
        sum(sentence_recalls) / len(sentence_recalls) if sentence_recalls else 0.0
    )
    avg_sentence_f1 = sum(sentence_f1s) / len(sentence_f1s) if sentence_f1s else 0.0

    results_summary = {
        "total_criteria_evaluated": total_criteria_evaluated,
        "eligibility_accuracy": eligibility_accuracy,
        "avg_sentence_precision": avg_sentence_precision,
        "avg_sentence_recall": avg_sentence_recall,
        "avg_sentence_f1": avg_sentence_f1,
        "gt_eligibility_labels": eligibility_labels_gt,  # For confusion matrix
        "pred_eligibility_labels": eligibility_labels_pred,  # For confusion matrix
    }

    return results_summary


# --- Reporting ---
def print_evaluation_summary(summary: dict):
    """Prints a summary of the evaluation results."""
    print("\n--- Evaluation Summary ---")
    print(
        f"Total Patient-Criterion Pairs Evaluated: {summary['total_criteria_evaluated']}"
    )
    print(f"Criterion Eligibility Accuracy: {summary['eligibility_accuracy']:.4f}")
    print("\nSentence Location Metrics (Macro-Averages):")
    print(f"  Precision: {summary['avg_sentence_precision']:.4f}")
    print(f"  Recall:    {summary['avg_sentence_recall']:.4f}")
    print(f"  F1-Score:  {summary['avg_sentence_f1']:.4f}")
    print("--- End of Summary ---")

    # Optional: Confusion Matrix (requires sklearn or similar)
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import pandas as pd

        if summary["gt_eligibility_labels"] and summary["pred_eligibility_labels"]:
            labels = sorted(
                list(
                    set(
                        summary["gt_eligibility_labels"]
                        + summary["pred_eligibility_labels"]
                    )
                )
            )
            cm = confusion_matrix(
                summary["gt_eligibility_labels"],
                summary["pred_eligibility_labels"],
                labels=labels,
            )
            print("\nConfusion Matrix for Eligibility Predictions:")
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            print(cm_df)

            print("\nClassification Report for Eligibility Predictions:")
            # Set zero_division=0 to handle cases where a label might not have support in preds
            report = classification_report(
                summary["gt_eligibility_labels"],
                summary["pred_eligibility_labels"],
                labels=labels,
                zero_division=0,
            )
            print(report)

    except ImportError:
        print(
            "\n(Skipping confusion matrix and classification report: scikit-learn not installed)"
        )


# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths
    MODEL_OUTPUT_FILEPATH = "results/matching_results_gpt-4-turbo_sigir_sample10.json"

    print(
        "Loading ground truth data from Hugging Face Hub (ncbi/TrialGPT-Criterion-Annotations)..."
    )
    ground_truth_data_list = (
        []
    )  # Initialize to prevent NameError if try fails before assignment
    try:
        # Load the dataset. It might return a DatasetDict or a Dataset object.
        hf_dataset_object = load_dataset("ncbi/TrialGPT-Criterion-Annotations")

        hf_dataset_split = None
        # Check if it's a DatasetDict (which is a subclass of dict)
        if isinstance(hf_dataset_object, dict):
            if "train" in hf_dataset_object:
                hf_dataset_split = hf_dataset_object["train"]
                print("Using 'train' split from the dataset.")
            else:
                available_splits = list(hf_dataset_object.keys())
                if not available_splits:
                    raise ValueError(
                        "No splits found in the loaded Hugging Face dataset (DatasetDict was empty)."
                    )
                selected_split_name = available_splits[0]
                hf_dataset_split = hf_dataset_object[selected_split_name]
                print(
                    f"Warning: 'train' split not found in DatasetDict. Using the first available split: {selected_split_name}"
                )
        # Check if it's a Dataset object directly
        elif isinstance(hf_dataset_object, Dataset):
            hf_dataset_split = hf_dataset_object
            print("Using the loaded object directly as a Hugging Face Dataset.")
        else:
            raise TypeError(
                f"Loaded dataset is of unexpected type: {type(hf_dataset_object)}. Expected a Hugging Face Dataset or DatasetDict."
            )

        if (
            hf_dataset_split is None
        ):  # Should ideally not be hit if logic above is correct
            raise ValueError(
                "Could not determine a valid dataset split to use from the loaded Hugging Face data."
            )

        ground_truth_data_list = [item for item in hf_dataset_split]

        if not ground_truth_data_list and hf_dataset_split is not None:
            print(
                f"Warning: The selected dataset split processed into an empty list of records."
            )
        elif ground_truth_data_list:
            print(
                f"Successfully loaded {len(ground_truth_data_list)} records from the Hugging Face dataset."
            )
        # No 'else' needed here as an empty list is handled by the check above or subsequent checks.

    except ImportError:
        print(
            "Error: The 'datasets' library is not installed. Please install it by running: pip install datasets"
        )
        import sys

        sys.exit(1)
    except (
        Exception
    ) as e:  # Catch other potential errors from load_dataset or processing
        print(
            f"Error loading or processing Hugging Face dataset 'ncbi/TrialGPT-Criterion-Annotations': {e}"
        )
        print(
            "Please ensure you have internet access, the 'datasets' library is installed, and the dataset identifier is correct."
        )
        import sys

        sys.exit(1)

    print(f"Loading model output from: {MODEL_OUTPUT_FILEPATH}")
    try:
        model_output_data = load_json_file(MODEL_OUTPUT_FILEPATH)
    except FileNotFoundError:
        print(
            f"Error: Model output file not found at {MODEL_OUTPUT_FILEPATH}. Please ensure it exists."
        )
        import sys

        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Error: Model output file at {MODEL_OUTPUT_FILEPATH} is not a valid JSON file."
        )
        import sys

        sys.exit(1)
    except Exception as e:  # Catch any other error during model output loading
        print(
            f"An unexpected error occurred while loading model output file {MODEL_OUTPUT_FILEPATH}: {e}"
        )
        import sys

        sys.exit(1)

    # Ensure ground_truth_data_list is not empty before proceeding
    if not ground_truth_data_list:
        print(
            "Ground truth data is empty after attempting to load from Hugging Face. Cannot perform evaluation."
        )
        import sys

        sys.exit(1)

    # Perform evaluation
    evaluation_results = evaluate_matching_component(
        ground_truth_data_list, model_output_data
    )

    # Print summary
    print_evaluation_summary(evaluation_results)
