# evaluation_pipeline.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score  # Modified import
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set OpenAI API environment variables if you're going to run TrialGPT itself
# os.environ["OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT_URL"
# os.environ["OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"


def load_criterion_annotations(file_path):
    """
    Load the ground truth dataset containing criterion-level annotations
    """
    with open(file_path, "r") as f:
        fp = json.load(f)
    df = pd.DataFrame(fp).transpose()
    print(f"Loaded {len(df)} criterion-level annotations from {file_path}")
    return df


def load_trialgpt_predictions(matching_results_path):
    """
    Load TrialGPT-Matching predictions
    """
    with open(matching_results_path, "r") as f:
        predictions = json.load(f)
    print(f"Loaded predictions from {matching_results_path}")
    return predictions


def evaluate_explanations(annotations_df):
    """
    Evaluate the correctness of TrialGPT explanations
    """
    if "explanation_correctness" not in annotations_df.columns:
        print(
            "Warning: 'explanation_correctness' column not found in annotations. Skipping explanation evaluation."
        )
        return {}, {}

    correctness_counts = annotations_df["explanation_correctness"].value_counts()
    total = len(annotations_df)

    results = {
        "correct": correctness_counts.get("Correct", 0) / total if total > 0 else 0,
        "partially_correct": (
            correctness_counts.get("Partially Correct", 0) / total if total > 0 else 0
        ),
        "incorrect": correctness_counts.get("Incorrect", 0) / total if total > 0 else 0,
    }

    results_by_type = {}
    if "criterion_type" in annotations_df.columns:
        for criterion_type in annotations_df["criterion_type"].unique():
            subset = annotations_df[annotations_df["criterion_type"] == criterion_type]
            counts = subset["explanation_correctness"].value_counts()
            total_subset = len(subset)

            results_by_type[criterion_type] = {
                "correct": (
                    counts.get("Correct", 0) / total_subset if total_subset > 0 else 0
                ),
                "partially_correct": (
                    counts.get("Partially Correct", 0) / total_subset
                    if total_subset > 0
                    else 0
                ),
                "incorrect": (
                    counts.get("Incorrect", 0) / total_subset if total_subset > 0 else 0
                ),
            }
    else:
        print(
            "Warning: 'criterion_type' column not found. Skipping explanation evaluation by type."
        )

    return results, results_by_type


def parse_sentence_list(sentence_str):
    """
    Parse sentence list from the string format in the dataset
    """
    if pd.isna(sentence_str) or sentence_str == "[]" or not sentence_str:
        return []
    try:
        # Handles strings like "[0, 1, 2]" or "0, 1, 2"
        return [
            int(x.strip())
            for x in str(sentence_str).strip("[]").split(",")
            if x.strip()
        ]
    except ValueError:
        print(f"Warning: Could not parse sentence list string: {sentence_str}")
        return []


def evaluate_sentence_identification(annotations_df):
    """
    Evaluate precision, recall, and F1 for relevant sentence identification
    """
    if not all(
        col in annotations_df.columns for col in ["expert_sentences", "gpt4_sentences"]
    ):
        print(
            "Warning: 'expert_sentences' or 'gpt4_sentences' column not found. Skipping sentence identification evaluation."
        )
        return {"precision": 0, "recall": 0, "f1": 0}

    # Filter to include only annotations where at least one expert sentence is relevant
    # or handle cases where expert_sentences might be missing/NaN before parsing
    annotations_df["parsed_expert_sentences"] = annotations_df[
        "expert_sentences"
    ].apply(parse_sentence_list)
    valid_annotations = annotations_df[
        annotations_df["parsed_expert_sentences"].apply(len) > 0
    ]

    if valid_annotations.empty:
        print(
            "Warning: No valid annotations with expert sentences found for sentence identification evaluation."
        )
        return {"precision": 0, "recall": 0, "f1": 0}

    precision_list, recall_list, f1_list = [], [], []

    for _, row in valid_annotations.iterrows():
        predicted_sentences = set(parse_sentence_list(row["gpt4_sentences"]))
        true_sentences = set(row["parsed_expert_sentences"])  # Use pre-parsed

        if not predicted_sentences and not true_sentences:
            # Both empty, perfect score for this instance if we consider it.
            # Or skip if problem defines it differently. Let's be consistent: if true_sentences is empty, it's skipped by filter.
            # If predicted is empty but true is not, precision is undefined by some, 0 by others.
            # The current logic handles this: precision = 0 if len(predicted_sentences) == 0.
            pass

        true_positives = len(predicted_sentences.intersection(true_sentences))

        precision = (
            true_positives / len(predicted_sentences)
            if len(predicted_sentences) > 0
            else 0
        )
        recall = (
            true_positives / len(true_sentences) if len(true_sentences) > 0 else 0
        )  # true_sentences guaranteed > 0 by filter
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    results = {
        "precision": np.mean(precision_list) if precision_list else 0,
        "recall": np.mean(recall_list) if recall_list else 0,
        "f1": np.mean(f1_list) if f1_list else 0,
    }

    return results


def evaluate_eligibility_classification(annotations_df):
    """
    Evaluate the eligibility classification accuracy and create a confusion matrix
    """
    if not all(
        col in annotations_df.columns
        for col in ["gpt4_eligibility", "expert_eligibility", "criterion_type"]
    ):
        print(
            "Warning: Required columns for eligibility classification missing. Skipping."
        )
        return 0, {}, {}

    # Overall accuracy
    accuracy = accuracy_score(
        annotations_df["expert_eligibility"], annotations_df["gpt4_eligibility"]
    )

    accuracy_by_type = {}
    confusion_matrices = {}

    for criterion_type in annotations_df["criterion_type"].unique():
        subset = annotations_df[annotations_df["criterion_type"] == criterion_type]
        if subset.empty:
            continue

        accuracy_by_type[criterion_type] = accuracy_score(
            subset["expert_eligibility"], subset["gpt4_eligibility"]
        )

        # Create confusion matrix
        # Ensure all potential labels are included, even if not present in both pred/true for a subset
        all_labels = sorted(
            pd.unique(
                annotations_df[["expert_eligibility", "gpt4_eligibility"]].values.ravel(
                    "K"
                )
            )
        )

        confusion_matrix_df = pd.DataFrame(0, index=all_labels, columns=all_labels)

        for _, row in subset.iterrows():
            true_label = row["expert_eligibility"]
            pred_label = row["gpt4_eligibility"]
            if (
                true_label in confusion_matrix_df.index
                and pred_label in confusion_matrix_df.columns
            ):
                confusion_matrix_df.loc[true_label, pred_label] += 1
            else:
                # This case should ideally not happen if all_labels is comprehensive
                print(
                    f"Warning: Unexpected label found: True='{true_label}', Pred='{pred_label}' for type '{criterion_type}'"
                )

        # Normalize by row (true label)
        row_sums = confusion_matrix_df.sum(axis=1)
        # Avoid division by zero for labels not present in true labels for this subset
        normalized_matrix = confusion_matrix_df.div(
            row_sums.replace(0, np.nan), axis=0
        ).fillna(0)
        confusion_matrices[criterion_type] = normalized_matrix

    return accuracy, accuracy_by_type, confusion_matrices


def analyze_errors(annotations_df):
    """
    Analyze errors made by TrialGPT and categorize them
    """
    if not all(
        col in annotations_df.columns
        for col in ["gpt4_eligibility", "expert_eligibility"]
    ):
        print("Warning: Required columns for error analysis missing. Skipping.")
        return pd.DataFrame(), {}, {}

    errors = annotations_df[
        annotations_df["gpt4_eligibility"] != annotations_df["expert_eligibility"]
    ]
    print(
        f"Found {len(errors)} errors out of {len(annotations_df)} annotations for eligibility classification."
    )

    error_categories = {
        "incorrect_reasoning": [],
        "lack_of_medical_knowledge": [],
        "ambiguous_label_definitions": [],
        "other": [],
    }

    # For a full implementation, you would need to manually categorize these errors
    # This is a placeholder for demonstration

    error_examples = {}  # Placeholder for actual examples

    return errors, error_categories, error_examples


def aggregate_criterion_predictions(annotations_df):
    """
    Aggregate criterion-level predictions to trial-level scores
    """
    if not all(
        col in annotations_df.columns
        for col in ["patient_id", "trial_id", "criterion_type", "gpt4_eligibility"]
    ):
        print(
            "Warning: Required columns for aggregation missing. Skipping trial-level aggregation."
        )
        return defaultdict(dict)

    trial_scores = defaultdict(
        lambda: defaultdict(dict)
    )  # Ensure inner dicts are also defaultdicts initially

    unique_patient_ids = annotations_df["patient_id"].unique()
    if len(unique_patient_ids) == 0:
        print("Warning: No patient_ids found for aggregation.")
        return trial_scores

    for patient_id in unique_patient_ids:
        patient_df = annotations_df[annotations_df["patient_id"] == patient_id]
        unique_trial_ids = patient_df["trial_id"].unique()
        if len(unique_trial_ids) == 0:
            continue

        for trial_id in unique_trial_ids:
            trial_df = patient_df[patient_df["trial_id"] == trial_id]

            inclusion_criteria = trial_df[trial_df["criterion_type"] == "inclusion"]
            exclusion_criteria = trial_df[trial_df["criterion_type"] == "exclusion"]

            pct_included = 0.0
            if not inclusion_criteria.empty:
                met_inclusion = inclusion_criteria["gpt4_eligibility"] == "included"
                pct_included = (
                    met_inclusion.sum() / len(inclusion_criteria)
                    if len(inclusion_criteria) > 0
                    else 0.0
                )

            pct_not_excluded = 0.0  # % of exclusion criteria that are 'not excluded' (patient meets them)
            if not exclusion_criteria.empty:
                not_excluded_from_exclusion = (
                    exclusion_criteria["gpt4_eligibility"] == "not excluded"
                )
                pct_not_excluded = (
                    not_excluded_from_exclusion.sum() / len(exclusion_criteria)
                    if len(exclusion_criteria) > 0
                    else 0.0
                )

            # Store aggregated scores
            # Original: trial_score = pct_included - (1 - pct_not_excluded)
            # Interpretation: if all inclusion met (1.0) and no exclusion criteria breached (pct_not_excluded = 1.0 for all 'not excluded'),
            # then score = 1.0 - (1 - 1.0) = 1.0.
            # If all inclusion met (1.0) and all exclusion criteria are breached (pct_not_excluded = 0.0 for all 'excluded'),
            # then score = 1.0 - (1 - 0.0) = 0.0.
            # This seems to represent eligibility based on how many criteria are favorably met.
            trial_score_val = pct_included - (1.0 - pct_not_excluded)

            trial_scores[patient_id][trial_id] = {
                "pct_included": pct_included,
                "pct_not_excluded": pct_not_excluded,  # This is "percentage of exclusion criteria NOT triggering exclusion"
                "trial_score": trial_score_val,
            }

    return trial_scores


def evaluate_trial_ranking(trial_scores, ground_truth_rankings):
    """
    Evaluate trial ranking performance using NDCG@10 and P@10
    """
    # This is a placeholder - you would need to implement the actual metrics
    # based on the TrialGPT-Ranking methodology
    print("Warning: Trial ranking evaluation is a placeholder and not implemented.")
    results = {
        "ndcg@10": 0.0,
        "p@10": 0.0,
        "auroc": 0.0,  # AUROC might be more for binary classification of eligibility
    }

    return results


def plot_explanation_correctness(results):
    """
    Plot explanation correctness results
    """
    if not results:
        print("No explanation results to plot.")
        return
    labels = ["Correct", "Partially Correct", "Incorrect"]
    values = [
        results.get("correct", 0),
        results.get("partially_correct", 0),
        results.get("incorrect", 0),
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=["green", "orange", "red"])
    plt.title("Explanation Correctness")
    plt.ylabel("Percentage")
    plt.savefig("explanation_correctness.png")
    plt.close()
    print("Saved plot: explanation_correctness.png")


def plot_confusion_matrix(confusion_matrix_df, criterion_type):
    """
    Plot confusion matrix
    """
    if confusion_matrix_df.empty:
        print(f"No confusion matrix data to plot for {criterion_type}.")
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title(f"Confusion Matrix - {criterion_type.capitalize()}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f'confusion_matrix_{criterion_type.replace(" ", "_").lower()}.png')
    plt.close()
    print(
        f"Saved plot: confusion_matrix_{criterion_type.replace(' ', '_').lower()}.png"
    )


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):  # Handle DataFrames in confusion matrices
        return obj.to_dict(orient="index")
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(
        obj, (pd.Timestamp, pd.Timedelta)
    ):  # Handle pandas time objects if any
        return str(obj)
    return obj


def run_evaluation_pipeline(
    annotations_file, matching_results_file=None, ranking_results_file=None
):
    """
    Run the complete evaluation pipeline
    """
    # Load data
    try:
        annotations_df = load_criterion_annotations(annotations_file)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file}")
        return None
    except Exception as e:
        print(f"Error loading annotations file {annotations_file}: {e}")
        return None

    if annotations_df.empty:
        print("Annotations DataFrame is empty. Aborting pipeline.")
        return None

    # The original code loads predictions but doesn't explicitly merge or use them
    # if gpt4_* columns are already in annotations_df.
    # This assumes annotations_df contains all necessary columns (expert_*, gpt4_*, explanation_correctness).
    if matching_results_file:
        try:
            predictions_data = load_trialgpt_predictions(matching_results_file)
            # Here, one might merge predictions_data into annotations_df if it's not already comprehensive.
            # For now, assuming annotations_df has all required columns.
            print(
                "Note: matching_results_file was loaded, but the script assumes 'annotations_file' contains 'gpt4_*' columns for evaluation."
            )
        except FileNotFoundError:
            print(
                f"Warning: Matching results file not found at {matching_results_file}. Proceeding without it."
            )
        except Exception as e:
            print(f"Error loading matching results file {matching_results_file}: {e}")

    # Evaluate explanations
    explanation_results, explanation_results_by_type = evaluate_explanations(
        annotations_df.copy()
    )  # Pass copy to avoid SettingWithCopyWarning
    print("\n=== Explanation Correctness ===")
    print(f"Overall: {explanation_results}")
    if explanation_results_by_type:
        print(f"By criterion type: {explanation_results_by_type}")

    # Evaluate relevant sentence identification
    sentence_results = evaluate_sentence_identification(annotations_df.copy())
    print("\n=== Relevant Sentence Identification ===")
    print(f"Results: {sentence_results}")

    # Evaluate eligibility classification
    accuracy, accuracy_by_type, confusion_matrices = (
        evaluate_eligibility_classification(annotations_df.copy())
    )
    print("\n=== Eligibility Classification ===")
    print(f"Overall Accuracy: {accuracy:.4f}")
    if accuracy_by_type:
        print(f"Accuracy by criterion type: {accuracy_by_type}")

    print("\n=== Generating Visualizations ===")
    if explanation_results:
        plot_explanation_correctness(explanation_results)

    if confusion_matrices:
        print("Confusion Matrices (normalized by true label count):")
        for criterion_type, matrix_df in confusion_matrices.items():
            print(f"\n{criterion_type.capitalize()}:")
            print(matrix_df.to_string(float_format="%.2f"))  # Print DataFrame
            plot_confusion_matrix(matrix_df, criterion_type)

    # Analyze errors
    errors_df, error_categories, error_examples = analyze_errors(annotations_df.copy())
    print("\n=== Error Analysis ===")
    # print(f"Error counts by category (placeholder): {[len(v) for k, v in error_categories.items()]}")
    if not errors_df.empty:
        print(f"Number of classification errors: {len(errors_df)}")
        # print("Sample errors:")
        # print(errors_df.head()) # Display some errors if needed

    # Aggregate to trial level
    trial_scores = aggregate_criterion_predictions(annotations_df.copy())
    print("\n=== Trial-Level Aggregated Scores (Example) ===")
    # Print a few examples of trial scores
    # for patient_id, trials in list(trial_scores.items())[:2]:
    #    print(f"Patient ID: {patient_id}")
    #    for trial_id, scores in list(trials.items())[:2]:
    #        print(f"  Trial ID: {trial_id}, Scores: {scores}")

    # Evaluate ranking (placeholder)
    if ranking_results_file:
        # Placeholder: Load ground_truth_rankings from ranking_results_file
        ground_truth_rankings_data = {}  # Replace with actual loading
        ranking_results = evaluate_trial_ranking(
            trial_scores, ground_truth_rankings_data
        )
        print("\n=== Trial Ranking Performance (Placeholder) ===")
        print(f"Results: {ranking_results}")

    pipeline_results = {
        "explanation_results": explanation_results,
        "explanation_results_by_type": explanation_results_by_type,
        "sentence_identification_results": sentence_results,
        "eligibility_classification_accuracy": accuracy,
        "eligibility_accuracy_by_type": accuracy_by_type,
        "eligibility_confusion_matrices": confusion_matrices,
        "trial_aggregation_scores": trial_scores,
        # "error_analysis": {"errors_df": errors_df.to_dict(orient='records'), "categories": error_categories} # errors_df can be large
    }
    # Limit what goes into JSON if some parts are too verbose or not primary metrics
    # For example, full error_df might be too large for summary JSON.

    return pipeline_results


# --- Utility functions (not part of the main pipeline execution here but provided in original text) ---
def convert_dataset_to_jsonl(dataset_df, output_file):
    """
    Convert the dataset (Pandas DataFrame) to JSONL format
    """
    with open(output_file, "w") as f:
        for _, row in dataset_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\\n")
    print(f"Converted dataset saved to {output_file}")


def load_your_dataset(file_path):
    """
    Load your specific dataset format (example placeholder)
    """
    # df = pd.read_csv(file_path)
    # df = pd.read_json(file_path)
    # df = df.rename(columns={'your_column': 'annotation_id', ...})
    print(f"Placeholder: Load your dataset from {file_path}")
    return pd.DataFrame()


if __name__ == "__main__":
    print("Starting TrialGPT Evaluation Pipeline Script...")

    # Replace with your actual file paths
    # Ensure these files exist and are in the expected format.
    # annotations_file is expected to contain 'expert_*' columns, 'gpt4_*' columns,
    # 'explanation_correctness', 'criterion_type', 'patient_id', 'trial_id'.
    annotations_file = "dataset/trial_info.json"  # Example: path/to/your_comprehensive_annotations.jsonl

    # matching_results_file is loaded but its content is not explicitly merged into annotations_df
    # by default in this version of the script. The script assumes gpt4_* columns are in annotations_file.
    # If you need to use this file to provide gpt4_* columns, you'll need to implement
    # a merge step within run_evaluation_pipeline.
    matching_results_file = "results/matching_results.json"  # Example: results/matching_output.json (optional)

    # ranking_results_file is for ground truth trial rankings, currently a placeholder
    ranking_results_file = "results/ranking_ground_truth.json"  # Example: path/to/ranking_ground_truth.json (optional)

    # Check if placeholder files exist, create dummy ones for basic script run if not.
    # This is for demonstration so the script can run end-to-end with placeholders.
    # In a real scenario, these files must be actual data.
    if not os.path.exists(annotations_file):
        print(
            f"Warning: Annotations file '{annotations_file}' not found. Creating a dummy file for demonstration."
        )
        dummy_annotations_data = [
            {
                "annotation_id": 1,
                "patient_id": "p1",
                "trial_id": "t1",
                "criterion_type": "inclusion",
                "expert_sentences": "[0,1]",
                "gpt4_sentences": "[0,2]",
                "expert_eligibility": "included",
                "gpt4_eligibility": "included",
                "explanation_correctness": "Correct",
            },
            {
                "annotation_id": 2,
                "patient_id": "p1",
                "trial_id": "t1",
                "criterion_type": "inclusion",
                "expert_sentences": "[3]",
                "gpt4_sentences": "[]",
                "expert_eligibility": "included",
                "gpt4_eligibility": "not included",
                "explanation_correctness": "Incorrect",
            },
            {
                "annotation_id": 3,
                "patient_id": "p1",
                "trial_id": "t1",
                "criterion_type": "exclusion",
                "expert_sentences": "[4,5]",
                "gpt4_sentences": "[5]",
                "expert_eligibility": "excluded",
                "gpt4_eligibility": "not excluded",
                "explanation_correctness": "Partially Correct",
            },  # expert says excluded, gpt says not_excluded from this criterion
        ]
        with open(annotations_file, "w") as f:
            for record in dummy_annotations_data:
                f.write(json.dumps(record) + "\\n")

    # Optional: create dummy matching_results_file if needed for a particular workflow
    # if not os.path.exists(matching_results_file):
    #    print(f"Warning: Matching results file '{matching_results_file}' not found. This file is optional if annotations_file is comprehensive.")
    # with open(matching_results_file, 'w') as f: json.dump([], f)

    final_results = run_evaluation_pipeline(
        annotations_file,
        matching_results_file=(
            matching_results_file if os.path.exists(matching_results_file) else None
        ),
        ranking_results_file=(
            ranking_results_file if os.path.exists(ranking_results_file) else None
        ),
    )

    if final_results:
        # Convert numpy types in results for JSON serialization
        results_serializable = convert_numpy_types(final_results)

        output_results_file = "results/evaluation_results.json"
        try:
            with open(output_results_file, "w") as f:
                json.dump(results_serializable, f, indent=2)
            print(f"\nEvaluation results saved to {output_results_file}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")
    else:
        print("Evaluation pipeline did not produce results.")

    print("\nEvaluation Pipeline Script finished.")
