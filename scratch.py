import json
import pandas as pd


def load_criterion_annotations(file_path):
    """
    Load the ground truth dataset containing criterion-level annotations
    """
    with open(file_path, "r") as f:
        fp = json.load(f)
    print(fp["NCT05817643"].keys(), fp["NCT05817643"])
    # df = pd.DataFrame(fp).transpose()
    # print(f"Loaded {len(df)} criterion-level annotations from {file_path}")
    # return df


if __name__ == "__main__":
    load_criterion_annotations("dataset/trial_info.json")
