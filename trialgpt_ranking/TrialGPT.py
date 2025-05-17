__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def convert_criteria_pred_to_string(
    prediction: dict,
    trial_info: dict,
) -> str:
    """Given the TrialGPT prediction, output the linear string of the criteria."""
    output = ""

    for inc_exc in ["inclusion", "exclusion"]:

        # first get the idx2criterion dict
        idx2criterion = {}
        criteria = trial_info[inc_exc + "_criteria"].split("\n\n")

        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()

            if (
                "inclusion criteria" in criterion.lower()
                or "exclusion criteria" in criterion.lower()
            ):
                continue

            if len(criterion) < 5:
                continue

            idx2criterion[str(idx)] = criterion
            idx += 1

        for idx, info in enumerate(prediction[inc_exc].items()):
            criterion_idx, preds = info

            if criterion_idx not in idx2criterion:
                continue

            criterion = idx2criterion[criterion_idx]

            if len(preds) != 3:
                continue

            output += f"{inc_exc} criterion {idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            if len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"

    return output


def convert_pred_to_prompt(
    patient: str,
    pred: dict,
    trial_info: dict,
) -> str:
    """Convert the prediction to a prompt string."""
    # get the trial string
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}"

    # then get the prediction strings
    pred = convert_criteria_pred_to_string(pred, trial_info)

    # construct the prompt
    prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
    prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
    prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
    prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'

    user_prompt = "Here is the patient note:\n"
    user_prompt += patient + "\n\n"
    user_prompt += "Here is the clinical trial description:\n"
    user_prompt += trial + "\n\n"
    user_prompt += "Here are the criterion-level eligibility prediction:\n"
    user_prompt += pred + "\n\n"
    user_prompt += "Plain JSON output:"

    return prompt, user_prompt


def get_trialgpt_ranking_result(
    messages,
    model,
    max_tokens=1024,  # Default from OpenAI, adjust if needed
    temperature=0.0,
    # Anthropic doesn't use top_p, frequency_penalty, presence_penalty in the same way for messages.create
):
    """Get the TrialGPT-Ranking result using the Anthropic API."""

    # Map OpenAI model names to Anthropic model names
    if model == "gpt-4-turbo":
        anthropic_model = "claude-3-opus-20240229"
    elif model == "gpt-35-turbo":
        anthropic_model = "claude-3-sonnet-20240229"
    else:
        anthropic_model = model

    try:
        system_prompt = None
        user_prompts = []
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            user_prompts = messages[1:]
        else:
            user_prompts = messages

        completion = client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": m["role"], "content": m["content"]} for m in user_prompts
            ],
        )
        return completion.content[0].text
    except Exception as e:
        print(f"Error in Anthropic API call for ranking: {e}")
        return ""


def main(patient_note, criteria_pred_string, model):
    system_prompt, user_prompt = convert_pred_to_prompt(
        patient_note, criteria_pred_string, trial_info
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call the new Anthropic wrapper function
    message = get_trialgpt_ranking_result(
        messages=messages,
        model=model,
        temperature=0.0,  # Explicitly pass, as in original logic
    )

    message = message.strip("`").strip("json")

    try:
        return json.loads(message)
    except:
        return message
