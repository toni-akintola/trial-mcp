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
        if inc_exc not in prediction or not isinstance(prediction[inc_exc], dict):
            output += f"Note: {inc_exc.capitalize()} criteria data is not in the expected dictionary format or is missing.\n"
            if inc_exc in prediction:
                output += f"  Raw {inc_exc} data: {str(prediction[inc_exc])[:200]}\n"  # Log a snippet of the raw data
            continue  # Skip processing for this inc_exc type

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


def get_ranking_llm_prompts(
    patient_note: str, criteria_pred_string: str, trial_info: dict
):
    """Prepares the system and user prompts for the ranking LLM call."""
    # get the trial string using parts of trial_info
    trial_description = f"Title: {trial_info.get('brief_title', '')}\n"
    trial_description += (
        f"Target conditions: {', '.join(trial_info.get('diseases_list', []))}\n"
    )
    trial_description += f"Summary: {trial_info.get('brief_summary', '')}"

    system_prompt = "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    system_prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
    system_prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
    system_prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
    system_prompt += 'Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'

    user_prompt = f"Here is the patient note:\n{patient_note}\n\n"
    user_prompt += f"Here is the clinical trial description:\n{trial_description}\n\n"
    user_prompt += f"Here are the criterion-level eligibility prediction:\n{criteria_pred_string}\n\n"
    user_prompt += "Plain JSON output:"
    return system_prompt, user_prompt


def get_trialgpt_ranking_result(
    messages,
    model,
    max_tokens=8192,  # Default from OpenAI, adjust if needed
    temperature=0.0,
    # Anthropic doesn't use top_p, frequency_penalty, presence_penalty in the same way for messages.create
):
    """Get the TrialGPT-Ranking result using the Anthropic API."""

    # Map OpenAI model names to Anthropic model names
    # if model == "gpt-4-turbo":
    #     anthropic_model = "claude-3-opus-20240229"
    # elif model == "gpt-35-turbo":
    #     anthropic_model = "claude-3-sonnet-20240229"
    # else:
    #     anthropic_model = model
    anthropic_model = model  # Use the model parameter directly

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


def trialgpt_aggregation(
    patient_note: str, trial_results: dict, trial_info: dict, model: str
):
    # Convert the structured trial_results (dict) from matching phase into a string representation
    criteria_pred_string = convert_criteria_pred_to_string(trial_results, trial_info)

    # Get prompts for the LLM call, now passing trial_info as well for full context if needed by get_ranking_llm_prompts
    system_prompt, user_prompt = get_ranking_llm_prompts(
        patient_note, criteria_pred_string, trial_info
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    message_content = get_trialgpt_ranking_result(
        messages=messages, model=model, temperature=0.0
    )

    # Stripping is now expected to be done by the caller if needed,
    # or ensure get_trialgpt_ranking_result does it consistently.
    # For now, retaining the stripping here as per original structure for this function.
    message_content = message_content.strip("`").strip("json")

    try:
        return json.loads(message_content)
    except json.JSONDecodeError:
        # If JSON loading fails, return the raw string (original behavior)
        return message_content
