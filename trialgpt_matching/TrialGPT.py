__author__ = "qiao"

"""
TrialGPT-Matching main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

from anthropic import Anthropic

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def parse_criteria(criteria):
    output = ""
    criteria = criteria.split("\n\n")

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

        output += f"{idx}. {criterion}\n"
        idx += 1

    return output


def print_trial(
    trial_info: dict,
    inc_exc: str,
) -> str:
    """Given a dict of trial information, returns a string of trial."""

    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}\n"

    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % parse_criteria(
            trial_info["inclusion_criteria"]
        )
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % parse_criteria(
            trial_info["exclusion_criteria"]
        )

    return trial


def get_matching_prompt(
    trial_info: dict,
    inc_exc: str,
    patient: str,
) -> str:
    """Output the prompt."""
    prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

    if inc_exc == "inclusion":
        prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    elif inc_exc == "exclusion":
        prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

    prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
    prompt += f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
    prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n"
    prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "

    if inc_exc == "inclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
    elif inc_exc == "exclusion":
        prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'

    prompt += "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}."

    user_prompt = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n\n"
    user_prompt += (
        f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n\n"
    )
    user_prompt += f"Plain JSON output:"

    return prompt, user_prompt


def get_trialgpt_matching_result(
    messages,
    model,
    max_tokens=256,
    temperature=0.0,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=None,
):
    """Get the TrialGPT-Matching result using the Anthropic API."""

    # Map OpenAI model names to Anthropic model names
    if model == "gpt-4-turbo":
        anthropic_model = "claude-3-opus-20240229"
    elif model == "gpt-35-turbo":
        anthropic_model = "claude-3-sonnet-20240229"
    else:
        # Default to the provided model name or raise an error
        anthropic_model = model

    try:
        # Separate system prompt if present, as Anthropic handles it differently
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
            system=system_prompt,  # Pass system prompt via the 'system' parameter
            messages=[  # Ensure messages are in the correct format
                {"role": m["role"], "content": m["content"]} for m in user_prompts
            ],
            # top_p is supported
            top_p=top_p,
            # stop_sequences is supported
            stop_sequences=stop_sequences if stop_sequences else [],
            # frequency_penalty and presence_penalty are not directly supported in the same way.
            # These would require different handling or custom logic if essential.
        )
        # Anthropic response structure
        response_text = completion.content[0].text
        response_text = response_text.strip("`").strip(
            "json"
        )  # Keep existing stripping
        return response_text
    except Exception as e:
        print(f"Error in Anthropic API call: {e}")
        # Existing error message was `message = ""` then `message.strip()...`
        # which would fail. Returning empty string directly.
        return ""


def trialgpt_matching(trial: dict, patient: str, model: str):
    results = {}

    # doing inclusions and exclusions in separate prompts
    for inc_exc in ["inclusion", "exclusion"]:
        # get_matching_prompt expects trial_info, inc_exc, patient
        # The 'trial' object passed to this function is the trial_info
        system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        message = get_trialgpt_matching_result(
            messages=messages, model=model, temperature=0.0
        )

        try:
            results[inc_exc] = json.loads(message)
        except:
            results[inc_exc] = message  # Store raw message if JSON parsing fails

    return results


# Placeholder for where 'trial', 'patient', and 'model' would be defined or passed
# For example:
# if __name__ == '__main__':
#     example_trial_info = {} # Populate with actual trial info
#     example_patient_info = "" # Populate with actual patient info
#     example_model_name = "gpt-4-turbo" # This will be mapped to claude-3-opus
#
#     # Need to assign to global 'trial', 'patient', 'model' if main() expects them as globals
#     # Or, better, pass them as arguments to main()
#     # For now, assuming they are somehow available in the scope main() is called from.
#     # trial = example_trial_info
#     # patient = example_patient_info
#     # model = example_model_name
#     # output = main()
#     # print(json.dumps(output, indent=4))
