__author__ = "qiao"

"""
TrialGPT-Matching main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os

from anthropic import Anthropic

# Assuming client.py is in the parent directory relative to trial_mcp_matching folder
# Adjust if your project structure is different.
import sys

# Get the directory of the current file (TrialMCP.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (trial_mcp_matching)
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (project root where client.py is expected)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from client import MCPClient


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


async def get_trialgpt_matching_result(
    mcp_client: MCPClient,  # Added MCPClient
    system_prompt_text: str,  # For clarity, passing system and user prompts separately
    user_prompt_text: str,
    # model, max_tokens, temperature, etc. are now handled by MCPClient's Claude call
    # or would need to be passed to process_query if MCPClient supports overriding them
):
    """Get the TrialGPT-Matching result using the MCPClient."""

    try:
        # The MCPClient's process_query will internally use its configured model and system prompt.
        # The system_prompt in MCPClient is general. Here, the 'system_prompt_text'
        # is more like specific instructions for this task. We'll prepend it to the user query.
        full_query = (
            f"{system_prompt_text}\n\n{user_prompt_text}"  # Corrected extra newline
        )

        # The MCPClient's process_query should handle the interaction with Claude, including tools.
        # For this specific matching task, it's primarily an LLM call.
        response_text = await mcp_client.process_query(full_query)

        # process_query is expected to return the final text string from Claude.
        # We need to ensure the stripping logic is still valid.
        response_text = response_text.strip("`").strip("json")
        return response_text
    except Exception as e:
        print(f"Error in MCPClient process_query call: {e}")
        return ""


async def trialgpt_matching(
    mcp_client: MCPClient, trial: dict, patient: str, model: str
):  # Added mcp_client, made async
    results = {}
    # MAX_MATCHING_TOKENS is not directly passed to MCPClient here.
    # It's assumed MCPClient's Claude call has appropriate token limits.

    # doing inclusions and exclusions in separate prompts
    for inc_exc in ["inclusion", "exclusion"]:
        system_prompt, user_prompt = get_matching_prompt(trial, inc_exc, patient)

        message = await get_trialgpt_matching_result(
            mcp_client=mcp_client,
            system_prompt_text=system_prompt,
            user_prompt_text=user_prompt,
            # model and other params are handled by MCPClient's configuration
        )

        try:
            results[inc_exc] = json.loads(message)
        except json.JSONDecodeError as e:  # More specific exception
            trial_id_for_log = trial.get(
                "NCTID", trial.get("nct_id", "UnknownNCTID")
            )  # Attempt to get trial ID for logging
            print(
                f"Warning: JSONDecodeError for {inc_exc} criteria in trial {trial_id_for_log}. Error: {e}. Raw message snippet: {message[:200]}..."
            )
            results[inc_exc] = {
                "error_json_parsing": str(e),
                "raw_response_snippet": message[:200],
            }  # Store an error dict
        except Exception as e_gen:  # Catch any other unexpected error during parsing
            trial_id_for_log = trial.get("NCTID", trial.get("nct_id", "UnknownNCTID"))
            print(
                f"Warning: Unexpected error parsing {inc_exc} criteria for trial {trial_id_for_log}. Error: {e_gen}. Raw message snippet: {message[:200]}..."
            )
            results[inc_exc] = {
                "error_unexpected_parsing": str(e_gen),
                "raw_response_snippet": message[:200],
            }

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
