__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time
import os
import sys

from anthropic import Anthropic

# Adjust sys.path to find client.py in the project root
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_file_dir))  # Up two levels
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from client import MCPClient  # Added MCPClient import

# Initialize Anthropic client (original, might be removed if MCPClient fully replaces)
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


async def get_trialgpt_ranking_result(  # Made async
    mcp_client: MCPClient,  # Added mcp_client
    system_prompt_text: str,
    user_prompt_text: str,
    # model, max_tokens, temperature are now handled by MCPClient or its internal call
):
    """Get the TrialGPT-Ranking result using the MCPClient."""
    anthropic_model = (
        "claude-3-opus-20240229"  # Defaulting, MCPClient will use its own config
    )

    try:
        # Combine system and user prompts for MCPClient's process_query
        full_query = f"{system_prompt_text}\\n\\n{user_prompt_text}"

        response_text = await mcp_client.process_query(full_query)

        # Assuming process_query returns the text content directly.
        # Stripping of backticks and "json" will be handled in trialgpt_aggregation
        return response_text
    except Exception as e:
        print(f"Error in MCPClient process_query for ranking: {e}")
        return ""


async def trialgpt_aggregation(  # Made async
    mcp_client: MCPClient,  # Added mcp_client
    patient_note: str,
    trial_results: dict,
    trial_info: dict,
    model: str,  # Kept model, though MCPClient uses its own config
):
    criteria_pred_string = convert_criteria_pred_to_string(trial_results, trial_info)

    system_prompt, user_prompt = get_ranking_llm_prompts(
        patient_note, criteria_pred_string, trial_info
    )

    # No longer constructing messages list for direct Anthropic call
    message_content = await get_trialgpt_ranking_result(
        mcp_client=mcp_client,
        system_prompt_text=system_prompt,
        user_prompt_text=user_prompt,
        # model and other params are handled by MCPClient's configuration
    )

    message_content = message_content.strip("`").strip("json")

    try:
        return json.loads(message_content)
    except json.JSONDecodeError:
        # If JSON loading fails, return the raw string (original behavior)
        return message_content
