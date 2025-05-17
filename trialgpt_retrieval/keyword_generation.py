__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
from anthropic import Anthropic
import sys

# Initialize Anthropic client
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def get_keyword_generation_messages(note):
    system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

    prompt = f"Here is the patient description: \n{note}\n\nJSON output:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    return messages


def get_keyword_generation_result(
    messages,
    model,
    max_tokens=1024,  # Default from OpenAI, adjust if needed for Anthropic
    temperature=0.0,
    # Anthropic doesn't use top_p, frequency_penalty, presence_penalty in the same way for messages.create
):
    """Get the keyword generation result using the Anthropic API."""

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
        print(f"Error in Anthropic API call for keyword generation: {e}")
        return ""  # Return empty or error string


if __name__ == "__main__":
    # the corpus: trec_2021, trec_2022, or sigir
    corpus = sys.argv[1]

    # the model index to use
    model = sys.argv[2]

    outputs = {}

    with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
        for line in f.readlines():
            entry = json.loads(line)
            messages = get_keyword_generation_messages(entry["text"])

            # New call using the Anthropic wrapper function
            message = get_keyword_generation_result(
                messages=messages, model=model, temperature=0.0
            )

            output = message.strip("`").strip("json")

            outputs[entry["_id"]] = json.loads(output)

            with open(f"results/retrieval_keywords_{model}_{corpus}.json", "w") as f:
                json.dump(outputs, f, indent=4)
