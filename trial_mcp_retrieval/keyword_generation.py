__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
from anthropic import Anthropic
import sys
import argparse  # Added for argument parsing

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
    parser = argparse.ArgumentParser(
        description="Generate search keywords for patient notes."
    )
    parser.add_argument(
        "corpus", type=str, help="The corpus to use (e.g., trec_2021, trec_2022, sigir)"
    )
    parser.add_argument(
        "model", type=str, help="The model name to use for keyword generation"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process from the queries file. Processes all if not specified.",
    )

    args = parser.parse_args()

    corpus = args.corpus
    model = args.model
    sample_size = args.sample_size

    outputs = {}
    processed_count = 0

    queries_file_path = f"dataset/{corpus}/queries.jsonl"
    if not os.path.exists(queries_file_path):
        print(f"Error: Queries file not found at {queries_file_path}")
        sys.exit(1)

    with open(queries_file_path, "r") as f:
        for line in f.readlines():
            if sample_size is not None and processed_count >= sample_size:
                print(f"Reached sample size of {sample_size}. Stopping.")
                break

            entry = json.loads(line)
            print(
                f"Processing patient ID: {entry['_id']} ({processed_count + 1}{f'/{sample_size}' if sample_size else ''})..."
            )
            messages = get_keyword_generation_messages(entry["text"])

            # New call using the Anthropic wrapper function
            message_content = get_keyword_generation_result(
                messages=messages, model=model, temperature=0.0
            )

            try:
                # Ensure the output is valid JSON before attempting to load
                # The original code stripped backticks and "json", which can be brittle.
                # A more robust approach is to find the JSON block if it's embedded.
                if message_content.startswith("```json\n"):
                    json_str = message_content[len("```json\n") : -len("```")]
                elif message_content.startswith("```") and message_content.endswith(
                    "```"
                ):
                    json_str = message_content[3:-3]
                else:
                    json_str = message_content

                outputs[entry["_id"]] = json.loads(json_str.strip())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for patient ID {entry['_id']}: {e}")
                print(f"Received content: {message_content}")
                outputs[entry["_id"]] = {
                    "error": "Failed to decode JSON from LLM",
                    "raw_content": message_content,
                }
            except Exception as e:
                print(
                    f"An unexpected error occurred for patient ID {entry['_id']}: {e}"
                )
                outputs[entry["_id"]] = {
                    "error": str(e),
                    "raw_content": message_content,
                }

            processed_count += 1

            # Save incrementally, especially useful for long runs or sampling
            output_file_path = f"results/retrieval_keywords_{model}_{corpus}{f'_sample{sample_size}' if sample_size else ''}.json"
            with open(output_file_path, "w") as out_f:
                json.dump(outputs, out_f, indent=4)
            print(f"Incrementally saved to {output_file_path}")

    final_output_file_path = f"results/retrieval_keywords_{model}_{corpus}{f'_sample{sample_size}' if sample_size else ''}.json"
    # Final save (might be redundant if sample_size causes early exit, but good for clarity)
    with open(final_output_file_path, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Finished keyword generation. Total processed: {processed_count}.")
    print(f"Results saved to {final_output_file_path}")
