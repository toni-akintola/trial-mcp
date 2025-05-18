__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
from anthropic import Anthropic  # Will be replaced by MCPClient for API calls
import sys
import argparse
import asyncio  # Added asyncio

# Adjust sys.path to find client.py in the project root
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from client import MCPClient  # Added MCPClient import

# Original Anthropic client, can be removed if MCPClient fully replaces its use cases here.
# client = Anthropic(
# api_key=os.getenv("ANTHROPIC_API_KEY"),
# )


def get_keyword_generation_messages(note):
    system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
    prompt = f"Here is the patient description: \\n{note}\\n\\nJSON output:"
    # This function now just returns the system and user prompt strings
    return system, prompt


async def get_keyword_generation_result(  # Made async
    mcp_client: MCPClient,  # Added mcp_client
    system_prompt_text: str,
    user_prompt_text: str,
    # model, max_tokens, temperature are handled by MCPClient's process_query or its config
):
    """Get the keyword generation result using the MCPClient."""
    try:
        # Combine system and user prompts for MCPClient's process_query
        full_query = f"{system_prompt_text}\\n\\n{user_prompt_text}"

        response_text = await mcp_client.process_query(full_query)

        # JSON parsing and stripping will be handled by the caller
        return response_text
    except Exception as e:
        print(f"Error in MCPClient process_query for keyword generation: {e}")
        return ""  # Return empty or error string


async def main_async():  # Main logic moved to async function
    parser = argparse.ArgumentParser(
        description="Generate search keywords for patient notes using MCPClient."
    )
    parser.add_argument(
        "corpus", type=str, help="The corpus to use (e.g., trec_2021, trec_2022, sigir)"
    )
    # Model argument is less critical now as MCPClient configures the model, but can be kept for logging/metadata
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-3-opus-20240229",
        help="Model name for reference (actual model from MCPClient config).",
    )
    parser.add_argument(
        "--mcp_server_url",
        default=os.environ.get("MCP_SERVER_URL", "http://localhost:8080/sse"),
        help="URL of the MCP SSE server.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process. Processes all if not specified.",
    )
    args = parser.parse_args()

    corpus = args.corpus
    model_name_for_log = args.model_name  # Used for filenames/logging
    mcp_server_url = args.mcp_server_url
    sample_size = args.sample_size

    mcp_client = MCPClient()
    outputs = {}
    processed_count = 0

    try:
        await mcp_client.connect_to_sse_server(server_url=mcp_server_url)
        print(f"Successfully connected to MCP Server at {mcp_server_url}")

        queries_file_path = f"dataset/{corpus}/queries.jsonl"
        if not os.path.exists(queries_file_path):
            print(f"Error: Queries file not found at {queries_file_path}")
            sys.exit(1)

        with open(queries_file_path, "r") as f:
            lines = f.readlines()
            total_lines = len(lines)
            lines_to_process = lines[:sample_size] if sample_size is not None else lines

            for i, line in enumerate(lines_to_process):
                entry = json.loads(line)
                current_processing_count = i + 1
                print(
                    f"Processing patient ID: {entry['_id']} ({current_processing_count}/{len(lines_to_process)} of sample, {current_processing_count}/{total_lines} total)..."
                )

                system_prompt, user_prompt = get_keyword_generation_messages(
                    entry["text"]
                )

                message_content = await get_keyword_generation_result(
                    mcp_client=mcp_client,
                    system_prompt_text=system_prompt,
                    user_prompt_text=user_prompt,
                )

                try:
                    if message_content.startswith(
                        "```json\n"
                    ) and message_content.endswith("```"):
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
                    print(f"Received content snippet: {message_content[:200]}")
                    outputs[entry["_id"]] = {
                        "error": "Failed to decode JSON from LLM",
                        "raw_content": message_content,
                    }
                except Exception as e:
                    print(f"An unexpected error for patient ID {entry['_id']}: {e}")
                    outputs[entry["_id"]] = {
                        "error": str(e),
                        "raw_content": message_content,
                    }

                processed_count += 1  # Increment after processing logic

                # Save incrementally
                # Using model_name_for_log in filename
                output_file_suffix = f"_sample{sample_size}" if sample_size else ""
                output_file_path = f"results/retrieval_keywords_mcp_{model_name_for_log}_{corpus}{output_file_suffix}.json"
                with open(output_file_path, "w") as out_f:
                    json.dump(outputs, out_f, indent=4)
                if (
                    current_processing_count % 10 == 0
                    or current_processing_count == len(lines_to_process)
                ):  # Log save less frequently
                    print(f"Incrementally saved to {output_file_path}")

        final_output_file_path = f"results/retrieval_keywords_mcp_{model_name_for_log}_{corpus}{f'_sample{sample_size}' if sample_size else ''}.json"
        with open(final_output_file_path, "w") as f:
            json.dump(outputs, f, indent=4)
        print(f"Finished keyword generation. Total processed: {processed_count}.")
        print(f"Results saved to {final_output_file_path}")

    except ConnectionRefusedError:
        print(
            f"Error: Connection refused. Is the MCP server running at {mcp_server_url}?"
        )
    except Exception as e:
        print(f"An unexpected error occurred in main_async: {e}")
    finally:
        if "mcp_client" in locals() and mcp_client.session:
            print("Cleaning up MCPClient session...")
            await mcp_client.cleanup()
            print("MCPClient session cleaned up.")


if __name__ == "__main__":
    asyncio.run(main_async())
