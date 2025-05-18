import asyncio
import json
import os
from typing import Optional, Any
from contextlib import AsyncExitStack
from datetime import datetime
from mcp import ClientSession
from mcp.client.sse import sse_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env
CLAUDE_MODEL = "claude-3-7-sonnet-latest"


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    @staticmethod
    def _unwrap_mcp_content(data: Any) -> Any:
        if hasattr(data, "payload"):  # E.g., mcp.JSONContent
            # Recursively unwrap the payload
            return MCPClient._unwrap_mcp_content(data.payload)
        elif hasattr(data, "text") and not isinstance(
            data, str
        ):  # E.g., mcp.TextContent but not a plain string already
            # Get the text and recursively unwrap (in case text itself could be structured, though unlikely for TextContent)
            return MCPClient._unwrap_mcp_content(data.text)
        elif isinstance(data, list):
            return [MCPClient._unwrap_mcp_content(item) for item in data]
        elif isinstance(data, dict):
            return {
                key: MCPClient._unwrap_mcp_content(value) for key, value in data.items()
            }
        else:
            # Base case: data is already a primitive (str, int, float, bool, None) or not an MCP content wrapper
            return data

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        system_prompt = f"You are a helpful assistant. Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        initial_user_query = query  # Store for later check
        messages = [{"role": "user", "content": query}]

        # list_tools_response is the MCP response object, not an LLM response
        list_tools_response = await self.session.list_tools()
        available_tools_for_anthropic = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in list_tools_response.tools
        ]

        last_text_response = ""

        # Loop to handle potential multiple tool calls and final text response
        while True:
            # Initial/current Claude API call
            current_llm_response = self.anthropic.messages.create(
                system=system_prompt,
                model=CLAUDE_MODEL,
                max_tokens=15000,  # Consider making this configurable
                messages=messages,
                tools=available_tools_for_anthropic,
            )

            assistant_action_taken = False  # Flag to check if assistant did anything other than just text/stop

            # The assistant's response (current_llm_response.content) can have multiple blocks
            # (e.g., text followed by tool_use, or multiple tool_uses)
            # We need to prepare the *entire* assistant message for history if tool use occurs.
            assistant_message_content_for_history = []

            new_tool_requests_in_this_turn = []

            for content_block in current_llm_response.content:
                if content_block.type == "text":
                    last_text_response = (
                        content_block.text
                    )  # Capture the latest text output
                    assistant_message_content_for_history.append(
                        {"type": "text", "text": content_block.text}
                    )
                elif content_block.type == "tool_use":
                    assistant_action_taken = True  # A tool is being requested
                    # Add to assistant message for history
                    assistant_message_content_for_history.append(
                        {
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input,
                        }
                    )
                    # Store this tool request to be processed
                    new_tool_requests_in_this_turn.append(content_block)

            # If there were tool requests in the assistant's last turn
            if new_tool_requests_in_this_turn:
                # First, add the complete assistant message (that contained the tool requests) to history
                if assistant_message_content_for_history:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message_content_for_history,
                        }
                    )

                tool_results_for_next_llm_call = []
                for tool_request_block in new_tool_requests_in_this_turn:
                    tool_name = tool_request_block.name
                    tool_args = tool_request_block.input
                    print(
                        f"[MCPClient is executing tool: {tool_name} with args: {json.dumps(tool_args)}]"
                    )

                    mcp_tool_result = await self.session.call_tool(tool_name, tool_args)

                    raw_tool_data = None
                    if hasattr(mcp_tool_result.content, "payload"):
                        raw_tool_data = mcp_tool_result.content.payload
                    elif hasattr(mcp_tool_result.content, "text") and not isinstance(
                        mcp_tool_result.content, str
                    ):
                        raw_tool_data = mcp_tool_result.content.text
                    elif mcp_tool_result.content is not None:
                        raw_tool_data = mcp_tool_result.content
                    else:
                        raw_tool_data = ""

                    unwrapped_tool_data = MCPClient._unwrap_mcp_content(raw_tool_data)

                    tool_output_content_str = ""
                    if isinstance(unwrapped_tool_data, (dict, list)):
                        try:
                            tool_output_content_str = json.dumps(unwrapped_tool_data)
                        except TypeError as e:
                            print(
                                f"Warning: Could not serialize unwrapped tool data ({type(unwrapped_tool_data)}) for {tool_name} to JSON: {e}. Using str()."
                            )
                            tool_output_content_str = str(unwrapped_tool_data)
                    elif unwrapped_tool_data is not None:
                        tool_output_content_str = str(unwrapped_tool_data)

                    tool_results_for_next_llm_call.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_request_block.id,
                            "content": tool_output_content_str,
                        }
                    )

                if tool_results_for_next_llm_call:
                    messages.append(
                        {"role": "user", "content": tool_results_for_next_llm_call}
                    )

                # After adding tool results, if the original query suggested a direct JSON output,
                # add a message to re-emphasize this before the next LLM call.
                if (
                    "JSON output:" in initial_user_query
                    or "MUST be ONLY the JSON" in system_prompt
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": "Thank you for the tool results. Now, please provide ONLY the JSON output as originally requested, based on all the information gathered.",
                        }
                    )

                continue
            else:
                # No tool calls were made in the assistant's last turn.
                # This means the assistant's response was text-only, or it stopped.
                # The last_text_response should hold the final answer.
                break

        return last_text_response

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>"
        )
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
