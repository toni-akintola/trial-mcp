import json
import os
import requests
import argparse
import uvicorn

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request

# Assuming SseServerTransport is part of your mcp library or a compatible library
# If this path is incorrect, please adjust it.
from mcp.server.sse import SseServerTransport

from transformers import (
    AutoTokenizer,
    pipeline as hf_pipeline,
)  # Renamed to avoid conflict
import torch  # Transformers often requires torch

# Initialize MCP Server
mcp = FastMCP()

BERT_MODEL = "pabRomero/BioClinicalBERT-full-finetuned-ner-pablo"

# --- Hugging Face Model Initialization ---
# Load once to be reused by the tool.
# Note: The first time this runs, it will download the model (can be several hundred MBs).
try:
    print("Loading Bio_ClinicalBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    print("Loading Bio_ClinicalBERT model for token classification pipeline...")
    # Using a pipeline for token classification.
    # The base emilyalsentzer/Bio_ClinicalBERT model might not have a readily available NER head
    # for the pipeline, or might require specific configuration if it does.
    # If this specific model doesn't work well out-of-the-box in the pipeline for NER,
    # consider using a model fine-tuned for clinical NER, e.g., "d4data/biomedical-ner-all"
    # or fine-tuning this base model on your NER task.
    ner_pipeline = hf_pipeline(
        "token-classification",
        model=BERT_MODEL,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    print("Bio_ClinicalBERT NER pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading Bio_ClinicalBERT model or creating pipeline: {e}")
    print(
        "Proceeding with NER pipeline as None. bio_ner_map will use placeholder logic."
    )
    ner_pipeline = None


# --- Placeholder/Stub Implementations ---
def bio_ner_map(text: str) -> list[str]:
    """Placeholder for BioClinicalBERT call + OMOP mapping."""
    print(f"Bio_ClinicalBERT bio_ner_map received text: '{text[:50]}...'")
    if ner_pipeline is None:
        print(
            "NER pipeline not available. Using fallback placeholder logic for OMOP codes."
        )
        if "headache" in text.lower():
            return [
                "OMOP:HeadacheCode",
                "OMOP:FallbackDiabetesCode",
            ]  # Differentiated for clarity
        if "diabetes" in text.lower():
            return ["OMOP:FallbackDiabetesCode"]
        return ["OMOP:FallbackGenericSymptomCode"]

    try:
        print("Running NER pipeline...")
        entities = ner_pipeline(text)
        print(f"Extracted entities: {entities}")

        omop_codes = []
        if not entities:
            omop_codes.append("OMOP:NoEntitiesFound")

        for entity in entities:
            entity_text = entity.get("word", "").lower()
            entity_group = entity.get(
                "entity_group", ""
            ).lower()  # e.g., 'DISEASE', 'SYMPTOM'

            # Placeholder mapping logic:
            # This needs to be a sophisticated mapping in a real scenario.
            if "headache" in entity_text:
                omop_codes.append("OMOP:HeadacheCode")
            elif "diabetes" in entity_text:
                omop_codes.append("OMOP:DiabetesCode")
            elif (
                entity_group == "problem" or entity_group == "disease"
            ):  # Example entity groups
                omop_codes.append(f"OMOP:Mapped_{entity_text.replace(' ', '_')}")
            else:
                omop_codes.append(f"OMOP:Unmapped_{entity_text.replace(' ', '_')}")

        if not omop_codes:  # If entities were found but none mapped
            omop_codes.append("OMOP:NoMatchingCodes")

        # Remove duplicates
        omop_codes = sorted(list(set(omop_codes)))
        return omop_codes if omop_codes else ["OMOP:DefaultCodeIfEmpty"]

    except Exception as e:
        print(f"Error during NER processing or OMOP mapping: {e}")
        return ["OMOP:ErrorInProcessing"]


class MockFabricClient:
    """Mock Hyperledger Fabric Client."""

    def __init__(self, net_profile: str | None = None):
        print(f"MockFabricClient initialized with profile: {net_profile}")

    def get_peer(self, peer_name: str):
        print(f"MockFabricClient: Getting peer: {peer_name}")
        return peer_name  # Return a mock peer object or name

    def chaincode_invoke(
        self,
        requestor: str,
        channel_name: str,
        peers: list[str],
        cc_name: str,
        fcn: str,
        args: list[str],
        wait_for_event: bool = False,
    ) -> str:
        print(
            f"MockFabricClient: Chaincode invoke called by {requestor} on channel {channel_name} for {cc_name}.{fcn}"
        )
        print(f"Args: {args}")
        return json.dumps({"status": "success", "transaction_id": "mock_tx_id_123"})


# --- Register MCP Tools ---
def register_all_tools(mcp_instance: FastMCP):
    @mcp_instance.tool(
        name="query_opentargets",
        description="Uses OpenTargets GraphQL API for systematic target–disease associations. target_ensembl_id: The Ensembl ID of the target.",
    )
    def query_opentargets(target_ensembl_id: str) -> dict:
        """
        Uses OpenTargets GraphQL API for systematic target–disease associations.
        target_ensembl_id: The Ensembl ID of the target.
        """
        api_url = "https://api.platform.opentargets.org/api/v4/graphql"
        # GraphQL query to get diseases associated with a target
        graphql_query = """
            query AssociatedDiseasesForTarget($ensemblId: String!) {
              target(ensemblId: $ensemblId) {
                id
                approvedSymbol
                associatedDiseases {
                  rows {
                    disease {
                      id
                      name
                      therapeuticAreas {
                        id
                        name
                      }
                    }
                    score # Association score
                  }
                  count
                }
              }
            }
        """
        variables = {"ensemblId": target_ensembl_id}
        print(
            f"Tool: query_opentargets called with target_ensembl_id: {target_ensembl_id}"
        )
        try:
            response = requests.post(
                api_url, json={"query": graphql_query, "variables": variables}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenTargets GraphQL API: {e}")
            return {"error": str(e), "target_ensembl_id": target_ensembl_id}
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON from OpenTargets API: {e}. Response text: {response.text}"
            )
            return {
                "error": f"JSONDecodeError: {str(e)}",
                "response_text": response.text[:500],
            }

    @mcp_instance.tool(
        name="symptom_to_biomarker",
        description="Processes clinical text using Bio_ClinicalBERT to extract entities, maps them to OMOP codes, and enriches with OpenTargets data.",
    )
    def symptom_to_biomarker(text: str) -> dict:
        """
        Wraps BioClinicalBERT for deep NER, maps to OMOP/CDS codes,
        then augments via OpenTargets.
        """
        print(f"Tool: symptom_to_biomarker called with text: '{text[:50]}...'")
        codes = bio_ner_map(text)

        ot_info_list = []
        if (
            codes
            and codes[0]
            and not codes[0].startswith("OMOP:Error")
            and not codes[0].startswith("OMOP:No")
        ):
            ot_info_list = query_opentargets(codes[0])

        return {
            "codes": codes,
            "opentargets_info": ot_info_list,
        }

    @mcp_instance.tool(
        name="query_clinicaltrials",
        description="Queries the ClinicalTrials.gov v2 API for studies matching the search_expression. Fetches specified fields and limits results by page_size.",
    )
    def query_clinicaltrials(
        search_expression: str, fields: list[str], page_size: int = 10
    ) -> dict:
        """
        Queries the ClinicalTrials.gov v2 API for studies matching the search_expression.
        Fetches specified fields and limits results by page_size.
        """
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": search_expression,
            "fields": ",".join(fields),
            "pageSize": page_size,
            # "pageToken" can be added here if pagination beyond the first page is needed
        }
        print(
            f"Tool: query_clinicaltrials called with expression: '{search_expression}', fields: {fields}, pageSize: {page_size}"
        )
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling ClinicalTrials.gov API: {e}")
            return {"error": str(e), "search_expression": search_expression}
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON from ClinicalTrials.gov API: {e}. Response text: {response.text}"
            )
            return {
                "error": f"JSONDecodeError: {str(e)}",
                "response_text": response.text[:500],
            }  # Return snippet of response


# --- Starlette App Creation ---
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        # Type casting _send to Any to satisfy SseServerTransport if it has stricter typing not known here
        from typing import Any

        _send_any: Any = request._send

        async with sse.connect_sse(request.scope, request.receive, _send_any) as (
            read_stream,
            write_stream,
        ):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),  # Assuming this method exists on Server
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),  # Corrected Mount path
        ],
    )


# --- Main Execution ---
if __name__ == "__main__":
    load_dotenv()

    mcp = FastMCP("TrialMCP Server Gateway")
    register_all_tools(mcp)

    mcp_server = mcp._mcp_server  # noqa: WPS437

    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    starlette_app = create_starlette_app(mcp_server, debug=True)

    print(f"Starting MCP SSE Server on http://{args.host}:{args.port}")
    uvicorn.run(starlette_app, host=args.host, port=args.port)
