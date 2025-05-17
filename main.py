import json
import os
import requests
from mcp.server.fastmcp import FastMCP
from transformers import (
    AutoTokenizer,
    pipeline as hf_pipeline,
)  # Renamed to avoid conflict
import torch  # Transformers often requires torch

# Placeholder for API keys - In a real scenario, use environment variables or a secrets manager
IRB_API_TOKEN = os.getenv("IRB_API_TOKEN", "your_irb_api_token_here")
DRUGBANK_KEY = os.getenv("DRUGBANK_KEY", "your_drugbank_key_here")

# Initialize MCP Server
mcp = FastMCP()

# --- Hugging Face Model Initialization ---
# Load once to be reused by the tool.
# Note: The first time this runs, it will download the model (can be several hundred MBs).
try:
    print("Loading Bio_ClinicalBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    print("Loading Bio_ClinicalBERT model for token classification pipeline...")
    # Using a pipeline for token classification.
    # The base emilyalsentzer/Bio_ClinicalBERT model might not have a readily available NER head
    # for the pipeline, or might require specific configuration if it does.
    # If this specific model doesn't work well out-of-the-box in the pipeline for NER,
    # consider using a model fine-tuned for clinical NER, e.g., "d4data/biomedical-ner-all"
    # or fine-tuning this base model on your NER task.
    ner_pipeline = hf_pipeline(
        "token-classification",
        model="emilyalsentzer/Bio_ClinicalBERT",
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


# --- MCP Tool Definitions ---


@mcp.tool(
    name="enrich_with_drugbank",
    description="Uses DrugBank Discovery API to retrieve drug-target relationships. Requires DRUGBANK_KEY environment variable for full functionality.",
)
def enrich_with_drugbank(biomarker: str) -> list[dict]:
    """
    Uses DrugBank Discovery API to retrieve drug-target relationships.
    (Placeholder - does not make a real API call without a valid key)
    """
    print(f"Tool: enrich_with_drugbank called with biomarker: {biomarker}")
    if DRUGBANK_KEY == "your_drugbank_key_here":
        print("Warning: DRUGBANK_KEY is a placeholder. Returning mock data.")
        return [{"drug_name": "MockDrugA", "target": biomarker}]

    headers = {"Authorization": f"Bearer {DRUGBANK_KEY}"}
    try:
        response = requests.get(
            "https://api.drugbank.com/discovery/v1/query",
            params={"query": biomarker},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling DrugBank API: {e}")
        return [{"error": str(e), "biomarker": biomarker}]


@mcp.tool(
    name="query_opentargets",
    description="Uses OpenTargets GraphQL API for systematic target–disease associations.",
)
def query_opentargets(target: str) -> dict:
    """
    Uses OpenTargets GraphQL API for systematic target–disease associations.
    (Placeholder - makes a real API call to a public endpoint)
    """
    print(f"Tool: query_opentargets called with target: {target}")
    query = """query($t:String!){target(ensemblId:$t){associatedDiseases{rows{disease{name}}}}}"""  # Adjusted query for typical use
    variables = {"t": target}
    try:
        response = requests.post(
            "https://api.platform.opentargets.org/api/v4/graphql",
            json={"query": query, "variables": variables},
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenTargets API: {e}")
        return {"error": str(e), "target": target}


@mcp.tool(
    name="symptom_to_biomarker",
    description="Processes clinical text using Bio_ClinicalBERT to extract entities, maps them to OMOP codes, and enriches with DrugBank and OpenTargets data.",
)
def symptom_to_biomarker(text: str) -> dict:
    """
    Wraps BioClinicalBERT for deep NER, maps to OMOP/CDS codes,
    then augments via DrugBank and OpenTargets.
    """
    print(f"Tool: symptom_to_biomarker called with text: '{text[:50]}...'")
    codes = bio_ner_map(text)

    drug_info_list = []
    ot_info_list = []

    # For simplicity, let's assume codes is a list of strings
    # and we process the first one for enrichment as an example.
    if codes:
        # These would ideally call the respective MCP tools through the MCP client
        # if this were an orchestrated workflow. Here, we're directly calling the Python functions
        # (which are also exposed as tools). This is fine for a single server implementation.
        # Assuming codes[0] is a biomarker string for DrugBank & OpenTargets
        # In reality, you might iterate or choose codes more selectively.
        if (
            codes[0]
            and not codes[0].startswith("OMOP:Error")
            and not codes[0].startswith("OMOP:No")
        ):  # Basic check
            drug_info_list = enrich_with_drugbank(codes[0])
            ot_info_list = query_opentargets(
                codes[0]
            )  # Assuming codes[0] is a suitable target string (e.g., Ensembl ID)

    return {
        "codes": codes,
        "drugbank_info": drug_info_list,
        "opentargets_info": ot_info_list,
    }


@mcp.tool(
    name="query_clinicaltrials",
    description="Calls ClinicalTrials.gov Data API (v2) to fetch live trial records based on an expression and specified fields.",
)
def query_clinicaltrials(expr: str, fields: list[str] | None = None) -> dict:
    """
    Calls ClinicalTrials.gov Data API to fetch live trial records.
    """
    print(f"Tool: query_clinicaltrials called with expr: {expr}, fields: {fields}")
    if fields is None:
        fields = [
            "NCTId",
            "BriefTitle",
            "OverallStatus",
            "Condition",
            "InterventionName",
        ]

    params = {
        "expr": expr,
        "fields": ",".join(fields),
        "min_rnk": 1,
        "max_rnk": 10,
        "fmt": "json",
    }
    try:
        response = requests.get(
            "https://clinicaltrials.gov/api/v2/studies", params=params
        )  # Updated API endpoint
        response.raise_for_status()
        # The new API returns a list of studies under a 'studies' key
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling ClinicalTrials.gov API: {e}")
        return {"error": str(e), "expr": expr}
    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON from ClinicalTrials.gov API: {e}. Response text: {response.text[:200]}"
        )
        return {
            "error": str(e),
            "expr": expr,
            "response_text_snippet": response.text[:200],
        }


@mcp.tool(
    name="fetch_irb_protocol",
    description="Queries OnCore CTMS REST API for inclusion/exclusion criteria and site metadata. Requires IRB_API_TOKEN environment variable and valid endpoint for full functionality.",
)
def fetch_irb_protocol(protocol_id: str) -> dict:
    """
    Queries OnCore CTMS REST API for inclusion/exclusion criteria and site metadata.
    (Placeholder - does not make a real API call without a valid endpoint/token)
    """
    print(f"Tool: fetch_irb_protocol called with protocol_id: {protocol_id}")
    if IRB_API_TOKEN == "your_irb_api_token_here":
        print(
            "Warning: IRB_API_TOKEN is a placeholder. Using mock OnCore CTMS URL and returning mock data."
        )
        # Mock data simulating an IRB protocol
        return {
            "protocol_id": protocol_id,
            "title": "Mock Protocol Title for " + protocol_id,
            "inclusion_criteria": ["Age >= 18 years", "Specific condition X diagnosed"],
            "exclusion_criteria": [
                "Has condition Y",
                "Received treatment Z within last 6 months",
            ],
            "site_metadata": {
                "location": "Mock Hospital",
                "contact": "irb@mockhospital.com",
            },
        }

    # Actual API call structure (requires valid URL and token)
    url = f"https://oncore.example.com/api/v1/protocols/{protocol_id}"  # Replace with actual OnCore API endpoint
    headers = {"Authorization": f"Bearer {IRB_API_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling OnCore CTMS API: {e}")
        return {"error": str(e), "protocol_id": protocol_id}


# Initialize mock Fabric client
# In a real setup, this would configure connection to a Hyperledger Fabric network
fabric_client = MockFabricClient(net_profile="network.json")


@mcp.tool(
    name="record_audit",
    description="Writes an audit entry to a (mocked) Hyperledger Fabric ledger.",
)
def record_audit(entry: dict) -> str:
    """
    Writes an audit entry to a Hyperledger Fabric ledger.
    (Placeholder - uses a mock client)
    """
    print(f"Tool: record_audit called with entry: {entry}")
    try:
        # This uses the mock client's method
        result_json_str = fabric_client.chaincode_invoke(
            requestor="AdminUser",  # Example requestor
            channel_name="mychannel",
            peers=[fabric_client.get_peer("peer0.org1.example.com")],  # Mock peer
            cc_name="auditcc",
            fcn="logEntry",
            args=[json.dumps(entry)],  # Args must be list of strings for chaincode
            wait_for_event=True,
        )
        return result_json_str  # Should already be a JSON string from mock
    except Exception as e:
        print(f"Error in record_audit (mock fabric call): {e}")
        return json.dumps({"status": "failure", "error": str(e)})


if __name__ == "__main__":
    print("Starting MCP Server...")
    # To run the server, you'd typically call mcp.run()

    # For now, we'll just print a message indicating it's ready for tools to be called if imported.
    # The actual running of the server to listen for HTTP requests will be part of testing.
    print(
        "MCP Server defined. To run and listen for requests, call 'mcp.run(host=\"localhost\", port=50051)' or similar."
    )
    print("You can also test tools directly if running this script, e.g.:")
    print("--- Testing symptom_to_biomarker ---")
    print(
        json.dumps(
            symptom_to_biomarker(
                "Patient has a persistent headache and history of diabetes. Also reports trouble breathing."
            ),
            indent=2,
        )
    )
    print("--- Testing query_clinicaltrials ---")
    print(
        json.dumps(
            query_clinicaltrials(
                expr="diabetes", fields=["NCTId", "BriefTitle", "Condition"]
            ),
            indent=2,
        )
    )
    print("--- Testing fetch_irb_protocol ---")
    print(json.dumps(fetch_irb_protocol(protocol_id="NCT12345"), indent=2))
    print("--- Testing record_audit ---")
    print(
        json.dumps(
            record_audit(entry={"user_id": "test_user", "action": "test_action"}),
            indent=2,
        )
    )
    mcp.run(host="0.0.0.0", port=50051)  # Changed port to 50051 for gRPC standard
