Below is an in-depth design document for integrating TrialGPT with MCP tools, detailing architecture, component specifications, and an implementation roadmap. This plan leverages live data sources and compliance workflows to transform TrialGPT into a dynamic, auditable clinical-trial matcher.

**Summary**: We propose “TrialGPT+MCP,” which enhances TrialGPT’s static three-stage pipeline (Retrieval, Matching, Ranking) with live FastMCP connectors to ClinicalTrials.gov, IRB/CTMS, DrugBank/OpenTargets, and a Hyperledger Fabric audit ledger. Deep symptom→biomarker translation via BioClinicalBERT further refines matching precision, and a compliance layer guarantees HIPAA/GDPR redaction with immutable logging.

---

## ##1. Objectives & Scope

* **Goal**: Upgrade TrialGPT into a real-time, compliance-driven trial matcher by wiring its Retrieval, Matching, and Ranking stages to live MCP tools instead of static JSON snapshots ([Nature][1]).
* **MVP Scope**: Implement core FastMCP tools (`query_clinicaltrials`, `fetch_irb_protocol`, `enrich_with_drugbank`, `query_opentargets`, `symptom_to_biomarker`, `record_audit`), integrate them into TrialGPT’s prompt chains, and provide a minimal UI with provenance links.

---

## ##2. Architecture Overview

```text
[Patient Input]
     ↓
[BioClinicalBERT MCP: symptom_to_biomarker] → OMOP Codes
     ↓
[TrialGPT-Retrieval LLM] → retrieval_expr
     ↓
[FastMCP Retrieval Fabric]
   ├→ query_clinicaltrials(expr)          (ClinicalTrials.gov API) :contentReference[oaicite:1]{index=1}
   ├→ fetch_irb_protocol(protocol_id)     (OnCore CTMS API)       :contentReference[oaicite:2]{index=2}
   └→ enrich_with_drugbank(biomarker)     (DrugBank Discovery)    :contentReference[oaicite:3]{index=3}
   └→ query_opentargets(target)           (OpenTargets GraphQL)   :contentReference[oaicite:4]{index=4}
     ↓
[TrialGPT-Matching LLM] (criterion evaluation with IRB data) :contentReference[oaicite:5]{index=5}
     ↓
[TrialGPT-Ranking LLM] (aggregates scores + provenance URLs) :contentReference[oaicite:6]{index=6}
     ↓
[Compliance MCP: record_audit]
     ↓
[Frontend/UI: displays redacted matches + audit links]
```

---

## ##3. Detailed Component Design

### 3.1 Symptom→Biomarker Translation

* **Tool**: `symptom_to_biomarker` wraps BioClinicalBERT for deep NER on clinical text, mapping to OMOP/CDS codes, then augments via DrugBank and OpenTargets ([PubMed Central][2], [PubMed][3]).
* **Flow**:

  1. **Input**: Free-text symptoms.
  2. **BioClinicalBERT NER**: Extract entities → candidate concepts.
  3. **OMOP Mapper**: Map to standardized codes.
  4. **Enrichment**: Invoke `enrich_with_drugbank` and `query_opentargets` for target–drug associations.

### 3.2 Retrieval Connector

* **Tool**: `query_clinicaltrials(expr, fields)` calls ClinicalTrials.gov Data API’s `/query/study_fields` endpoint with `expr=<biomarker>` to fetch live trial records ([ClinicalTrials.gov][4], [ClinicalTrials.gov][5]).
* **Output**: JSON with fields like `NCTId`, `BriefTitle`, `OverallStatus`, limited to top-N results.

### 3.3 IRB/CTMS Connector

* **Tool**: `fetch_irb_protocol(protocol_id)` queries OnCore CTMS REST API for inclusion/exclusion criteria and site metadata ([Advarra - Advancing Better Research][6], [Medical College of Wisconsin][7]).
* **Usage**: TrialGPT-Matching invokes this tool per candidate trial to obtain official protocol text.

### 3.4 Pharma Partner Enrichment

* **DrugBank**: `enrich_with_drugbank(biomarker)` uses DrugBank Discovery API to retrieve drug-target relationships ([docs.drugbank.com][8]).
* **OpenTargets**: `query_opentargets(target)` uses GraphQL API for systematic target–disease associations ([Open Targets Platform][9]).

### 3.5 Compliance & Audit

* **Tool**: `record_audit(entry)` writes `{timestamp, userId, codes, trialIds}` to a Hyperledger Fabric ledger via the Fabric-Python SDK (`fabric-sdk-py`) ([GitHub][10], [Fabric SDK Py][11]).
* **Redaction**: A PHI scrubber layer removes any direct identifiers before logging.

### 3.6 Prompt Engineering

* **Retrieval Prompt**: “Given OMOP codes X, generate a ClinicalTrials.gov search expression targeting relevant trials.”
* **Matching Prompt**: “For trial ID Y with criteria Z (from fetch\_irb\_protocol), determine patient eligibility for each criterion.”
* **Ranking Prompt**: “Aggregate criterion-level labels into a trial-level score and provide a justification link to NCTId page.”

---

## ##4. MCP Tool Specifications

```python
from fastmcp import MCPServer
import requests, json
from hfc.fabric import Client

mcp = MCPServer()

@mcp.tool()
def symptom_to_biomarker(text: str) -> dict:
    # BioClinicalBERT call + OMOP mapping stub
    codes = bio_ner_map(text)
    # Enrichment
    drug_info = enrich_with_drugbank(codes[0])
    ot_info   = query_opentargets(codes[0])
    return {"codes": codes, "drugbank": drug_info, "opentargets": ot_info}

@mcp.tool()
def query_clinicaltrials(expr: str, fields: list[str]) -> dict:
    params = {"expr": expr, "fields": ",".join(fields), "min_rnk": 1, "max_rnk": 10}
    return requests.get("https://clinicaltrials.gov/api/query/study_fields", params=params).json()

@mcp.tool()
def fetch_irb_protocol(protocol_id: str) -> dict:
    url = f"https://oncore.example.com/api/v1/protocols/{protocol_id}"
    headers = {"Authorization": f"Bearer {IRB_API_TOKEN}"}
    return requests.get(url, headers=headers).json()

@mcp.tool()
def enrich_with_drugbank(biomarker: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {DRUGBANK_KEY}"}
    return requests.get("https://api.drugbank.com/discovery/v1/query", params={"query": biomarker}, headers=headers).json()

@mcp.tool()
def query_opentargets(target: str) -> dict:
    query = '''query($t:String!){target(ensemblId:$t){diseases {disease{name}}}}'''
    return requests.post("https://api.platform.opentargets.org/graphql", json={"query": query, "variables": {"t": target}}).json()

fabric_client = Client(net_profile="network.json")
@mcp.tool()
def record_audit(entry: dict) -> str:
    return fabric_client.chaincode_invoke(requestor='Admin', channel_name='mychannel',
                                          peers=[fabric_client.get_peer('peer0.org1.example.com')],
                                          cc_name='auditcc', fcn='logEntry', args=[json.dumps(entry)], wait_for_event=True)
```

---

## ##5. Implementation Roadmap & Milestones

| Phase       | Tasks                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------- |
| **Setup**   | Fork `ncbi-nlp/TrialGPT`, install FastMCP 2.0, configure `mcp_config.json` ([GitHub][12]) |
| **Phase 1** | Integrate `query_clinicaltrials` into Retrieval; validate live data retrieval             |
| **Phase 2** | Build `symptom_to_biomarker` with BioClinicalBERT stub; swap into Retrieval flow          |
| **Phase 3** | Add `fetch_irb_protocol` to Matching; update prompts to call MCP tools per trial          |
| **Phase 4** | Incorporate `enrich_with_drugbank` & `query_opentargets` into Translation                 |
| **Phase 5** | Wire `record_audit` into final step; ensure PHI redaction before logging                  |
| **Phase 6** | UI updates: provenance links, match explanations, audit status banner                     |
| **Testing** | End-to-end trials with synthetic and real patient texts; compliance audits; latency tests |

---

By embedding live MCP connectors into TrialGPT’s pipeline and layering in deep NER-based translation plus blockchain auditing, “TrialGPT+MCP” becomes a high-precision, real-time, and fully auditable clinical trial matcher—ready for rigorous hackathon development and beyond.

[1]: https://www.nature.com/articles/s41467-024-53081-z?utm_source=chatgpt.com "Matching patients to clinical trials with large language models - Nature"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9074854/?utm_source=chatgpt.com "A Deep Language Model for Symptom Extraction from Clinical Text ..."
[3]: https://pubmed.ncbi.nlm.nih.gov/34705659/?utm_source=chatgpt.com "A Deep Language Model for Symptom Extraction From Clinical Text ..."
[4]: https://clinicaltrials.gov/data-api/api?utm_source=chatgpt.com "ClinicalTrials.gov API"
[5]: https://clinicaltrials.gov/data-api/about-api/api-migration?utm_source=chatgpt.com "API Migration Guide | ClinicalTrials.gov"
[6]: https://www.advarra.com/sites/ctms/oncore/?utm_source=chatgpt.com "OnCore Clinical Trial Management System (CTMS) - Advarra"
[7]: https://www.mcw.edu/departments/research-systems/oncore/oncore-integrations?utm_source=chatgpt.com "OnCore Integrations | Research Systems"
[8]: https://docs.drugbank.com/discovery/v1/?utm_source=chatgpt.com "Discovery API Reference | DrugBank Help Center"
[9]: https://platform-docs.opentargets.org/data-access/graphql-api?utm_source=chatgpt.com "GraphQL API | Open Targets Platform Documentation"
[10]: https://github.com/hyperledger/fabric-sdk-py?utm_source=chatgpt.com "hyperledger/fabric-sdk-py - GitHub"
[11]: https://fabric-sdk-py.readthedocs.io/en/latest/tutorial.html?utm_source=chatgpt.com "Tutorial of Using Fabric Python SDK"
[12]: https://github.com/jlowin/fastmcp?utm_source=chatgpt.com "GitHub - jlowin/fastmcp: The fast, Pythonic way to build MCP servers ..."
