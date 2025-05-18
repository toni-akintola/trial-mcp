# TrialGPT: Matching Patients to Clinical Trials with Large Language Models

## Introduction
Clinical trials are often hindered by the challenge of patient recruitment. In this work, we introduce TrialGPT, a novel end-to-end framework for zero-shot patient-to-trial matching with large language models (LLMs). TrialGPT consists of three key components: it first performs filtering of irrelevant clinical trials at scale (TrialGPT-Retrieval), then predicts the patient eligibility on a criterion-by-criterion basis (TrialGPT-Matching); and finally aggregates criterion-level predictions into trial-level scores for ranking the clinical trials (TrialGPT-Ranking). We evaluate TrialGPT on three publicly available cohorts of 183 synthetic patients with over 75,000 trial eligibility annotations. TrialGPT-Retrieval can efficiently recall over 90% of relevant clinical trials using only less than 6% of the initial clinical trial collection. Over 1,000 patient-criterion pairs were manually annotated by three physicians to evaluate TrialGPT-Matching, which achieves a criterion-level accuracy of 87.3% with faithful explanations, close to the expert performance (88.7%–90.0%). For TrialGPT-Ranking, the aggregated trial-level scores are highly correlated with human eligibility judgments, and they outperform the best-competing models by 28.8% to 53.5% in ranking and excluding clinical trials. Furthermore, our user study reveals that TrialGPT can significantly reduce the screening time by 42.6% in a real-life clinical trial matching task. Taken together, these results have demonstrated promising opportunities for clinical trial matching with LLMs via the TrialGPT framework.

![image](https://github.com/user-attachments/assets/66b01b03-1871-4ccc-be05-10e17e077370)

## Configuration

To run TrialGPT, one needs to set up the Anthropic API. Please set the environment variable accordingly:

```bash
export ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
```

The code has been tested with Python 3.9.13 using CentOS Linux release 7.9.2009 (Core). Please install the required Python packages by:

```bash
pip install -r requirements.txt
```

## Datasets

We used the clinical trial information on https://clinicaltrials.gov/. Please download our parsed dataset by:

```bash
wget -O dataset/trial_info.json https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trial_info.json
```

Three publicly available datasets are used in the study (please properly cite these datasets if you use them; see details about citations in the bottom):
- The SIGIR 2016 corpus, available at: https://data.csiro.au/collection/csiro:17152
- The TREC Clinical Trials 2021 corpus, available at: https://www.trec-cds.org/2021.html
- The TREC Clinical Trials 2022 corpus, available at: https://www.trec-cds.org/2022.html

The SIGIR dataset is already in `/dataset/`, please download the corpora of TREC CT 2021 and 2022 by:

```bash
wget -O dataset/trec_2021/corpus.jsonl https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl
wget -O dataset/trec_2022/corpus.jsonl https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2022_corpus.jsonl
```

## TrialGPT-Retrieval

Given a patient summary and an initial collection of clinical trials, the first step is TrialGPT-Retrieval, which generates a list of keywords for the patient and utilizes a hybrid-fusion retrieval mechanism to get relevant trials (component a in the figure). 

Specifically, one can run the code below for keyword generation. The generated keywords will be saved in the `./results/` directory.

```bash
# syntax: python trialgpt_retrieval/keyword_generation.py ${corpus} ${model}  
# ${corpus} can be sigir, trec_2021, and trec_2022
# ${model} should be an Anthropic model name (e.g., claude-3-opus-20240229, claude-3-sonnet-20240229)
# examples below (replace with actual Anthropic model names)
python trialgpt_retrieval/keyword_generation.py sigir claude-3-opus-20240229
python trialgpt_retrieval/keyword_generation.py trec_2021 claude-3-opus-20240229
python trialgpt_retrieval/keyword_generation.py trec_2022 claude-3-opus-20240229
```

After generating the keywords, one can run the code below for retrieving relevant clinical trials. The retrieved trials will be saved in the `./results/` directory. The code below will use our cached results of keyword generation that are located in `./dataset/{corpus}/id2queries.json`. (Note: cached results may still refer to OpenAI models, new keywords should be generated with Anthropic models for consistency).

```bash
# syntax: python trialgpt_retrieval/hybrid_fusion_retrieval.py ${corpus} ${q_type} ${k} ${bm25_weight} ${medcpt_weight} 
# ${corpus} can be sigir, trec_2021, and trec_2022
# ${q_type} can be raw, or an Anthropic model name used for generation (e.g., claude-3-opus-20240229). Cached OpenAI model names (gpt-35-turbo, gpt-4-turbo) might not work correctly without re-generation.
# ${k} is the constant in the reciprocal rank fusion, and we recommend using 20
# ${bm25_weight} is the weight for the BM25 retriever, it should be set as 1 unless in ablation experiments
# ${medcpt_weight} is the weight for the MedCPT retriever, it should be set as 1 unless in ablation experiments
# examples below (replace with actual Anthropic model names if not using 'raw' or re-generated keywords)
python trialgpt_retrieval/hybrid_fusion_retrieval.py sigir claude-3-opus-20240229 20 1 1
python trialgpt_retrieval/hybrid_fusion_retrieval.py trec_2021 claude-3-opus-20240229 20 1 1
python trialgpt_retrieval/hybrid_fusion_retrieval.py trec_2022 claude-3-opus-20240229 20 1 1
```

## TrialGPT-Matching

After retrieving the candidate clinical trials with TrialGPT-Retrieval, the next step is to use TrialGPT-Matching to perform fine-grained criterion-by-criterion analyses on each patient-trial pair (component b in the figure). We have also made the retrieved trials by GPT-4-based TrialGPT-Retrieval available at `./dataset/{corpus}/retrieved_trials.json`. (Note: these cached trials were based on OpenAI models).
One can run the following commands to use TrialGPT-Matching, and the results will be saved in `./results/`:

```bash
# syntax: python trialgpt_matching/run_matching.py ${corpus} ${retrieved_nctids_file_path}
# ${corpus} can be sigir, trec_2021, and trec_2022
# ${retrieved_nctids_file_path} is the path to the retrieved NCTIDs (e.g. ./dataset/{corpus}/retrieved_trials.json or results/retrieved_nctids_...json)
# The model is now hardcoded to "claude-3-7-sonnet-latest"
# examples below
python trialgpt_matching/run_matching.py sigir ./dataset/sigir/retrieved_trials.json
python trialgpt_matching/run_matching.py trec_2021 ./dataset/trec_2021/retrieved_trials.json
python trialgpt_matching/run_matching.py trec_2022 ./dataset/trec_2022/retrieved_trials.json
```

## TrialGPT-Ranking

The final step is to use TrialGPT-Ranking to aggregate the criterion-level predictions into trial-level scores for ranking (component c in the figure). To get the LLM-aggregation scores for TrialGPT-Ranking, one can run the following commands. The results will be saved in `./results/`:

```bash
# syntax: python trialgpt_ranking/run_aggregation.py ${corpus} ${matching_results_path}
# ${corpus} can be sigir, trec_2021, and trec_2022
# ${matching_results_path} is the path to the TrialGPT matching results 
# The model is now hardcoded to "claude-3-7-sonnet-latest"
# example below (ensure matching results were generated using an Anthropic model)
python trialgpt_ranking/run_aggregation.py sigir results/matching_results_sigir_claude-3-7-sonnet-latest.json 
```

Once the matching results and the aggregation results are complete, one can run the following code to get the final ranking of clinical trials for each patient:

```bash
# syntax: python trialgpt_ranking/rank_results.py ${matching_results_path} ${aggregation_results_path}
# ${matching_results_path} is the path to the TrialGPT matching results (e.g. results/matching_results_sigir_claude-3-7-sonnet-latest.json)
# ${aggregation_results_path} is the path to the aggregation results generated above (e.g. results/aggregation_results_sigir_claude-3-7-sonnet-latest.json)
# example below (ensure results were generated using an Anthropic model)
python trialgpt_ranking/rank_results.py results/matching_results_sigir_claude-3-7-sonnet-latest.json results/aggregation_results_sigir_claude-3-7-sonnet-latest.json
```

Example output:

```bash
Patient ID: sigir-20141
Clinical trial ranking:
NCT00185120 2.8999999995
NCT02144636 2.8999999995
NCT02608255 2.84999999975
NCT01724996 2.7999999998
...
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

## Citation

If you find this repo helpful, please cite TrialGPT by:
```bibtex
@article{jin2024matching,
  title={Matching patients to clinical trials with large language models},
  author={Jin, Qiao and Wang, Zifeng and Floudas, Charalampos S and Chen, Fangyuan and Gong, Changlin and Bracken-Clarke, Dara and Xue, Elisabetta and Yang, Yifan and Sun, Jimeng and Lu, Zhiyong},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={9074},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

If you use the SIGIR cohort, please cite the original dataset papers by:
```bibtex
@inproceedings{koopman2016test,
  title={A test collection for matching patients to clinical trials},
  author={Koopman, Bevan and Zuccon, Guido},
  booktitle={Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval},
  pages={669--672},
  year={2016}
}
@inproceedings{roberts2015overview,
  title={Overview of the TREC 2015 Clinical Decision Support Track},
  author={Roberts, Kirk and Simpson, Matthew S and Voorhees, Ellen M and Hersh, William R},
  booktitle={Proceedings of the Twenty-Fourth Text REtrieval Conference (TREC 2015)},
  year={2015}
}
@inproceedings{simpson2014overview,
  title={Overview of the TREC 2014 Clinical Decision Support Track},
  author={Simpson, Matthew S and Voorhees, Ellen M and Hersh, William R},
  booktitle={Proceedings of the Twenty-Third Text REtrieval Conference (TREC 2014)},
  year={2014}
}
```

If you use the TREC cohorts, please cite the original dataset papers by:
```bibtex
@inproceedings{roberts2021overview,
  title={Overview of the TREC 2021 clinical trials track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Bedrick, Steven and Hersh, Willian R},
  booktitle={Proceedings of the Thirtieth Text REtrieval Conference (TREC 2021)},
  year={2021}
}
@inproceedings{roberts2022overview,
  title={Overview of the TREC 2022 clinical trials track},
  author={Roberts, Kirk and Demner-Fushman, Dina and Voorhees, Ellen M and Bedrick, Steven and Hersh, Willian R},
  booktitle={Proceedings of the Thirty-first Text REtrieval Conference (TREC 2022)},
  year={2022}
}
```

# TrialGPT+MCP Implementation

This project integrates TrialGPT with a FastMCP server to provide powerful tools for clinical trial matching.

## Project Structure

- `trialgpt_matching/`: Original TrialGPT matching logic.
- `trialgpt_ranking/`: Original TrialGPT ranking logic.
- `main.py`: Main entry point, **now also acts as the MCP server with integrated Bio_ClinicalBERT for NER.**
- `design.md`: The design document for MCP integration.
- `mcp_client_tester.py`: A script to test the MCP server tools (requires generated client stubs).
- `requirements.txt`: Python dependencies.

## Setup and Running

### 1. Prerequisites

- Python 3.8+
- Pip (Python package installer)
- Git

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

You will need to install `fastmcp` and its dependencies, including gRPC tools for client generation.
Additionally, `transformers` (for Bio_ClinicalBERT) and `torch` are now required.
```bash
pip install fastmcp requests google-api-python-client # google-api for Struct if needed by client
pip install grpcio grpcio-tools betterproto # For FastMCP server and client stub generation
pip install transformers torch # For Hugging Face Transformers and PyTorch backend
```
(Note: `hfc.fabric` for Hyperledger is mocked, so full Fabric setup is not needed for the current server version.)

### 3. Generate MCP Client Stubs (Important Step)

The `main.py` (acting as `mcp_server.py`) uses FastMCP, which dynamically generates gRPC services based on your Python tool definitions. To create a Python client to interact with these services, you need to:

**a. Obtain the .proto definition:**
   - When you run `main.py` (the MCP server) with FastMCP, it might save the generated `.proto` file (e.g., `mcp.proto` or `fastmcp_service.proto`) in the current directory or a specified output directory. Check the FastMCP documentation or server startup logs for its location.
   - Alternatively, FastMCP might have a command to dump the .proto definition.

**b. Compile the .proto file:**
   - Once you have the `.proto` file (let's assume it's named `mcp_service.proto`), use `protoc` with the `betterproto` plugin (as FastMCP v2 often uses it) to generate the Python client stubs:
     ```bash
     python -m grpc_tools.protoc -I. --python_betterproto_out=. mcp_service.proto
     ```
     This will generate a Python file like `mcp_service_pb2.py` (the name depends on the package and service definition in the proto file). This file contains the client stub and message classes. You'll need to move this generated file into your project where it can be imported by `mcp_client_tester.py`.

### 4. Running the MCP Server

Open a terminal and run:
```bash
python main.py
```
The server should start and listen on `0.0.0.0:50051` (or the configured port).
**Note:** The first time you run this, it will attempt to download the Bio_ClinicalBERT model from Hugging Face, which can be several hundred megabytes and may take some time.

### 5. Testing the MCP Server with the Python Client

**a. Update Client Imports:**
   - Modify `mcp_client_tester.py` to import the correct generated stub and message classes from the file created in step 3b (e.g., `from mcp_service_pb2 import McpStub, SymptomToBiomarkerRequest, ...`).

**b. Run the Client Tester:**
   - Once the server is running and the client script is updated with correct imports, open another terminal and run:
     ```bash
     python mcp_client_tester.py
     ```
   This script will attempt to call each tool on the MCP server.

### 6. (Next Steps) Integrating MCP Client into TrialGPT

After the MCP server and client are tested:
- Create a dedicated MCP client module/class within the TrialGPT application.
- Modify `trialgpt_matching/TrialGPT.py` and `trialgpt_ranking/TrialGPT.py` to use this MCP client for functionalities like `symptom_to_biomarker`, fetching trial data, etc., instead of direct LLM calls or static data.

## Tool Details (from main.py)

- `symptom_to_biomarker(text: str) -> dict`: Now uses Bio_ClinicalBERT for NER.
- `query_clinicaltrials(expr: str, fields: list[str] | None = None) -> dict`
- `fetch_irb_protocol(protocol_id: str) -> dict`
- `enrich_with_drugbank(biomarker: str) -> list[dict]`
- `query_opentargets(target: str) -> dict`
- `record_audit(entry: dict) -> str`

Refer to `design.md` for more details on the intended functionality of each tool.
