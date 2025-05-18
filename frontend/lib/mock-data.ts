import type { MatchResponse } from "./types";

export const mockMatchResponse: MatchResponse = {
  biomarkerCodes: ["R05", "R50.9", "R53.83"],
  matchedTrials: [
    {
      nctId: "NCT04312997",
      title:
        "COVID-19 Convalescent Plasma for the Treatment of Hospitalized Patients",
      status: "Recruiting",
      matchScore: 0.87,
      citations: [
        {
          source: "ClinicalTrials.gov API",
          description:
            "Data retrieved from the official ClinicalTrials.gov API",
          reference: "ClinicalTrials.gov Identifier: NCT04312997",
          url: "https://clinicaltrials.gov/study/NCT04312997",
        },
        {
          source: "ICD-10 Codes",
          description:
            "International Classification of Diseases, 10th Revision",
          reference:
            "R05 (Cough), R50.9 (Fever, unspecified), R53.83 (Fatigue)",
          url: "https://www.icd10data.com/",
        },
      ],
    },
    {
      nctId: "NCT04401501",
      title:
        "Efficacy of Remdesivir in COVID-19 Patients With Respiratory Symptoms",
      status: "Active, not recruiting",
      matchScore: 0.82,
      citations: [
        {
          source: "ClinicalTrials.gov API",
          description:
            "Data retrieved from the official ClinicalTrials.gov API",
          reference: "ClinicalTrials.gov Identifier: NCT04401501",
          url: "https://clinicaltrials.gov/study/NCT04401501",
        },
        {
          source: "MedDRA Terms",
          description:
            "Medical Dictionary for Regulatory Activities Terminology",
          reference: "Persistent cough (10034825), Pyrexia (10037660)",
          url: "https://www.meddra.org/",
        },
      ],
    },
    {
      nctId: "NCT04280705",
      title: "Adaptive COVID-19 Treatment Trial (ACTT)",
      status: "Completed",
      matchScore: 0.79,
      citations: [
        {
          source: "ClinicalTrials.gov API",
          description:
            "Data retrieved from the official ClinicalTrials.gov API",
          reference: "ClinicalTrials.gov Identifier: NCT04280705",
          url: "https://clinicaltrials.gov/study/NCT04280705",
        },
        {
          source: "SNOMED CT",
          description:
            "Systematized Nomenclature of Medicine -- Clinical Terms",
          reference: "Fever (386661006), Cough (49727002), Fatigue (84229001)",
          url: "https://www.snomed.org/",
        },
      ],
    },
  ],
  chainOfThought: [
    {
      step: 1,
      tool: "SymptomToBiomarker",
      prompt:
        'Extract biomarkers from the following patient symptoms: "Patient presents with persistent cough, fever, and fatigue for the past 2 weeks."',
      result:
        "Extracted the following biomarkers:\n- R05 (Cough)\n- R50.9 (Fever, unspecified)\n- R53.83 (Other fatigue)",
      sourceCitation: {
        source: "ICD-10 Codes",
        description: "International Classification of Diseases, 10th Revision",
        reference: "R05 (Cough), R50.9 (Fever, unspecified), R53.83 (Fatigue)",
        url: "https://www.icd10data.com/",
      },
    },
    {
      step: 2,
      tool: "TrialRetriever",
      prompt:
        "Retrieve clinical trials relevant to the following biomarkers: R05 (Cough), R50.9 (Fever), R53.83 (Fatigue)",
      result:
        "Retrieved 15 potentially relevant clinical trials. Top matches include:\n- NCT04312997 (COVID-19 Convalescent Plasma)\n- NCT04401501 (Remdesivir for COVID-19)\n- NCT04280705 (Adaptive COVID-19 Treatment Trial)",
      sourceCitation: {
        source: "ClinicalTrials.gov API",
        description: "Data retrieved from the official ClinicalTrials.gov API",
        reference:
          "Query parameters: condition=covid&biomarkers=R05,R50.9,R53.83",
        url: "https://clinicaltrials.gov/api/query/study_fields",
      },
    },
    {
      step: 3,
      tool: "CriteriaEvaluator",
      prompt:
        'Evaluate if the patient with "persistent cough, fever, and fatigue for 2 weeks" meets the inclusion criteria for trial NCT04312997',
      result:
        "Inclusion criteria analysis:\n1. Adult patient (assumed): YES\n2. Respiratory symptoms present: YES (persistent cough)\n3. Fever present: YES\n4. Duration of symptoms appropriate: YES (2 weeks is within range)\n\nConclusion: Patient likely meets primary inclusion criteria with a confidence score of 0.87",
      sourceCitation: {
        source: "Trial Eligibility Criteria",
        description: "Inclusion and exclusion criteria from the trial protocol",
        reference: "NCT04312997 - Section 5.1: Inclusion Criteria",
        url: "https://clinicaltrials.gov/study/NCT04312997",
      },
    },
    {
      step: 4,
      tool: "TrialRanker",
      prompt:
        "Rank the following trials by relevance to patient with biomarkers R05, R50.9, R53.83: NCT04312997, NCT04401501, NCT04280705",
      result:
        "Ranked trials by relevance:\n1. NCT04312997 (Score: 0.87)\n2. NCT04401501 (Score: 0.82)\n3. NCT04280705 (Score: 0.79)",
      sourceCitation: {
        source: "Ranking Algorithm",
        description: "Internal algorithm for ranking trial relevance",
        reference:
          "TrialMCP Ranking v2.1 - Weighted criteria matching with symptom duration factor",
        url: null,
      },
    },
    {
      step: 5,
      tool: "AuditLogger",
      prompt:
        'Log the matching process for patient with symptoms "persistent cough, fever, and fatigue" to trials NCT04312997, NCT04401501, NCT04280705',
      result:
        "Successfully logged matching process to blockchain ledger. Audit record ID: audit-123456",
      sourceCitation: null,
    },
  ],
  auditRecordId: "audit-123456",
};
