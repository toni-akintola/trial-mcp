export interface Trial {
  nctId: string;
  title: string;
  status: string;
  matchScore: number;
  citations: Citation[];
}

export interface Citation {
  source: string;
  description: string;
  reference: string;
  url?: string;
}

export interface ChainOfThoughtStep {
  step: number;
  tool: string;
  prompt: string;
  result: string;
  sourceCitation: Citation | null;
}

export interface MatchResponse {
  biomarkerCodes: string[];
  matchedTrials: Trial[];
  chainOfThought: ChainOfThoughtStep[];
  auditRecordId: string;
}
