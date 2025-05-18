"use client";

import { useState } from "react";
import { SymptomForm } from "@/components/demo/symptom-form";
import { TrialResults } from "@/components/demo/trial-results";
import { ChainOfThoughtPanel } from "@/components/demo/chain-of-thought-panel";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mockMatchResponse } from "@/lib/mock-data";
import type { MatchResponse } from "@/lib/types";
import { Loader2 } from "lucide-react";

export default function Demo() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<MatchResponse | null>(null);

  const handleSubmit = async (symptoms: string) => {
    setIsLoading(true);

    // Simulate API call with a delay
    setTimeout(() => {
      setResults(mockMatchResponse);
      setIsLoading(false);
    }, 2000);

    // In the future, this would be a real API call:
    // const response = await fetch('/api/match', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ symptoms }),
    // })
    // const data = await response.json()
    // setResults(data)
    // setIsLoading(false)
  };

  return (
    <div className="container py-10 space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Try the Demo</h1>
        <p className="text-muted-foreground">
          Enter patient symptoms to find matching clinical trials with full
          transparency
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Symptom Input</CardTitle>
          <CardDescription>
            Describe the patient's symptoms in natural language
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SymptomForm onSubmit={handleSubmit} isLoading={isLoading} />
        </CardContent>
      </Card>

      {isLoading && (
        <div className="flex justify-center py-12">
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Processing symptoms...
            </p>
          </div>
        </div>
      )}

      {results && (
        <div className="space-y-6">
          <Tabs defaultValue="trials" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="trials">Matching Trials</TabsTrigger>
              <TabsTrigger value="chain">Chain of Thought</TabsTrigger>
            </TabsList>

            <TabsContent value="trials" className="mt-6">
              <TrialResults trials={results.matchedTrials} />
            </TabsContent>

            <TabsContent value="chain" className="mt-6">
              <ChainOfThoughtPanel chainOfThought={results.chainOfThought} />
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}
