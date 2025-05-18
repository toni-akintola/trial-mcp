"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

interface SymptomFormProps {
  onSubmit: (symptoms: string) => void;
  isLoading: boolean;
}

export function SymptomForm({ onSubmit, isLoading }: SymptomFormProps) {
  const [symptoms, setSymptoms] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (symptoms.trim()) {
      onSubmit(symptoms);
    }
  };

  const handleExampleClick = (example: string) => {
    setSymptoms(example);
  };

  const examples = [
    "Patient presents with persistent cough, fever, and fatigue for the past 2 weeks.",
    "65-year-old male with chest pain, shortness of breath, and swelling in the ankles.",
    "34-year-old female with recurring headaches, blurred vision, and dizziness.",
  ];

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Textarea
          placeholder="Describe patient symptoms here..."
          className="min-h-[150px] resize-none"
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
        />
        <Button
          type="submit"
          disabled={isLoading || !symptoms.trim()}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            "Find Matching Trials"
          )}
        </Button>
      </form>

      <div className="pt-4">
        <p className="text-sm font-medium mb-2">Example inputs:</p>
        <div className="grid gap-2">
          {examples.map((example, index) => (
            <Card
              key={index}
              className="cursor-pointer hover:bg-muted/50"
              onClick={() => handleExampleClick(example)}
            >
              <CardContent className="p-3">
                <p className="text-sm truncate">{example}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
