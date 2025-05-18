import { TrialCard } from "@/components/demo/trial-card";
import type { Trial } from "@/lib/types";

interface TrialResultsProps {
  trials: Trial[];
}

export function TrialResults({ trials }: TrialResultsProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Matching Clinical Trials</h2>
        <span className="text-sm text-muted-foreground">
          {trials.length} matches found
        </span>
      </div>

      <div className="grid gap-4">
        {trials.map((trial) => (
          <TrialCard key={trial.nctId} trial={trial} />
        ))}
      </div>
    </div>
  );
}
