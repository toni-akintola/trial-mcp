import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Info } from "lucide-react";
import { CitationBadge } from "@/components/demo/citation-badge";
import type { Trial } from "@/lib/types";

interface TrialCardProps {
  trial: Trial;
}

export function TrialCard({ trial }: TrialCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-lg">{trial.title}</CardTitle>
            <CardDescription className="mt-1">
              <span className="font-mono">{trial.nctId}</span>
            </CardDescription>
          </div>
          <Badge
            variant={trial.status === "Recruiting" ? "default" : "secondary"}
          >
            {trial.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Match Score:</span>
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full"
                style={{ width: `${trial.matchScore * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium">
              {Math.round(trial.matchScore * 100)}%
            </span>
          </div>

          <div className="flex flex-wrap gap-2 mt-3">
            {trial.citations.map((citation, index) => (
              <CitationBadge key={index} citation={citation} />
            ))}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" size="sm" className="gap-1">
          <Info className="h-4 w-4" />
          Details
        </Button>
        <Button variant="outline" size="sm" className="gap-1" asChild>
          <a
            href={`https://clinicaltrials.gov/study/${trial.nctId}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            View on ClinicalTrials.gov
            <ExternalLink className="h-4 w-4" />
          </a>
        </Button>
      </CardFooter>
    </Card>
  );
}
