import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CitationBadge } from "@/components/demo/citation-badge";
import type { ChainOfThoughtStep } from "@/lib/types";

interface ChainOfThoughtPanelProps {
  chainOfThought: ChainOfThoughtStep[];
}

export function ChainOfThoughtPanel({
  chainOfThought,
}: ChainOfThoughtPanelProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Chain of Thought</h2>
        <span className="text-sm text-muted-foreground">
          {chainOfThought.length} steps
        </span>
      </div>

      <div className="space-y-4">
        {chainOfThought.map((step, index) => (
          <Card key={index}>
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-base">
                    Step {step.step}: {step.tool}
                  </CardTitle>
                  <CardDescription>
                    Tool invocation and reasoning
                  </CardDescription>
                </div>
                <Badge variant="outline">{`#${step.step}`}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Prompt</h4>
                <div className="rounded-md bg-muted p-3">
                  <p className="text-sm whitespace-pre-wrap">
                    {step.prompt.length > 200
                      ? `${step.prompt.substring(0, 200)}...`
                      : step.prompt}
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium">Result</h4>
                <div className="rounded-md bg-muted p-3">
                  <p className="text-sm whitespace-pre-wrap">
                    {step.result.length > 200
                      ? `${step.result.substring(0, 200)}...`
                      : step.result}
                  </p>
                </div>
              </div>

              <div className="pt-2">
                <h4 className="text-sm font-medium mb-2">Citations</h4>
                <div className="flex flex-wrap gap-2">
                  {step.sourceCitation && (
                    <CitationBadge citation={step.sourceCitation} />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
