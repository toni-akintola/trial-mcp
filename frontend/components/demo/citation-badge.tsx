"use client";

import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ExternalLink } from "lucide-react";
import type { Citation } from "@/lib/types";

interface CitationBadgeProps {
  citation: Citation;
}

export function CitationBadge({ citation }: CitationBadgeProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Badge variant="outline" className="cursor-pointer hover:bg-muted">
          {citation.source}
        </Badge>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>{citation.source}</DialogTitle>
          <DialogDescription>
            Reference information and documentation
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Description</h4>
            <p className="text-sm text-muted-foreground">
              {citation.description}
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-medium">Reference</h4>
            <div className="rounded-md bg-muted p-3">
              <p className="text-sm font-mono break-all">
                {citation.reference}
              </p>
            </div>
          </div>

          {citation.url && (
            <div className="flex justify-end">
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline inline-flex items-center gap-1"
              >
                View Documentation
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
