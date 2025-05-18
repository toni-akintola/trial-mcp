import Link from "next/link";
import { Brain } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <Brain className="h-5 w-5" />
          <p className="text-center text-sm leading-loose md:text-left">
            &copy; {new Date().getFullYear()} TrialMCP. All rights reserved.
          </p>
        </div>
        <div className="flex gap-4">
          <Link
            href="/docs"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Documentation
          </Link>
          <Link
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            GitHub
          </Link>
        </div>
      </div>
    </footer>
  );
}
