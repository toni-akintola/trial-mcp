"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/mode-toggle";
import { Brain } from "lucide-react";

export function MainNav() {
  const pathname = usePathname();

  return (
    <div className="flex w-full items-center justify-between">
      <div className="flex items-center gap-6 md:gap-10">
        <Link href="/" className="flex items-center space-x-2">
          <Brain className="h-6 w-6" />
          <span className="font-bold inline-block">TrialMCP</span>
        </Link>
        <nav className="hidden md:flex gap-6">
          <Link
            href="/"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              pathname === "/" ? "text-foreground" : "text-muted-foreground",
            )}
          >
            Home
          </Link>
          <Link
            href="/model-insight"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              pathname === "/model-insight"
                ? "text-foreground"
                : "text-muted-foreground",
            )}
          >
            Model Insight
          </Link>
          <Link
            href="/demo"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              pathname === "/demo"
                ? "text-foreground"
                : "text-muted-foreground",
            )}
          >
            Demo
          </Link>
          <Link
            href="/docs"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary",
              pathname === "/docs"
                ? "text-foreground"
                : "text-muted-foreground",
            )}
          >
            Docs
          </Link>
        </nav>
      </div>
      <div className="flex items-center gap-2">
        <ModeToggle />
        <Button variant="outline" size="sm" asChild className="hidden md:flex">
          <Link
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </Link>
        </Button>
      </div>
    </div>
  );
}
