import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function HeroSection() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-background to-muted/30">
      <div className="container px-4 md:px-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
          <div className="flex flex-col justify-center space-y-4">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                AI-Powered Clinical Trial Matching
              </h1>
              <p className="max-w-[600px] text-muted-foreground md:text-xl">
                TrialMCP uses advanced AI to match patient symptoms with
                clinical trials, providing transparent reasoning and citations
                for every recommendation.
              </p>
            </div>
            <div className="flex flex-col gap-2 min-[400px]:flex-row">
              <Button asChild size="lg" className="gap-1">
                <Link href="/demo">
                  Try the Demo
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <Link href="/model-insight">Explore the Model</Link>
              </Button>
            </div>
          </div>
          <div className="flex items-center justify-center">
            <div className="relative w-full h-[300px] md:h-[400px] lg:h-[500px] rounded-lg overflow-hidden bg-muted">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-muted flex items-center justify-center">
                <div className="text-center space-y-2 p-4">
                  <div className="inline-block rounded-lg bg-background/90 px-3 py-1 text-sm backdrop-blur">
                    Futuristic AI Visualization
                  </div>
                  <div className="h-40 w-40 mx-auto rounded-full bg-primary/20 flex items-center justify-center">
                    <div className="h-32 w-32 rounded-full bg-primary/30 flex items-center justify-center">
                      <div className="h-24 w-24 rounded-full bg-primary/40 flex items-center justify-center">
                        <div className="h-16 w-16 rounded-full bg-primary/50 flex items-center justify-center">
                          <div className="h-8 w-8 rounded-full bg-primary/60"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
