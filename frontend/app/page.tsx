import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { HeroSection } from "@/components/hero-section";
import { FeatureCard } from "@/components/feature-card";
import {
  ArrowRight,
  Brain,
  FlaskRoundIcon as Flask,
  Search,
} from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col gap-12 pb-8">
      <HeroSection />

      <section className="container space-y-6 py-8 md:py-12 lg:py-16">
        <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-4 text-center">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            Advanced Clinical Trial Matching
          </h2>
          <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
            TrialMCP uses state-of-the-art AI to match patient symptoms with
            relevant clinical trials, providing transparent reasoning and
            citations for every recommendation.
          </p>
        </div>

        <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3">
          <FeatureCard
            icon={<Brain className="h-10 w-10" />}
            title="AI-Powered Matching"
            description="Advanced language models translate symptoms to biomarkers and match to trial criteria."
          />
          <FeatureCard
            icon={<Search className="h-10 w-10" />}
            title="Transparent Process"
            description="See the full chain-of-thought with citations and tool invocations for every match."
          />
          <FeatureCard
            icon={<Flask className="h-10 w-10" />}
            title="Evidence-Based"
            description="All recommendations include citations to medical literature and trial documentation."
          />
        </div>

        <div className="mx-auto flex justify-center gap-4">
          <Button asChild size="lg" className="gap-1">
            <Link href="/model-insight">
              Model Insight
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
          <Button asChild size="lg" variant="secondary" className="gap-1">
            <Link href="/demo">
              Try the Demo
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>

      <section className="container py-8 md:py-12 lg:py-16">
        <div className="mx-auto grid gap-6 lg:grid-cols-2 lg:gap-12">
          <Card>
            <CardHeader>
              <CardTitle>Model Insight</CardTitle>
              <CardDescription>
                Understand how TrialMCP works through interactive visualizations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Explore component diagrams, state transitions, and network
                visualizations that explain the inner workings of our AI-powered
                matching system.
              </p>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline">
                <Link href="/model-insight">Explore Model</Link>
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Try the Demo</CardTitle>
              <CardDescription>
                Input symptoms and see matching clinical trials in real-time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Enter patient symptoms and watch as TrialMCP translates them to
                biomarkers, retrieves relevant trials, and ranks matches with
                full transparency.
              </p>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline">
                <Link href="/demo">Start Demo</Link>
              </Button>
            </CardFooter>
          </Card>
        </div>
      </section>
    </div>
  );
}
