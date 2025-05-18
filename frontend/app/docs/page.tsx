import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function Docs() {
  return (
    <div className="container py-10 space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Documentation</h1>
        <p className="text-muted-foreground">
          Learn more about TrialMCP, its API, and how to integrate it into your
          applications
        </p>
      </div>

      <Tabs defaultValue="faq" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="faq">FAQ</TabsTrigger>
          <TabsTrigger value="api">API Reference</TabsTrigger>
          <TabsTrigger value="github">GitHub</TabsTrigger>
        </TabsList>

        <TabsContent value="faq" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Frequently Asked Questions</CardTitle>
              <CardDescription>
                Common questions about TrialMCP and its capabilities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger>What is TrialMCP?</AccordionTrigger>
                  <AccordionContent>
                    TrialMCP is an AI-powered clinical trial matching platform
                    that translates patient symptoms into biomarkers, retrieves
                    relevant clinical trials, and ranks matches with full
                    transparency and citations.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                  <AccordionTrigger>
                    How accurate is the symptom-to-biomarker translation?
                  </AccordionTrigger>
                  <AccordionContent>
                    Our system achieves over 90% accuracy in translating common
                    symptoms to standardized biomarkers, based on validation
                    against medical literature and expert review.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-3">
                  <AccordionTrigger>
                    Can I integrate TrialMCP into my healthcare application?
                  </AccordionTrigger>
                  <AccordionContent>
                    Yes, TrialMCP offers a REST API that allows for seamless
                    integration with electronic health records, patient portals,
                    and other healthcare applications.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-4">
                  <AccordionTrigger>
                    How does TrialMCP ensure data privacy?
                  </AccordionTrigger>
                  <AccordionContent>
                    TrialMCP processes all data in compliance with HIPAA
                    regulations. Patient data is encrypted in transit and at
                    rest, and we maintain a comprehensive audit trail of all
                    data access.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-5">
                  <AccordionTrigger>
                    What sources does TrialMCP use for clinical trial data?
                  </AccordionTrigger>
                  <AccordionContent>
                    TrialMCP retrieves clinical trial data from
                    ClinicalTrials.gov, the WHO International Clinical Trials
                    Registry Platform, and other authorized trial registries.
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>API Reference</CardTitle>
              <CardDescription>
                Documentation for the TrialMCP REST API
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Endpoint: /api/match</h3>
                  <p className="text-sm text-muted-foreground">
                    POST request to match patient symptoms to clinical trials
                  </p>
                  <div className="rounded-md bg-muted p-4">
                    <pre className="text-sm">
                      {`POST /api/match
Content-Type: application/json

{
  "symptoms": "Patient presents with persistent cough, fever, and fatigue for the past 2 weeks."
}`}
                    </pre>
                  </div>
                </div>

                <div className="space-y-2">
                  <h3 className="text-lg font-medium">Response Format</h3>
                  <p className="text-sm text-muted-foreground">
                    The API returns matched trials, biomarkers, and the full
                    chain of thought
                  </p>
                  <div className="rounded-md bg-muted p-4">
                    <pre className="text-sm">
                      {`{
  "biomarkerCodes": ["R05", "R50.9", "R53.83"],
  "matchedTrials": [
    {
      "nctId": "NCT04312997",
      "title": "COVID-19 Convalescent Plasma for the Treatment of Hospitalized Patients",
      "status": "Recruiting",
      "matchScore": 0.87,
      "citations": [...]
    },
    ...
  ],
  "chainOfThought": [
    {
      "step": 1,
      "tool": "SymptomToBiomarker",
      "prompt": "...",
      "result": "...",
      "sourceCitation": "..."
    },
    ...
  ],
  "auditRecordId": "audit-123456"
}`}
                    </pre>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="github" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>GitHub Repository</CardTitle>
              <CardDescription>
                Access the source code and contribute to TrialMCP
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p>
                  TrialMCP is an open-source project. You can find the source
                  code, report issues, and contribute to the project on GitHub.
                </p>
                <div className="rounded-md bg-muted p-4 flex items-center justify-between">
                  <span className="font-mono text-sm">
                    github.com/trialmcp/trialmcp
                  </span>
                  <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline text-sm"
                  >
                    Visit Repository
                  </a>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
