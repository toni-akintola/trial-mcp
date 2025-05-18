import { ComponentDiagram } from "@/components/diagrams/component-diagram";
import { StateDiagram } from "@/components/diagrams/state-diagram";
import { NetworkGraph } from "@/components/diagrams/network-graph";
import { MetricsCharts } from "@/components/diagrams/metrics-charts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function ModelInsight() {
  return (
    <div className="container py-10 space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Insight</h1>
        <p className="text-muted-foreground">
          Interactive visualizations explaining the TrialMCP architecture and
          data flow
        </p>
      </div>

      <Tabs defaultValue="component" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="component">Component Diagram</TabsTrigger>
          <TabsTrigger value="state">State Diagram</TabsTrigger>
          <TabsTrigger value="network">Network Graph</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="component" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Component Architecture</CardTitle>
              <CardDescription>
                Visualization of the TrialMCP system architecture and component
                interactions
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <ComponentDiagram />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="state" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>State Transitions</CardTitle>
              <CardDescription>
                Flow diagram showing how data transforms through the TrialMCP
                pipeline
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <StateDiagram />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Tool Network</CardTitle>
              <CardDescription>
                Interactive network graph showing MCP tools as nodes and data
                transformations as edges
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <NetworkGraph />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>
                Charts showing mapping accuracy, source contribution, and other
                key metrics
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <MetricsCharts />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
