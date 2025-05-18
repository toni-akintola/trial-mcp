"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler,
);

export function MetricsCharts() {
  // --- Mock Data for TrialMCP vs SOTA ---

  // 1. Matching Accuracy (Bar Chart)
  const matchingAccuracyData = {
    labels: ["TrialMCP", "SOTA Model", "Expert Average"],
    datasets: [
      {
        label: "Criterion-Level Accuracy (%)",
        data: [87.3, 73.1, 89.3], // TrialMCP: 87.3, SOTA: 73.1 (from sklearn), Expert: 88.7-90.0
        backgroundColor: [
          "rgba(14, 165, 233, 0.6)", // sky-500
          "rgba(249, 115, 22, 0.6)", // orange-500
          "rgba(22, 163, 74, 0.6)", // green-600
        ],
        borderColor: [
          "rgba(14, 165, 233, 1)",
          "rgba(249, 115, 22, 1)",
          "rgba(22, 163, 74, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };

  // 2. Sentence Location Metrics (Table)
  const sentenceLocationMetrics = [
    {
      metric: "Precision (Macro Avg)",
      trialMCP: "0.8500", // Placeholder
      sotaModel: "0.4467",
    },
    {
      metric: "Recall (Macro Avg)",
      trialMCP: "0.8800", // Placeholder
      sotaModel: "0.5000",
    },
    {
      metric: "F1-Score (Macro Avg)",
      trialMCP: "0.8647", // Placeholder
      sotaModel: "0.4587",
    },
  ];

  // 3. Screening Time Reduction (Bar Chart)
  const screeningTimeData = {
    labels: ["TrialMCP", "SOTA Model"],
    datasets: [
      {
        label: "Screening Time Reduction (%)",
        data: [42.6, 15.0], // TrialMCP: 42.6%, SOTA: less or none
        backgroundColor: [
          "rgba(14, 165, 233, 0.6)", // sky-500
          "rgba(249, 115, 22, 0.6)", // orange-500
        ],
        borderColor: ["rgba(14, 165, 233, 1)", "rgba(249, 115, 22, 1)"],
        borderWidth: 1,
      },
    ],
  };

  // --- Chart Options ---
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            let label = context.dataset.label || "";
            if (label) {
              label += ": ";
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y + "%";
            }
            return label;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function (value: any) {
            return value + "%";
          },
        },
        grid: {
          color: "rgba(200, 200, 200, 0.2)", // Lighter grid lines
        },
      },
      x: {
        grid: {
          display: false, // No vertical grid lines
        },
      },
    },
  };

  const matchingOptions = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: { display: true, text: "Criterion-Level Matching Accuracy" },
    },
  };
  const screeningOptions = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: { display: true, text: "Clinical Trial Screening Time Reduction" },
    },
  };

  return (
    <div className="space-y-6">
      <Tabs defaultValue="matching" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          {" "}
          {/* Adjusted grid-cols to 2 */}
          <TabsTrigger value="matching">Matching Accuracy</TabsTrigger>
          <TabsTrigger value="screening">Screening Time</TabsTrigger>
        </TabsList>

        <TabsContent value="matching" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Criterion-Level Matching Accuracy</CardTitle>
              <CardDescription>
                Comparison of TrialMCP, SOTA model, and expert performance in
                predicting patient eligibility on a criterion-by-criterion
                basis.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="h-[350px] p-4 bg-background rounded-lg shadow">
                <Bar options={matchingOptions} data={matchingAccuracyData} />
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2">
                  Sentence Location Metrics (Macro-Averages)
                </h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead className="text-right">
                        TrialMCP (Placeholder)
                      </TableHead>
                      <TableHead className="text-right">SOTA Model</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sentenceLocationMetrics.map((item) => (
                      <TableRow key={item.metric}>
                        <TableCell className="font-medium">
                          {item.metric}
                        </TableCell>
                        <TableCell className="text-right text-sky-600 font-semibold">
                          {item.trialMCP}
                        </TableCell>
                        <TableCell className="text-right text-orange-600 font-semibold">
                          {item.sotaModel}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="screening" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Screening Time Reduction</CardTitle>
              <CardDescription>
                Percentage reduction in clinical trial screening time comparing
                TrialMCP to a SOTA model.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[350px] p-4 bg-background rounded-lg shadow">
                <Bar options={screeningOptions} data={screeningTimeData} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
