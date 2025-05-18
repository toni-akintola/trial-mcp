"use client";

import React, { useCallback } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
  BackgroundVariant,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

// Define initial nodes and edges in React Flow format
const initialNodes: Node[] = [
  {
    id: "symptom",
    position: { x: 200, y: 150 },
    data: { label: "Symptom Parser" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
  {
    id: "biomarker",
    position: { x: 400, y: 100 },
    data: { label: "Biomarker Translator" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
  {
    id: "retrieval",
    position: { x: 600, y: 150 },
    data: { label: "Trial Retriever" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
  {
    id: "matcher",
    position: { x: 600, y: 300 },
    data: { label: "Criteria Matcher" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
  {
    id: "ranker",
    position: { x: 400, y: 350 },
    data: { label: "Trial Ranker" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
  {
    id: "audit",
    position: { x: 200, y: 300 },
    data: { label: "Audit Logger" },
    style: {
      backgroundColor: "#0ea5e920",
      borderColor: "#0ea5e9",
      borderWidth: 2,
      borderRadius: "50%",
      width: 80,
      height: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
  },
];

const initialEdges: Edge[] = [
  {
    id: "e-symptom-biomarker",
    source: "symptom",
    target: "biomarker",
    label: "extracts",
  },
  {
    id: "e-biomarker-retrieval",
    source: "biomarker",
    target: "retrieval",
    label: "queries",
  },
  {
    id: "e-retrieval-matcher",
    source: "retrieval",
    target: "matcher",
    label: "provides",
  },
  {
    id: "e-matcher-ranker",
    source: "matcher",
    target: "ranker",
    label: "scores",
  },
  { id: "e-ranker-audit", source: "ranker", target: "audit", label: "logs" },
  {
    id: "e-audit-symptom",
    source: "audit",
    target: "symptom",
    label: "verifies",
  },
];

export function NetworkGraph() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: any) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  return (
    <div className="w-full h-[500px] rounded-lg border">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}
