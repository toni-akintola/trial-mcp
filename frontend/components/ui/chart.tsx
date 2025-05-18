import type React from "react";

export const Bar = () => <rect />;
export const BarChart = () => <svg />;
export const CartesianGrid = () => <g />;
export const Cell = () => <rect />;
export const Legend = () => <div></div>;
export const Line = () => <path />;
export const LineChart = () => <svg />;
export const Pie = () => <path />;
export const PieChart = () => <svg />;
export const ResponsiveContainer = ({
  children,
}: {
  children: React.ReactNode;
}) => <div>{children}</div>;
export const Tooltip = () => <div></div>;
export const XAxis = () => <g />;
export const YAxis = () => <g />;
