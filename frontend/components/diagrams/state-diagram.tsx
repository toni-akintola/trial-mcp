"use client";

import { useTheme } from "next-themes";

export function StateDiagram() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Colors based on theme
  const bgColor = isDark ? "#1e293b" : "#f8fafc";
  const textColor = isDark ? "#e2e8f0" : "#334155";
  const borderColor = isDark ? "#475569" : "#cbd5e1";
  const accentColor = isDark ? "#0ea5e9" : "#0284c7";
  const highlightColor = isDark ? "#f97316" : "#ea580c";

  return (
    <div className="w-full h-[500px] overflow-auto rounded-lg border p-4">
      <svg
        width="800"
        height="400"
        viewBox="0 0 800 400"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Background */}
        <rect width="800" height="400" fill={bgColor} rx="8" />

        {/* States */}
        <g transform="translate(50, 50)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Symptoms
          </text>
        </g>

        <g transform="translate(250, 50)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Biomarkers
          </text>
        </g>

        <g transform="translate(450, 50)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Retrieval
          </text>
        </g>

        <g transform="translate(650, 50)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Trials
          </text>
        </g>

        <g transform="translate(650, 170)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Matching
          </text>
        </g>

        <g transform="translate(450, 170)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Ranking
          </text>
        </g>

        <g transform="translate(250, 170)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={highlightColor}
            fillOpacity="0.2"
            stroke={highlightColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Results
          </text>
        </g>

        <g transform="translate(50, 170)">
          <rect
            width="120"
            height="60"
            rx="30"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="60"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Audit
          </text>
        </g>

        {/* Example Data */}
        <g transform="translate(50, 280)">
          <rect
            width="120"
            height="60"
            rx="8"
            fill="transparent"
            stroke={borderColor}
            strokeWidth="1"
          />
          <text
            x="60"
            y="30"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Persistent cough
          </text>
          <text
            x="60"
            y="45"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Fever
          </text>
        </g>

        <g transform="translate(250, 280)">
          <rect
            width="120"
            height="60"
            rx="8"
            fill="transparent"
            stroke={borderColor}
            strokeWidth="1"
          />
          <text
            x="60"
            y="30"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            R05 (Cough)
          </text>
          <text
            x="60"
            y="45"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            R50.9 (Fever)
          </text>
        </g>

        <g transform="translate(450, 280)">
          <rect
            width="120"
            height="60"
            rx="8"
            fill="transparent"
            stroke={borderColor}
            strokeWidth="1"
          />
          <text
            x="60"
            y="30"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Query:
          </text>
          <text
            x="60"
            y="45"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            COVID-19 trials
          </text>
        </g>

        <g transform="translate(650, 280)">
          <rect
            width="120"
            height="60"
            rx="8"
            fill="transparent"
            stroke={borderColor}
            strokeWidth="1"
          />
          <text
            x="60"
            y="30"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            NCT04312997
          </text>
          <text
            x="60"
            y="45"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Score: 0.87
          </text>
        </g>

        {/* Connections */}
        <path
          d="M170 80 L250 80"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="250,80 240,75 240,85" fill={borderColor} />

        <path
          d="M370 80 L450 80"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="450,80 440,75 440,85" fill={borderColor} />

        <path
          d="M570 80 L650 80"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="650,80 640,75 640,85" fill={borderColor} />

        <path
          d="M710 110 L710 170"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="710,170 705,160 715,160" fill={borderColor} />

        <path
          d="M650 200 L570 200"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="570,200 580,195 580,205" fill={borderColor} />

        <path
          d="M450 200 L370 200"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="370,200 380,195 380,205" fill={borderColor} />

        <path
          d="M250 200 L170 200"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="170,200 180,195 180,205" fill={borderColor} />

        {/* Data Flow */}
        <path
          d="M110 110 L110 280"
          stroke={borderColor}
          strokeWidth="1"
          strokeDasharray="4,4"
          fill="none"
        />
        <path
          d="M310 110 L310 280"
          stroke={borderColor}
          strokeWidth="1"
          strokeDasharray="4,4"
          fill="none"
        />
        <path
          d="M510 110 L510 280"
          stroke={borderColor}
          strokeWidth="1"
          strokeDasharray="4,4"
          fill="none"
        />
        <path
          d="M710 230 L710 280"
          stroke={borderColor}
          strokeWidth="1"
          strokeDasharray="4,4"
          fill="none"
        />
      </svg>
    </div>
  );
}
