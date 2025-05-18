"use client";

import { useTheme } from "next-themes";

export function ComponentDiagram() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Colors based on theme
  const bgColor = isDark ? "#1e293b" : "#f8fafc";
  const textColor = isDark ? "#e2e8f0" : "#334155";
  const borderColor = isDark ? "#475569" : "#cbd5e1";
  const accentColor = isDark ? "#0ea5e9" : "#0284c7";

  return (
    <div className="w-full h-[500px] overflow-auto rounded-lg border p-4">
      <svg
        width="800"
        height="600"
        viewBox="0 0 800 600"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Background */}
        <rect width="800" height="600" fill={bgColor} rx="8" />

        {/* Input Node */}
        <g transform="translate(100, 100)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Symptom Input
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Natural language
          </text>
        </g>

        {/* Biomarker Translation Node */}
        <g transform="translate(100, 250)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Biomarker
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Translation
          </text>
        </g>

        {/* Retrieval Node */}
        <g transform="translate(325, 250)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Trial Retrieval
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            API & Database
          </text>
        </g>

        {/* Matching Node */}
        <g transform="translate(550, 250)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Criteria Matching
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            LLM-based
          </text>
        </g>

        {/* Ranking Node */}
        <g transform="translate(325, 400)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Trial Ranking
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Relevance scoring
          </text>
        </g>

        {/* Output Node */}
        <g transform="translate(550, 400)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Results
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Matched trials
          </text>
        </g>

        {/* Audit Node */}
        <g transform="translate(325, 100)">
          <rect
            width="150"
            height="80"
            rx="8"
            fill={accentColor}
            fillOpacity="0.2"
            stroke={accentColor}
            strokeWidth="2"
          />
          <text
            x="75"
            y="35"
            textAnchor="middle"
            fill={textColor}
            fontWeight="bold"
          >
            Audit System
          </text>
          <text
            x="75"
            y="55"
            textAnchor="middle"
            fill={textColor}
            fontSize="12"
          >
            Blockchain ledger
          </text>
        </g>

        {/* Connections */}
        {/* Input to Biomarker */}
        <path
          d="M175 180 L175 250"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="175,250 170,240 180,240" fill={borderColor} />

        {/* Biomarker to Retrieval */}
        <path
          d="M250 290 L325 290"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="325,290 315,285 315,295" fill={borderColor} />

        {/* Retrieval to Matching */}
        <path
          d="M475 290 L550 290"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="550,290 540,285 540,295" fill={borderColor} />

        {/* Matching to Ranking */}
        <path
          d="M625 330 L625 440 L475 440"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="475,440 485,435 485,445" fill={borderColor} />

        {/* Ranking to Results */}
        <path
          d="M475 440 L550 440"
          stroke={borderColor}
          strokeWidth="2"
          fill="none"
        />
        <polygon points="550,440 540,435 540,445" fill={borderColor} />

        {/* Audit connections */}
        <path
          d="M400 180 L400 250"
          stroke={borderColor}
          strokeWidth="2"
          strokeDasharray="5,5"
          fill="none"
        />
        <path
          d="M400 180 L625 250"
          stroke={borderColor}
          strokeWidth="2"
          strokeDasharray="5,5"
          fill="none"
        />
        <path
          d="M400 180 L625 400"
          stroke={borderColor}
          strokeWidth="2"
          strokeDasharray="5,5"
          fill="none"
        />
      </svg>
    </div>
  );
}
