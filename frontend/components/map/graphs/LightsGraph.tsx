"use client";
import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip
} from "recharts";
import { Sun } from "lucide-react";
import { SENSOR_COLORS, ROOM_COLORS } from "@/components/map/constants";

interface LightsGraphProps {
  artifact: any;
}

const LIGHT_LABELS: Record<number, string> = {
  0: "Dark",
  1: "Dim",
  2: "Normal",
  3: "Bright",
  4: "Very Bright",
  5: "Very Sunny"
};

export default function LightsGraph({ artifact }: LightsGraphProps) {
  // --- 1. HIGH-DENSITY CONTINUOUS HOVER & EXCEPTION ENGINE ---
  const formattedData = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0) return [];

    const sorted = [...artifact.series].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const parsedSorted = sorted.map(pt => ({
      ...pt,
      timeMs: new Date(pt.timestamp).getTime()
    }));

    // Identify dynamic sensor keys (ignoring metadata keys like capacity or timeframe)
    const sensorKeys = Object.keys(artifact.metadata || {}).filter(k => k !== 'capacity' && k !== 'timeframe');

    // ---> CRITICAL EXCEPTION: For 30d & 90d, overwrite Last Day (today) with Previous Day's telemetry! <---
    if (["30d", "90d"].includes(artifact.timeframe) && parsedSorted.length >= 2) {
      const prevDay = parsedSorted[parsedSorted.length - 2];
      const lastDay = parsedSorted[parsedSorted.length - 1];
      sensorKeys.forEach(k => {
        lastDay[k] = prevDay[k];
      });
    }

    const startTime = parsedSorted[0].timeMs;
    let majorStepMs = 0;
    let minorStepMs = 0;
    let majorBucketsCount = 0;

    switch (artifact.timeframe) {
      case "2h":
        majorStepMs = 10 * 60 * 1000;
        minorStepMs = 30 * 1000;
        majorBucketsCount = 13;
        break;
      case "24h":
        majorStepMs = 2 * 60 * 60 * 1000;
        minorStepMs = 2 * 60 * 1000;
        majorBucketsCount = 13;
        break;
      case "7d":
        majorStepMs = 2 * 60 * 60 * 1000;
        minorStepMs = 10 * 60 * 1000;
        majorBucketsCount = 85;
        break;
      case "30d":
        majorStepMs = 24 * 60 * 60 * 1000;
        minorStepMs = 1 * 60 * 60 * 1000;
        majorBucketsCount = 31;
        break;
      case "90d":
        majorStepMs = 24 * 60 * 60 * 1000;
        minorStepMs = 3 * 60 * 60 * 1000;
        majorBucketsCount = 91;
        break;
      default:
        return sorted;
    }

    const endTime = startTime + majorBucketsCount * majorStepMs;
    const totalDurationMs = endTime - startTime;
    const count = Math.round(totalDurationMs / minorStepMs) + 1;
    const grid = [];

    let currentMatchIndex = 0;

    for (let i = 0; i < count; i++) {
      const gridTime = startTime + i * minorStepMs;
      const timeFromStart = gridTime - startTime;

      const isMajorBoundary =
        Math.abs(timeFromStart % majorStepMs) < 100 ||
        Math.abs(gridTime - endTime) < 100;
      
      const isClosingBoundary = i === count - 1;

      while (
        currentMatchIndex < parsedSorted.length - 1 &&
        parsedSorted[currentMatchIndex + 1].timeMs <= gridTime
      ) {
        currentMatchIndex++;
      }

      const match = parsedSorted[currentMatchIndex];
      
      // Calculate aggregate plot value if multiple light sensors exist
      let sum = 0;
      let validSensors = 0;
      sensorKeys.forEach(k => {
        if (match[k] !== undefined && match[k] !== null) {
          sum += match[k];
          validSensors++;
        }
      });
      let plotValue = validSensors > 0 ? sum / validSensors : 0;

      const majorBucketIndex = Math.floor((timeFromStart - (isClosingBoundary ? 1 : 0)) / majorStepMs);
      const bucketStartTime = startTime + majorBucketIndex * majorStepMs;
      const bucketEndTime = bucketStartTime + majorStepMs;

      grid.push({
        ...match,
        timestamp: new Date(gridTime).toISOString(),
        PlotValue: plotValue,
        isMajorBoundary,
        isClosingBoundary,
        bucketStartTime,
        bucketEndTime
      });
    }

    // ---> EPSILON HACK: If all values are identical, add 0.0001 to point 0. <---
    if (grid.length > 0) {
      const firstVal = grid[0].PlotValue;
      const allSame = grid.every((pt: any) => pt.PlotValue === firstVal);
      if (allSame) {
        grid[0].PlotValue += 0.0001;
      }
    }

    return grid;
  }, [artifact]);

  // --- 2. MAJOR TICKS ---
  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timestamp);
  }, [formattedData]);

  if (!artifact || !artifact.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center pb-32 select-none">
        <Sun size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        <p className="text-xs font-mono uppercase tracking-wider">No Telemetry Recorded</p>
      </div>
    );
  }

  const formatXAxisTick = (timestamp: string) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const tf = artifact.timeframe;

    if (["30d", "90d"].includes(tf)) {
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    }
    if (tf === "7d") {
      return date.toLocaleDateString("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
    }
    return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  };

  // --- 3. TOOLTIP WITH BASELINE ALIGNMENT & 30D/90D EXACT DAY DISPLAY ---
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const startDate = new Date(dataPoint.bucketStartTime);
      const endDate = new Date(dataPoint.bucketEndTime);
      const tf = artifact.timeframe;

      const isLongTerm = ["30d", "90d"].includes(tf);
      let timeStr = "";

      if (isLongTerm) {
        timeStr = startDate.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      } else if (tf === "7d" || tf === "24h") {
        const startDay = startDate.toLocaleDateString("en-US", { weekday: "short" });
        const endDay = endDate.toLocaleDateString("en-US", { weekday: "short" });
        const startTime = startDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const endTime = endDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${startDay} ${startTime} – ${endDay} ${endTime}`;
      } else {
        const startTime = startDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const endTime = endDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${startTime} – ${endTime}`;
      }

      // Round out the 0.0001 epsilon hack so tooltips display clean integers
      const val = Math.round(dataPoint.PlotValue);
      const semanticLabel = LIGHT_LABELS[val] || `Level ${val}`;
      const badgeColor = SENSOR_COLORS.good;
      const badgeBg = ROOM_COLORS.good;

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[150px]">
          <div className="text-center text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5">
            {timeStr}
          </div>

          <div className="flex items-center justify-center gap-3 pt-0.5">
            <span className="font-bold text-base leading-none" style={{ color: badgeColor }}>
              {val} <span className="text-[10px] text-[#A3B8B2]/60 font-normal">level</span>
            </span>
            <span
              className="px-3 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 shadow-sm"
              style={{
                backgroundColor: `${badgeBg}90`,
                borderColor: badgeColor,
                color: badgeColor
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: badgeColor }} />
              {semanticLabel}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };
  // --- 4. CUSTOM ACTIVE DOT ---
  const CustomActiveDot = (props: any) => {
    const { cx, cy } = props;
    if (cx === undefined || cy === undefined) return null;
    return (
      <circle 
        cx={cx} 
        cy={cy} 
        r={6} 
        fill={SENSOR_COLORS.good} 
        stroke="#0A0A0A" 
        strokeWidth={2} 
      />
    );
  };

  return (
    <div className="w-full h-full flex flex-col justify-center bg-transparent p-4 pb-32 select-none">
      <div className="w-full h-full min-h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={formattedData}
            margin={{
              top: 30,
              right: 30,
              left: -10,
              bottom: 20
            }}
          >
            <defs>
              <linearGradient id="lightAreaFade" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={SENSOR_COLORS.good} stopOpacity={0.35} />
                <stop offset="95%" stopColor={SENSOR_COLORS.good} stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="timestamp"
              ticks={majorTicks}
              tickFormatter={formatXAxisTick}
              interval="preserveStartEnd"
              minTickGap={25}
              stroke="#A3B8B2"
              strokeOpacity={0.4}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dy={10}
            />

            <YAxis
              stroke="#A3B8B2"
              strokeOpacity={0.6}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              domain={[0, 5]}
              ticks={[0, 1, 2, 3, 4, 5]}
            />

            <Tooltip
              shared={true}
              content={<CustomTooltip />}
              cursor={{ stroke: "#A3B8B2", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.4 }}
            />

            <Area
              type="stepAfter"
              dataKey="PlotValue"
              stroke="none"
              fill="url(#lightAreaFade)"
              isAnimationActive={false}
              activeDot={false}
            />

            <Line
              type="stepAfter"
              dataKey="PlotValue"
              stroke={SENSOR_COLORS.good}
              strokeWidth={2.5}
              isAnimationActive={false}
              dot={false}
              activeDot={<CustomActiveDot />}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}