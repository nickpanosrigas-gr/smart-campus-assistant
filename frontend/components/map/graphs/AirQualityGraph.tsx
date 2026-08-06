// frontend/components/map/graphs/AirQualityGraph.tsx
"use client";
import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { Wind } from "lucide-react";
import { SENSOR_COLORS, ROOM_COLORS } from "@/components/map/constants";

interface AirQualityGraphProps {
  artifact: any;
}

// 1. Define the 6 specific graphs in the requested order with hard-coded standard domains
const METRICS = [
  { key: "co2", label: "Indoor CO₂", unit: "ppm", color: SENSOR_COLORS.good, bg: ROOM_COLORS.good, domain: [0, 5000] },
  { key: "pm2_5", label: "Indoor PM2.5", unit: "µg/m³", color: SENSOR_COLORS.good, bg: ROOM_COLORS.good, domain: [0, 100] },
  { key: "outdoor_pm2_5", label: "Outdoor PM2.5", unit: "µg/m³", color: SENSOR_COLORS.critical, bg: ROOM_COLORS.critical, domain: [0, 100] },
  { key: "pm10", label: "Indoor PM10", unit: "µg/m³", color: SENSOR_COLORS.good, bg: ROOM_COLORS.good, domain: [0, 100] },
  { key: "outdoor_pm10", label: "Outdoor PM10", unit: "µg/m³", color: SENSOR_COLORS.critical, bg: ROOM_COLORS.critical, domain: [0, 100] },
  { key: "tvoc", label: "Indoor TVOC", unit: "ppb", color: SENSOR_COLORS.good, bg: ROOM_COLORS.good, domain: [0, 30] },
];

export default function AirQualityGraph({ artifact }: AirQualityGraphProps) {
  // --- 1. CONTINUOUS TIME BUCKETING ENGINE ---
  const formattedData = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0) return [];

    const sorted = [...artifact.series].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const parsedSorted = sorted.map((pt: any) => ({
      ...pt,
      timeMs: new Date(pt.timestamp).getTime()
    }));

    // CRITICAL EXCEPTION: For 30d & 90d, overwrite Last Day (today) with Previous Day's telemetry
    if (["30d", "90d"].includes(artifact.timeframe) && parsedSorted.length >= 2) {
      const prevDay = parsedSorted[parsedSorted.length - 2];
      const lastDay = parsedSorted[parsedSorted.length - 1];
      METRICS.forEach(m => {
        if (prevDay[m.key] !== undefined) {
          lastDay[m.key] = prevDay[m.key];
        }
      });
    }

    // Forward-fill missing fields across the payload for isolated series drops (e.g., outdoor sensor ticks separately)
    const lastKnown: Record<string, number | undefined> = {};
    const filledSorted = parsedSorted.map(pt => {
      const newPt = { ...pt };
      METRICS.forEach(m => {
        if (newPt[m.key] !== undefined) lastKnown[m.key] = newPt[m.key];
        else if (lastKnown[m.key] !== undefined) newPt[m.key] = lastKnown[m.key];
      });
      return newPt;
    });

    const startTime = filledSorted[0].timeMs;
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
        return filledSorted;
    }

    const endTime = startTime + majorBucketsCount * majorStepMs;
    const totalDurationMs = endTime - startTime;
    const count = Math.round(totalDurationMs / minorStepMs) + 1;
    const grid: any[] = [];

    let currentMatchIndex = 0;

    for (let i = 0; i < count; i++) {
      const gridTime = startTime + i * minorStepMs;
      const timeFromStart = gridTime - startTime;

      const isMajorBoundary =
        Math.abs(timeFromStart % majorStepMs) < 100 ||
        Math.abs(gridTime - endTime) < 100;
      
      const isClosingBoundary = i === count - 1;

      while (
        currentMatchIndex < filledSorted.length - 1 &&
        filledSorted[currentMatchIndex + 1].timeMs <= gridTime
      ) {
        currentMatchIndex++;
      }

      const match = filledSorted[currentMatchIndex];
      const majorBucketIndex = Math.floor((timeFromStart - (isClosingBoundary ? 1 : 0)) / majorStepMs);
      const bucketStartTime = startTime + majorBucketIndex * majorStepMs;
      const bucketEndTime = bucketStartTime + majorStepMs;

      const dataPoint: any = {
        timestamp: new Date(gridTime).toISOString(),
        timeMs: gridTime,
        isMajorBoundary,
        isClosingBoundary,
        bucketStartTime,
        bucketEndTime
      };

      METRICS.forEach(m => {
        dataPoint[m.key] = match[m.key] !== undefined ? match[m.key] : null;
      });

      grid.push(dataPoint);
    }

    // EPSILON HACK: Prevent SVG render crashes on perfectly flat lines
    if (grid.length > 0) {
      METRICS.forEach(m => {
        const firstVal = grid[0][m.key];
        if (firstVal !== null && firstVal !== undefined) {
          const allSame = grid.every((pt: any) => pt[m.key] === firstVal);
          if (allSame) {
            grid[0][m.key] += 0.0001;
          }
        }
      });
    }

    return grid;
  }, [artifact]);

  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timeMs);
  }, [formattedData]);

  if (!artifact || !artifact.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center select-none">
        <Wind size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        <p className="text-xs font-mono uppercase tracking-wider">No Air Quality Telemetry Recorded</p>
      </div>
    );
  }

  // --- 2. X-AXIS FORMATTING ---
  const formatXAxisTick = (timestamp: number) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const tf = artifact.timeframe;

    if (["30d", "90d"].includes(tf)) {
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    }
    if (["7d", "24h"].includes(tf)) {
      return date.toLocaleDateString("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
    }
    return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  };

  // --- 3. EXACT TOOLTIP ENGINE ---
  const CustomTooltip = ({ active, payload, metric }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const startDate = new Date(dataPoint.bucketStartTime);
      const endDate = new Date(dataPoint.bucketEndTime);
      const tf = artifact.timeframe;

      let timeStr = "";

      if (["30d", "90d"].includes(tf)) {
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

      // Filter out the Epsilon Hack decimal
      const rawVal = payload[0].value;
      const displayVal = typeof rawVal === "number" ? Number(rawVal.toFixed(2)) : rawVal;

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[140px] text-center">
          <div className="text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5 border-b border-white/10">
            {timeStr}
          </div>
          <div className="flex items-center justify-center pt-0.5">
            <span className="font-bold text-lg leading-none" style={{ color: metric.color }}>
              {displayVal}
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider ml-1 opacity-80" style={{ color: metric.color }}>
                {metric.unit}
              </span>
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  // --- 4. CUSTOM ACTIVE DOT ---
  const CustomActiveDot = (props: any) => {
    const { cx, cy, metric } = props;
    if (cx === undefined || cy === undefined) return null;
    return (
      <circle 
        cx={cx} cy={cy} r={5} 
        fill={metric.color} 
        stroke="#0A0A0A" 
        strokeWidth={2} 
        style={{ pointerEvents: "none" }}
      />
    );
  };

  return (
    <div className="w-full h-full flex flex-col bg-transparent p-4 pb-4 select-none overflow-hidden">
      {/* --- Scrollable Graphs Container --- */}
      <div className="flex-1 w-full overflow-y-auto overflow-x-hidden chat-scrollbar pr-2 flex flex-col gap-6">
        {METRICS.map((metric) => (
          <div key={metric.key} className="w-full shrink-0 min-h-[160px] relative mt-2">
            <h3 className="absolute top-1 left-8 text-[10px] font-mono uppercase tracking-widest text-[#A3B8B2]/60 z-10">
              {metric.label}
            </h3>
            
            <ResponsiveContainer width="100%" height="100%">
              {/* NOTE: No syncId is used here, ensuring tooltips/dots only appear on the graph currently hovered */}
              <ComposedChart
                data={formattedData}
                margin={{ top: 25, right: 30, left: 0, bottom: 5 }}
              >
                <defs>
                  <linearGradient id={`areaFadeAQ-${metric.key}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={metric.color} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={metric.color} stopOpacity={0.0} />
                  </linearGradient>
                </defs>

                <XAxis
                  dataKey="timeMs"
                  type="number"
                  domain={[formattedData[0]?.timeMs, formattedData[formattedData.length - 1]?.timeMs]}
                  ticks={majorTicks}
                  tickFormatter={formatXAxisTick}
                  stroke="#A3B8B2"
                  strokeOpacity={0.4}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tick={false} // Ticks are mapped to the fixed bottom axis
                  minTickGap={25}
                  interval="preserveStartEnd"
                />

                <YAxis
                  width={45} // Fixed width guarantees correct left alignment without clipping
                  domain={metric.domain}
                  stroke="#A3B8B2"
                  strokeOpacity={0.6}
                  fontSize={10}
                  axisLine={false}
                  tickLine={false}
                  allowDataOverflow={true}
                />

                <Tooltip
                  content={<CustomTooltip metric={metric} />}
                  cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }}
                />

                <Area
                  type="monotone"
                  dataKey={metric.key}
                  stroke="none"
                  fill={`url(#areaFadeAQ-${metric.key})`}
                  isAnimationActive={false}
                  activeDot={false}
                  connectNulls={true}
                />

                <Line
                  type="monotone"
                  dataKey={metric.key}
                  stroke={metric.color}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls={true}
                  activeDot={<CustomActiveDot metric={metric} />}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* --- Fixed Bottom X-Axis Container --- */}
      <div className="w-full h-[24px] shrink-0 mt-1 pointer-events-none pr-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={formattedData} margin={{ top: 0, right: 30, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="timeMs"
              type="number"
              domain={[formattedData[0]?.timeMs, formattedData[formattedData.length - 1]?.timeMs]}
              ticks={majorTicks}
              tickFormatter={formatXAxisTick}
              stroke="#A3B8B2"
              strokeOpacity={0.4}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              minTickGap={25}
              interval="preserveStartEnd"
            />
            {/* Hidden YAxis with identical explicit width perfectly aligns the grids */}
            <YAxis width={45} hide domain={[0, 1]} /> 
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}