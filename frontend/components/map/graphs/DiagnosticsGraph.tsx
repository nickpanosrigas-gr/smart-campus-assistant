// frontend/components/map/graphs/DiagnosticsGraph.tsx
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
import { Activity } from "lucide-react";
import { SENSOR_COLORS } from "@/components/map/constants";

interface DiagnosticsGraphProps {
  artifact: any;
}

const getBatteryColor = (val: number, isPluggedIn: boolean) => {
  if (isPluggedIn) return SENSOR_COLORS.good;
  if (val < 15) return SENSOR_COLORS.error;
  if (val < 50) return SENSOR_COLORS.critical;
  return SENSOR_COLORS.good;
};

export default function DiagnosticsGraph({ artifact }: DiagnosticsGraphProps) {
  const isBuilding = artifact?.room_id === "building";

  // --- 1. DYNAMIC METRIC CONFIGURATION ---
  const METRICS = useMemo(() => {
    if (isBuilding) {
      return [
        { key: "good", label: "Healthy Sensors", unit: "sensors", color: SENSOR_COLORS.good, isDynamic: false, isPluggedIn: false, domain: [0, 150] as any },
        { key: "warning", label: "Warning Sensors", unit: "sensors", color: SENSOR_COLORS.warning, isDynamic: false, isPluggedIn: false, domain: [0, 150] as any },
        { key: "critical", label: "Critical Sensors", unit: "sensors", color: SENSOR_COLORS.critical, isDynamic: false, isPluggedIn: false, domain: [0, 150] as any },
        { key: "error", label: "Offline/Error Sensors", unit: "sensors", color: SENSOR_COLORS.error, isDynamic: false, isPluggedIn: false, domain: [0, 150] as any },
        { key: "average_battery", label: "Average Battery", unit: "%", color: SENSOR_COLORS.good, isDynamic: true, isPluggedIn: false, domain: [0, 100] as any }
      ];
    } else {
      const keys = Object.keys(artifact?.metadata || {});
      return keys.map(k => {
        const isPlugged = artifact.metadata[k] === "Plugged In";
        return {
          key: k,
          label: k, // Sensor name
          unit: isPlugged ? "" : "%",
          color: SENSOR_COLORS.good, // Fallback color
          isDynamic: !isPlugged,
          isPluggedIn: isPlugged,
          domain: [0, 100] as any
        };
      });
    }
  }, [artifact, isBuilding]);

  // --- 2. CONTINUOUS TIME BUCKETING ENGINE ---
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

    // Forward-fill missing fields across the payload for isolated series drops
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

    return grid;
  }, [artifact, METRICS]);

  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timeMs);
  }, [formattedData]);

  if (!artifact || !artifact.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center select-none">
        <Activity size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        <p className="text-xs font-mono uppercase tracking-wider">No Diagnostic Telemetry Recorded</p>
      </div>
    );
  }

  // --- 3. FORMATTING HELPERS & GENERATORS ---
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

  const renderGradientStops = (dataKey: string, isDynamic: boolean, fixedColor: string) => {
    const len = formattedData.length;
    if (len === 0) return null;

    if (!isDynamic) {
      return (
        <>
          <stop offset="0%" stopColor={fixedColor} />
          <stop offset="100%" stopColor={fixedColor} />
        </>
      );
    }

    const stops = [];
    let prevColor = getBatteryColor(formattedData[0][dataKey], false);
    stops.push(<stop key="start" offset="0%" stopColor={prevColor} />);

    for (let i = 1; i < len; i++) {
      const currVal = formattedData[i][dataKey];
      if (currVal === null || currVal === undefined) continue;
      const currColor = getBatteryColor(currVal, false);
      const prevY = formattedData[i - 1][dataKey];
      const currY = currVal;
      const currPct = (i / (len - 1)) * 100;

      if (prevColor !== currColor) {
        if (prevY >= currY) {
          const endPrev = Math.min(100, currPct + 0.15);
          const startCurr = Math.min(100, currPct + 0.16);
          stops.push(<stop key={`drop-prev-${i}`} offset={`${endPrev}%`} stopColor={prevColor} />);
          stops.push(<stop key={`drop-curr-${i}`} offset={`${startCurr}%`} stopColor={currColor} />);
        } else {
          const endPrev = Math.max(0, currPct - 0.16);
          const startCurr = Math.max(0, currPct - 0.15);
          stops.push(<stop key={`jump-prev-${i}`} offset={`${endPrev}%`} stopColor={prevColor} />);
          stops.push(<stop key={`jump-curr-${i}`} offset={`${startCurr}%`} stopColor={currColor} />);
        }
        prevColor = currColor;
      }
    }
    stops.push(<stop key="end" offset="100%" stopColor={prevColor} />);
    return stops;
  };

  // --- 4. EXACT TOOLTIP ENGINE ---
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

      const rawVal = payload[0].value;
      const displayVal = typeof rawVal === "number" ? (metric.isPluggedIn ? 100 : Number(rawVal.toFixed(1))) : rawVal;

      // Determine tooltip color context
      const activeColor = metric.isDynamic 
        ? getBatteryColor(rawVal, metric.isPluggedIn) 
        : metric.color;

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[140px] text-center">
          <div className="text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5 border-b border-white/10">
            {timeStr}
          </div>
          <div className="flex items-center justify-center pt-0.5">
            <span className="font-bold text-lg leading-none" style={{ color: activeColor }}>
              {metric.isPluggedIn ? "Plugged In" : displayVal}
              {!metric.isPluggedIn && (
                <span className="text-[11px] font-mono font-bold uppercase tracking-wider ml-1 opacity-80" style={{ color: activeColor }}>
                  {metric.unit}
                </span>
              )}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomActiveDot = (props: any) => {
    const { cx, cy, payload, metric } = props;
    if (cx === undefined || cy === undefined || !payload) return null;
    
    const activeColor = metric.isDynamic 
      ? getBatteryColor(payload[metric.key], metric.isPluggedIn) 
      : metric.color;

    return (
      <circle 
        cx={cx} cy={cy} r={5} 
        fill={activeColor} 
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
        {METRICS.map((metric) => {
          // SOLID FLATLINE BYPASS: Eliminates the 0.0001 hack and fixes SVG clipping natively
          const firstVal = formattedData[0]?.[metric.key];
          const isFlat = formattedData.every(pt => pt[metric.key] === firstVal);

          const getStrokeFill = () => {
            if (!isFlat) return `url(#gradLineDiag-${metric.key})`;
            if (!metric.isDynamic) return metric.color;
            return getBatteryColor(firstVal ?? 0, metric.isPluggedIn);
          };

          const strokeFill = getStrokeFill();

          return (
            <div key={metric.key} className="w-full shrink-0 min-h-[140px] relative mt-2">
              <h3 className="absolute top-1 left-8 text-[10px] font-mono uppercase tracking-widest text-[#A3B8B2]/60 z-10">
                {metric.label}
              </h3>

              {/* Plugged In Centered Label */}
              {metric.isPluggedIn && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                  <span className="text-[15px] font-mono font-bold uppercase tracking-widest text-[#14C89B]/50 bg-[#0A0A0A]/50 px-4 py-1.5 rounded-full border border-[#14C89B]/10 backdrop-blur-sm shadow-md">
                    Plugged In
                  </span>
                </div>
              )}
              
              <ResponsiveContainer width="100%" height="100%">
                {/* NOTE: No syncId is used here, ensuring tooltips/dots only appear on the graph currently hovered */}
                <ComposedChart
                  data={formattedData}
                  margin={{ top: 25, right: 30, left: 0, bottom: 5 }}
                >
                  {/* Define Localized SVG Masks and Gradients ensuring guaranteed cross-browser rendering */}
                  <defs>
                    <linearGradient id={`verticalFadeMask-${metric.key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0.0} />
                    </linearGradient>
                    <mask id={`diagAreaMask-${metric.key}`}>
                      <rect x="0" y="0" width="100%" height="100%" fill={`url(#verticalFadeMask-${metric.key})`} />
                    </mask>

                    {!isFlat && (
                      <linearGradient id={`gradLineDiag-${metric.key}`} x1="0%" y1="0%" x2="100%" y2="0%">
                        {renderGradientStops(metric.key, metric.isDynamic, metric.color)}
                      </linearGradient>
                    )}
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
                    tick={false} // Never show ticks here to save vertical space; mapped to fixed bottom axis
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
                    allowDataOverflow={false}
                  />

                  <Tooltip
                    content={<CustomTooltip metric={metric} />}
                    cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }}
                  />

                  {/* Area Shadowing Effect */}
                  <Area
                    type="stepAfter"
                    dataKey={metric.key}
                    stroke="none"
                    fill={strokeFill}
                    mask={`url(#diagAreaMask-${metric.key})`}
                    isAnimationActive={false}
                    activeDot={false}
                    connectNulls={true}
                  />

                  <Line
                    type="stepAfter"
                    dataKey={metric.key}
                    stroke={strokeFill}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                    connectNulls={true}
                    activeDot={<CustomActiveDot metric={metric} />}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          );
        })}
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