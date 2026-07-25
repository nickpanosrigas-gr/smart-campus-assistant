// frontend/components/map/graphs/DoorsWindowsGraph.tsx
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
import { Lock, Layers } from "lucide-react";
import { SENSOR_COLORS, ROOM_COLORS } from "@/components/map/constants";

interface DoorsWindowsGraphProps {
  artifact: any;
}

// Dedicated palette: Orange for Doors (Security/Entry), Sky Blue for Windows (Ventilation)
const DW_PALETTE = {
  door: { stroke: "#E8863A", fill: "#A8651D", label: "Door" },
  window: { stroke: "#38BDF8", fill: "#0284C7", label: "Window" },
  default: { stroke: "#14C89B", fill: "#0A664F", label: "Sensor" }
};

export default function DoorsWindowsGraph({ artifact }: DoorsWindowsGraphProps) {
  // --- 1. EXTRACT SENSORS & ASSIGN DEDICATED GANTT ROW SLICES ---
  const sensorSeries = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0) return [];
    
    const keys = new Set<string>();
    artifact.series.forEach((pt: any) => {
      Object.keys(pt).forEach(k => {
        if (k !== "timestamp" && k !== "timeMs") keys.add(k);
      });
    });

    return Array.from(keys).map((key, idx) => {
      const metaObj = artifact?.metadata?.[key];
      let friendlyLabel = key;
      let devType = "Sensor";

      if (metaObj && typeof metaObj === "object") {
        friendlyLabel = metaObj.label || key;
        devType = metaObj.type || "Sensor";
      } else if (typeof metaObj === "string") {
        friendlyLabel = metaObj;
        if (key.toLowerCase().includes("door")) devType = "Door";
        else if (key.toLowerCase().includes("window")) devType = "Window";
      } else {
        if (key.toLowerCase().includes("door")) devType = "Door";
        else if (key.toLowerCase().includes("window")) devType = "Window";
      }

      let theme = DW_PALETTE.default;
      if (devType.toLowerCase() === "door" || key.toLowerCase().includes("door")) theme = DW_PALETTE.door;
      else if (devType.toLowerCase() === "window" || key.toLowerCase().includes("window")) theme = DW_PALETTE.window;

      // Assign non-overlapping mathematical Y-axis slices: 
      // Each row gets 20 units total (10 unit bar height + 10 unit gap)
      const baseValue = idx * 20;
      const openValue = baseValue + 10;
      const centerY = baseValue + 5;

      return { 
        key, 
        friendlyLabel, 
        type: devType, 
        baseValue, 
        openValue, 
        centerY, 
        ...theme 
      };
    });
  }, [artifact]);

  // --- 2. STATE RETENTION & GANTT BLOCK ENGINE ---
  const formattedData = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0 || sensorSeries.length === 0) return [];

    const sorted = [...artifact.series].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const parsedSorted = sorted.map(pt => ({
      ...pt,
      timeMs: new Date(pt.timestamp).getTime()
    }));

    // Forward-filling state retention for sparse delta payloads
    const runningState: Record<string, number> = {};
    sensorSeries.forEach(s => { runningState[s.key] = 0; });

    const filledSorted = parsedSorted.map(pt => {
      sensorSeries.forEach(s => {
        if (pt[s.key] !== undefined && pt[s.key] !== null) {
          runningState[s.key] = Number(pt[s.key]);
        }
      });
      return {
        ...pt,
        ...runningState
      };
    });

    // For 30d & 90d, overwrite Last Day with Previous Day's telemetry
    if (["30d", "90d"].includes(artifact.timeframe) && filledSorted.length >= 2) {
      const prevDay = filledSorted[filledSorted.length - 2];
      const lastDay = filledSorted[filledSorted.length - 1];
      sensorSeries.forEach(s => { lastDay[s.key] = prevDay[s.key]; });
    }

    const startTime = filledSorted[0].timeMs;
    let majorStepMs = 0;
    let minorStepMs = 0;
    let majorBucketsCount = 0;

    switch (artifact.timeframe) {
      case "2h":
        majorStepMs = 10 * 60 * 1000; minorStepMs = 30 * 1000; majorBucketsCount = 13; break;
      case "24h":
        majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 2 * 60 * 1000; majorBucketsCount = 13; break;
      case "7d":
        majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 10 * 60 * 1000; majorBucketsCount = 85; break;
      case "30d":
        majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 1 * 60 * 60 * 1000; majorBucketsCount = 31; break;
      case "90d":
        majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 3 * 60 * 60 * 1000; majorBucketsCount = 91; break;
      default:
        return filledSorted;
    }

    const endTime = startTime + majorBucketsCount * majorStepMs;
    const count = Math.round((endTime - startTime) / minorStepMs) + 1;
    const grid: Record<string, any>[] = [];

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
      const point: Record<string, any> = {
        timestamp: new Date(gridTime).toISOString(),
        isMajorBoundary,
        isClosingBoundary,
        bucketStartTime: startTime + Math.floor((timeFromStart - (isClosingBoundary ? 1 : 0)) / majorStepMs) * majorStepMs,
        bucketEndTime: startTime + (Math.floor((timeFromStart - (isClosingBoundary ? 1 : 0)) / majorStepMs) + 1) * majorStepMs
      };

      // Map each sensor state to its exact Gantt Row Y-axis coordinates!
      sensorSeries.forEach(s => {
        const rawVal = match[s.key] !== undefined ? match[s.key] : 0;
        const isOpen = Math.round(rawVal) >= 1;
        
        point[s.key + "_val"] = isOpen ? 1 : 0;           // Clean boolean for tooltip
        point[s.key + "_plot"] = isOpen ? s.openValue : s.baseValue; // Solid rectangular Gantt bar
        point[s.key + "_bg"] = s.openValue;               // Constant height for background track
      });

      grid.push(point);
    }

    // Epsilon hack to prevent SVG bounding box height from collapsing to 0
    if (grid.length > 0) {
      sensorSeries.forEach(s => {
        const firstVal = grid[0][s.key + "_plot"];
        if (grid.every(pt => pt[s.key + "_plot"] === firstVal)) {
          grid[0][s.key + "_plot"] += 0.0001;
        }
      });
    }

    return grid;
  }, [artifact, sensorSeries]);

  // --- 3. DYNAMIC Y-AXIS ROW SCALING ---
  const { yAxisMin, yAxisMax, yTicks } = useMemo(() => {
    const len = Math.max(sensorSeries.length, 1);
    return {
      yTicks: sensorSeries.map(s => s.centerY),
      yAxisMin: -2,
      yAxisMax: len * 20 - 8
    };
  }, [sensorSeries]);

  // --- 4. MAJOR TICKS ---
  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timestamp);
  }, [formattedData]);

  if (!artifact || !artifact.series || artifact.series.length === 0 || sensorSeries.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center pb-32 select-none">
        <Lock size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        <p className="text-xs font-mono uppercase tracking-wider">No Door/Window Telemetry Recorded</p>
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

  const formatYAxisTick = (val: number) => {
    const sensor = sensorSeries.find(s => s.centerY === val);
    if (!sensor) return "";
    return `${sensor.friendlyLabel} (${sensor.type})`;
  };

  // --- 5. DAW / GANTT TIMELINE TOOLTIP ---
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
        const startTime = startDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const endTime = endDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${startDay} ${startTime} – ${endTime}`;
      } else {
        const startTime = startDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const endTime = endDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${startTime} – ${endTime}`;
      }

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[220px]">
          {/* Top: Date & Time Interval */}
          <div className="text-left text-xs font-mono font-semibold text-[#A3B8B2]/80 pb-1.5 border-b border-white/10 flex items-center justify-between">
            <span>{timeStr}</span>
            <Layers size={13} className="text-[#14C89B]" />
          </div>

          {/* Bottom: Sensor Rows sorted top-to-bottom exactly as they appear on chart */}
          <div className="flex flex-col gap-2 pt-0.5">
            {[...sensorSeries].reverse().map(s => {
              const isOpen = dataPoint[s.key + "_val"] === 1;

              const badgeColor = isOpen ? s.stroke : SENSOR_COLORS.good;
              const badgeBg = isOpen ? (s.fill || ROOM_COLORS.warning) : ROOM_COLORS.good;
              const badgeText = `${isOpen ? "OPEN" : "CLOSED"} • ${s.type.toUpperCase()}`;

              return (
                <div key={s.key} className="flex items-center justify-between gap-4 text-xs font-mono">
                  <div className="flex items-center gap-2 truncate">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: s.stroke }} />
                    <span className="text-white/90 truncate font-medium">{s.friendlyLabel}</span>
                  </div>
                  <span
                    className="px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider border shrink-0 flex items-center gap-1 shadow-sm"
                    style={{
                      backgroundColor: `${badgeBg}90`,
                      borderColor: badgeColor,
                      color: badgeColor
                    }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: badgeColor }} />
                    {badgeText}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    return null;
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
              left: 20,
              bottom: 20
            }}
          >
            <defs>
              {sensorSeries.map(s => (
                <linearGradient key={`grad-${s.key}`} id={`grad-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={s.stroke} stopOpacity={0.85} />
                  <stop offset="95%" stopColor={s.fill || s.stroke} stopOpacity={0.45} />
                </linearGradient>
              ))}
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

            {/* Dedicated Y-Axis displaying friendly sensor names and types! */}
            <YAxis
              stroke="#A3B8B2"
              strokeOpacity={0.6}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              domain={[yAxisMin, yAxisMax]}
              ticks={yTicks}
              tickFormatter={formatYAxisTick}
              width={160}
            />

            {/* Delicate hairline guide aligning all rows at the hovered timestamp */}
            <Tooltip
              shared={true}
              cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }}
              content={<CustomTooltip />}
            />

            {/* --- GENERATE DEDICATED GANTT ROW FOR EACH SENSOR --- */}
            {sensorSeries.map(s => (
              <React.Fragment key={s.key}>
                {/* 1. Translucent Background Track (shows sensor is online & monitored when Closed) */}
                <Area
                  type="stepAfter"
                  dataKey={s.key + "_bg"}
                  baseValue={s.baseValue}
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeWidth={1}
                  fill="#ffffff"
                  fillOpacity={0.03}
                  isAnimationActive={false}
                  activeDot={false}
                />

                {/* 2. Vibrant Solid Gantt Block (instantly fills height 10 when Open) */}
                <Area
                  type="stepAfter"
                  dataKey={s.key + "_plot"}
                  baseValue={s.baseValue}
                  stroke="none"
                  fill={`url(#grad-${s.key})`}
                  isAnimationActive={false}
                  activeDot={false}
                />

                {/* 3. Sharp DAW Border Outline along top of active block */}
                <Line
                  type="stepAfter"
                  dataKey={s.key + "_plot"}
                  stroke={s.stroke}
                  strokeWidth={2}
                  isAnimationActive={false}
                  dot={false}
                  activeDot={{ r: 5, fill: s.stroke, stroke: "#0A0A0A", strokeWidth: 2 }}
                />
              </React.Fragment>
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}