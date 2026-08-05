// frontend/components/map/graphs/DoorsWindowsGraph.tsx
"use client";
import React, { useMemo, useState, useRef } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip
} from "recharts";
import { Lock } from "lucide-react";
import { SENSOR_COLORS, ROOM_COLORS } from "@/components/map/constants";

interface DoorsWindowsGraphProps {
  artifact: any;
}

// Unified palette: Green for Open (Active), Orange for Closed (Critical)
const COLOR_OPEN = SENSOR_COLORS.good;
const COLOR_CLOSED = SENSOR_COLORS.critical;
const BG_OPEN = ROOM_COLORS.good;
const BG_CLOSED = ROOM_COLORS.critical;

// --- NEW: BUILDING AGGREGATE VIEW ---
const BuildingAggregateGraph = ({ artifact }: { artifact: any }) => {
  const { formattedData, startTime, endTime, maxDoors, maxWindows } = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0) {
      return { formattedData: [], startTime: 0, endTime: 0, maxDoors: 0, maxWindows: 0 };
    }

    // 1. Sort chronologically
    const parsedSorted = [...artifact.series]
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map(pt => ({ ...pt, timeMs: new Date(pt.timestamp).getTime() }));

    // ---> NEW: Boundary Back-fill Hack <---
    // The backend injects an artificial boundary point at index 0 to stretch the graph, 
    // but the aggregate sums incorrectly default to 0. 
    // We fix this by finding the first *real* telemetry value and copying it backward.
    if (["2h", "24h", "7d"].includes(artifact.timeframe) && parsedSorted.length > 1) {
      // slice(1) ensures we skip the artificial boundary point and look for actual data
      const firstRealDoors = parsedSorted.slice(1).find(pt => pt.open_doors !== undefined)?.open_doors;
      const firstRealWindows = parsedSorted.slice(1).find(pt => pt.open_windows !== undefined)?.open_windows;
      
      if (firstRealDoors !== undefined) parsedSorted[0].open_doors = firstRealDoors;
      if (firstRealWindows !== undefined) parsedSorted[0].open_windows = firstRealWindows;
    }

    // 2. Fill-forward delta logic: if a value is missing, carry over the last known state
    let currentDoors = parsedSorted[0].open_doors || 0;
    let currentWindows = parsedSorted[0].open_windows || 0;
    
    const filledSorted = parsedSorted.map(pt => {
      if (pt.open_doors !== undefined) currentDoors = pt.open_doors;
      if (pt.open_windows !== undefined) currentWindows = pt.open_windows;
      return { ...pt, open_doors: currentDoors, open_windows: currentWindows };
    });

    const sTime = filledSorted[0].timeMs;
    let majorStepMs = 0;
    let minorStepMs = 0;
    let majorBucketsCount = 0;

    switch (artifact.timeframe) {
      case "2h": majorStepMs = 10 * 60 * 1000; minorStepMs = 30 * 1000; majorBucketsCount = 12; break;
      case "24h": majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 2 * 60 * 1000; majorBucketsCount = 12; break;
      case "7d": majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 10 * 60 * 1000; majorBucketsCount = 84; break;
      case "30d": majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 1 * 60 * 60 * 1000; majorBucketsCount = 31; break;
      case "90d": majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 3 * 60 * 60 * 1000; majorBucketsCount = 91; break;
      default: return { formattedData: filledSorted, startTime: sTime, endTime: filledSorted[filledSorted.length - 1].timeMs, maxDoors: 0, maxWindows: 0 };
    }

    const eTime = sTime + majorBucketsCount * majorStepMs;

    // 3. Grid Generation (Including Exact Timestamps Fix)
    const timeSet = new Set<number>();
    const count = Math.round((eTime - sTime) / minorStepMs) + 1;
    for (let i = 0; i < count; i++) timeSet.add(sTime + i * minorStepMs);
    filledSorted.forEach(pt => timeSet.add(pt.timeMs));
    
    const sortedGridTimes = Array.from(timeSet).sort((a, b) => a - b);
    
    let matchIdx = 0;
    const grid = sortedGridTimes.map(gridTime => {
      while (matchIdx < filledSorted.length - 1 && filledSorted[matchIdx + 1].timeMs <= gridTime) {
        matchIdx++;
      }
      const match = filledSorted[matchIdx];
      
      return {
        timeMs: gridTime,
        timestamp: new Date(gridTime).toISOString(),
        open_doors: match.open_doors,
        open_windows: match.open_windows,
        isMajorBoundary: Math.abs((gridTime - sTime) % majorStepMs) < 100 || Math.abs(gridTime - eTime) < 100
      };
    });

    return { 
      formattedData: grid, 
      startTime: sTime, 
      endTime: eTime,
      maxDoors: artifact.total_doors || 20,
      maxWindows: artifact.total_windows || 10
    };
  }, [artifact]);

  const majorTicks = useMemo(() => formattedData.filter((pt: any) => pt.isMajorBoundary).map((pt: any) => pt.timeMs), [formattedData]);

  const formatXAxisTick = (timestamp: number) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const tf = artifact.timeframe;
    if (["30d", "90d"].includes(tf)) return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    if (["7d", "24h"].includes(tf)) return date.toLocaleDateString("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
    return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  };

  const CustomTooltip = ({ active, payload, type }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const val = payload[0].value;
      const date = new Date(dataPoint.timeMs);
      const tf = artifact.timeframe;
      
      let timeStr = "";
      if (["30d", "90d"].includes(tf)) {
        timeStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      } else if (["7d", "24h"].includes(tf)) {
        const day = date.toLocaleDateString("en-US", { weekday: "short" });
        const time = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${day} ${time}`;
      } else {
        timeStr = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
      }

      const label = type === "doors" ? "Open Doors" : "Open Windows";

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[120px] text-center">
          <div className="text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5">{timeStr}</div>
          <div className="flex items-center justify-center gap-1.5 pt-0.5">
            <span className="font-bold text-base leading-none" style={{ color: COLOR_OPEN }}>{val}</span>
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider" style={{ color: COLOR_OPEN }}>{label}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomActiveDot = (props: any) => {
    const { cx, cy } = props;
    if (cx === undefined || cy === undefined) return null;
    return <circle cx={cx} cy={cy} r={5} fill={COLOR_OPEN} stroke="#0A0A0A" strokeWidth={2} style={{ pointerEvents: "none" }} />;
  };

  return (
    <div className="w-full h-full flex flex-col justify-center gap-6 bg-transparent p-4 pb-8 select-none">
      {/* 1. DOORS GRAPH */}
      <div className="w-full flex-1 min-h-[140px] relative">
        <h3 className="absolute -top-4 left-4 text-[10px] font-mono uppercase tracking-widest text-[#A3B8B2]/60 z-10">Total Open Doors</h3>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={formattedData} margin={{ top: 15, right: 30, left: -20, bottom: 5 }}>
            <defs>
              <linearGradient id="areaFadeDoors" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={COLOR_OPEN} stopOpacity={0.35} />
                <stop offset="95%" stopColor={COLOR_OPEN} stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="timeMs" type="number" domain={[startTime, endTime]} ticks={majorTicks} tickFormatter={formatXAxisTick} tick={false} axisLine={false} tickLine={false} />
            <YAxis domain={[0, maxDoors]} stroke="#A3B8B2" strokeOpacity={0.6} fontSize={10} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip content={<CustomTooltip type="doors" />} cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }} />
            <Area type="stepAfter" dataKey="open_doors" stroke="none" fill="url(#areaFadeDoors)" isAnimationActive={false} activeDot={false} />
            <Line type="stepAfter" dataKey="open_doors" stroke={COLOR_OPEN} strokeWidth={2} dot={false} isAnimationActive={false} activeDot={<CustomActiveDot />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* 2. WINDOWS GRAPH */}
      <div className="w-full flex-1 min-h-[140px] relative">
        <h3 className="absolute -top-4 left-4 text-[10px] font-mono uppercase tracking-widest text-[#A3B8B2]/60 z-10">Total Open Windows</h3>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={formattedData} margin={{ top: 15, right: 30, left: -20, bottom: 20 }}>
            <defs>
              <linearGradient id="areaFadeWindows" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={COLOR_OPEN} stopOpacity={0.35} />
                <stop offset="95%" stopColor={COLOR_OPEN} stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="timeMs" type="number" domain={[startTime, endTime]} ticks={majorTicks} tickFormatter={formatXAxisTick} stroke="#A3B8B2" strokeOpacity={0.4} fontSize={11} axisLine={false} tickLine={false} dy={10} minTickGap={25} interval="preserveStartEnd" />
            <YAxis domain={[0, maxWindows]} stroke="#A3B8B2" strokeOpacity={0.6} fontSize={10} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip content={<CustomTooltip type="windows" />} cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }} />
            <Area type="stepAfter" dataKey="open_windows" stroke="none" fill="url(#areaFadeWindows)" isAnimationActive={false} activeDot={false} />
            <Line type="stepAfter" dataKey="open_windows" stroke={COLOR_OPEN} strokeWidth={2} dot={false} isAnimationActive={false} activeDot={<CustomActiveDot />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default function DoorsWindowsGraph({ artifact }: DoorsWindowsGraphProps) {
  // ---> NEW: Intercept building-level queries <---
  if (artifact?.room_id === "building") {
    return <BuildingAggregateGraph artifact={artifact} />;
  }

  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // --- 1. EXTRACT SENSORS & ASSIGN DYNAMIC Y-AXIS SLICES ---
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

      if (metaObj && typeof metaObj === "object") {
        friendlyLabel = metaObj.label || key;
      } else if (typeof metaObj === "string") {
        friendlyLabel = metaObj;
      }

      // Format name: remove underscores for clean display
      friendlyLabel = friendlyLabel.replace(/_/g, ' ');

      // Assign non-overlapping mathematical Y-axis slices normalized to 100-units per row.
      const rowMin = idx * 100;
      const rowMax = (idx + 1) * 100;
      
      const baseValue = rowMin + 15; // 15% padding at bottom
      const openValue = rowMin + 85; // 15% padding at top
      const centerY = rowMin + 50;

      return { 
        key, 
        friendlyLabel, 
        rowMin,
        rowMax,
        baseValue, 
        openValue, 
        centerY
      };
    });
  }, [artifact]);

  // --- 2. EXACT EVENT TIMESTAMPS & TIMELINE GENERATION ---
  const { formattedData, startTime, endTime } = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0 || sensorSeries.length === 0) {
      return { formattedData: [], startTime: 0, endTime: 0 };
    }

    const parsedSorted = [...artifact.series]
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map(pt => ({ ...pt, timeMs: new Date(pt.timestamp).getTime() }));

    const sTime = parsedSorted[0].timeMs;

    // A. Define bucket parameters and strict counts
    let majorStepMs = 0;
    let minorStepMs = 0;
    let majorBucketsCount = 0;

    switch (artifact.timeframe) {
      case "2h": majorStepMs = 10 * 60 * 1000; minorStepMs = 30 * 1000; majorBucketsCount = 12; break;
      case "24h": majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 2 * 60 * 1000; majorBucketsCount = 12; break;
      case "7d": majorStepMs = 2 * 60 * 60 * 1000; minorStepMs = 10 * 60 * 1000; majorBucketsCount = 84; break;
      case "30d": majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 1 * 60 * 60 * 1000; majorBucketsCount = 31; break;
      case "90d": majorStepMs = 24 * 60 * 60 * 1000; minorStepMs = 3 * 60 * 60 * 1000; majorBucketsCount = 91; break;
      default: return { formattedData: parsedSorted, startTime: sTime, endTime: parsedSorted[parsedSorted.length - 1].timeMs };
    }

    const eTime = sTime + majorBucketsCount * majorStepMs;

    // B. Pre-calculate exact state durations for precise tooltips
    const sensorIntervals: Record<string, any[]> = {};
    sensorSeries.forEach(s => {
      const events = parsedSorted.filter(pt => pt[s.key] !== undefined);
      const intervals = [];
      
      let currentState = events.length > 0 ? (events[0][s.key] >= 1 ? 1 : 0) : 0;
      let currentStart = sTime;

      for (let i = 0; i < events.length; i++) {
        const ev = events[i];
        const state = ev[s.key] >= 1 ? 1 : 0;
        
        if (state !== currentState) {
          intervals.push({ start: currentStart, end: ev.timeMs, state: currentState });
          currentStart = ev.timeMs;
          currentState = state;
        }
      }
      
      intervals.push({ start: currentStart, end: eTime, state: currentState });
      sensorIntervals[s.key] = intervals;
    });

    // C. Build the high-density grid for smooth rendering & continuous hovering
    const count = Math.round((eTime - sTime) / minorStepMs) + 1;
    
    // 1. Create a Set to hold all our grid timestamps
    const timeSet = new Set<number>();
    
    // 2. Add the mathematical step intervals
    for (let i = 0; i < count; i++) {
      timeSet.add(sTime + i * minorStepMs);
    }
    
    // 3. THE FIX: Inject the exact event timestamps so short spikes aren't missed
    parsedSorted.forEach(pt => timeSet.add(pt.timeMs));
    
    // 4. Sort all the timestamps chronologically
    const sortedGridTimes = Array.from(timeSet).sort((a, b) => a - b);
    
    const grid: any[] = [];

    // 5. Build the grid using the merged timestamps
    sortedGridTimes.forEach(gridTime => {
      const timeFromStart = gridTime - sTime;

      const isMajorBoundary =
        Math.abs(timeFromStart % majorStepMs) < 100 ||
        Math.abs(gridTime - eTime) < 100;
      
      const point: any = { 
        timeMs: gridTime, 
        timestamp: new Date(gridTime).toISOString(), 
        isMajorBoundary 
      };
      
      sensorSeries.forEach(s => {
        let iv = sensorIntervals[s.key].find(interval => gridTime >= interval.start && gridTime < interval.end);
        if (!iv) iv = sensorIntervals[s.key][sensorIntervals[s.key].length - 1]; 
        
        point[s.key + "_val"] = iv.state;
        point[s.key + "_plot"] = iv.state === 1 ? s.openValue : s.baseValue;
        point[s.key + "_bg"] = s.openValue; 
        point[s.key + "_startTime"] = iv.start;
        point[s.key + "_endTime"] = iv.end;
      });

      grid.push(point);
    });

    return { formattedData: grid, startTime: sTime, endTime: eTime };
  }, [artifact, sensorSeries]);

  // --- 3. ANALYZE FLAT LINES TO FIX SVG BOUNDING-BOX CULLING BUG ---
  const flatStatus = useMemo(() => {
    const status: Record<string, { isFlat: boolean, state: number }> = {};
    if (formattedData.length > 0) {
      sensorSeries.forEach(s => {
        const firstVal = formattedData[0][s.key + "_val"];
        const isFlat = formattedData.every((pt: any) => pt[s.key + "_val"] === firstVal);
        status[s.key] = { isFlat, state: firstVal };
      });
    }
    return status;
  }, [formattedData, sensorSeries]);

  // --- 4. DYNAMIC Y-AXIS ROW SCALING ---
  const { yAxisMin, yAxisMax, yTicks } = useMemo(() => {
    const ticks: number[] = [];
    sensorSeries.forEach(s => {
      ticks.push(s.baseValue, s.centerY, s.openValue);
    });

    return {
      yTicks: ticks,
      yAxisMin: 0,
      yAxisMax: Math.max(100, sensorSeries.length * 100)
    };
  }, [sensorSeries]);

  // --- 5. MAJOR TICKS ---
  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timeMs);
  }, [formattedData]);

  if (!formattedData.length) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center pb-8 select-none">
        <Lock size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        <p className="text-xs font-mono uppercase tracking-wider">No Door/Window Telemetry Recorded</p>
      </div>
    );
  }

  // --- 6. FORMATTING HELPERS ---
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

  const renderGradientStops = (sensorKey: string) => {
    const stops = [];
    const len = formattedData.length;
    if (len === 0) return null;

    // 1. Calculate the total duration of the graph
    const totalTime = endTime - startTime;

    let prevColor = formattedData[0][`${sensorKey}_val`] === 1 ? COLOR_OPEN : COLOR_CLOSED;
    stops.push(<stop key="start" offset="0%" stopColor={prevColor} />);

    for (let i = 1; i < len; i++) {
      const currColor = formattedData[i][`${sensorKey}_val`] === 1 ? COLOR_OPEN : COLOR_CLOSED;
      const prevY = formattedData[i - 1][`${sensorKey}_plot`];
      const currY = formattedData[i][`${sensorKey}_plot`];
      
      // 2. THE FIX: Base the gradient percentage on exact time, not array index
      const currTime = formattedData[i].timeMs;
      const currPct = totalTime > 0 ? ((currTime - startTime) / totalTime) * 100 : 0;

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

  // --- 7. CUSTOM Y-AXIS TICK (Multi-line Name & Exact Boundaries) ---
  const CustomYAxisTick = (props: any) => {
    const { x, y, payload } = props;
    const val = payload.value;
    
    const sensor = sensorSeries.find(s => s.baseValue === val || s.openValue === val || s.centerY === val);
    if (!sensor) return null;
    
    if (val === sensor.centerY) {
      const name = sensor.friendlyLabel;
      const words = name.split(' ');
      const lines: string[] = [];
      
      if (name.length <= 16) {
        lines.push(name);
      } else if (words.length <= 4) {
        const mid = Math.ceil(words.length / 2);
        lines.push(words.slice(0, mid).join(' '));
        lines.push(words.slice(mid).join(' '));
      } else {
        const third = Math.ceil(words.length / 3);
        lines.push(words.slice(0, third).join(' '));
        lines.push(words.slice(third, third * 2).join(' '));
        if (third * 2 < words.length) {
          lines.push(words.slice(third * 2).join(' '));
        }
      }
      
      return (
        <text x={x - 55} y={y} textAnchor="end" fill="#A3B8B2" fontSize={12} className="font-mono font-semibold">
          {lines.map((line, i) => {
            let dy = "1.2em";
            if (i === 0) {
              if (lines.length === 1) dy = "0.3em";
              if (lines.length === 2) dy = "-0.3em";
              if (lines.length === 3) dy = "-0.9em";
            }
            return (
              <tspan key={i} x={x - 55} dy={dy}>
                {line}
              </tspan>
            );
          })}
        </text>
      );
    } 
    else if (val === sensor.openValue) {
      return (
        <text x={x - 10} y={y} dy={4} textAnchor="end" fill={COLOR_OPEN} fontSize={10} className="font-mono font-bold">
          Open
        </text>
      );
    } 
    else if (val === sensor.baseValue) {
      return (
        <text x={x - 10} y={y} dy={4} textAnchor="end" fill={COLOR_CLOSED} fontSize={10} className="font-mono font-bold">
          Closed
        </text>
      );
    }
    return null;
  };

  // --- 8. ISOLATED TOOLTIP WITH SPECIFIC STATE DURATIONS ---
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length > 0 && hoveredKey) {
      const dataPoint = payload[0].payload; 
      const sensor = sensorSeries.find(s => s.key === hoveredKey);
      if (!sensor) return null;

      const isOpen = dataPoint[hoveredKey + "_val"] === 1;
      const startDate = new Date(dataPoint[hoveredKey + "_startTime"]);
      const endDate = new Date(dataPoint[hoveredKey + "_endTime"]);
      const tf = artifact.timeframe;

      let timeStr = "";
      if (["30d", "90d"].includes(tf)) {
        const startDay = startDate.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        const endDay = endDate.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        timeStr = `${startDay} – ${endDay}`;
      } else {
        const startDay = startDate.toLocaleDateString("en-US", { weekday: "short" });
        const endDay = endDate.toLocaleDateString("en-US", { weekday: "short" });
        const startTime = startDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const endTime = endDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        timeStr = `${startDay} ${startTime} – ${endDay} ${endTime}`;
      }
      
      const statusColor = isOpen ? COLOR_OPEN : COLOR_CLOSED;
      const statusBg = isOpen ? BG_OPEN : BG_CLOSED;
      const statusText = isOpen ? "OPEN" : "CLOSED";

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[150px] text-center">
          <div className="text-center text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5">
            {timeStr}
          </div>
          
          <div className="flex items-center justify-center pt-0.5">
            <span
              className="px-3 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 shadow-sm"
              style={{
                backgroundColor: `${statusBg}90`,
                borderColor: statusColor,
                color: statusColor
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: statusColor }} />
              {statusText}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  // --- 9. NATIVE HOVER MATH ---
  const handleNativeMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (sensorSeries.length === 1) {
      setHoveredKey(sensorSeries[0].key);
      return;
    }

    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const chartTopMargin = 20;
    const chartBottomMargin = 20;
    
    const usableHeight = rect.height - chartTopMargin - chartBottomMargin;
    if (usableHeight <= 0) return;

    const relativeY = (e.clientY - rect.top) - chartTopMargin;
    const pct = Math.max(0, Math.min(1, relativeY / usableHeight)); 

    const dataY = yAxisMax - pct * (yAxisMax - yAxisMin);
    const hoveredSensor = sensorSeries.find(s => dataY >= s.rowMin && dataY <= s.rowMax);
    
    if (hoveredSensor) {
      setHoveredKey(hoveredSensor.key);
    } else {
      setHoveredKey(null);
    }
  };

  // --- 10. CUSTOM ACTIVE DOT (Multi-Graph Aware) ---
  const CustomActiveDot = (props: any) => {
    const { cx, cy, value, sensor } = props;
    if (cx === undefined || cy === undefined || !sensor) return null;
    
    // Determine the color based on the current Y value
    const dotColor = value === sensor.openValue ? COLOR_OPEN : COLOR_CLOSED;

    return (
      <circle 
        cx={cx} 
        cy={cy} 
        r={6} 
        fill={dotColor} 
        stroke="#0A0A0A" 
        strokeWidth={2} 
        style={{ pointerEvents: "none" }}
      />
    );
  };

  return (
    <div className="w-full h-full flex flex-col justify-center bg-transparent p-4 pb-8 select-none">
      <div 
        className="w-full h-full relative" 
        ref={containerRef}
        onMouseMove={handleNativeMouseMove}
        onMouseLeave={() => setHoveredKey(null)}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={formattedData}
            margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
          >
            <defs>
              {sensorSeries.map(s => {
                const flatInfo = flatStatus[s.key];
                if (flatInfo && flatInfo.isFlat) return null;
                return (
                  <linearGradient key={`gradLine-${s.key}`} id={`gradLine-${s.key}`} x1="0%" y1="0%" x2="100%" y2="0%">
                    {renderGradientStops(s.key)}
                  </linearGradient>
                );
              })}
            </defs>

            <XAxis
              type="number"
              scale="time"
              domain={[startTime, endTime]}
              dataKey="timeMs"
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
              domain={[yAxisMin, yAxisMax]}
              ticks={yTicks}
              tick={<CustomYAxisTick />}
              width={260}
            />

            <Tooltip
              shared={true}
              cursor={{ stroke: "#ffffff", strokeWidth: 1, strokeDasharray: "3 3", strokeOpacity: 0.25 }}
              content={<CustomTooltip />}
            />

            {sensorSeries.map(s => {
              const flatInfo = flatStatus[s.key];
              const strokeFill = (flatInfo && flatInfo.isFlat) 
                ? (flatInfo.state === 1 ? COLOR_OPEN : COLOR_CLOSED) 
                : `url(#gradLine-${s.key})`;

              return (
                <React.Fragment key={s.key}>
                  <Area
                    type="stepAfter"
                    dataKey={s.key + "_bg"}
                    baseValue={s.baseValue}
                    stroke={COLOR_CLOSED}
                    strokeOpacity={0.25}
                    strokeWidth={1}
                    fill={BG_CLOSED}
                    fillOpacity={0.15}
                    isAnimationActive={false}
                    activeDot={false}
                  />

                  <Area
                    type="stepAfter"
                    dataKey={s.key + "_plot"}
                    baseValue={s.baseValue}
                    stroke="none"
                    fill={strokeFill}
                    fillOpacity={0.4}
                    isAnimationActive={false}
                    activeDot={false}
                  />

                  <Line
                    type="stepAfter"
                    dataKey={s.key + "_plot"}
                    stroke={strokeFill}
                    strokeWidth={2}
                    isAnimationActive={false}
                    dot={false}
                    activeDot={
                      hoveredKey === s.key 
                        ? <CustomActiveDot sensor={s} /> 
                        : false
                    }
                  />
                </React.Fragment>
              );
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}