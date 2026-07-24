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
import { Users, Activity } from "lucide-react";
import { SENSOR_COLORS, ROOM_COLORS } from "@/components/map/constants";

interface OccupancyGraphProps {
  artifact: any;
}

export default function OccupancyGraph({ artifact }: OccupancyGraphProps) {
  // --- 1. AUTOMATIC POLYMORPHIC DETECTION ---
  const isMotionOnly = useMemo(() => {
    if (!artifact) return false;
    if (artifact.metadata) {
      return !("Occupancy" in artifact.metadata) && "Motion" in artifact.metadata;
    }
    if (artifact.series && artifact.series.length > 0) {
      return !artifact.series.some((pt: any) => pt.Occupancy !== undefined && pt.Occupancy !== null);
    }
    return false;
  }, [artifact]);

  // --- 2. HIGH-DENSITY CONTINUOUS HOVER & EXCEPTION ENGINE ---
  const formattedData = useMemo(() => {
    if (!artifact || !artifact.series || artifact.series.length === 0) return [];

    const sorted = [...artifact.series].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const parsedSorted = sorted.map(pt => ({
      ...pt,
      timeMs: new Date(pt.timestamp).getTime()
    }));

    // ---> CRITICAL EXCEPTION: For 30d & 90d, overwrite Last Day (today) with Previous Day's telemetry! <---
    if (["30d", "90d"].includes(artifact.timeframe) && parsedSorted.length >= 2) {
      const prevDay = parsedSorted[parsedSorted.length - 2];
      const lastDay = parsedSorted[parsedSorted.length - 1];
      lastDay.Occupancy = prevDay.Occupancy;
      lastDay.Motion = prevDay.Motion;
    }

    const startTime = parsedSorted[0].timeMs;
    let majorStepMs = 0;
    let minorStepMs = 0;
    let majorBucketsCount = 0;

    switch (artifact.timeframe) {
      case "2h":
        majorStepMs = 10 * 60 * 1000; // 10-min major buckets
        minorStepMs = 30 * 1000;      // 30-sec continuous hover points
        majorBucketsCount = 13;       // 13 intervals -> 14 boundary timestamps
        break;
      case "24h":
        majorStepMs = 2 * 60 * 60 * 1000; // 2-hour major buckets
        minorStepMs = 2 * 60 * 1000;      // 2-min continuous hover points
        majorBucketsCount = 13;           // 13 intervals -> 14 boundary timestamps
        break;
      case "7d":
        majorStepMs = 2 * 60 * 60 * 1000; // 2-hour major buckets
        minorStepMs = 10 * 60 * 1000;     // 10-min continuous hover points
        majorBucketsCount = 85;           // 85 intervals -> 86 boundary timestamps
        break;
      case "30d":
        majorStepMs = 24 * 60 * 60 * 1000; // 1-day major buckets
        minorStepMs = 1 * 60 * 60 * 1000;  // 1-hour continuous hover points
        majorBucketsCount = 31;            // ---> UPDATED: 31 intervals -> 32 timestamps (closes at Jul 26) <---
        break;
      case "90d":
        majorStepMs = 24 * 60 * 60 * 1000; // 1-day major buckets
        minorStepMs = 3 * 60 * 60 * 1000;  // 3-hour continuous hover points
        majorBucketsCount = 91;            // ---> UPDATED: 91 intervals -> 92 timestamps (closes at Jul 26) <---
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
      const occ = match.Occupancy !== undefined ? match.Occupancy : 0;
      const mot = match.Motion !== undefined ? match.Motion : 0;
      const plotValue = isMotionOnly ? mot : occ;

      const majorBucketIndex = Math.floor((timeFromStart - (isClosingBoundary ? 1 : 0)) / majorStepMs);
      const bucketStartTime = startTime + majorBucketIndex * majorStepMs;
      const bucketEndTime = bucketStartTime + majorStepMs;

      grid.push({
        ...match,
        timestamp: new Date(gridTime).toISOString(),
        Occupancy: occ,
        Motion: mot,
        PlotValue: plotValue,
        isMajorBoundary,
        isClosingBoundary,
        bucketStartTime,
        bucketEndTime
      });
    }

    // ---> EPSILON HACK: If all values are identical, add 0.0001 to point 0. <---
    // This gives the SVG DOM a valid height (>0) for objectBoundingBox without altering visual flatness!
    if (grid.length > 0) {
      const firstVal = grid[0].PlotValue;
      const allSame = grid.every((pt: any) => pt.PlotValue === firstVal);
      if (allSame) {
        grid[0].PlotValue += 0.0001;
      }
    }

    return grid;
  }, [artifact, isMotionOnly]);

  // --- 3. VERTICAL AXIS DOMAIN WITH BREATHING BUFFER ---
  const { yAxisMin, yAxisMax, capacity } = useMemo(() => {
    if (isMotionOnly) {
      return { capacity: null, yAxisMin: -0.05, yAxisMax: 1.05 };
    }
    const cap = artifact?.metadata?.capacity || null;
    const maxDataVal = Math.max(...formattedData.map((d: any) => d.PlotValue), 0);
    const max = cap ? cap : Math.max(maxDataVal, 5);
    return {
      capacity: cap,
      yAxisMin: -max * 0.02,
      yAxisMax: max * 1.02
    };
  }, [artifact, formattedData, isMotionOnly]);

  // --- 4. MAJOR TICKS ---
  const majorTicks = useMemo(() => {
    return formattedData
      .filter((pt: any) => pt.isMajorBoundary)
      .map((pt: any) => pt.timestamp);
  }, [formattedData]);

  if (!artifact || !artifact.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/50 p-8 text-center pb-32 select-none">
        {isMotionOnly ? (
          <Activity size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        ) : (
          <Users size={32} className="mb-2 text-[#14C89B]/30 animate-pulse" />
        )}
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

  // --- 5. TOP-LINE PRIORITY GRADIENT GENERATOR ---
  const renderGradientStops = () => {
    const stops = [];
    const len = formattedData.length;
    if (len === 0) return null;

    let prevColor = formattedData[0].Motion === 1 ? SENSOR_COLORS.warning : SENSOR_COLORS.good;
    stops.push(<stop key="start" offset="0%" stopColor={prevColor} />);

    for (let i = 1; i < len; i++) {
      const currColor = formattedData[i].Motion === 1 ? SENSOR_COLORS.warning : SENSOR_COLORS.good;
      const prevY = formattedData[i - 1].PlotValue;
      const currY = formattedData[i].PlotValue;
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

  // --- 6. TOOLTIP WITH BASELINE ALIGNMENT & 30D/90D EXACT DAY DISPLAY ---
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const startDate = new Date(dataPoint.bucketStartTime);
      const endDate = new Date(dataPoint.bucketEndTime);
      const tf = artifact.timeframe;

      const isLongTerm = ["30d", "90d"].includes(tf);
      let timeStr = "";

      if (isLongTerm) {
        // ---> Correctly displays the bucket start date (e.g., "Jul 25") <---
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

      // Round out the 0.0001 epsilon hack so tooltips display clean integers!
      const val = Math.round(dataPoint.Occupancy);
      const isMotionActive = dataPoint.Motion === 1;

      const badgeColor = isMotionActive ? SENSOR_COLORS.warning : SENSOR_COLORS.good;
      const badgeBg = isMotionActive ? ROOM_COLORS.warning : ROOM_COLORS.good;
      const badgeText = isMotionActive ? "Motion Active" : "Motion Idle";

      return (
        <div className="flex flex-col gap-2 bg-[#0A0A0A]/95 p-3 rounded-2xl border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.8)] pointer-events-none backdrop-blur-md min-w-[150px]">
          {isMotionOnly ? (
            <>
              <div className="text-center text-xs font-mono font-semibold text-[#A3B8B2]/90 pb-0.5">
                {timeStr}
              </div>
              <div className="flex items-center justify-center pt-0.5">
                <span
                  className="px-3 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 shadow-sm"
                  style={{
                    backgroundColor: `${badgeBg}90`,
                    borderColor: badgeColor,
                    color: badgeColor
                  }}
                >
                  <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: badgeColor }} />
                  {badgeText}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-baseline justify-between gap-4 text-xs font-mono font-semibold pb-1">
                <span className="text-[#A3B8B2]/80 leading-none">{timeStr}</span>
                <span className="font-bold text-base leading-none" style={{ color: badgeColor }}>
                  {val} <span className="text-[10px] text-[#A3B8B2]/60 font-normal">people</span>
                </span>
              </div>

              <div className="flex items-center justify-center pt-0.5">
                <span
                  className="px-3 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border flex items-center gap-1.5 shadow-sm"
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
            </>
          )}
        </div>
      );
    }
    return null;
  };

  const CustomActiveDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (cx === undefined || cy === undefined || !payload) return null;
    const color = payload.Motion === 1 ? SENSOR_COLORS.warning : SENSOR_COLORS.good;
    return <circle cx={cx} cy={cy} r={6} fill={color} stroke="#0A0A0A" strokeWidth={2} />;
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
              left: isMotionOnly ? 15 : -10,
              bottom: 20
            }}
          >
            <defs>
              {/* Reverted to default objectBoundingBox for 100% pixel-perfect X-coordinate alignment! */}
              <linearGradient id="motionStrokeGradMinimal" x1="0%" y1="0%" x2="100%" y2="0%">
                {renderGradientStops()}
              </linearGradient>

              <linearGradient id="verticalFadeMask" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ffffff" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#ffffff" stopOpacity={0.0} />
              </linearGradient>

              <mask id="occAreaMask">
                <rect x="0" y="0" width="100%" height="100%" fill="url(#verticalFadeMask)" />
              </mask>
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
              domain={[yAxisMin, yAxisMax]}
              ticks={isMotionOnly ? [0, 1] : undefined}
              tickFormatter={
                isMotionOnly
                  ? (val) => (val === 1 ? "Active" : val === 0 ? "Idle" : "")
                  : undefined
              }
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
              fill="url(#motionStrokeGradMinimal)"
              mask="url(#occAreaMask)"
              isAnimationActive={false}
              activeDot={false}
            />

            <Line
              type="stepAfter"
              dataKey="PlotValue"
              stroke="url(#motionStrokeGradMinimal)"
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