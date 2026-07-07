// frontend/components/map/floors/DataOverlay.tsx
import React from "react";
import { SENSOR_COLORS, RoomHealth } from "../constants";

interface DataOverlayProps {
  artifact?: any;
  roomId?: string; // Optional context for room-specific labels
}

export default function DataOverlay({ artifact, roomId }: DataOverlayProps) {
  if (!artifact) return null;

  if (artifact.view_type === "error") {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center font-sans">
        <span className="text-xs font-light text-center leading-tight whitespace-normal text-[#C84B5E] drop-shadow-md">
          {artifact.message || "Offline"}
        </span>
      </div>
    );
  }

  const domain = artifact.domain;
  // Fallback ensures it works for both BuildingView and Floor2Base
  const aggs = artifact.building_aggregates || artifact.room_aggregates; 
  const results = artifact.results;
  const status = (artifact.status as RoomHealth) || "good";

  // If there are no aggregates/results and it's not a schedule tool, don't render
  if (!aggs && !results && domain !== "Schedule") return null;

  const textColor = SENSOR_COLORS[status] || SENSOR_COLORS.good;

  // Render a clean, borderless, typography-focused UI
  const renderContent = () => {
    if (domain === "Schedule") {
      // Look directly for the schedule payload we defined in the backend
      const activeClass = artifact.schedule_data;
      
      if (!activeClass) {
        // No class is taking place: Display the dynamic backend message
        return (
          <div className="flex flex-col items-center justify-center gap-1 p-2 text-center">
            <span className="text-lg font-light tracking-tight leading-snug" style={{ color: textColor }}>
              {artifact.message || "Free"}
            </span>
          </div>
        );
      } else {
        // Class is currently active
        return (
          <div className="flex flex-col items-start justify-center gap-1 w-full max-w-full overflow-hidden">
            <span className="text-sm font-medium tracking-wide truncate w-full" style={{ color: textColor }}>
              {activeClass.course_name}
            </span>
            <div className="text-[10px] font-light text-[#A3B8B2]/80 leading-tight w-full">
              <p className="truncate text-white mb-1">{activeClass.start_time} - {activeClass.end_time}</p>
              <p className="truncate">{activeClass.instructor_name}</p>
              <p className="text-[#A3B8B2]/50 mt-1 uppercase tracking-widest text-[8px] truncate">{activeClass.course_type}</p>
            </div>
          </div>
        );
      }
    }

    switch (domain) {
      case "Climate":
        return (
          <div className="flex flex-col items-center justify-center gap-1">
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-light tracking-tighter" style={{ color: textColor }}>
                {aggs?.temperature ? `${aggs.temperature}°C` : "--"}
              </span>
              <span className="text-2xl font-light text-[#A3B8B2]/70">
                {aggs?.humidity ? `${aggs.humidity}%` : "--"}
              </span>
            </div>
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#A3B8B2]/50">
              Climate
            </span>
          </div>
        );

      case "Occupancy":
        return (
          <div className="flex flex-col items-center justify-center gap-1">
            <span className="text-4xl font-light tracking-tighter" style={{ color: textColor }}>
              {aggs?.occupancy ?? "--"} <span className="text-xl text-[#A3B8B2]/60">/ {aggs?.capacity ?? "--"}</span>
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#A3B8B2]/50">
              Occupancy
            </span>
          </div>
        );

      case "Air Quality":
        return (
          <div className="flex flex-col items-center justify-center gap-2">
            <div className="flex items-baseline gap-5">
              <div className="flex flex-col items-center">
                <span className="text-3xl font-light" style={{ color: textColor }}>{aggs?.co2 || "--"}</span>
                <span className="text-[9px] uppercase tracking-widest text-[#A3B8B2]/50">CO2</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-3xl font-light" style={{ color: textColor }}>{aggs?.pm2_5 || "--"}</span>
                <span className="text-[9px] uppercase tracking-widest text-[#A3B8B2]/50">PM2.5</span>
              </div>
            </div>
          </div>
        );

      case "Lights":
      case "Doors/Windows":
        const topVal = domain === "Lights" ? `${aggs?.light_level ?? "0"} Lvl` : `${aggs?.open_count ?? "0"} Open`;
        const label = domain === "Lights" ? "Illumination" : "Active Access";
        return (
          <div className="flex flex-col items-center justify-center gap-1">
            <span className="text-3xl font-light tracking-tight" style={{ color: textColor }}>{topVal}</span>
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#A3B8B2]/50">{label}</span>
          </div>
        );

      case "Diagnostics":
        return (
          <div className="flex flex-col items-center justify-center gap-1 text-sm font-light text-[#A3B8B2]">
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] mb-1 text-white/40">Status: {aggs?.total || 0} Total</span>
            <div className="flex gap-3">
               <div className="flex flex-col items-center"><span style={{ color: SENSOR_COLORS.good }} className="text-xl">{aggs?.good || 0}</span><span className="text-[8px] uppercase">OK</span></div>
               <div className="flex flex-col items-center"><span style={{ color: SENSOR_COLORS.warning }} className="text-xl">{aggs?.warning || 0}</span><span className="text-[8px] uppercase">Warn</span></div>
               <div className="flex flex-col items-center"><span style={{ color: SENSOR_COLORS.error }} className="text-xl">{aggs?.error || 0}</span><span className="text-[8px] uppercase">Err</span></div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="w-full h-full flex items-center justify-center font-sans drop-shadow-lg p-1">
      {renderContent()}
    </div>
  );
}