"use client";
import React from "react";
import OccupancyGraph from "./OccupancyGraph";
import { BarChart3, MousePointerClick, Loader2 } from "lucide-react";

interface GraphViewProps {
  activeTools: string[];
  selectedRooms: string[];
  roomArtifacts: Record<string, any>;
  allArtifacts: Record<string, Record<string, Record<string, any>>>;
  timeframe: string;
}

export default function GraphView({
  activeTools,
  selectedRooms,
  roomArtifacts,
  allArtifacts,
  timeframe
}: GraphViewProps) {
  
  // ---> UPDATED: Removed background/border, removed bounce animation, updated copy <---
  if (!selectedRooms || selectedRooms.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-transparent text-[#A3B8B2] select-none">
        <MousePointerClick size={40} className="mb-3 text-[#14C89B]" />
        <h3 className="text-lg font-bold text-white">Select a Room</h3>
        <p className="text-xs text-[#A3B8B2]/60 max-w-sm mt-1">
          Use the Sidebar to select a Floor and then a Room
        </p>
      </div>
    );
  }

  // ---> UPDATED: Removed background/border, renamed title, updated copy <---
  if (!activeTools || activeTools.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-transparent text-[#A3B8B2] select-none">
        <BarChart3 size={40} className="mb-3 text-[#14C89B]" />
        <h3 className="text-lg font-bold text-white">Select a Tool</h3>
        <p className="text-xs text-[#A3B8B2]/60 max-w-sm mt-1">
          Use the Toggle below to select a Tool
        </p>
      </div>
    );
  }

  const targetRoom = selectedRooms[0];
  const activeTool = activeTools[0];

  // Case-insensitive 3D Cache Lookup
  const fullRoomCache = allArtifacts?.[targetRoom] || {};
  let artifact = undefined;

  for (const [domainKey, tfMap] of Object.entries(fullRoomCache)) {
    if (domainKey.toLowerCase() === activeTool?.toLowerCase()) {
      artifact = (tfMap as any)?.[timeframe];
      break;
    }
  }

  // Fallback to active view artifacts if needed
  if (!artifact && roomArtifacts?.[targetRoom]) {
    const fallbackData = roomArtifacts[targetRoom];
    if (fallbackData.domain?.toLowerCase() === activeTool?.toLowerCase() && fallbackData.timeframe === timeframe) {
      artifact = fallbackData;
    } else {
      for (const [domainKey, val] of Object.entries(fallbackData)) {
        if (domainKey.toLowerCase() === activeTool?.toLowerCase() && (val as any)?.timeframe === timeframe) {
          artifact = val;
          break;
        }
      }
    }
  }

  // ---> UPDATED: Removed background/border on loading state for visual consistency <---
  if (!artifact) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-transparent text-[#A3B8B2] select-none">
        <Loader2 size={36} className="mb-3 text-[#14C89B] animate-spin" />
        <h3 className="text-base font-bold text-white">Loading Telemetry...</h3>
      </div>
    );
  }

  switch (activeTool.toLowerCase()) {
    case "occupancy":
      return <OccupancyGraph artifact={artifact} />;

    case "air quality":
    case "climate":
    case "lights":
    case "doors/windows":
    case "diagnostics":
      return (
        <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-[#061C16]/40 rounded-3xl border border-[#0A664F]/50 text-[#A3B8B2]">
          <BarChart3 size={36} className="mb-3 text-[#14C89B]" />
          <h3 className="text-lg font-bold text-white">{activeTool} Graph</h3>
          <p className="text-xs text-[#A3B8B2]/70 mt-1 max-w-md">
            The data payload for Room {targetRoom.toUpperCase()} ({artifact.timeframe}) was successfully received! 
            The dedicated UI chart for <strong className="text-[#14C89B]">{activeTool}</strong> will be added next.
          </p>
          <pre className="mt-4 p-3 bg-black/50 rounded-xl border border-white/5 text-[10px] font-mono text-left w-full max-w-md overflow-hidden text-ellipsis">
            {JSON.stringify({ timeframe: artifact.timeframe, dataPoints: artifact.series?.length || 0 }, null, 2)}
          </pre>
        </div>
      );

    default:
      return (
        <div className="w-full h-full flex items-center justify-center p-8 text-[#A3B8B2]/60 italic">
          Unsupported Graph Domain: {activeTool}
        </div>
      );
  }
}