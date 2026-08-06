"use client";
import React from "react";
import OccupancyGraph from "./OccupancyGraph";
import DoorsWindowsGraph from "./DoorsWindowsGraph";
import LightsGraph from "./LightsGraph";
import ClimateGraph from "./ClimateGraph";
import AirQualityGraph from "./AirQualityGraph";
import { BarChart3, MousePointerClick, Loader2 } from "lucide-react";
import DiagnosticsGraph from "./DiagnosticsGraph";

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

    case "doors/windows":
    case "doors & windows":
      return <DoorsWindowsGraph artifact={artifact} />;
    
    case "lights":
      return <LightsGraph artifact={artifact} />;

    case "climate":
      return <ClimateGraph artifact={artifact} />;

    case "air quality":
      return <AirQualityGraph artifact={artifact} />;

    case "diagnostics":
      return < DiagnosticsGraph artifact={artifact}/>;

    default:
      return (
        <div className="w-full h-full flex items-center justify-center p-8 text-[#A3B8B2]/60 italic">
          Unsupported Graph Domain: {activeTool}
        </div>
      );
  }
}