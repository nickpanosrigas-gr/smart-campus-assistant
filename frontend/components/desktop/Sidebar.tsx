"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Timeframe, ViewType } from "@/app/page";

interface SidebarProps {
  activeLevel: string;
  setActiveLevel: (level: string) => void;
  selectedRooms: string[];
  onRoomToggle: (roomId: string) => void;
  activeTools: string[];
  floorStates: Record<string, { selectedRooms: string[]; activeTools: string[] }>;
  timeframe: Timeframe;
  onTimeframeChange: (tf: Timeframe) => void;
  viewMode: ViewType;
  onViewModeChange: (mode: ViewType) => void;
  artifactCache: Record<string, Record<string, Record<string, any>>>;
  lastHistoricalTimeframe: Timeframe;
}

const FLOORS = [
  { id: "B", label: "B", rooms: [{ id: "building", name: "Building" }] },
  { id: "5", label: "5", rooms: [{ id: "5.6", name: "Room 5.6" }, { id: "5.7", name: "Room 5.7" }] },
  { id: "4", label: "4", rooms: [{ id: "4.9", name: "Room 4.9" }] },
  { id: "3", label: "3", rooms: [{ id: "3.7", name: "Room 3.7" }, { id: "3.8", name: "Room 3.8" }, { id: "3.9", name: "Room 3.9" }] },
  { id: "2", label: "2", rooms: [{ id: "2.1", name: "Room 2.1" }, { id: "2.2", name: "Room 2.2" }, { id: "2.3", name: "Room 2.3" }, { id: "2.4", name: "Room 2.4" }] },
  { id: "1", label: "1", rooms: [{ id: "1.1", name: "Room 1.1" }, { id: "1.2", name: "Room 1.2" }] },
  { id: "0", label: "0", rooms: [{ id: "entrance", name: "Entrance" }, { id: "restaurant", name: "Restaurant" }] },
  { id: "-1", label: "-1", rooms: [{ id: "kitchen", name: "Kitchen" }, { id: "data_center", name: "Data Center" }] },
  { id: "-2", label: "-2", rooms: [{ id: "parkin.b", name: "Parking B" }] },
  { id: "-3", label: "-3", rooms: [{ id: "parkin.c", name: "Parking C" }] },
];

const TIMEFRAMES: Timeframe[] = ["2h", "24h", "7d", "30d", "90d"];

// ---> Human-Readable Timeframe Labels <---
const TIMEFRAME_LABELS: Record<Timeframe, string> = {
  "now": "Now",
  "2h": "2 Hours",
  "24h": "24 Hours",
  "7d": "7 Days",
  "30d": "30 Days",
  "90d": "90 Days",
};

export default function Sidebar({ 
  activeLevel, 
  setActiveLevel, 
  selectedRooms, 
  onRoomToggle, 
  activeTools, 
  floorStates,
  timeframe,
  onTimeframeChange,
  viewMode,
  onViewModeChange,
  artifactCache,
  lastHistoricalTimeframe
}: SidebarProps) {
  const [isTimeframeMenuOpen, setIsTimeframeMenuOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(true);

  const bottomSectionRef = useRef<HTMLDivElement>(null);
  const mode = viewMode === "graph" ? "graph" : "map";

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (bottomSectionRef.current && !bottomSectionRef.current.contains(event.target as Node)) {
        setIsTimeframeMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const floorHasData = (floorId: string) => {
    if (!artifactCache) return false;
    const floorObj = FLOORS.find(f => f.id === floorId);
    if (!floorObj) return false;

    return floorObj.rooms.some(room => {
      const roomCache = artifactCache[room.id];
      if (!roomCache) return false;
      if (activeTools.length > 0) {
        const activeTool = activeTools[0];
        return Object.keys(roomCache).some(k => k.toLowerCase() === activeTool.toLowerCase() && !!roomCache[k]?.[timeframe]);
      }
      return Object.values(roomCache).some(toolCache => !!toolCache?.[timeframe]);
    });
  };

  const hasCachedData = (roomId: string) => {
    if (!artifactCache[roomId]) return false;
    if (activeTools.length > 0) {
      const activeTool = activeTools[0];
      return Object.keys(artifactCache[roomId]).some(
        k => k.toLowerCase() === activeTool.toLowerCase() && !!artifactCache[roomId][k]?.[timeframe]
      );
    }
    return Object.values(artifactCache[roomId]).some((toolCache: any) => !!toolCache?.[timeframe]);
  };

  const hasCachedDataForTimeframe = (tf: Timeframe) => {
    if (!artifactCache) return false;
    if (selectedRooms.length > 0 && activeTools.length > 0) {
      const room = selectedRooms[0];
      const tool = activeTools[0];
      const roomMap = artifactCache[room] || {};
      const toolKey = Object.keys(roomMap).find(k => k.toLowerCase() === tool.toLowerCase());
      if (toolKey && roomMap[toolKey]?.[tf]) return true;
    }
    return Object.values(artifactCache).some((roomMap) =>
      Object.values(roomMap || {}).some((domainMap) => !!domainMap?.[tf])
    );
  };

  return (
    <div 
      className={`h-full flex flex-col bg-[#0A0A0A] shrink-0 z-10 shadow-2xl transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-[88px]" : "w-[clamp(260px,18vw,320px)]"
      }`}
    >
      <div className="p-3 pt-4 pb-1 shrink-0">
        <div
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={`w-full h-12 rounded-2xl bg-white/5 hover:bg-white/10 cursor-pointer flex items-center transition-all duration-300 shadow-sm border border-white/5 ${
            isCollapsed ? "justify-center px-0" : "px-3.5 gap-3"
          }`}
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          <img src="/icon.png" alt="HUAssistant Logo" className="w-7 h-7 rounded-xl shrink-0 object-contain" />
          {!isCollapsed && (
            <span className="font-bold text-white text-base tracking-wide truncate">
              HUAssistant
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden chat-scrollbar pb-4 pt-2">
        {FLOORS.map((floor) => {
          const isActiveFloor = activeLevel === floor.id;
          const hasData = floorHasData(floor.id);

          return (
            <div 
              key={floor.id} 
              onClick={() => { if (!isActiveFloor) setActiveLevel(floor.id); }}
              className={`flex py-2.5 transition-colors duration-300 group cursor-pointer rounded-3xl ${
                isCollapsed ? "mx-2 justify-center" : "mx-3 mb-1"
              } ${isActiveFloor ? "bg-[#0A664F]" : "hover:bg-[#0A664F]/40"}`}
            >
              <div className={`${isCollapsed ? "flex justify-center" : "w-16 shrink-0 flex justify-center items-start"}`}>
                <div
                  className={`w-10 h-10 flex items-center justify-center font-bold text-base transition-all duration-300 rounded-xl ${
                    isActiveFloor ? "bg-[#14C89B] text-[#0A0A0A] shadow-lg" : 
                    hasData ? "bg-[#0A664F] text-white shadow-md" : 
                    "text-[#A3B8B2] bg-transparent group-hover:bg-[#0A664F] group-hover:text-white group-hover:shadow-md"
                  }`}
                >
                  {floor.label}
                </div>
              </div>

              {!isCollapsed && (
                <div className="flex-1 flex flex-col pr-4 gap-1.5 min-w-0">
                  {isActiveFloor ? (
                    floor.rooms.map((room) => {
                      const isSelected = selectedRooms.includes(room.id);
                      const cantUnselect = viewMode === "snapshot" && isSelected && activeTools.length > 0;
                      const isCached = !isSelected && hasCachedData(room.id);

                      let roomStyle = "bg-black/20 text-[#A3B8B2] border-transparent hover:bg-[#14C89B] hover:text-[#0A0A0A]";
                      if (isSelected) {
                        roomStyle = "bg-[#14C89B] text-[#0A0A0A] font-bold shadow-md border-transparent";
                      } else if (isCached) {
                        if (mode === "graph") {
                          roomStyle = "bg-transparent text-[#14C89B] font-semibold border border-[#14C89B] shadow-[0_0_10px_rgba(20,200,155,0.15)] hover:bg-[#14C89B] hover:text-[#0A0A0A]";
                        } else {
                          roomStyle = "bg-[#053D2F]/80 text-[#14C89B] font-semibold border border-[#0A664F] shadow-[0_0_10px_rgba(20,200,155,0.15)] hover:bg-[#14C89B] hover:text-[#0A0A0A]";
                        }
                      }

                      return (
                        <div
                          key={room.id}
                          onClick={(e) => { e.stopPropagation(); onRoomToggle(room.id); }}
                          className={`h-10 px-4 transition-all duration-300 flex items-center justify-between rounded-full truncate border ${roomStyle} ${
                            cantUnselect ? "cursor-not-allowed opacity-90" : "cursor-pointer"
                          }`}
                        >
                          <span className="text-xs truncate block">{room.name}</span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="h-10 flex items-center">
                      <span className="text-xs text-[#A3B8B2] opacity-40 truncate group-hover:opacity-100 transition-opacity w-full block">
                        {floor.rooms.map((r) => r.name).join(", ")}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div 
        ref={bottomSectionRef} 
        className={`bg-[#0A664F] shrink-0 relative flex flex-col gap-3.5 rounded-t-3xl border-t border-[#14C89B]/20 shadow-[0_-10px_30px_rgba(0,0,0,0.3)] transition-all duration-300 py-5 ${
          isCollapsed ? "px-3 items-center" : "px-5"
        }`}
      >
        {isCollapsed ? (
          <div className="flex flex-col items-center gap-3.5 w-full">
            <motion.button
              whileTap={{ scale: 0.92 }}
              onClick={() => {
                if (mode === "map") {
                  const targetTf = timeframe === "now" ? lastHistoricalTimeframe : timeframe;
                  onTimeframeChange(targetTf);
                  onViewModeChange("graph");
                  setIsCollapsed(false); 
                  setIsTimeframeMenuOpen(true);
                } else {
                  onTimeframeChange("now");
                  onViewModeChange("snapshot");
                }
              }}
              className="w-10 h-10 rounded-xl bg-[#14C89B] text-[#0A0A0A] flex items-center justify-center font-bold text-base shadow-lg hover:brightness-110 transition-all"
              title={mode === "graph" ? "Graph Mode (Click for Map)" : "Map Mode (Click for Graph)"}
            >
              <AnimatePresence mode="wait">
                <motion.span
                  key={mode}
                  initial={{ opacity: 0, scale: 0.6, rotate: -45 }}
                  animate={{ opacity: 1, scale: 1, rotate: 0 }}
                  exit={{ opacity: 0, scale: 0.6, rotate: 45 }}
                  transition={{ duration: 0.15 }}
                  className="block"
                >
                  {mode === "graph" ? "G" : "M"}
                </motion.span>
              </AnimatePresence>
            </motion.button>

            <div className="w-10 h-10 rounded-xl bg-black/20 flex items-center justify-center shrink-0 text-white shadow-md cursor-default">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </div>
          </div>
        ) : (
          <>
            <div className="relative grid grid-cols-2 bg-black/20 p-1 rounded-full gap-1 shadow-inner mx-1 h-10">
              
              {/* TIMEFRAME DROP-UP MENU WITH GREY TYPOGRAPHY & DESCRIPTIVE LABELS */}
              {isTimeframeMenuOpen && (
                <div className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-48 bg-[#0A664F] border border-[#14C89B]/30 rounded-2xl shadow-[0_-10px_30px_rgba(0,0,0,0.5)] p-2 flex flex-col gap-1.5 z-50">
                  {TIMEFRAMES.map(tf => {
                    const isSelected = timeframe === tf;
                    const isCached = !isSelected && hasCachedDataForTimeframe(tf);

                    // ---> UPDATED: Replaced text-white with text-[#A3B8B2] for never selected items <---
                    let buttonStyle = "bg-black/20 text-[#A3B8B2] font-medium border border-transparent hover:bg-[#14C89B] hover:text-[#0A0A0A]";
                    if (isSelected) {
                      buttonStyle = "bg-[#14C89B] text-[#0A0A0A] font-bold shadow-[0_0_12px_rgba(20,200,155,0.4)] border border-transparent";
                    } else if (isCached) {
                      buttonStyle = "bg-transparent text-[#14C89B] font-semibold border border-[#14C89B] shadow-[0_0_10px_rgba(20,200,155,0.15)] hover:bg-[#14C89B] hover:text-[#0A0A0A]";
                    }

                    return (
                      <button
                        key={tf}
                        onClick={() => { 
                          onTimeframeChange(tf); 
                          setIsTimeframeMenuOpen(false); 
                        }}
                        className={`w-full text-left px-3.5 py-2 rounded-full text-xs transition-all flex items-center justify-between ${buttonStyle}`}
                      >
                        {/* ---> UPDATED: Explicit human-readable labels ("2 Hours", "7 Days", etc.) <--- */}
                        <span>{TIMEFRAME_LABELS[tf]}</span>
                      </button>
                    );
                  })}
                </div>
              )}

              <div className={`flex items-center justify-between rounded-full overflow-hidden transition-all duration-300 h-full ${
                mode === "graph" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/90 hover:bg-[#14C89B] hover:text-[#0A0A0A]"
              }`}>
                <button
                  onClick={() => {
                    const targetTf = timeframe === "now" ? lastHistoricalTimeframe : timeframe;
                    onTimeframeChange(targetTf);
                    onViewModeChange("graph");
                  }}
                  className="flex-1 flex items-center justify-center h-full pl-3 text-xs font-bold whitespace-nowrap"
                >
                  Graph ({timeframe === "now" ? lastHistoricalTimeframe : timeframe})
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsTimeframeMenuOpen(!isTimeframeMenuOpen);
                  }}
                  className="px-2 h-full flex items-center justify-center transition-transform duration-200"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    {isTimeframeMenuOpen ? <polyline points="18 15 12 9 6 15" /> : <polyline points="6 9 12 15 18 9" />}
                  </svg>
                </button>
              </div>

              <button
                onClick={() => {
                  onTimeframeChange("now");
                  onViewModeChange("snapshot");
                }}
                className={`flex items-center justify-center h-full rounded-full text-xs font-bold transition-all duration-300 whitespace-nowrap ${
                  mode === "map" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/90 hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                }`}
              >
                Map
              </button>
            </div>

            <div className="flex items-center gap-2.5 bg-transparent h-10">
              <div className="w-10 h-10 rounded-xl bg-black/20 flex items-center justify-center shrink-0 shadow-sm">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              </div>
              
              <div className="flex-1 text-xs text-white font-medium line-clamp-2 break-all leading-tight">
                it2022094@hua.gr
              </div>

              <button 
                className="w-10 h-10 rounded-xl bg-[#8E2F3E] text-white flex items-center justify-center shrink-0 transition-all hover:bg-[#C84B5E] hover:text-[#0A0A0A] shadow-md"
                title="Log Out"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}