"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface SidebarProps {
  activeLevel: string;
  setActiveLevel: (level: string) => void;
  selectedRooms: string[];
  onRoomToggle: (roomId: string) => void;
  activeTools: string[];
  floorStates: Record<string, { selectedRooms: string[]; activeTools: string[] }>;
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

const TIMEFRAMES = ["2h", "24h", "7d", "30d", "90d"];

export default function Sidebar({ activeLevel, setActiveLevel, selectedRooms, onRoomToggle, activeTools, floorStates }: SidebarProps) {
  const [isTimeframeMenuOpen, setIsTimeframeMenuOpen] = useState(false);
  const [timeframe, setTimeframe] = useState("24h");
  const [mode, setMode] = useState<"now" | "historical">("now");
  
  const [isCollapsed, setIsCollapsed] = useState(true);

  const bottomSectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (bottomSectionRef.current && !bottomSectionRef.current.contains(event.target as Node)) {
        setIsTimeframeMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const floorHasData = (floorId: string) => {
    const state = floorStates[floorId];
    return state && state.selectedRooms?.length > 0 && state.activeTools?.length > 0;
  };

  return (
    <div 
      className={`h-full flex flex-col bg-[#0A0A0A] shrink-0 z-10 shadow-2xl transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-[88px]" : "w-[320px]"
      }`}
    >
      
      {/* Top App Logo & Name Section */}
      <div className="p-3 pt-4 pb-1 shrink-0">
        <div
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={`w-full h-14 rounded-2xl bg-white/5 hover:bg-white/10 cursor-pointer flex items-center transition-all duration-300 shadow-sm border border-white/5 ${
            isCollapsed ? "justify-center px-0" : "px-4 gap-3"
          }`}
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          <img src="/icon.png" alt="HUAssistant Logo" className="w-8 h-8 rounded-xl shrink-0 object-contain" />
          {!isCollapsed && (
            <span className="font-bold text-white text-lg tracking-wide truncate">
              HUAssistant
            </span>
          )}
        </div>
      </div>

      {/* Scrollable Container */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden chat-scrollbar pb-4 pt-2">
        {FLOORS.map((floor) => {
          const isActiveFloor = activeLevel === floor.id;
          const hasData = floorHasData(floor.id);

          return (
            <div 
              key={floor.id} 
              onClick={() => { if (!isActiveFloor) setActiveLevel(floor.id); }}
              // FIXED: py-3 and mb-1 are now applied identically in BOTH open and closed states to guarantee pixel-perfect vertical alignment!
              className={`flex py-3 mb-1 transition-colors duration-300 group cursor-pointer rounded-3xl ${
                isCollapsed ? "mx-2 justify-center" : "mx-3"
              } ${isActiveFloor ? "bg-[#0A664F]" : "hover:bg-[#0A664F]/40"}`}
            >
              
              {/* Left Column: Floor Number */}
              <div className={`${isCollapsed ? "flex justify-center" : "w-20 shrink-0 flex justify-center items-start"}`}>
                <div
                  // UPDATED: Added group-hover:text-[#0A0A0A] and hover:text-[#0A0A0A] so the letter/number turns black on hover!
                  className={`w-12 h-12 flex items-center justify-center font-bold text-lg transition-all duration-300 rounded-2xl ${
                    isActiveFloor ? "bg-[#14C89B] text-[#0A0A0A] shadow-lg" : 
                    hasData ? "bg-[#0A664F] text-white shadow-md group-hover:text-[#0A0A0A] hover:text-[#0A0A0A]" : 
                    "text-[#A3B8B2] bg-transparent group-hover:bg-[#0A664F] group-hover:text-[#0A0A0A] group-hover:shadow-md hover:text-[#0A0A0A]"
                  }`}
                >
                  {floor.label}
                </div>
              </div>

              {/* Right Column: Rooms */}
              {!isCollapsed && (
                <div className="flex-1 flex flex-col pr-5 gap-2 min-w-0">
                  {isActiveFloor ? (
                    floor.rooms.map((room) => {
                      const isSelected = selectedRooms.includes(room.id);
                      const cantUnselect = isSelected && activeTools.length > 0;

                      return (
                        <div
                          key={room.id}
                          onClick={(e) => { e.stopPropagation(); onRoomToggle(room.id); }}
                          className={`h-12 px-5 transition-all duration-300 flex items-center justify-between rounded-full truncate ${
                            isSelected ? "bg-[#14C89B] text-[#0A0A0A] font-bold shadow-md" : "bg-black/20 text-[#A3B8B2] hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                          } ${cantUnselect ? "cursor-not-allowed opacity-90" : ""}`}
                        >
                          <span className="text-sm truncate block">{room.name}</span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="h-12 flex items-center">
                      <span className="text-sm text-[#A3B8B2] opacity-40 truncate group-hover:opacity-100 transition-opacity w-full block">
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

      {/* Bottom Profile & Timeframe Section */}
      <div 
        ref={bottomSectionRef} 
        className={`bg-[#0A664F] shrink-0 relative flex flex-col gap-4 rounded-t-3xl border-t border-[#14C89B]/20 shadow-[0_-10px_30px_rgba(0,0,0,0.3)] transition-all duration-300 py-6 ${
          isCollapsed ? "px-3 items-center" : "px-6"
        }`}
      >
        
        {isCollapsed ? (
          /* COLLAPSED BOTTOM VIEW */
          <div className="flex flex-col items-center gap-4 w-full">
            
            {/* H / N Squircle Toggle */}
            <motion.button
              whileTap={{ scale: 0.92 }}
              onClick={() => {
                if (mode === "now") {
                  setMode("historical");
                  setIsCollapsed(false); 
                } else {
                  setMode("now");
                }
              }}
              className="w-12 h-12 rounded-2xl bg-[#14C89B] text-[#0A0A0A] flex items-center justify-center font-bold text-lg shadow-lg hover:brightness-110 transition-all"
              title={mode === "historical" ? "Historical Mode (Click for Now)" : "Now Mode (Click for Historical)"}
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
                  {mode === "historical" ? "H" : "N"}
                </motion.span>
              </AnimatePresence>
            </motion.button>

            {/* User Icon Squircle */}
            <div 
              className="w-12 h-12 rounded-2xl bg-black/20 flex items-center justify-center shrink-0 text-white shadow-md cursor-default"
              title="admin@smartcampus.gr"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </div>

          </div>
        ) : (
          /* EXPANDED BOTTOM VIEW */
          <>
            {/* Pill-Shaped Toggles: Historical vs Now */}
            <div className="relative grid grid-cols-2 bg-black/20 p-1 rounded-full gap-1 shadow-inner mx-1 h-12">
              {isTimeframeMenuOpen && (
                <div className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-48 bg-[#0A664F] border border-[#14C89B]/30 rounded-2xl shadow-2xl overflow-hidden z-50">
                  {TIMEFRAMES.map(tf => (
                    <button
                      key={tf}
                      onClick={() => { 
                        setTimeframe(tf); 
                        setIsTimeframeMenuOpen(false); 
                        setMode("historical"); 
                      }}
                      className={`w-full text-left px-5 py-3 text-sm transition-colors ${
                        timeframe === tf ? "bg-[#14C89B] text-[#0A0A0A] font-bold" : "text-white hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              )}

              {/* Historical Grouped Pill */}
              <div className={`flex items-center justify-between rounded-full overflow-hidden transition-all duration-300 h-full ${
                mode === "historical" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/90 hover:bg-[#14C89B] hover:text-[#0A0A0A]"
              }`}>
                <button
                  onClick={() => setMode("historical")}
                  className="flex-1 flex items-center justify-center h-full pl-3 text-xs font-bold whitespace-nowrap"
                >
                  Historical ({timeframe})
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsTimeframeMenuOpen(!isTimeframeMenuOpen);
                    setMode("historical"); 
                  }}
                  className="px-2 h-full flex items-center justify-center transition-transform duration-200"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    {isTimeframeMenuOpen ? (
                      <polyline points="18 15 12 9 6 15" /> 
                    ) : (
                      <polyline points="6 9 12 15 18 9" /> 
                    )}
                  </svg>
                </button>
              </div>

              {/* Now Pill */}
              <button
                onClick={() => setMode("now")}
                className={`flex items-center justify-center h-full rounded-full text-xs font-bold transition-all duration-300 whitespace-nowrap ${
                  mode === "now" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/90 hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                }`}
              >
                Now
              </button>
            </div>

            {/* Profile Element */}
            <div className="flex items-center gap-3 bg-transparent h-12">
              <div className="w-12 h-12 rounded-2xl bg-black/20 flex items-center justify-center shrink-0 shadow-sm">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              </div>
              
              <div className="flex-1 text-sm text-white font-medium line-clamp-2 break-all leading-tight">
                it2022094@hua.gr
              </div>

              <button 
                className="w-12 h-12 rounded-2xl bg-[#8E2F3E] text-white flex items-center justify-center shrink-0 transition-all hover:bg-[#C84B5E] hover:text-[#0A0A0A] shadow-md"
                title="Log Out"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
              </button>
            </div>
          </>
        )}

      </div>
    </div>
  );
}