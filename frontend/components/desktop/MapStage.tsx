"use client";
import { useState, useEffect } from "react";
import InteractiveMap from "./InteractiveMap"; 
import { AppState } from "@/app/page";
import { motion, LayoutGroup } from "framer-motion";

interface MapStageProps {
  appState: AppState;
  activeTools: string[];
  setActiveTools: (tools: string[]) => void;
  activeFloor: number;
  setActiveFloor: (floor: number) => void;
}

const ALL_FLOORS = [5, 4, 3, 2, 1, 0, -1, -2, -3];
const ALL_TOGGLES = ["Air Quality", "Doors/Windows", "Lights", "Occupancy", "Climate", "Schedule", "Diagnostics"];

export default function MapStage({ appState, activeTools, setActiveTools, activeFloor, setActiveFloor }: MapStageProps) {
  const [currentView, setCurrentView] = useState<string>("");

  useEffect(() => {
    if (activeTools.length > 0 && !activeTools.includes(currentView)) {
      setCurrentView(activeTools[activeTools.length - 1]); 
    }
  }, [activeTools, currentView]);

  const availableToggles = activeTools;
  const unavailableToggles = ALL_TOGGLES.filter(t => !activeTools.includes(t));

  const sortedAvailable = currentView
    ? [currentView, ...availableToggles.filter(t => t !== currentView)]
    : availableToggles;

  const handleToggleClick = (toggle: string, isAvailable: boolean) => {
    if (isAvailable) {
      setCurrentView(toggle);
    } else {
      setActiveTools([...activeTools, toggle]);
      setCurrentView(toggle);
    }
  };

  return (
    <div className="w-full h-full flex flex-col relative p-6">
      
      <div className="flex-1 flex gap-8 min-h-0 mb-6 relative">
        <div className="w-16 flex flex-col justify-center items-center gap-2 border-r border-[#A3B8B2]/10 pr-6 shrink-0">
          {ALL_FLOORS.map(floor => (
            <button
              key={floor}
              onClick={() => setActiveFloor(floor)}
              className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg transition-all ${
                activeFloor === floor 
                  ? "bg-[#14C89B] text-black shadow-[0_0_15px_rgba(20,200,155,0.4)] scale-110" 
                  : "bg-transparent text-[#A3B8B2]/50 hover:bg-[#1E1E1E] hover:text-white"
              }`}
            >
              {floor}
            </button>
          ))}
          <span className="text-[#A3B8B2]/30 text-xs font-semibold uppercase tracking-widest mt-4 -rotate-90 origin-bottom w-32 translate-y-12">
            Floors
          </span>
        </div>

        {/* MAP CANVAS */}
        <div className="flex-1 flex items-center justify-center relative">
          <div className="w-full max-h-full flex items-center justify-center">
             {activeFloor === 2 ? (
                <InteractiveMap appState={appState} activeTools={activeTools} currentView={currentView} />
             ) : (
               <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/40 text-lg border-2 border-dashed border-[#A3B8B2]/10 rounded-3xl">
                 <span>Floor {activeFloor} map data</span>
                 <span className="text-sm">Not yet generated</span>
               </div>
             )}
          </div>
        </div>
      </div>

      <LayoutGroup>
        <div className="w-full shrink-0 flex items-center justify-center gap-4 flex-wrap">
          {sortedAvailable.length > 0 && (
            <motion.div
              layout
              className="flex items-center bg-[#053D2F]/80 border border-[#0A664F] rounded-full p-1.5 shadow-[0_4px_20px_rgba(20,200,155,0.15)]"
            >
              {sortedAvailable.map(toggle => {
                const isSelected = toggle === currentView;
                return (
                  <motion.button
                    layout="position" 
                    key={toggle}
                    onClick={() => handleToggleClick(toggle, true)}
                    className={`px-6 py-2.5 rounded-full text-sm font-semibold transition-colors duration-300 ${
                      isSelected
                        ? "bg-[#14C89B] text-[#0A0A0A] shadow-[0_0_15px_rgba(20,200,155,0.4)]"
                        : "bg-transparent text-[#14C89B] hover:bg-[#0A664F]/60"
                    }`}
                  >
                    {toggle}
                  </motion.button>
                );
              })}
            </motion.div>
          )}

          {unavailableToggles.length > 0 && (
            <motion.div
              layout
              className="flex items-center bg-[#1A1A1A]/80 border border-[#333333] rounded-full p-1.5 shadow-inner"
            >
              {unavailableToggles.map(toggle => (
                <motion.button
                  layout="position"
                  key={toggle}
                  onClick={() => handleToggleClick(toggle, false)}
                  className="px-5 py-2.5 rounded-full text-sm font-medium bg-transparent text-[#A3B8B2]/50 hover:text-[#A3B8B2] hover:bg-[#2A2A2A] transition-colors duration-300"
                >
                  {toggle}
                </motion.button>
              ))}
            </motion.div>
          )}
        </div>
      </LayoutGroup>
    </div>
  );
}