// frontend/components/desktop/MapStage.tsx
"use client";
import { useState, useEffect, useRef } from "react";
import InteractiveMap from "@/components/map/InteractiveMap"; 
import { AppState, ViewType } from "@/app/page"; 
import { motion, LayoutGroup } from "framer-motion";
import { RoomHealth } from "@/components/map/constants";

interface MapStageProps {
  appState: AppState;
  activeTools: string[];
  activeLevel: string;
  setActiveLevel: (level: string) => void;
  selectedRooms: string[];
  onRoomToggle: (roomId: string) => void; 
  viewMode: ViewType;
  setViewMode: (mode: ViewType) => void;
  isZoomed: boolean;
  setIsZoomed: (zoom: boolean) => void;
  roomHealthData: Record<string, RoomHealth>;
  onToggleSelect: (toggle: string) => void;
  roomArtifacts: Record<string, any>;
  allArtifacts: Record<string, Record<string, any>>;
}

const ALL_TOGGLES = ["Air Quality", "Doors/Windows", "Lights", "Occupancy", "Climate", "Schedule", "Diagnostics"];

export default function MapStage(props: MapStageProps) {
  const [currentView, setCurrentView] = useState<string>("");
  const previousToolsLength = useRef(props.activeTools.length);

  useEffect(() => {
    if (props.activeTools.length > 0 && props.activeTools[0] !== currentView) {
      setCurrentView(props.activeTools[0]);
    } else if (props.activeTools.length === 0) {
      setCurrentView("");
    }
    previousToolsLength.current = props.activeTools.length;
  }, [props.activeTools, currentView]);

  const handleToggleClick = (toggle: string) => {
    setCurrentView(toggle);
    props.onToggleSelect(toggle);
  };

  const availableToggles = props.activeTools;
  const unavailableToggles = ALL_TOGGLES.filter(t => !props.activeTools.includes(t));

  return (
    // FIX: Replaced the bg-gradient classes with bg-transparent
    <div className="w-full h-full relative overflow-hidden bg-transparent">
      
      {/* MAP LAYER */}
      <div className="absolute inset-0 z-0"> 
        <InteractiveMap 
          appState={props.appState}
          activeTools={currentView ? [currentView] : []}
          activeLevel={props.activeLevel}
          setActiveLevel={props.setActiveLevel}
          selectedRooms={props.selectedRooms}
          onRoomToggle={props.onRoomToggle} 
          viewMode={props.viewMode === "snapshot" ? "map" : "graph"}
          setViewMode={(mode) => props.setViewMode(mode === "map" ? "snapshot" : "graph")}
          isZoomed={props.isZoomed}
          setIsZoomed={props.setIsZoomed}
          roomHealthData={props.roomHealthData}
          roomArtifacts={props.roomArtifacts}
          allArtifacts={props.allArtifacts}
        />
      </div>

      {/* TOGGLES LAYER */}
      <div className="absolute bottom-6 left-0 right-0 z-10 pointer-events-none flex flex-col items-center justify-end">
        <LayoutGroup>
           <div className="w-full shrink-0 flex items-center justify-center gap-4 flex-wrap pb-2 pointer-events-auto">
            {availableToggles.length > 0 && (
              <motion.div layout className="flex items-center bg-[#053D2F]/80 border border-[#0A664F] rounded-full p-1.5 shadow-[0_4px_20px_rgba(20,200,155,0.15)]">
                {availableToggles.map(toggle => {
                  const isSelected = toggle === currentView;
                  return (
                    <motion.button 
                      layout="position" 
                      key={toggle} 
                      onClick={() => handleToggleClick(toggle)}
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
              <motion.div layout className="flex items-center bg-[#1A1A1A]/80 border border-[#333333] rounded-full p-1.5 shadow-inner">
                {unavailableToggles.map(toggle => (
                  <motion.button 
                    layout="position" 
                    key={toggle} 
                    onClick={() => handleToggleClick(toggle)}
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
    </div>
  );
}