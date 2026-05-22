"use client";
import { useState, useEffect } from "react";
import InteractiveMap from "@/components/map/InteractiveMap"; 
import { AppState } from "@/app/page";
import { motion, LayoutGroup } from "framer-motion";
import { RoomHealth } from "@/components/map/constants";

interface MapStageProps {
  appState: AppState;
  activeTools: string[];
  setActiveTools: (tools: string[]) => void;
  activeLevel: string;
  setActiveLevel: (level: string) => void;
  selectedRooms: string[];
  setSelectedRooms: (rooms: string[]) => void;
  viewMode: "map" | "graph";
  setViewMode: (mode: "map" | "graph") => void;
  isZoomed: boolean;
  setIsZoomed: (zoom: boolean) => void;
  roomHealthData: Record<string, RoomHealth>;
  onToggleSelect: (toggle: string) => void;
}

const ALL_TOGGLES = ["Air Quality", "Doors/Windows", "Lights", "Occupancy", "Climate", "Schedule", "Diagnostics"];

export default function MapStage(props: MapStageProps) {
  const [currentView, setCurrentView] = useState<string>("");

  useEffect(() => {
    if (props.activeTools.length > 0) {
      setCurrentView(props.activeTools[0]); // Always highlight the first tool in the list
    }
  }, [props.activeTools]);

  const handleToggleClick = (toggle: string) => {
    setCurrentView(toggle);
    props.onToggleSelect(toggle);
  };

  const availableToggles = props.activeTools;
  const unavailableToggles = ALL_TOGGLES.filter(t => !props.activeTools.includes(t));

  return (
    <div className="w-full h-full flex flex-col relative p-6 bg-gradient-to-br from-[#0d0d0d] to-[#141414]">
      
      {/* CANVAS ELEMENT VIEW AREA */}
      <div className="flex-1 flex min-h-0 mb-6 relative items-center justify-center">
        <InteractiveMap 
          appState={props.appState}
          activeTools={currentView ? [currentView] : props.activeTools}
          activeLevel={props.activeLevel}
          setActiveLevel={props.setActiveLevel}
          selectedRooms={props.selectedRooms}
          setSelectedRooms={props.setSelectedRooms}
          viewMode={props.viewMode}
          setViewMode={props.setViewMode}
          isZoomed={props.isZoomed}
          setIsZoomed={props.setIsZoomed}
          roomHealthData={props.roomHealthData}
        />
      </div>

      {/* DUAL TOGGLE GROUPS (ANIMATED) */}
      <LayoutGroup>
        <div className="w-full shrink-0 flex items-center justify-center gap-4 flex-wrap pb-2">
          
          {/* Available / Selected Group */}
          {availableToggles.length > 0 && (
            <motion.div
              layout
              className="flex items-center bg-[#053D2F]/80 border border-[#0A664F] rounded-full p-1.5 shadow-[0_4px_20px_rgba(20,200,155,0.15)]"
            >
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

          {/* Unavailable Group */}
          {unavailableToggles.length > 0 && (
            <motion.div
              layout
              className="flex items-center bg-[#1A1A1A]/80 border border-[#333333] rounded-full p-1.5 shadow-inner"
            >
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
  );
}