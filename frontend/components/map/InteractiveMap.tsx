// frontend/components/map/InteractiveMap.tsx
"use client";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import BuildingView from "./floors/BuildingView";
import FloorMinus3Base from "./floors/FloorMinus3";
import FloorMinus2Base from "./floors/FloorMinus2";
import FloorMinus1Base from "./floors/FloorMinus1";
import Floor0Base from "./floors/Floor0";
import Floor1Base from "./floors/Floor1";
import Floor2Base from "./floors/Floor2";
import Floor3Base from "./floors/Floor3";
import Floor4Base from "./floors/Floor4";
import Floor5Base from "./floors/Floor5";
import { BUILDING_LEVELS, RoomHealth } from "./constants"; 

interface InteractiveMapProps {
  appState: "idle" | "routing" | "tool_execution" | "resolved";
  activeTools: string[];
  activeLevel: string;
  setActiveLevel: (lvl: string) => void;
  selectedRooms: string[];
  onRoomToggle: (roomId: string) => void;
  viewMode: "map" | "graph";
  setViewMode: (mode: "map" | "graph") => void;
  isZoomed: boolean;
  setIsZoomed: (zoom: boolean) => void;
  roomHealthData: Record<string, RoomHealth>; 
  roomArtifacts: Record<string, any>;
}

export default function InteractiveMap(props: InteractiveMapProps) {
  const { 
    activeLevel, setActiveLevel, 
    selectedRooms, onRoomToggle, 
    viewMode, 
    isZoomed, setIsZoomed,
    roomHealthData,
    roomArtifacts
  } = props;

  let zoomOrigin = "50% 50%";
  if (selectedRooms.length === 1 && isZoomed) {
    const room = selectedRooms[0];
    if (room === "2.4") zoomOrigin = "80% 30%"; 
    if (room === "2.3") zoomOrigin = "30% 30%"; 
    if (room === "2.2") zoomOrigin = "20% 70%"; 
    if (room === "2.1") zoomOrigin = "70% 70%"; 
  }

  const buildingArtifact = roomArtifacts["building"] || 
    Object.values(roomArtifacts || {}).find((a: any) => a.floor === "B" || a.room_id === "building");

  return (
    <div className="w-full h-full relative overflow-hidden flex items-center justify-center">
      
      <div className="absolute left-4 top-1/2 -translate-y-1/2 z-20 flex flex-col gap-2 bg-[#0A0A0A]/60 backdrop-blur-xl p-2 rounded-2xl border border-[#A3B8B2]/10 shadow-lg">
        {BUILDING_LEVELS.map(lvl => {
          const isCurrent = activeLevel === lvl;
          // Check if ANY artifact in memory belongs to this specific floor level
          const hasData = Object.values(roomArtifacts || {}).some((artifact: any) => artifact.floor === lvl);
          
          let buttonStyles = "text-[#A3B8B2] hover:bg-[#14C89B]/20";
          if (isCurrent) {
             // Selected Floor -> Bright Green
             buttonStyles = "bg-[#14C89B] text-black shadow-[0_0_15px_rgba(20,200,155,0.4)]";
          } else if (hasData) {
             // Not Selected, but HAS data -> Dark Green
             buttonStyles = "bg-[#0A664F] text-white shadow-sm"; 
          }

          return (
            <button
              key={lvl}
              onClick={() => setActiveLevel(lvl)} 
              className={`w-10 h-10 rounded-xl font-bold transition-all ${buttonStyles}`}
            >
              {lvl}
            </button>
          );
        })}
      </div>

      <AnimatePresence>
        {isZoomed && (
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            onClick={() => setIsZoomed(false)}
            className="absolute top-6 left-20 z-20 flex items-center gap-2 bg-[#0A0A0A]/80 backdrop-blur-md px-4 py-2 rounded-full text-[#A3B8B2] hover:text-[#14C89B] border border-[#A3B8B2]/20 hover:border-[#14C89B]/50 transition-colors shadow-lg"
          >
            <ArrowLeft size={16} />
            <span className="text-sm font-medium">Back to Floor View</span>
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {viewMode === "map" ? (
          <motion.div
            key="map"
            exit={{ opacity: 0, rotateY: 90 }}
            className="w-full max-w-4xl absolute transition-transform duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
            style={{ transformOrigin: zoomOrigin }}
            animate={{ scale: isZoomed ? 2.0 : (activeLevel === "B" ? 0.68 : 1.25) }}
          >
            {activeLevel === "B" ? (
              /* 👇 Fixed: Now hands down the actual aggregate metrics */
              <BuildingView buildingArtifact={buildingArtifact} />
            ) : activeLevel === "-3" ? (
              <FloorMinus3Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "-2" ? (
              <FloorMinus2Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "-1" ? (
              <FloorMinus1Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "0" ? (
              <Floor0Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "1" ? (
              <Floor1Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "2" ? (
              <Floor2Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "3" ? (
              <Floor3Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "4" ? (
              <Floor4Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : activeLevel === "5" ? (
              <Floor5Base 
                activeTools={props.activeTools} 
                selectedRooms={selectedRooms} 
                onToggleRoom={onRoomToggle}
                roomHealthData={roomHealthData}
                roomArtifacts={roomArtifacts} 
              />
            ) : (
              <div className="w-full text-center text-[#A3B8B2]/50 italic p-20">
                Floor {activeLevel} Data Not Uploaded Yet
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div 
            key="graph" 
            initial={{ opacity: 0, rotateY: -90 }} 
            animate={{ opacity: 1, rotateY: 0 }}
            className="w-full h-full flex items-center justify-center p-20"
          >
             <div className="w-full h-64 border border-dashed border-[#14C89B]/40 rounded-2xl flex items-center justify-center text-[#14C89B]">
                [Historical Graph View Placeholder]
             </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}