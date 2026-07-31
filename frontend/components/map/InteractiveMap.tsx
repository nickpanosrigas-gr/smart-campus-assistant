"use client";
import { useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
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
import { RoomHealth } from "./constants";
import GraphView from "./graphs/GraphView";

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
  allArtifacts?: Record<string, Record<string, any>>;
  timeframe: string; 
}

export default function InteractiveMap(props: InteractiveMapProps) {
  const {
    activeLevel,
    selectedRooms, onRoomToggle,
    viewMode,
    roomHealthData,
    roomArtifacts,
    allArtifacts
  } = props;

  const isDragging = useRef(false);

  const buildingArtifact = roomArtifacts["building"] ||
    Object.values(roomArtifacts || {}).find((a: any) => String(a.floor) === "B" || a.room_id === "building");

  return (
    <div className="w-full h-full relative overflow-hidden flex items-center justify-center">
      <AnimatePresence mode="wait">
        {viewMode === "map" ? (
          <TransformWrapper
            key={`transform-wrapper-${activeLevel}`}
            centerOnInit={true}
            initialScale={1.15} // Zoomed in a bit more initially
            minScale={1}
            maxScale={4}
            wheel={{ step: 0.002 }}
            panning={{ velocityDisabled: false }}
            doubleClick={{ disabled: true }}
            onPanning={() => {
              isDragging.current = true;
            }}
            onPanningStop={() => {
              setTimeout(() => {
                isDragging.current = false;
              }, 100);
            }}
          >
            <TransformComponent
              wrapperStyle={{ width: "100%", height: "100%" }}
              contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <motion.div
                key="map"
                exit={{ opacity: 0, rotateY: 90 }}
                // Increased max bounds so the zoomed map takes up more space
                className="w-full h-full max-w-[90%] max-h-[82%] transition-transform duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)] flex items-center justify-center"
                onClickCapture={(e) => {
                  if (isDragging.current) {
                    e.stopPropagation();
                  }
                }}
              >
                {activeLevel === "B" ? (
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
            </TransformComponent>
          </TransformWrapper>
        ) : (
          <motion.div
            key="graph"
            initial={{ opacity: 0, rotateY: -90 }}
            animate={{ opacity: 1, rotateY: 0 }}
            exit={{ opacity: 0, rotateY: 90 }}
            transition={{ duration: 0.5, ease: [0.34, 1.56, 0.64, 1] }}
            className="w-full h-full flex items-center justify-center p-4 max-h-[90%]"
          >
            <GraphView
              activeTools={props.activeTools}
              selectedRooms={selectedRooms}
              roomArtifacts={roomArtifacts}
              allArtifacts={allArtifacts || {}}
              timeframe={props.timeframe}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}