"use client";
import { useState, useEffect, useRef } from "react";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";

export type AppState = "idle" | "routing" | "tool_execution" | "resolved";
export type ViewMode = "map" | "graph";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/chat";

// --- PER-FLOOR STATE INTERFACE ---
interface FloorState {
  selectedRooms: string[];
  activeTools: string[];
  roomHealthData: Record<string, RoomHealth>;
  isZoomed: boolean;
}

// --- HELPER: MAP FLOORS TO ROOMS ---
const getRoomsForFloor = (floor: string) => {
  if (floor === "-3") return ["parkin.c"];
  if (floor === "-2") return ["parkin.b"];
  if (floor === "-1") return ["data_center"];
  if (floor === "0") return ["entrance", "restaurant"];
  if (floor === "1") return ["1.1", "1.2", "kitchen"];
  if (floor === "2") return ["2.1", "2.2", "2.3", "2.4"];
  if (floor === "3") return ["3.7", "3.8", "3.9"];
  if (floor === "4") return ["4.9"];
  if (floor === "5") return ["5.6", "5.7"];
  if (floor === "B") return ["building"];
  return [];
};

export default function DesktopDashboard() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  
  // Floor States Dictionary & Tracker
  const [floorStates, setFloorStates] = useState<Record<string, FloorState>>({});
  const [activeLevel, setActiveLevel] = useState<string>("B"); 

  const [contextData, setContextData] = useState({ tokens: 0 });
  const [sessionTools, setSessionTools] = useState<{tool: string, room: string}[]>([]);
  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>([]);
  
  const ws = useRef<WebSocket | null>(null);
  
  // --- NEW: REF TO PREVENT STALE CLOSURES IN WEBSOCKET ---
  const activeLevelRef = useRef(activeLevel);

  // Derived current state for the active floor
  const currentFloor = floorStates[activeLevel] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false };
  const { selectedRooms, activeTools, roomHealthData, isZoomed } = currentFloor;

  // Helper to update state for a specific floor without losing others
  const updateFloor = (level: string, updates: Partial<FloorState>) => {
    setFloorStates(prev => ({
      ...prev,
      [level]: { ...(prev[level] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false }), ...updates }
    }));
  };

  // --- BROWSER CACHING LOGIC ---
  useEffect(() => {
    const cachedFloors = sessionStorage.getItem("floorStates");
    if (cachedFloors) setFloorStates(JSON.parse(cachedFloors));
    const cachedLevel = sessionStorage.getItem("activeLevel");
    if (cachedLevel) setActiveLevel(cachedLevel);
  }, []);

  useEffect(() => {
    sessionStorage.setItem("floorStates", JSON.stringify(floorStates));
  }, [floorStates]);

  useEffect(() => {
    sessionStorage.setItem("activeLevel", activeLevel);
    // Keep the ref strictly in sync with the state so the WebSocket always sees the latest floor
    activeLevelRef.current = activeLevel; 
  }, [activeLevel]);
  // ------------------------------

  useEffect(() => {
    ws.current = new WebSocket(WS_URL);
    ws.current.onopen = () => console.log("Connected to Smart Campus Backend");

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "tool_start") {
          setAppState("tool_execution");
          if (data.tools_used) {
             setFloorStates(prev => {
               // Use the ref here to avoid stale closures
               const currentLvl = activeLevelRef.current;
               const floor = prev[currentLvl] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false };
               const newTools = data.tools_used.filter((t: string) => !floor.activeTools.includes(t));
               return { ...prev, [currentLvl]: { ...floor, activeTools: [...newTools, ...floor.activeTools] }};
             });
          }
        }
        
        if (data.type === "map_update" || data.room_data || data.target_rooms) {
          let targetLevel = activeLevelRef.current;
          
          // Auto-Switch Floors
          if (data.target_rooms && data.target_rooms.length > 0) {
            const firstRoom = data.target_rooms[0];
            if (firstRoom === "building") targetLevel = "B";
            else if (firstRoom.startsWith("5.")) targetLevel = "5";
            else if (firstRoom.startsWith("4.")) targetLevel = "4";
            else if (firstRoom.startsWith("3.")) targetLevel = "3";
            else if (firstRoom.startsWith("2.")) targetLevel = "2";
            else if (firstRoom.startsWith("1.") || firstRoom === "kitchen") targetLevel = "1";
            else if (firstRoom === "entrance" || firstRoom === "restaurant") targetLevel = "0";
            else if (firstRoom === "data_center") targetLevel = "-1";
            else if (firstRoom === "parkin.b") targetLevel = "-2";
            else if (firstRoom === "parkin.c") targetLevel = "-3";
            
            setActiveLevel(targetLevel);
          }

          setFloorStates(prev => {
            const floor = prev[targetLevel] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false };
            let newZoom = floor.isZoomed;
            if (data.target_rooms) {
              newZoom = (data.target_rooms.length === 1 && data.target_rooms[0] !== "building");
            }
            return {
              ...prev,
              [targetLevel]: {
                ...floor,
                selectedRooms: data.target_rooms || floor.selectedRooms,
                isZoomed: newZoom,
                roomHealthData: { ...floor.roomHealthData, ...(data.room_data || {}) }
              }
            };
          });
        }

        if (data.text) {
          const replyText = data.text;
          setMessages(prev => {
            if (prev.length > 0 && prev[prev.length - 1].sender === "agent") {
              const updated = [...prev];
              updated[updated.length - 1] = { sender: "agent", text: replyText };
              return updated;
            }
            return [...prev, { sender: "agent", text: replyText }];
          });
          setAppState("resolved");
        }

        if (data.type === "resolved") setAppState("resolved");

        if (data.type === "context_update") {
           setContextData({ tokens: data.tokens });
           setSessionTools(data.session_tools);
        }

      } catch (err) {
        console.error("Error parsing websocket message", err);
      }
    };
    
    return () => { if (ws.current) ws.current.close(); };
  }, []); // <--- CRITICAL FIX: Empty dependency array so it only mounts once

  const handleUserMessage = (msg: string) => {
    if (!msg.trim()) return;
    setMessages(prev => [...prev, { sender: "user", text: msg }]);
    setAppState("routing");

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current?.send(JSON.stringify({
        type: "chat_message",
        query: msg,
        context: { 
          activeLevel, 
          selectedRooms: selectedRooms.length > 0 ? selectedRooms : ["ALL"] 
        }
      }));
    }
  };

  const handleToggleSelect = (toggle: string) => {
    const isActivating = !activeTools.includes(toggle);
    const newTools = isActivating 
      ? [toggle, ...activeTools] 
      : [toggle, ...activeTools.filter(t => t !== toggle)];
    
    let roomsToFetch = selectedRooms;

    // --- AUTO-SELECT ALL ROOMS IF NONE SELECTED ---
    if (isActivating && selectedRooms.length === 0) {
      roomsToFetch = getRoomsForFloor(activeLevel);
      updateFloor(activeLevel, { activeTools: newTools, selectedRooms: roomsToFetch });
    } else {
      updateFloor(activeLevel, { activeTools: newTools });
    }

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current?.send(JSON.stringify({
        type: "map_interaction",
        rooms: roomsToFetch.length > 0 ? roomsToFetch : ["ALL"],
        floor: activeLevel,
        domain: toggle
      }));
    }
  };

  const handleRoomSelect = (roomId: string) => {
    const hasActiveTools = activeTools.length > 0;
    const isCurrentlySelected = selectedRooms.includes(roomId);

    // --- BLOCK UNSELECTING IF CONTEXT WAS FETCHED ---
    if (isCurrentlySelected && hasActiveTools) {
       return; // Silently ignore the unselect attempt
    }

    let newSelection = [];
    let newZoom = isZoomed;
    let isSelecting = false;

    if (isCurrentlySelected) {
        // Allowed to unselect ONLY because hasActiveTools is false
        newSelection = selectedRooms.filter(r => r !== roomId);
        if (newSelection.length !== 1 && isZoomed) newZoom = false;
    } else {
        newSelection = [...selectedRooms, roomId];
        isSelecting = true;
    }

    updateFloor(activeLevel, { selectedRooms: newSelection, isZoomed: newZoom });

    // Auto-fetch tools for the newly selected room
    if (isSelecting && ws.current && ws.current.readyState === WebSocket.OPEN && hasActiveTools) {
        activeTools.forEach(tool => {
            ws.current?.send(JSON.stringify({
                type: "map_interaction",
                rooms: [roomId], 
                floor: activeLevel,
                domain: tool
            }));
        });
    }
  };

  const handleResetSession = () => {
    // 1. Reset all frontend variables to their defaults
    setFloorStates({});
    setActiveLevel("B");
    setMessages([]);
    setSessionTools([]);
    setContextData({ tokens: 0 });
    setAppState("idle");
    
    // 2. Clear out local storage so it doesn't immediately reload the old state
    sessionStorage.removeItem("floorStates");
    sessionStorage.removeItem("activeLevel");

    // 3. Notify backend to clear LangGraph memory checkpointer
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "reset_session" }));
    }
  };

  return (
    <main className="w-full h-screen flex overflow-hidden bg-gradient-to-b from-[#0A664F] to-[#0A0A0A] text-[#A3B8B2] p-4 gap-4">
      
      {/* LEFT SIDE: MAP CONTAINER */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0A0A0A]/40 border border-[#A3B8B2]/10 rounded-3xl backdrop-blur-md overflow-hidden relative shadow-2xl h-full">
        <MapStage 
          appState={appState} 
          activeTools={activeTools}
          activeLevel={activeLevel}
          setActiveLevel={setActiveLevel} 
          selectedRooms={selectedRooms}
          onRoomToggle={handleRoomSelect}
          viewMode={viewMode}
          setViewMode={setViewMode}
          isZoomed={isZoomed}
          setIsZoomed={(z) => updateFloor(activeLevel, { isZoomed: z })}
          roomHealthData={roomHealthData}
          onToggleSelect={handleToggleSelect}
        />
      </div>

      {/* RIGHT SIDE: CHAT INTERFACE */}
      <div className="w-[630px] flex-shrink-0 h-full transition-all duration-500 ease-in-out">
        <ChatPanel 
          appState={appState} 
          onSendMessage={handleUserMessage}
          activeTools={activeTools}
          messages={messages}
          contextData={contextData}  
          sessionTools={sessionTools} 
          onResetSession={handleResetSession}
        />
      </div>
    </main>
  );
}