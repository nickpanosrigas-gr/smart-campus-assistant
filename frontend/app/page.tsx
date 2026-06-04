"use client";
import { useState, useEffect, useRef } from "react";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";

export type AppState = "idle" | "routing" | "tool_execution" | "resolved";
export type ViewType = "snapshot" | "graph" | "schedule"; // NEW: Expanded views

// --- NEW INTERFACES ---
export interface LLMStatus {
  state: "thinking" | "tool_use";
  message: string;
  tool_name?: string;
}

interface FloorState {
  selectedRooms: string[];
  activeTools: string[];
  roomHealthData: Record<string, RoomHealth>;
  isZoomed: boolean;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/chat";

// Helper for Manual Map Clicks
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
  
  // --- NEW: ARTIFACT & VIEW STATES ---
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);
  const [roomArtifacts, setRoomArtifacts] = useState<Record<string, any>>({});
  const [currentViewType, setCurrentViewType] = useState<ViewType>("snapshot");
  
  // Floor States Dictionary & Tracker
  const [floorStates, setFloorStates] = useState<Record<string, FloorState>>({});
  const [activeLevel, setActiveLevel] = useState<string>("B"); 

  const [contextData, setContextData] = useState({ tokens: 0 });
  const [sessionTools, setSessionTools] = useState<{tool: string, room: string}[]>([]);
  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>([]);
  
  const ws = useRef<WebSocket | null>(null);
  const activeLevelRef = useRef(activeLevel);

  const currentFloor = floorStates[activeLevel] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false };
  const { selectedRooms, activeTools, roomHealthData, isZoomed } = currentFloor;

  const updateFloor = (level: string, updates: Partial<FloorState>) => {
    setFloorStates(prev => ({
      ...prev,
      [level]: { ...(prev[level] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false }), ...updates }
    }));
  };

// --- COMPREHENSIVE BROWSER CACHING LOGIC (page.tsx) ---
  
  // 1. Load Everything on Mount
  useEffect(() => {
    const cachedFloors = sessionStorage.getItem("floorStates");
    if (cachedFloors) setFloorStates(JSON.parse(cachedFloors));
    
    const cachedLevel = sessionStorage.getItem("activeLevel");
    if (cachedLevel) setActiveLevel(cachedLevel);

    const cachedMessages = sessionStorage.getItem("chatMessages");
    if (cachedMessages) setMessages(JSON.parse(cachedMessages));

    const cachedArtifacts = sessionStorage.getItem("roomArtifacts");
    if (cachedArtifacts) setRoomArtifacts(JSON.parse(cachedArtifacts));

    const cachedTools = sessionStorage.getItem("sessionTools");
    if (cachedTools) setSessionTools(JSON.parse(cachedTools));

    const cachedContext = sessionStorage.getItem("contextData");
    if (cachedContext) setContextData(JSON.parse(cachedContext));
    
    const cachedStatus = sessionStorage.getItem("llmStatus");
    if (cachedStatus && cachedStatus !== "null") setLlmStatus(JSON.parse(cachedStatus));

    const cachedViewType = sessionStorage.getItem("currentViewType");
    if (cachedViewType) setCurrentViewType(cachedViewType as ViewType);
  }, []);

  // 2. Save Everything on Change
  useEffect(() => {
    sessionStorage.setItem("floorStates", JSON.stringify(floorStates));
  }, [floorStates]);

  useEffect(() => {
    sessionStorage.setItem("activeLevel", activeLevel);
    activeLevelRef.current = activeLevel; 
  }, [activeLevel]);

  useEffect(() => {
    sessionStorage.setItem("chatMessages", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    sessionStorage.setItem("roomArtifacts", JSON.stringify(roomArtifacts));
  }, [roomArtifacts]);

  useEffect(() => {
    sessionStorage.setItem("sessionTools", JSON.stringify(sessionTools));
  }, [sessionTools]);

  useEffect(() => {
    sessionStorage.setItem("contextData", JSON.stringify(contextData));
  }, [contextData]);

  useEffect(() => {
    sessionStorage.setItem("llmStatus", JSON.stringify(llmStatus));
  }, [llmStatus]);

  useEffect(() => {
    sessionStorage.setItem("currentViewType", currentViewType);
  }, [currentViewType]);
  // --------------------------------------------------------

  useEffect(() => {
    ws.current = new WebSocket(WS_URL);
    ws.current.onopen = () => console.log("Connected to Smart Campus Backend");

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // --- 1. LLM STATUS STREAM ---
        if (data.type === "llm_status") {
          setAppState(data.state === "thinking" ? "routing" : "tool_execution");
          setLlmStatus({
            state: data.state,
            message: data.message,
            tool_name: data.tool_name
          });
        }
        
        // --- 2. THE SMART SERVER ARTIFACT PIPELINE ---
        if (data.type === "map_update" && data.artifact) {
          const artifact = data.artifact;
          const targetLevel = artifact.floor || activeLevelRef.current;
          const roomId = artifact.room_id;
          
          // A. Auto-Switch Floors based on backend calculation
          if (artifact.floor && artifact.floor !== activeLevelRef.current) {
            setActiveLevel(artifact.floor);
          }

          // B. Auto-Route the UI View (Snapshot, Graph, Schedule)
          if (artifact.view_type) {
            setCurrentViewType(artifact.view_type as ViewType);
          }

          // C. Save the raw artifact payload for the UI components to render
          if (roomId) {
            setRoomArtifacts(prev => ({
              ...prev,
              [roomId]: artifact
            }));
          }

          // D. Update the Visual Map State (Colors, Zoom, Active Tools)
          setFloorStates(prev => {
            const floor = prev[targetLevel] || { selectedRooms: [], activeTools: [], roomHealthData: {}, isZoomed: false };
            
            // Zoom in unless it's a building-wide macro view
            const newZoom = roomId && roomId !== "building";
            
            // Color the map polygon if the tool provided a status (e.g. good, warning)
            const updatedHealthData = { ...floor.roomHealthData };
            if (artifact.status && roomId) {
               updatedHealthData[roomId] = artifact.status;
            }

            // Ensure the tool is visually toggled "ON" in the pill menu
            const newActiveTools = [...floor.activeTools];
            if (artifact.domain && !newActiveTools.includes(artifact.domain)) {
               newActiveTools.unshift(artifact.domain);
            }

            return {
              ...prev,
              [targetLevel]: {
                ...floor,
                selectedRooms: roomId ? [roomId] : floor.selectedRooms,
                isZoomed: newZoom,
                roomHealthData: updatedHealthData,
                activeTools: newActiveTools
              }
            };
          });
        }

        // --- 3. CHAT TEXT UPDATES ---
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
          setLlmStatus(null); // Clear the dynamic status message
        }

        if (data.type === "resolved") {
          setAppState("resolved");
          setLlmStatus(null);
        }

        // --- 4. TELEMETRY & CONTEXT UPDATES ---
        if (data.type === "context_update") {
           setContextData({ tokens: data.tokens });
           setSessionTools(data.session_tools);
        }

      } catch (err) {
        console.error("Error parsing websocket message", err);
      }
    };
    
    return () => { if (ws.current) ws.current.close(); };
  }, []); 

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

    if (isCurrentlySelected && hasActiveTools) {
       return; 
    }

    let newSelection = [];
    let newZoom = isZoomed;
    let isSelecting = false;

    if (isCurrentlySelected) {
        newSelection = selectedRooms.filter(r => r !== roomId);
        if (newSelection.length !== 1 && isZoomed) newZoom = false;
    } else {
        newSelection = [...selectedRooms, roomId];
        isSelecting = true;
    }

    updateFloor(activeLevel, { selectedRooms: newSelection, isZoomed: newZoom });

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
    setFloorStates({});
    setActiveLevel("B");
    setMessages([]);
    setSessionTools([]);
    setContextData({ tokens: 0 });
    setRoomArtifacts({});
    setCurrentViewType("snapshot");
    setLlmStatus(null);
    setAppState("idle");
    
    sessionStorage.removeItem("floorStates");
    sessionStorage.removeItem("activeLevel");

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "reset_session" }));
    }
  };

  return (
    <main className="w-full h-screen flex overflow-hidden bg-gradient-to-b from-[#0A664F] to-[#0A0A0A] text-[#A3B8B2] p-4 gap-4">
      
      {/* LEFT SIDE: MAP & DATA STAGE CONTAINER */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0A0A0A]/40 border border-[#A3B8B2]/10 rounded-3xl backdrop-blur-md overflow-hidden relative shadow-2xl h-full">
        {/* IMPORTANT: You will eventually update MapStage to accept currentViewType 
          and roomArtifacts so it knows whether to render the 3D Map, the Graph, or the Schedule List. 
        */}
        <MapStage 
          appState={appState} 
          activeTools={activeTools}
          activeLevel={activeLevel}
          setActiveLevel={setActiveLevel} 
          selectedRooms={selectedRooms}
          onRoomToggle={handleRoomSelect}
          
          // FIX: Pass the state directly to MapStage
          viewMode={currentViewType} 
          setViewMode={setCurrentViewType}
          
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
          llmStatus={llmStatus} /* NEW: Passing down the dynamic text */
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