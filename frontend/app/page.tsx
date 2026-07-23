"use client";
import { useState, useEffect, useRef } from "react";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";
import Sidebar from "@/components/desktop/Sidebar";

export type AppState = "idle" | "routing" | "tool_execution" | "resolved";
export type ViewType = "snapshot" | "graph" | "schedule"; 

export interface LLMStatus {
  state: "thinking" | "tool_use" | "transcribing";
  message: string;
  tool_name?: string;
}

interface FloorState {
  selectedRooms: string[];
  activeTools: string[];
  isZoomed: boolean;
}

const getRoomsForFloor = (floor: string) => {
  if (floor === "-3") return ["parkin.c"];
  if (floor === "-2") return ["parkin.b"];
  if (floor === "-1") return ["data_center", "kitchen"];
  if (floor === "0") return ["entrance", "restaurant"];
  if (floor === "1") return ["1.1", "1.2"];
  if (floor === "2") return ["2.1", "2.2", "2.3", "2.4"];
  if (floor === "3") return ["3.7", "3.8", "3.9"];
  if (floor === "4") return ["4.9"];
  if (floor === "5") return ["5.6", "5.7"];
  if (floor === "B") return ["building"];
  return [];
};

const getFloorForRoom = (roomId: string) => {
  if (["parkin.c"].includes(roomId)) return "-3";
  if (["parkin.b"].includes(roomId)) return "-2";
  if (["data_center", "kitchen"].includes(roomId)) return "-1";
  if (["entrance", "restaurant"].includes(roomId)) return "0";
  if (["1.1", "1.2"].includes(roomId)) return "1";
  if (["2.1", "2.2", "2.3", "2.4"].includes(roomId)) return "2";
  if (["3.7", "3.8", "3.9"].includes(roomId)) return "3";
  if (["4.9"].includes(roomId)) return "4";
  if (["5.6", "5.7"].includes(roomId)) return "5";
  if (["building"].includes(roomId)) return "B";
  return null;
};

export default function DesktopDashboard() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);
  
  // CACHING ARCHITECTURE: Maps roomId -> domain -> artifact
  const [roomArtifacts, setRoomArtifacts] = useState<Record<string, Record<string, any>>>({});
  const [currentViewType, setCurrentViewType] = useState<ViewType>("snapshot");
  
  const [floorStates, setFloorStates] = useState<Record<string, FloorState>>({});
  const [activeLevel, setActiveLevel] = useState<string>("B"); 

  const [contextData, setContextData] = useState({ tokens: 0 });
  const [sessionTools, setSessionTools] = useState<{tool: string, room: string}[]>([]);
  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>([]);
  const [transcribedText, setTranscribedText] = useState<string | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const activeLevelRef = useRef(activeLevel);

  const currentFloor = floorStates[activeLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };
  const { selectedRooms, activeTools, isZoomed } = currentFloor;

  const updateFloor = (level: string, updates: Partial<FloorState>) => {
    setFloorStates(prev => ({
      ...prev,
      [level]: { ...(prev[level] || { selectedRooms: [], activeTools: [], isZoomed: false }), ...updates }
    }));
  };

  // --- BROWSER CACHING LOGIC ---
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

  useEffect(() => { sessionStorage.setItem("floorStates", JSON.stringify(floorStates)); }, [floorStates]);
  useEffect(() => { 
    sessionStorage.setItem("activeLevel", activeLevel);
    activeLevelRef.current = activeLevel; 
  }, [activeLevel]);
  useEffect(() => { sessionStorage.setItem("chatMessages", JSON.stringify(messages)); }, [messages]);
  useEffect(() => { sessionStorage.setItem("roomArtifacts", JSON.stringify(roomArtifacts)); }, [roomArtifacts]);
  useEffect(() => { sessionStorage.setItem("sessionTools", JSON.stringify(sessionTools)); }, [sessionTools]);
  useEffect(() => { sessionStorage.setItem("contextData", JSON.stringify(contextData)); }, [contextData]);
  useEffect(() => { sessionStorage.setItem("llmStatus", JSON.stringify(llmStatus)); }, [llmStatus]);
  useEffect(() => { sessionStorage.setItem("currentViewType", currentViewType); }, [currentViewType]);

  // --- WEBSOCKET CONNECTION ---
  useEffect(() => {
    const getWsUrl = () => {
      if (typeof window === "undefined") return ""; 
      if (window.location.hostname === "localhost") {
        return "ws://localhost:8000/ws/chat"; 
      }
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${window.location.host}/ws/chat`;
    };

    ws.current = new WebSocket(getWsUrl());
    ws.current.onopen = () => console.log("🟢 Connected to Smart Campus Backend");

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("📥 INCOMING WS PAYLOAD:", data);

        if (data.type === "llm_status") {
          setAppState(data.state === "thinking" || data.state === "transcribing" ? "routing" : "tool_execution");
          setLlmStatus({ state: data.state, message: data.message, tool_name: data.tool_name });
        }
        
        // Handles Voice Message Text Echo (when transcribing + sending to LLM)
        if (data.type === "transcription_result") {
          setMessages(prev => [...prev, { sender: "user", text: data.text }]);
        }

        // Handles Transcribe Only mode
        if (data.type === "transcription_only_result") {
          setTranscribedText(data.text);
          setAppState("idle");
          setLlmStatus(null);
        }
        
        if (data.type === "map_update" && data.artifact) {
          const artifact = data.artifact;
          const roomId = artifact.room_id;
          const domain = artifact.domain || "Unknown";

          let resolvedFloor = activeLevelRef.current;
          
          if (artifact.floor !== undefined && artifact.floor !== null) {
            resolvedFloor = String(artifact.floor); 
          } else if (roomId) {
            const derivedFloor = getFloorForRoom(roomId); 
            if (derivedFloor) resolvedFloor = derivedFloor;
          }

          const targetLevel = resolvedFloor;

          if (targetLevel !== activeLevelRef.current) {
            setActiveLevel(targetLevel);
          }
          
          if (artifact.view_type) {
            if (artifact.view_type === "error") {
              setCurrentViewType("snapshot");
            } else {
              setCurrentViewType(artifact.view_type as ViewType);
            }
          }

          if (roomId) {
            setRoomArtifacts(prev => ({
              ...prev,
              [roomId]: {
                ...(prev[roomId] || {}),
                [domain]: artifact 
              }
            }));
          }

          setFloorStates(prev => {
            const floor = prev[targetLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };
            const newZoom = roomId && roomId !== "building";
            
            const newActiveTools = [...floor.activeTools];
            if (domain && domain !== "Unknown") {
               const filteredTools = newActiveTools.filter(t => t !== domain);
               newActiveTools.splice(0, newActiveTools.length, domain, ...filteredTools);
            }

            return {
              ...prev,
              [targetLevel]: {
                ...floor,
                selectedRooms: roomId && !floor.selectedRooms.includes(roomId) ? [...floor.selectedRooms, roomId] : floor.selectedRooms,
                isZoomed: newZoom,
                activeTools: newActiveTools
              }
            };
          });
        }

        if (data.type === "text" && data.text) {
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
          setLlmStatus(null); 
        }

        if (data.type === "resolved") {
          setAppState("resolved");
          setLlmStatus(null);
        }

        if (data.type === "context_update") {
           setContextData({ tokens: data.tokens });
           setSessionTools(data.session_tools);
        }

      } catch (err) {
        console.error("❌ Error parsing websocket message:", err);
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
        context: { activeLevel, selectedRooms: selectedRooms.length > 0 ? selectedRooms : ["ALL"] }
      }));
    }
  };

  const handleSendAudio = (audioBase64: string, sendToLLM: boolean, currentInput: string) => {
    setAppState("routing");
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: sendToLLM ? "voice_message" : "transcribe_audio",
        audio: audioBase64,
        format: "webm",
        prepend_text: currentInput
      }));
    }
  };

  const handleToggleSelect = (toggle: string) => {
    const isNewTool = !activeTools.includes(toggle);
    const newTools = [toggle, ...activeTools.filter(t => t !== toggle)];
    let roomsToFetch = selectedRooms;

    if (isNewTool && selectedRooms.length === 0) {
      roomsToFetch = getRoomsForFloor(activeLevel);
      updateFloor(activeLevel, { activeTools: newTools, selectedRooms: roomsToFetch });
    } else {
      updateFloor(activeLevel, { activeTools: newTools });
    }

    const roomsRequiringFetch = roomsToFetch.filter(roomId => {
      const roomData = roomArtifacts[roomId] || {};
      const hasKey = Object.keys(roomData).some(k => k.toLowerCase() === toggle.toLowerCase());
      return !hasKey;
    });

    if (roomsRequiringFetch.length > 0 && ws.current && ws.current.readyState === WebSocket.OPEN) {
      const payload = {
        type: "map_interaction",
        rooms: roomsRequiringFetch,
        floor: activeLevel,
        domain: toggle
      };
      ws.current?.send(JSON.stringify(payload));
    }
  };

  const handleRoomSelect = (roomId: string) => {
    const hasActiveTools = activeTools.length > 0;
    const isCurrentlySelected = selectedRooms.includes(roomId);

    if (isCurrentlySelected && hasActiveTools) return; 

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
            const roomData = roomArtifacts[roomId] || {};
            const hasCachedData = Object.keys(roomData).some(k => k.toLowerCase() === tool.toLowerCase());
            
            if (!hasCachedData) {
                const payload = {
                    type: "map_interaction",
                    rooms: [roomId], 
                    floor: activeLevel,
                    domain: tool
                };
                ws.current?.send(JSON.stringify(payload));
            }
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
    setTranscribedText(null);
    sessionStorage.clear();
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "reset_session" }));
    }
  };

  const visuallyActiveTool = activeTools[0]; 
  const activeViewArtifacts: Record<string, any> = {};
  const currentRoomHealthData: Record<string, RoomHealth> = {};

  if (visuallyActiveTool) {
    Object.keys(roomArtifacts).forEach(roomId => {
      const toolKey = Object.keys(roomArtifacts[roomId]).find(
          key => key.toLowerCase() === visuallyActiveTool.toLowerCase()
      );

      if (toolKey) {
        const artifact = roomArtifacts[roomId][toolKey];
        activeViewArtifacts[roomId] = artifact;
        
        if (artifact.status) {
          currentRoomHealthData[roomId] = artifact.status.toLowerCase() as RoomHealth;
        } else if (artifact.view_type === "error") {
          currentRoomHealthData[roomId] = "error";
        }
      }
    });
  }

  return (
    <main 
      className="w-full h-screen flex overflow-hidden text-[#A3B8B2]"
      style={{
        background: "radial-gradient(circle at 30% 20%, #064E3B 0%, #020604 50%, #000000 100%)"
      }}
    >
      {/* Sidebar sits completely flush to the left, top, and bottom edge */}
      <Sidebar 
        activeLevel={activeLevel}
        setActiveLevel={setActiveLevel}
        selectedRooms={selectedRooms}
        onRoomToggle={handleRoomSelect}
        activeTools={activeTools}
        floorStates={floorStates}
      />

      {/* MapStage Wrapper: Retains full 16px padding on all sides */}
      <div className="flex-1 flex flex-col min-w-0 relative overflow-hidden h-full py-4 pl-4 pr-2">
        <div className="flex-1 flex flex-col min-w-0 relative overflow-hidden h-full rounded-3xl">
          <MapStage 
            appState={appState} 
            activeTools={activeTools}
            activeLevel={activeLevel}
            setActiveLevel={setActiveLevel} 
            selectedRooms={selectedRooms}
            onRoomToggle={handleRoomSelect}
            viewMode={currentViewType} 
            setViewMode={setCurrentViewType}
            isZoomed={isZoomed}
            setIsZoomed={(z) => updateFloor(activeLevel, { isZoomed: z })}
            onToggleSelect={handleToggleSelect}
            roomHealthData={currentRoomHealthData}
            roomArtifacts={activeViewArtifacts} 
            allArtifacts={roomArtifacts}
          />
        </div>
      </div>

      {/* ChatPanel Wrapper: Retains top and right padding (pt-4 pr-4), but sits flush at pb-0! */}
      <div className="w-[630px] flex-shrink-0 h-full pt-4 pr-4 pb-0 pl-2 transition-all duration-500 ease-in-out flex flex-col justify-end">
        <ChatPanel 
          appState={appState} 
          llmStatus={llmStatus}
          onSendMessage={handleUserMessage}
          onSendAudio={handleSendAudio}
          activeTools={activeTools}
          messages={messages}
          contextData={contextData}  
          sessionTools={sessionTools} 
          onResetSession={handleResetSession}
          transcribedText={transcribedText}
          onClearTranscribedText={() => setTranscribedText(null)}
        />
      </div>
    </main>
  );
}
