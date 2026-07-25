"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";
import Sidebar from "@/components/desktop/Sidebar";

export type Timeframe = "now" | "2h" | "24h" | "7d" | "30d" | "90d";
export type HistoricalTimeframe = "2h" | "24h" | "7d" | "30d" | "90d";
export type AppState = "idle" | "routing" | "tool_execution" | "resolved";
export type ViewType = "snapshot" | "graph" | "schedule";

export interface LLMStatus {
  state: "thinking" | "tool_use" | "transcribing";
  message: string;
  tool_name?: string;
}

interface MapSandboxState {
  selectedRooms: string[];
  activeTools: string[];
  isZoomed: boolean;
}

// ---> UPGRADED: Separates tool memory for every single room in each timeframe! <---
interface GraphSandboxState {
  selectedRoom: string | null;
  roomTools: Record<string, string[]>; 
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

// ---> UPGRADED: Initialized with empty roomTools objects <---
const INITIAL_GRAPH_SANDBOXES: Record<HistoricalTimeframe, GraphSandboxState> = {
  "2h": { selectedRoom: null, roomTools: {} },
  "24h": { selectedRoom: null, roomTools: {} },
  "7d": { selectedRoom: null, roomTools: {} },
  "30d": { selectedRoom: null, roomTools: {} },
  "90d": { selectedRoom: null, roomTools: {} },
};

export default function DesktopDashboard() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null);
  
  const [timeframe, setTimeframe] = useState<Timeframe>("now");
  const [lastHistoricalTimeframe, setLastHistoricalTimeframe] = useState<HistoricalTimeframe>("24h");
  const [currentViewType, setCurrentViewType] = useState<ViewType>("snapshot");
  const [activeLevel, setActiveLevel] = useState<string>("B"); 

  const [artifactCache, setArtifactCache] = useState<
    Record<string, Record<string, Record<string, any>>>
  >({});
  
  const [mapSandbox, setMapSandbox] = useState<Record<string, MapSandboxState>>({});
  const [graphSandboxes, setGraphSandboxes] = useState<Record<HistoricalTimeframe, GraphSandboxState>>(INITIAL_GRAPH_SANDBOXES);

  const [contextData, setContextData] = useState({ tokens: 0 });
  const [sessionTools, setSessionTools] = useState<{tool: string, room: string}[]>([]);
  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>([]);
  const [transcribedText, setTranscribedText] = useState<string | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const activeLevelRef = useRef(activeLevel);
  const inFlightRequests = useRef<Set<string>>(new Set());

  const isGraphMode = currentViewType === "graph" && timeframe !== "now";
  const currentGraphBox = isGraphMode ? graphSandboxes[timeframe as HistoricalTimeframe] : { selectedRoom: null, roomTools: {} };
  const currentMapFloor = mapSandbox[activeLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };

  const selectedRooms = useMemo(() => {
    return isGraphMode 
      ? (currentGraphBox.selectedRoom ? [currentGraphBox.selectedRoom] : []) 
      : currentMapFloor.selectedRooms;
  }, [isGraphMode, currentGraphBox.selectedRoom, currentMapFloor.selectedRooms]);

  // ---> UPGRADED: Pulls active tools from roomTools memory per selected room <---
  const activeTools = useMemo(() => {
    if (isGraphMode) {
      const room = currentGraphBox.selectedRoom;
      if (!room) return [];
      if (currentGraphBox.roomTools && currentGraphBox.roomTools[room]) {
        return currentGraphBox.roomTools[room];
      }
      // Backwards compatibility fallback for old sessionStorage formats
      if ((currentGraphBox as any).activeTool) {
        return [(currentGraphBox as any).activeTool];
      }
      return [];
    }
    return currentMapFloor.activeTools;
  }, [isGraphMode, currentGraphBox, currentMapFloor.activeTools]);

  const isZoomed = isGraphMode ? false : currentMapFloor.isZoomed;

  const updateMapFloor = (level: string, updates: Partial<MapSandboxState>) => {
    setMapSandbox(prev => ({
      ...prev,
      [level]: { ...(prev[level] || { selectedRooms: [], activeTools: [], isZoomed: false }), ...updates }
    }));
  };

  const updateGraphSandbox = (tf: HistoricalTimeframe, updates: Partial<GraphSandboxState>) => {
    setGraphSandboxes(prev => ({
      ...prev,
      [tf]: { ...prev[tf], ...updates }
    }));
  };

  useEffect(() => {
    const cachedMap = sessionStorage.getItem("mapSandbox");
    if (cachedMap) setMapSandbox(JSON.parse(cachedMap));
    
    const cachedGraph = sessionStorage.getItem("graphSandboxes");
    if (cachedGraph) setGraphSandboxes(JSON.parse(cachedGraph));

    const cachedLevel = sessionStorage.getItem("activeLevel");
    if (cachedLevel) setActiveLevel(cachedLevel);

    const cachedMessages = sessionStorage.getItem("chatMessages");
    if (cachedMessages) setMessages(JSON.parse(cachedMessages));

    const cached3DCache = sessionStorage.getItem("artifactCache");
    if (cached3DCache) setArtifactCache(JSON.parse(cached3DCache));

    const cachedTools = sessionStorage.getItem("sessionTools");
    if (cachedTools) setSessionTools(JSON.parse(cachedTools));

    const cachedContext = sessionStorage.getItem("contextData");
    if (cachedContext) setContextData(JSON.parse(cachedContext));
    
    const cachedStatus = sessionStorage.getItem("llmStatus");
    if (cachedStatus && cachedStatus !== "null") setLlmStatus(JSON.parse(cachedStatus));

    const cachedViewType = sessionStorage.getItem("currentViewType");
    if (cachedViewType) setCurrentViewType(cachedViewType as ViewType);

    const cachedTimeframe = sessionStorage.getItem("timeframe");
    if (cachedTimeframe) setTimeframe(cachedTimeframe as Timeframe);

    const cachedLastHistTf = sessionStorage.getItem("lastHistoricalTimeframe");
    if (cachedLastHistTf) setLastHistoricalTimeframe(cachedLastHistTf as HistoricalTimeframe);
  }, []);

  useEffect(() => { sessionStorage.setItem("mapSandbox", JSON.stringify(mapSandbox)); }, [mapSandbox]);
  useEffect(() => { sessionStorage.setItem("graphSandboxes", JSON.stringify(graphSandboxes)); }, [graphSandboxes]);
  useEffect(() => { 
    sessionStorage.setItem("activeLevel", activeLevel);
    activeLevelRef.current = activeLevel; 
  }, [activeLevel]);
  useEffect(() => { sessionStorage.setItem("chatMessages", JSON.stringify(messages)); }, [messages]);
  useEffect(() => { sessionStorage.setItem("artifactCache", JSON.stringify(artifactCache)); }, [artifactCache]);
  useEffect(() => { sessionStorage.setItem("sessionTools", JSON.stringify(sessionTools)); }, [sessionTools]);
  useEffect(() => { sessionStorage.setItem("contextData", JSON.stringify(contextData)); }, [contextData]);
  useEffect(() => { sessionStorage.setItem("llmStatus", JSON.stringify(llmStatus)); }, [llmStatus]);
  useEffect(() => { sessionStorage.setItem("currentViewType", currentViewType); }, [currentViewType]);
  useEffect(() => { sessionStorage.setItem("timeframe", timeframe); }, [timeframe]);
  useEffect(() => { sessionStorage.setItem("lastHistoricalTimeframe", lastHistoricalTimeframe); }, [lastHistoricalTimeframe]);

  // ---> SELF-HEALING TELEMETRY FETCH <---
  useEffect(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    if (activeTools.length === 0 || selectedRooms.length === 0) return;

    activeTools.forEach(tool => {
      selectedRooms.forEach(room => {
        const roomMap = artifactCache[room] || {};
        const hasData = Object.keys(roomMap).some(
          k => k.toLowerCase() === tool.toLowerCase() && !!roomMap[k]?.[timeframe]
        );

        // Create a unique identifier for this exact network request
        const requestKey = `${room}-${tool}-${timeframe}`.toLowerCase();

        // ---> FIX: Only fetch if we lack data AND the request is not already in-flight! <---
        if (!hasData && !inFlightRequests.current.has(requestKey)) {
          console.log(`[SELF-HEALING FETCH] Requesting missing telemetry: Room ${room} | Tool: ${tool} | TF: ${timeframe}`);
          
          // Mark this request as in-flight before sending
          inFlightRequests.current.add(requestKey);

          ws.current?.send(JSON.stringify({
            type: "map_interaction",
            rooms: [room],
            floor: activeLevel,
            domain: tool,
            timeframe: timeframe
          }));
        }
      });
    });
  }, [timeframe, selectedRooms, activeTools, activeLevel, artifactCache]);

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
        if (data.type === "llm_status") {
          setAppState(data.state === "thinking" || data.state === "transcribing" ? "routing" : "tool_execution");
          setLlmStatus({ state: data.state, message: data.message, tool_name: data.tool_name });
        }
        if (data.type === "transcription_result") {
          setMessages(prev => [...prev, { sender: "user", text: data.text }]);
        }
        if (data.type === "transcription_only_result") {
          setTranscribedText(data.text);
          setAppState("idle");
          setLlmStatus(null);
        }
        
        if (data.type === "map_update" && data.artifact) {
          const artifact = data.artifact;
          const roomId = artifact.room_id;
          const domain = artifact.domain || "Unknown";
          const tf = (artifact.timeframe || "now") as Timeframe;

          // ---> FIX: Clear the request from in-flight memory as soon as data arrives! <---
          const requestKey = `${roomId}-${domain}-${tf}`.toLowerCase();
          inFlightRequests.current.delete(requestKey);

          let resolvedFloor = activeLevelRef.current;
          if (artifact.floor !== undefined && artifact.floor !== null) {
            resolvedFloor = String(artifact.floor); 
          } else if (roomId) {
            const derivedFloor = getFloorForRoom(roomId); 
            if (derivedFloor) resolvedFloor = derivedFloor;
          }

          const targetLevel = resolvedFloor || "B";
          if (targetLevel !== activeLevelRef.current) {
            setActiveLevel(targetLevel);
          }
          
          // 1. Sync View Type & Timeframe State automatically from LLM
          setTimeframe(tf);
          if (tf !== "now") {
            setLastHistoricalTimeframe(tf as HistoricalTimeframe);
            setCurrentViewType("graph");
          } else {
            setCurrentViewType(artifact.view_type === "error" ? "snapshot" : (artifact.view_type as ViewType));
          }

          // 2. Store securely in 3D Cache
          if (roomId && domain !== "Unknown") {
            setArtifactCache(prev => {
              const roomMap = prev[roomId] || {};
              const domainMap = roomMap[domain] || {};
              return {
                ...prev,
                [roomId]: {
                  ...roomMap,
                  [domain]: {
                    ...domainMap,
                    [tf]: artifact
                  }
                }
              };
            });
          }

          // 3. Update Sandbox state immediately so the UI switches cleanly
          if (tf === "now") {
            setMapSandbox(prev => {
              const floor = prev[targetLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };
              const newZoom = roomId && roomId !== "building";
              
              // ---> FLICKER FIX: Case-insensitive check prevents background tools from jumping to index 0! <---
              let newActiveTools = floor.activeTools;
              if (domain !== "Unknown") {
                const existingIndex = floor.activeTools.findIndex(
                  t => t.toLowerCase() === domain.toLowerCase()
                );
                // Only if it is a completely NEW tool (not in activeTools at all), add it to index 0
                if (existingIndex === -1) {
                  newActiveTools = [domain, ...floor.activeTools];
                }
                // If it IS already in activeTools (even at index 1 or 2), leave the array untouched!
              }

              // ---> MAP MODE FIX: Append/preserve rooms instead of overwriting! <---
              let newRooms = floor.selectedRooms;
              if (roomId) {
                if (roomId === "building" || roomId === "ALL") {
                  newRooms = [roomId];
                } else if (!floor.selectedRooms.includes(roomId)) {
                  const currentSpecifics = floor.selectedRooms.filter(r => r !== "building" && r !== "ALL");
                  newRooms = [...currentSpecifics, roomId];
                }
              }

              // ---> SILENT CACHING BAILOUT: If state didn't change, abort re-render! <---
              if (
                newActiveTools === floor.activeTools &&
                newRooms === floor.selectedRooms &&
                newZoom === floor.isZoomed
              ) {
                return prev; // Data sits quietly in artifactCache without flickering the map UI!
              }

              return {
                ...prev,
                [targetLevel]: {
                  ...floor,
                  selectedRooms: newRooms,
                  isZoomed: newZoom,
                  activeTools: newActiveTools
                }
              };
            });
          } else {
            // ---> GRAPH MODE FIX: Appends tool to the specific room's memory! <---
            setGraphSandboxes(prev => {
              const box = prev[tf as HistoricalTimeframe] || { selectedRoom: null, roomTools: {} };
              const targetRoom = roomId || box.selectedRoom;
              if (!targetRoom) return prev;

              const currentRoomTools = box.roomTools?.[targetRoom] || [];
              let newRoomTools = currentRoomTools;
              if (domain !== "Unknown" && !currentRoomTools.includes(domain)) {
                newRoomTools = [domain, ...currentRoomTools];
              }

              return {
                ...prev,
                [tf]: {
                  ...box,
                  selectedRoom: targetRoom,
                  roomTools: {
                    ...(box.roomTools || {}),
                    [targetRoom]: newRoomTools
                  }
                }
              };
            });
          }
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

  // ---> UPGRADED: Handles tool toggling safely for both Map and Graph modes <---
  const handleToggleSelect = (toggle: string) => {
    if (isGraphMode) {
      const histTf = timeframe as HistoricalTimeframe;
      let room = currentGraphBox.selectedRoom;
      // Auto-select first room on the active floor if none selected yet
      if (!room) {
        const floorRooms = getRoomsForFloor(activeLevel);
        if (floorRooms.length > 0) room = floorRooms[0];
      }
      if (!room) return;

      const currentTools = currentGraphBox.roomTools?.[room] || [];
      const newTools = [toggle, ...currentTools.filter(t => t !== toggle)];

      updateGraphSandbox(histTf, {
        selectedRoom: room,
        roomTools: {
          ...(currentGraphBox.roomTools || {}),
          [room]: newTools
        }
      });
    } else {
      const isNewTool = !activeTools.includes(toggle);
      const newTools = [toggle, ...activeTools.filter(t => t !== toggle)];
      let roomsToFetch = selectedRooms;

      if (isNewTool && selectedRooms.length === 0) {
        roomsToFetch = getRoomsForFloor(activeLevel);
        updateMapFloor(activeLevel, { activeTools: newTools, selectedRooms: roomsToFetch });
      } else {
        updateMapFloor(activeLevel, { activeTools: newTools });
      }
    }
  };

  // ---> UPGRADED: Smart tool inheritance when opening a room for the first time in Graph mode <---
  const handleRoomSelect = (roomId: string) => {
    if (isGraphMode) {
      const histTf = timeframe as HistoricalTimeframe;
      if (currentGraphBox.selectedRoom === roomId) return; 

      let existingTools = currentGraphBox.roomTools?.[roomId] || [];
      if (existingTools.length === 0 && activeTools.length > 0) {
        existingTools = [activeTools[0]];
      }

      updateGraphSandbox(histTf, { 
        selectedRoom: roomId,
        roomTools: {
          ...(currentGraphBox.roomTools || {}),
          [roomId]: existingTools
        }
      });
    } else {
      const hasActiveTools = activeTools.length > 0;
      const isCurrentlySelected = selectedRooms.includes(roomId);
      if (isCurrentlySelected && hasActiveTools) return; 

      let newSelection = [];
      let newZoom = isZoomed;

      if (isCurrentlySelected) {
          newSelection = selectedRooms.filter(r => r !== roomId);
          if (newSelection.length !== 1 && isZoomed) newZoom = false;
      } else {
          newSelection = [...selectedRooms, roomId];
      }

      updateMapFloor(activeLevel, { selectedRooms: newSelection, isZoomed: newZoom });
    }
  };

  // ---> UPGRADED: Seamlessly inherits room & tools when exploring a new timeframe <---
  const handleTimeframeChange = (newTf: Timeframe) => {
    setTimeframe(newTf);
    if (newTf === "now") {
      setCurrentViewType("snapshot");
    } else {
      const histTf = newTf as HistoricalTimeframe;
      setLastHistoricalTimeframe(histTf);
      setCurrentViewType("graph");
      
      const targetBox = graphSandboxes[histTf] || { selectedRoom: null, roomTools: {} };
      
      if (!targetBox.selectedRoom) {
        const prevHistTf = timeframe !== "now" ? (timeframe as HistoricalTimeframe) : lastHistoricalTimeframe;
        const prevBox = graphSandboxes[prevHistTf] || { selectedRoom: null, roomTools: {} };
        
        const room = prevBox.selectedRoom;
        if (room) {
          const prevTools = prevBox.roomTools?.[room] || [];
          if (prevTools.length > 0) {
            setGraphSandboxes(prev => ({
              ...prev,
              [histTf]: {
                selectedRoom: room,
                roomTools: {
                  ...(prev[histTf]?.roomTools || {}),
                  [room]: prevTools
                }
              }
            }));
          }
        }
      }
    }
  };

  const handleViewModeChange = (newMode: ViewType) => {
    setCurrentViewType(newMode);
    if (newMode === "graph" && timeframe === "now") {
      handleTimeframeChange(lastHistoricalTimeframe);
    } else if (newMode === "snapshot") {
      handleTimeframeChange("now");
    }
  };

  const handleResetSession = () => {
    setMapSandbox({});
    setGraphSandboxes(INITIAL_GRAPH_SANDBOXES);
    setActiveLevel("B");
    setMessages([]);
    setSessionTools([]);
    setContextData({ tokens: 0 });
    setArtifactCache({});
    setCurrentViewType("snapshot");
    setLlmStatus(null);
    setAppState("idle");
    setTranscribedText(null);
    setTimeframe("now");
    setLastHistoricalTimeframe("24h");
    sessionStorage.clear();
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "reset_session" }));
    }
  };

  const activeViewArtifacts: Record<string, any> = {};
  const currentRoomHealthData: Record<string, RoomHealth> = {};
  const visuallyActiveTool = activeTools[0]; 

  if (visuallyActiveTool) {
    Object.keys(artifactCache).forEach(roomId => {
      const toolKey = Object.keys(artifactCache[roomId] || {}).find(k => k.toLowerCase() === visuallyActiveTool.toLowerCase());
      const artifact = toolKey ? artifactCache[roomId]?.[toolKey]?.[timeframe] : undefined;
      if (artifact) {
        activeViewArtifacts[roomId] = artifact;
        currentRoomHealthData[roomId] = (artifact.status?.toLowerCase() || "good") as RoomHealth;
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
      <Sidebar 
        activeLevel={activeLevel}
        setActiveLevel={setActiveLevel}
        selectedRooms={selectedRooms}
        onRoomToggle={handleRoomSelect}
        activeTools={activeTools}
        floorStates={mapSandbox}
        timeframe={timeframe}
        onTimeframeChange={handleTimeframeChange}
        viewMode={currentViewType}
        onViewModeChange={handleViewModeChange}
        artifactCache={artifactCache}
        lastHistoricalTimeframe={lastHistoricalTimeframe}
      />

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
            setViewMode={handleViewModeChange}
            isZoomed={isZoomed}
            setIsZoomed={(z) => updateMapFloor(activeLevel, { isZoomed: z })}
            onToggleSelect={handleToggleSelect}
            roomHealthData={currentRoomHealthData}
            roomArtifacts={activeViewArtifacts} 
            allArtifacts={artifactCache}
            timeframe={timeframe}
          />
        </div>
      </div>

      <div className="w-[clamp(380px,30vw,630px)] flex-shrink-0 h-full pt-4 pr-4 pb-0 pl-2 transition-all duration-500 ease-in-out flex flex-col justify-end">
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