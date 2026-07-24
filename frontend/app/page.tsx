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

interface GraphSandboxState {
  selectedRoom: string | null;
  activeTool: string | null;
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

const INITIAL_GRAPH_SANDBOXES: Record<HistoricalTimeframe, GraphSandboxState> = {
  "2h": { selectedRoom: null, activeTool: null },
  "24h": { selectedRoom: null, activeTool: null },
  "7d": { selectedRoom: null, activeTool: null },
  "30d": { selectedRoom: null, activeTool: null },
  "90d": { selectedRoom: null, activeTool: null },
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

  const isGraphMode = currentViewType === "graph" && timeframe !== "now";
  const currentGraphBox = isGraphMode ? graphSandboxes[timeframe as HistoricalTimeframe] : { selectedRoom: null, activeTool: null };
  const currentMapFloor = mapSandbox[activeLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };

  const selectedRooms = useMemo(() => {
    return isGraphMode 
      ? (currentGraphBox.selectedRoom ? [currentGraphBox.selectedRoom] : []) 
      : currentMapFloor.selectedRooms;
  }, [isGraphMode, currentGraphBox.selectedRoom, currentMapFloor.selectedRooms]);

  const activeTools = useMemo(() => {
    return isGraphMode 
      ? (currentGraphBox.activeTool ? [currentGraphBox.activeTool] : []) 
      : currentMapFloor.activeTools;
  }, [isGraphMode, currentGraphBox.activeTool, currentMapFloor.activeTools]);

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

  // ---> NEW: SELF-HEALING TELEMETRY FETCH <---
  // Automatically fetches data if the UI lands on a room/tool/timeframe that isn't cached yet!
  useEffect(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    if (activeTools.length === 0 || selectedRooms.length === 0) return;

    activeTools.forEach(tool => {
      selectedRooms.forEach(room => {
        const roomMap = artifactCache[room] || {};
        const hasData = Object.keys(roomMap).some(
          k => k.toLowerCase() === tool.toLowerCase() && !!roomMap[k]?.[timeframe]
        );

        if (!hasData) {
          console.log(`[SELF-HEALING FETCH] Requesting missing telemetry: Room ${room} | Tool: ${tool} | TF: ${timeframe}`);
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
              const newActiveTools = domain !== "Unknown" ? [domain] : floor.activeTools;
              const newRooms = roomId ? [roomId] : floor.selectedRooms;
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
            setGraphSandboxes(prev => ({
              ...prev,
              [tf]: {
                selectedRoom: roomId || prev[tf as HistoricalTimeframe]?.selectedRoom || null,
                activeTool: domain !== "Unknown" ? domain : (prev[tf as HistoricalTimeframe]?.activeTool || null)
              }
            }));
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

  const handleToggleSelect = (toggle: string) => {
    if (isGraphMode) {
      const histTf = timeframe as HistoricalTimeframe;
      updateGraphSandbox(histTf, { activeTool: toggle });
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

  const handleRoomSelect = (roomId: string) => {
    if (isGraphMode) {
      const histTf = timeframe as HistoricalTimeframe;
      if (currentGraphBox.selectedRoom === roomId) return; 
      updateGraphSandbox(histTf, { selectedRoom: roomId });
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

  const handleTimeframeChange = (newTf: Timeframe) => {
    setTimeframe(newTf);
    if (newTf === "now") {
      setCurrentViewType("snapshot");
    } else {
      const histTf = newTf as HistoricalTimeframe;
      setLastHistoricalTimeframe(histTf);
      setCurrentViewType("graph");
      
      const targetBox = graphSandboxes[histTf];
      
      if (!targetBox.selectedRoom && !targetBox.activeTool) {
        const prevHistTf = timeframe !== "now" ? (timeframe as HistoricalTimeframe) : lastHistoricalTimeframe;
        const prevBox = graphSandboxes[prevHistTf] || { selectedRoom: null, activeTool: null };
        
        const room = prevBox.selectedRoom;
        const tool = prevBox.activeTool;

        if (room && tool) {
          const hasDataInNewTf = Object.keys(artifactCache[room] || {}).some(
            k => k.toLowerCase() === tool.toLowerCase() && !!artifactCache[room][k]?.[newTf]
          );
          
          if (hasDataInNewTf) {
            setGraphSandboxes(prev => ({
              ...prev,
              [histTf]: {
                selectedRoom: room,
                activeTool: tool
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