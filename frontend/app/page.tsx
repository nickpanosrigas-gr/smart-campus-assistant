"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import LandingPage from "@/components/LandingPage";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";
import Sidebar from "@/components/desktop/Sidebar";

const API_BASE_URL = process.env.NODE_ENV === "production" ? "" : "http://localhost:8000";

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

const INITIAL_GRAPH_SANDBOXES: Record<HistoricalTimeframe, GraphSandboxState> = {
  "2h": { selectedRoom: null, roomTools: {} },
  "24h": { selectedRoom: null, roomTools: {} },
  "7d": { selectedRoom: null, roomTools: {} },
  "30d": { selectedRoom: null, roomTools: {} },
  "90d": { selectedRoom: null, roomTools: {} },
};

const SESSION_TTL_MS = 60 * 60 * 1000; // 1 Hour

function isCacheValid(): boolean {
  if (typeof window === "undefined") return false;
  const lastActive = localStorage.getItem("lastActiveTimestamp");
  if (!lastActive) return false;
  return Date.now() - parseInt(lastActive, 10) < SESSION_TTL_MS;
}

// ==========================================
// 1. MAIN APP WRAPPER (Handles Auth State)
// ==========================================
export default function Page() {
  const [user, setUser] = useState<{ sub: string; name?: string; picture?: string } | null | undefined>(undefined);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
          credentials: "include", 
        });
        if (res.ok) {
          const data = await res.json();
          setUser(data);
        } else {
          setUser(null);
        }
      } catch (error) {
        setUser(null);
      }
    };
    checkAuth();
  }, []);

  if (user === undefined) {
    return (
      <div className="w-full h-[100dvh] flex items-center justify-center bg-[#020604]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  if (user === null) {
    return <LandingPage onLoginSuccess={() => window.location.reload()} />;
  }

  return <DesktopDashboard user={user} />;
}

// ==========================================
// 2. MAIN DASHBOARD COMPONENT
// ==========================================
function DesktopDashboard({ user }: { user: { sub: string; name?: string; picture?: string } }) {

  const [appState, setAppState] = useState<AppState>("idle");
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("llmStatus");
      if (cached && cached !== "null") return JSON.parse(cached);
    }
    return null;
  });
  
  const [timeframe, setTimeframe] = useState<Timeframe>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      return (localStorage.getItem("timeframe") as Timeframe) || "now";
    }
    return "now";
  });

  const [lastHistoricalTimeframe, setLastHistoricalTimeframe] = useState<HistoricalTimeframe>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      return (localStorage.getItem("lastHistoricalTimeframe") as HistoricalTimeframe) || "24h";
    }
    return "24h";
  });

  const [currentViewType, setCurrentViewType] = useState<ViewType>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      return (localStorage.getItem("currentViewType") as ViewType) || "snapshot";
    }
    return "snapshot";
  });

  const [activeLevel, setActiveLevel] = useState<string>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      return localStorage.getItem("activeLevel") || "B";
    }
    return "B";
  }); 

  const [artifactCache, setArtifactCache] = useState<Record<string, Record<string, Record<string, any>>>>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("artifactCache");
      if (cached) return JSON.parse(cached);
    }
    return {};
  });
  
  const [mapSandbox, setMapSandbox] = useState<Record<string, MapSandboxState>>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("mapSandbox");
      if (cached) return JSON.parse(cached);
    }
    return {};
  });

  const [graphSandboxes, setGraphSandboxes] = useState<Record<HistoricalTimeframe, GraphSandboxState>>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("graphSandboxes");
      if (cached) return JSON.parse(cached);
    }
    return INITIAL_GRAPH_SANDBOXES;
  });

  const [contextData, setContextData] = useState(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("contextData");
      if (cached) return JSON.parse(cached);
    }
    return { tokens: 0 };
  });

  const [sessionTools, setSessionTools] = useState<{tool: string, room: string}[]>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("sessionTools");
      if (cached) return JSON.parse(cached);
    }
    return [];
  });

  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>(() => {
    if (typeof window !== "undefined" && isCacheValid()) {
      const cached = localStorage.getItem("chatMessages");
      if (cached) return JSON.parse(cached);
    }
    return [];
  });

  const [transcribedText, setTranscribedText] = useState<string | null>(null);

  const [ollamaOnline, setOllamaOnline] = useState(true);
  const [whisperOnline, setWhisperOnline] = useState(true);
  
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

  const activeTools = useMemo(() => {
    if (isGraphMode) {
      const room = currentGraphBox.selectedRoom;
      if (!room) return [];
      if (currentGraphBox.roomTools && currentGraphBox.roomTools[room]) {
        return currentGraphBox.roomTools[room];
      }
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

  // --- MULTI-TAB CROSS-SYNC EVENT LISTENER ---
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      try {
        if (e.key === "mapSandbox") setMapSandbox(e.newValue ? JSON.parse(e.newValue) : {});
        if (e.key === "graphSandboxes") setGraphSandboxes(e.newValue ? JSON.parse(e.newValue) : INITIAL_GRAPH_SANDBOXES);
        if (e.key === "activeLevel") setActiveLevel(e.newValue || "B");
        if (e.key === "chatMessages") setMessages(e.newValue ? JSON.parse(e.newValue) : []);
        if (e.key === "artifactCache") setArtifactCache(e.newValue ? JSON.parse(e.newValue) : {});
        if (e.key === "sessionTools") setSessionTools(e.newValue ? JSON.parse(e.newValue) : []);
        if (e.key === "contextData") setContextData(e.newValue ? JSON.parse(e.newValue) : { tokens: 0 });
        if (e.key === "llmStatus") setLlmStatus(e.newValue && e.newValue !== "null" ? JSON.parse(e.newValue) : null);
        if (e.key === "currentViewType") setCurrentViewType((e.newValue as ViewType) || "snapshot");
        if (e.key === "timeframe") setTimeframe((e.newValue as Timeframe) || "now");
        if (e.key === "lastHistoricalTimeframe") setLastHistoricalTimeframe((e.newValue as HistoricalTimeframe) || "24h");
      } catch (err) {
        console.error("Multi-tab sync error:", err);
      }
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  // --- PERSIST STATE WRITES TO LOCALSTORAGE ---
  useEffect(() => { localStorage.setItem("mapSandbox", JSON.stringify(mapSandbox)); }, [mapSandbox]);
  useEffect(() => { localStorage.setItem("graphSandboxes", JSON.stringify(graphSandboxes)); }, [graphSandboxes]);
  useEffect(() => { 
    localStorage.setItem("activeLevel", activeLevel);
    activeLevelRef.current = activeLevel; 
  }, [activeLevel]);
  useEffect(() => { localStorage.setItem("chatMessages", JSON.stringify(messages)); }, [messages]);
  useEffect(() => { localStorage.setItem("artifactCache", JSON.stringify(artifactCache)); }, [artifactCache]);
  useEffect(() => { localStorage.setItem("sessionTools", JSON.stringify(sessionTools)); }, [sessionTools]);
  useEffect(() => { localStorage.setItem("contextData", JSON.stringify(contextData)); }, [contextData]);
  useEffect(() => { localStorage.setItem("llmStatus", JSON.stringify(llmStatus)); }, [llmStatus]);
  useEffect(() => { localStorage.setItem("currentViewType", currentViewType); }, [currentViewType]);
  useEffect(() => { localStorage.setItem("timeframe", timeframe); }, [timeframe]);
  useEffect(() => { localStorage.setItem("lastHistoricalTimeframe", lastHistoricalTimeframe); }, [lastHistoricalTimeframe]);

  useEffect(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    if (activeTools.length === 0 || selectedRooms.length === 0) return;

    activeTools.forEach(tool => {
      // 1. Create an array to batch all rooms that need fetching
      const roomsToFetch: string[] = [];

      selectedRooms.forEach(room => {
        const roomMap = artifactCache[room] || {};
        const hasData = Object.keys(roomMap).some(
          k => k.toLowerCase() === tool.toLowerCase() && !!roomMap[k]?.[timeframe]
        );

        const requestKey = `${room}-${tool}-${timeframe}`.toLowerCase();

        if (!hasData && !inFlightRequests.current.has(requestKey)) {
          // 2. Add to our batch array and mark as in-flight
          roomsToFetch.push(room);
          inFlightRequests.current.add(requestKey);
        }
      });

      // 3. Send ONE WebSocket message containing all missing rooms
      if (roomsToFetch.length > 0) {
        console.log(`[SELF-HEALING FETCH] Requesting missing telemetry: Rooms [${roomsToFetch.join(', ')}] | Tool: ${tool} | TF: ${timeframe}`);
        
        ws.current?.send(JSON.stringify({
          type: "map_interaction",
          rooms: roomsToFetch,
          floor: activeLevel,
          domain: tool,
          timeframe: timeframe
        }));
      }
    });
  }, [timeframe, selectedRooms, activeTools, activeLevel, artifactCache]);

  // --- WEBSOCKET CONNECTION & EVENT HANDLER ---
  useEffect(() => {
    const getWsUrl = () => {
      if (typeof window === "undefined") return ""; 
      
      if (process.env.NODE_ENV === "production") {
        // In production, connect to the Next.js proxy
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        return `${protocol}//${window.location.host}/ws/chat`;
      }
      
      // Local development fallback
      return "ws://localhost:8000/ws/chat";
    };

    ws.current = new WebSocket(getWsUrl());
    ws.current.onopen = () => console.log("🟢 Connected to Smart Campus Backend");

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // --- SESSION HANDSHAKE HANDLER ---
        if (data.type === "session_init") {
          if (data.is_new) {
            console.log("🧹 New backend session detected. Resetting local cache...");
            handleResetSession(false);
          } else {
            // Update timestamp for active session
            localStorage.setItem("lastActiveTimestamp", Date.now().toString());
          }
          return;
        }

        // Record activity on any incoming message
        localStorage.setItem("lastActiveTimestamp", Date.now().toString());

        // --- SESSION TIMEOUT HANDLER ---
        if (data.type === "session_expired") {
          console.log("⏱️ Backend session expired due to inactivity. Resetting dashboard to default state...");
          handleResetSession(false);
          return;
        }

        // --- HEALTH MONITOR HANDLER ---
        if (data.type === "model_health") {
          setOllamaOnline(data.ollama);
          setWhisperOnline(data.whisper);
        }

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
          
          setTimeframe(tf);
          if (tf !== "now") {
            setLastHistoricalTimeframe(tf as HistoricalTimeframe);
            setCurrentViewType("graph");
          } else {
            setCurrentViewType(artifact.view_type === "error" ? "snapshot" : (artifact.view_type as ViewType));
          }

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

          if (tf === "now") {
            setMapSandbox(prev => {
              const floor = prev[targetLevel] || { selectedRooms: [], activeTools: [], isZoomed: false };
              const newZoom = roomId && roomId !== "building";
              
              let newActiveTools = floor.activeTools;
              if (domain !== "Unknown") {
                const existingIndex = floor.activeTools.findIndex(
                  t => t.toLowerCase() === domain.toLowerCase()
                );
                if (existingIndex === -1) {
                  newActiveTools = [domain, ...floor.activeTools];
                }
              }

              let newRooms = floor.selectedRooms;
              if (roomId) {
                if (roomId === "building" || roomId === "ALL") {
                  newRooms = [roomId];
                } else if (!floor.selectedRooms.includes(roomId)) {
                  const currentSpecifics = floor.selectedRooms.filter(r => r !== "building" && r !== "ALL");
                  newRooms = [...currentSpecifics, roomId];
                }
              }

              if (
                newActiveTools === floor.activeTools &&
                newRooms === floor.selectedRooms &&
                newZoom === floor.isZoomed
              ) {
                return prev; 
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
    
    localStorage.setItem("lastActiveTimestamp", Date.now().toString());
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
    localStorage.setItem("lastActiveTimestamp", Date.now().toString());
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
      let room = currentGraphBox.selectedRoom;
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

  const handleFloorChange = (newLevel: string) => {
    setActiveLevel(newLevel);

    if (isGraphMode) {
      const histTf = timeframe as HistoricalTimeframe;
      const currentBox = graphSandboxes[histTf] || { selectedRoom: null, roomTools: {} };
      const currentSelectedRoom = currentBox.selectedRoom;

      if (!currentSelectedRoom || getFloorForRoom(currentSelectedRoom) !== newLevel) {
        
        const floorRoomsWithData = Object.keys(currentBox.roomTools || {}).filter(
          (roomId) => getFloorForRoom(roomId) === newLevel
        );

        if (floorRoomsWithData.length > 0) {
          updateGraphSandbox(histTf, { 
            selectedRoom: floorRoomsWithData[floorRoomsWithData.length - 1] 
          });
        } else {
          updateGraphSandbox(histTf, { selectedRoom: null });
        }
      }
    }
  };

  const handleResetSession = (notifyBackend: boolean = true) => {
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
    
    setOllamaOnline(true);
    setWhisperOnline(true);
    
    localStorage.removeItem("mapSandbox");
    localStorage.removeItem("graphSandboxes");
    localStorage.removeItem("activeLevel");
    localStorage.removeItem("chatMessages");
    localStorage.removeItem("artifactCache");
    localStorage.removeItem("sessionTools");
    localStorage.removeItem("contextData");
    localStorage.removeItem("llmStatus");
    localStorage.removeItem("currentViewType");
    localStorage.removeItem("timeframe");
    localStorage.removeItem("lastHistoricalTimeframe");
    localStorage.removeItem("ollamaOnline");
    localStorage.removeItem("whisperOnline");

    localStorage.setItem("lastActiveTimestamp", Date.now().toString());

    if (notifyBackend && ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "reset_session" }));
    }
  };

  const handleStopResponse = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop_response" }));
    }
    setAppState("resolved");
    setLlmStatus(null);
  };

  const handleLogout = async () => {
    localStorage.clear();
    try {
      await fetch(`${API_BASE_URL}/api/auth/logout`, { method: "POST", credentials: "include" });
      window.location.reload();
    } catch (err) {
      console.error("Failed to logout:", err);
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
      className="fixed inset-0 w-full h-[100dvh] flex overflow-hidden text-[#A3B8B2] overscroll-none"
      style={{
        background: "radial-gradient(circle at 30% 20%, #064E3B 0%, #020604 50%, #000000 100%)"
      }}
    >
      {/* Hide Sidebar on mobile */}
      <div className="hidden md:block h-full shrink-0 z-10">
        <Sidebar 
          activeLevel={activeLevel}
          setActiveLevel={handleFloorChange}
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
          userEmail={user.sub} 
          userPicture={user.picture}
          onLogout={handleLogout}
        />
      </div>

      {/* Hide Map on mobile */}
      <div className="hidden md:flex flex-1 flex-col min-w-0 relative overflow-hidden h-full py-4 pl-4 pr-2">
        <div className="flex-1 flex flex-col min-w-0 relative overflow-hidden h-full rounded-3xl">
          <MapStage 
            appState={appState} 
            activeTools={activeTools}
            activeLevel={activeLevel}
            setActiveLevel={handleFloorChange} 
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

      {/* Chat Panel - Full width on mobile, Fixed width on desktop */}
      <div className="w-full md:w-[clamp(380px,30vw,630px)] flex-shrink-0 h-full p-0 md:pt-4 md:pr-4 md:pb-0 md:pl-2 transition-all duration-500 ease-in-out flex flex-col justify-end">
        <ChatPanel 
          appState={appState} 
          llmStatus={llmStatus}
          onSendMessage={handleUserMessage}
          onSendAudio={handleSendAudio}
          onStopResponse={handleStopResponse}
          activeTools={activeTools}
          messages={messages}
          contextData={contextData}  
          sessionTools={sessionTools} 
          onResetSession={() => handleResetSession(true)}
          transcribedText={transcribedText}
          onClearTranscribedText={() => setTranscribedText(null)}
          ollamaOnline={ollamaOnline}
          whisperOnline={whisperOnline}
          userName={user?.name?.split(' ')[0]} 
          activeLevel={activeLevel}
          selectedRooms={selectedRooms}
          timeframe={timeframe}
          onLogout={handleLogout}
        />
      </div>
    </main>
  );
}