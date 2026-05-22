"use client";
import { useState, useEffect, useRef } from "react";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";
import { RoomHealth } from "@/components/map/constants";

export type AppState = "idle" | "routing" | "tool_execution" | "resolved";
export type ViewMode = "map" | "graph";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/chat";

export default function DesktopDashboard() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [activeLevel, setActiveLevel] = useState<string>("B"); 
  const [selectedRooms, setSelectedRooms] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [isZoomed, setIsZoomed] = useState<boolean>(false);
  const [roomHealthData, setRoomHealthData] = useState<Record<string, RoomHealth>>({});

  const [messages, setMessages] = useState<Array<{ sender: "user" | "agent"; text: string }>>([]);
  const ws = useRef<WebSocket | null>(null);

  // --- BROWSER CACHING LOGIC ---
  useEffect(() => {
    const cachedTools = sessionStorage.getItem("activeTools");
    if (cachedTools) setActiveTools(JSON.parse(cachedTools));

    const cachedData = sessionStorage.getItem("roomHealthData");
    if (cachedData) setRoomHealthData(JSON.parse(cachedData));

    const cachedLevel = sessionStorage.getItem("activeLevel");
    if (cachedLevel) setActiveLevel(cachedLevel);
  }, []);

  useEffect(() => {
    sessionStorage.setItem("activeTools", JSON.stringify(activeTools));
  }, [activeTools]);

  useEffect(() => {
    sessionStorage.setItem("roomHealthData", JSON.stringify(roomHealthData));
  }, [roomHealthData]);

  useEffect(() => {
    sessionStorage.setItem("activeLevel", activeLevel);
  }, [activeLevel]);
  // ------------------------------

  useEffect(() => {
    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => console.log("✅ Connected to Smart Campus Backend");

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Received from backend:", data);

        if (data.type === "thinking" || data.type === "tool_start") {
          setAppState("tool_execution");
          if (data.tools_used) {
             setActiveTools(prev => {
                const newTools = data.tools_used.filter((t: string) => !prev.includes(t));
                return [...newTools, ...prev];
             });
          }
        }
        
        if (data.type === "map_update" || data.room_data || data.target_rooms) {
          if (data.target_rooms) {
            setSelectedRooms(data.target_rooms);
            if (data.target_rooms.length === 1) setIsZoomed(true);
          }
          if (data.room_data) {
            setRoomHealthData(prev => ({ ...prev, ...data.room_data }));
          }
        }

        if (data.reply || data.text) {
          const replyText = data.reply || data.text;
          setMessages(prev => {
            if (prev.length > 0 && prev[prev.length - 1].sender === "agent") {
              const updated = [...prev];
              updated[updated.length - 1] = { sender: "agent", text: replyText };
              return updated;
            } else {
              return [...prev, { sender: "agent", text: replyText }];
            }
          });
          setAppState("resolved");
        }

        if (data.type === "resolved") setAppState("resolved");

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
      ws.current.send(JSON.stringify({
        type: "chat_message",
        query: msg,
        context: { 
          activeLevel, 
          // If 0 rooms selected, tell the LLM it's targeting "ALL" rooms on this floor
          selectedRooms: selectedRooms.length > 0 ? selectedRooms : ["ALL"] 
        }
      }));
    }
  };

  const handleToggleSelect = (toggle: string) => {
    if (!activeTools.includes(toggle)) {
      setActiveTools(prev => [toggle, ...prev]);
    } else {
      setActiveTools(prev => [toggle, ...prev.filter(t => t !== toggle)]);
    }
    
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: "map_interaction",
        // Send the array of rooms. If empty, send ["ALL"] to trigger floor-wide data
        rooms: selectedRooms.length > 0 ? selectedRooms : ["ALL"],
        floor: activeLevel,
        domain: toggle
      }));
    }
  };

  // --- HARD RESET ON LEVEL CHANGE ---
  const handleLevelChange = (newLevel: string) => {
    if (newLevel !== activeLevel) {
      setActiveLevel(newLevel);
      setSelectedRooms([]);
      setIsZoomed(false);
      setActiveTools([]); // Send toggles back to unavailable group
      setRoomHealthData({}); // Clear map data from previous floor
    }
  };

  return (
    <main className="w-full h-screen flex overflow-hidden bg-gradient-to-b from-[#0A664F] to-[#0A0A0A] text-[#A3B8B2] p-4 gap-4">
      
      {/* LEFT SIDE: MAP CONTAINER */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0A0A0A]/40 border border-[#A3B8B2]/10 rounded-3xl backdrop-blur-md overflow-hidden relative shadow-2xl h-full">
        <MapStage 
          appState={appState} 
          activeTools={activeTools}
          setActiveTools={setActiveTools}
          activeLevel={activeLevel}
          setActiveLevel={handleLevelChange} // Inject the reset function here
          selectedRooms={selectedRooms}
          setSelectedRooms={setSelectedRooms}
          viewMode={viewMode}
          setViewMode={setViewMode}
          isZoomed={isZoomed}
          setIsZoomed={setIsZoomed}
          roomHealthData={roomHealthData}
          onToggleSelect={handleToggleSelect}
        />
      </div>

      {/* RIGHT SIDE: CHAT INTERFACE */}
      <div className="w-[420px] flex-shrink-0 h-full transition-all duration-500 ease-in-out">
        <ChatPanel 
          appState={appState} 
          onSendMessage={handleUserMessage}
          activeTools={activeTools}
          messages={messages}
        />
      </div>

    </main>
  );
}