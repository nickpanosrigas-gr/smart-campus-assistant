"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ChatPanel from "@/components/desktop/ChatPanel";
import MapStage from "@/components/desktop/MapStage";

export type AppState = "idle" | "routing" | "tool_execution" | "resolved";

export default function DesktopDashboard() {
  // We start at idle, but jumping to resolved for this milestone
  const [appState, setAppState] = useState<AppState>("idle");
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [activeFloor, setActiveFloor] = useState<number>(2);

  const handleUserMessage = (msg: string) => {
    // For now, jump straight to State 4 (Resolved) to see the final layout
    setAppState("resolved");
    setActiveTools(["Air Quality", "Occupancy"]);
  };

  return (
    <main className="w-full h-screen flex overflow-hidden bg-gradient-to-b from-[#0A664F] to-[#0A0A0A] text-[#A3B8B2] p-4 gap-4">
      
      {/* LEFT SIDE: MAP STAGE (Only visible > idle) */}
      <AnimatePresence mode="popLayout">
        {appState !== "idle" && (
          <motion.div
            initial={{ opacity: 0, x: -50, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="flex-1 flex flex-col min-w-0 bg-[#0A0A0A]/40 border border-[#A3B8B2]/10 rounded-3xl backdrop-blur-md overflow-hidden relative shadow-2xl"
          >
            <MapStage 
              appState={appState} 
              activeTools={activeTools}
              setActiveTools={setActiveTools}
              activeFloor={activeFloor}
              setActiveFloor={setActiveFloor}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* RIGHT SIDE (or CENTER if idle): CHAT PANEL */}
      <motion.div
        layout
        transition={{ type: "spring", bounce: 0.15, duration: 0.6 }}
        className={`flex flex-col h-full ${
          appState === "idle" ? "w-full max-w-3xl mx-auto justify-center" : "w-[400px] flex-shrink-0"
        }`}
      >
        <ChatPanel 
          appState={appState} 
          onSendMessage={handleUserMessage}
          activeTools={activeTools}
        />
      </motion.div>
      
    </main>
  );
}