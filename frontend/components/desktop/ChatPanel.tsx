"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Mic, ChevronDown, Wrench } from "lucide-react";
import { AppState } from "@/app/page";

interface ChatPanelProps {
  appState: AppState;
  onSendMessage: (msg: string) => void;
  activeTools: string[];
}

export default function ChatPanel({ appState, onSendMessage, activeTools }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [showAllActions, setShowAllActions] = useState(false);

  const isIdle = appState === "idle";

  return (
    <div className={`flex flex-col w-full ${isIdle ? "h-auto" : "h-full bg-[#0A0A0A]/60 border border-[#A3B8B2]/10 rounded-3xl p-4"}`}>
      
      {/* IDLE GREETING */}
      {isIdle && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-light text-white mb-4">
            Hello, <span className="font-semibold text-[#14C89B]">Nick</span>.
          </h1>
          <p className="text-xl text-[#A3B8B2]/80">How can I assist you with the campus today?</p>
        </motion.div>
      )}

      {/* ACTIVE CHAT HEADER (AGENT TRACKER) */}
      {!isIdle && (
        <div className="flex flex-col gap-2 mb-4 border-b border-[#A3B8B2]/10 pb-4">
          <div 
            className="flex items-center justify-between cursor-pointer text-[#14C89B] hover:text-[#14C89B]/80 transition-colors"
            onClick={() => setShowAllActions(!showAllActions)}
          >
            <div className="flex items-center gap-2 text-sm font-medium">
              <Wrench size={16} className={appState === "tool_execution" ? "animate-pulse" : ""} />
              <span>{appState === "tool_execution" ? "Agent is thinking..." : "Agent Actions Completed"}</span>
            </div>
            <ChevronDown size={16} className={`transition-transform ${showAllActions ? "rotate-180" : ""}`} />
          </div>
          
          <AnimatePresence>
            {showAllActions && (
              <motion.div 
                initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <ul className="text-xs text-[#A3B8B2]/60 space-y-1 mt-2 pl-6 list-disc">
                  <li>Query routed to Map Topology Agent</li>
                  <li>Extracted Target: Floor 2</li>
                  {activeTools.map(tool => (
                    <li key={tool}>Executed Tool: fetch_{tool.toLowerCase().replace(' ', '_')}</li>
                  ))}
                  <li>JSON payload injected to UI renderer</li>
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* CHAT HISTORY AREA */}
      {!isIdle && (
        <div className="flex-1 overflow-y-auto mb-4 space-y-4 no-scrollbar">
          {/* Simulated History */}
          <div className="bg-[#14C89B]/10 border border-[#14C89B]/20 p-3 rounded-2xl rounded-tr-sm ml-8 text-[#A3B8B2] text-sm">
            Show me the air quality and occupancy on the 2nd floor.
          </div>
          <div className="bg-[#1E1E1E] border border-[#A3B8B2]/10 p-3 rounded-2xl rounded-tl-sm mr-8 text-white text-sm">
            I've loaded the map for Floor 2. You can view the Air Quality and Occupancy data now.
          </div>
        </div>
      )}

      {/* INPUT BOX */}
      <div className={`flex items-center gap-3 ${isIdle ? "max-w-2xl mx-auto w-full shadow-2xl" : "mt-auto"}`}>
        <button className="p-3 rounded-full bg-[#1E1E1E] text-[#14C89B] hover:bg-[#333] transition-colors border border-[#A3B8B2]/20">
          <Mic size={20} />
        </button>
        
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the campus..."
          className="flex-1 bg-[#1E1E1E] border border-[#A3B8B2]/20 rounded-full px-5 py-4 text-white placeholder-[#A3B8B2]/50 focus:outline-none focus:border-[#14C89B] transition-colors"
          onKeyDown={(e) => {
            if (e.key === "Enter" && input) {
              onSendMessage(input);
              setInput("");
            }
          }}
        />

        <button 
          onClick={() => { if(input) { onSendMessage(input); setInput(""); } }}
          className="p-4 rounded-full bg-[#14C89B] text-black hover:bg-[#14C89B]/80 transition-colors shadow-lg shadow-[#14C89B]/20"
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  );
}