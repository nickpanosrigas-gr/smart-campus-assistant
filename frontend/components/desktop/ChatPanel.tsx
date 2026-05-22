"use client";
import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Send, Mic, Wrench } from "lucide-react";
import { AppState } from "@/app/page";

interface Message {
  sender: "user" | "agent";
  text: string;
}

interface ChatPanelProps {
  appState: AppState;
  onSendMessage: (msg: string) => void;
  activeTools: string[];
  messages: Message[];
}

export default function ChatPanel({ appState, onSendMessage, messages }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col w-full h-full bg-[#0A0A0A]/60 border border-[#A3B8B2]/10 rounded-3xl p-4 overflow-hidden shadow-xl backdrop-blur-xl">
      
      {/* SYSTEM HEADER INDICATOR */}
      <div className="flex items-center gap-3 mb-4 border-b border-[#A3B8B2]/10 pb-4 shrink-0 px-2">
        <div className={`p-2 rounded-xl ${appState === "tool_execution" ? "bg-[#14C89B]/20 text-[#14C89B]" : "bg-[#1E1E1E] text-[#A3B8B2]/60"}`}>
          <Wrench size={18} className={appState === "tool_execution" ? "animate-spin" : ""} style={{ animationDuration: "3s" }} />
        </div>
        <div className="flex flex-col">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#A3B8B2]/40">System Status</span>
          <span className="text-sm font-medium text-white">
            {appState === "idle" && "Ready for request"}
            {appState === "routing" && "Routing intent..."}
            {appState === "tool_execution" && "Executing smart tools..."}
            {appState === "resolved" && "Result processed"}
          </span>
        </div>
      </div>

      {/* MESSAGE STREAM LOG AREA */}
      <div ref={historyRef} className="flex-1 overflow-y-auto mb-4 space-y-4 pr-1 no-scrollbar flex flex-col justify-start">
        {messages.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, y: 15 }} 
            animate={{ opacity: 1, y: 0 }}
            className="my-auto text-center px-4"
          >
            <h1 className="text-4xl font-light text-white mb-3">
              Hello, <span className="font-semibold text-[#14C89B]">Nick</span>.
            </h1>
            <p className="text-md text-[#A3B8B2]/60 leading-relaxed max-w-xs mx-auto">
              I am connected and ready. Ask me anything about the campus infrastructure or specific domains.
            </p>
          </motion.div>
        ) : (
          messages.map((msg, index) => {
            const isUser = msg.sender === "user";
            return (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key={index}
                className={`max-w-[85%] p-3.5 rounded-2xl text-sm leading-relaxed shadow-md ${
                  isUser
                    ? "bg-[#14C89B]/10 border border-[#14C89B]/20 text-[#e4f9f4] self-end rounded-tr-sm"
                    : "bg-[#1E1E1E] border border-[#A3B8B2]/10 text-white self-start rounded-tl-sm"
                }`}
              >
                {msg.text}
              </motion.div>
            );
          })
        )}
      </div>

      {/* INPUT CONTROL CONTROLLER */}
      <div className="flex items-center gap-2 shrink-0 pt-2 border-t border-[#A3B8B2]/10">
        <button className="p-3.5 rounded-full bg-[#1E1E1E] text-[#14C89B] hover:bg-[#252525] transition-colors border border-[#A3B8B2]/10 hover:border-[#14C89B]/30 shrink-0">
          <Mic size={18} />
        </button>
        
        <div className="flex-1 relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your instruction..."
            className="w-full bg-[#1E1E1E] border border-[#A3B8B2]/10 rounded-full pl-5 pr-14 py-3.5 text-sm text-white placeholder-[#A3B8B2]/40 focus:outline-none focus:border-[#14C89B] transition-colors"
            onKeyDown={(e) => {
              if (e.key === "Enter" && input.trim()) {
                onSendMessage(input.trim());
                setInput("");
              }
            }}
          />

          <button 
            onClick={() => { if (input.trim()) { onSendMessage(input.trim()); setInput(""); } }}
            className="absolute right-1.5 p-2.5 rounded-full bg-[#14C89B] text-black hover:bg-[#14C89B]/90 transition-colors shadow-md"
          >
            <Send size={16} />
          </button>
        </div>
      </div>

    </div>
  );
}