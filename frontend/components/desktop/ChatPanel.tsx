import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChevronDown, ChevronUp, Cpu } from 'lucide-react';
import { SENSOR_COLORS } from '@/components/map/constants';

interface ChatPanelProps {
  appState: "idle" | "routing" | "tool_execution" | "resolved";
  onSendMessage: (msg: string) => void;
  activeTools: string[];
  messages: Array<{ sender: "user" | "agent"; text: string }>;
  contextData?: { tokens: number }; // <-- This fixes the red highlight!
  sessionTools?: { tool: string; room: string }[];
}

export default function ChatPanel({ appState, onSendMessage, messages, contextData, sessionTools }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [isStatusExpanded, setIsStatusExpanded] = useState(false); 
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Read config from .env.local
  const LLM_MODEL = process.env.NEXT_PUBLIC_LLM_MODEL || "LLM";
  const MAX_TOKENS = Number(process.env.NEXT_PUBLIC_LLM_CONTEXT_SIZE) || 8192;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, appState, isStatusExpanded]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input);
    setInput("");
  };

  // --- CONTEXT MATH & DYNAMIC SENSOR COLORS ---
  const tokens = contextData?.tokens || 0;
  const pct = MAX_TOKENS > 0 ? (tokens / MAX_TOKENS) * 100 : 0;
  
  // Default to Green
  let statusColor = SENSOR_COLORS?.Good || "#14C89B"; 
  let statusText = "Good";
  
  if (pct >= 85) {
    statusColor = SENSOR_COLORS?.Error || "#ef4444";
    statusText = "Error";
  } else if (pct >= 50) {
    statusColor = SENSOR_COLORS?.Warning || "#f97316";
    statusText = "Warning";
  }

  return (
    <div className="flex flex-col h-full bg-[#0A0A0A]/80 border border-[#A3B8B2]/10 rounded-3xl backdrop-blur-md overflow-hidden shadow-2xl">
      
      {/* --- CONTEXT WINDOW TRACKER --- */}
      <div className="p-2 border-b border-[#A3B8B2]/10">
        <button 
          onClick={() => setIsStatusExpanded(!isStatusExpanded)}
          className={`w-full flex flex-col p-4 transition-colors ${isStatusExpanded ? 'rounded-t-2xl' : 'rounded-2xl'}`}
          style={{ backgroundColor: `${statusColor}1A`, color: statusColor }}
        >
          <div className="flex justify-between w-full items-center mb-3">
            <span className="font-semibold text-sm flex items-center gap-2">
              <Cpu size={18} style={{ color: statusColor }} /> {LLM_MODEL} Context ({statusText})
            </span>
            <div className="flex items-center gap-3">
               <span className="text-xs font-mono">{tokens.toLocaleString()} / {MAX_TOKENS.toLocaleString()} ctx</span>
               {isStatusExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden">
            <div className="h-full transition-all duration-500 ease-out" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: statusColor }} />
          </div>
        </button>
        
        {/* Expandable Session Tools List */}
        <div className={`overflow-hidden transition-all duration-300 ease-in-out bg-black/40 ${isStatusExpanded ? 'max-h-64 border border-[#A3B8B2]/10 border-t-0 rounded-b-2xl' : 'max-h-0'}`}>
          <div className="p-4 text-xs font-mono text-[#A3B8B2] space-y-3 overflow-y-auto max-h-64">
             {(!sessionTools || sessionTools.length === 0) ? (
                 <p className="opacity-50 text-center py-2">No tools used in this session yet.</p>
             ) : (
                 // --- NEW: ADJACENT TOOL GROUPING ---
                 sessionTools.reduce((acc, curr) => {
                   const last = acc[acc.length - 1];
                   if (last && last.tool === curr.tool) {
                     if (!last.rooms.includes(curr.room)) last.rooms.push(curr.room);
                   } else {
                     acc.push({ tool: curr.tool, rooms: [curr.room] });
                   }
                   return acc;
                 }, [] as { tool: string; rooms: string[] }[]).map((st, idx) => (
                     <div key={idx} className="flex flex-col gap-1 pb-3 border-b border-[#A3B8B2]/10 last:border-0 last:pb-0">
                         <span className="font-bold" style={{ color: statusColor }}>➜ {st.tool}</span>
                         <span className="pl-5 opacity-70 border-l border-[#A3B8B2]/20 ml-1">Target: {st.rooms.join(", ")}</span>
                     </div>
                 ))
             )}
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[90%] p-4 rounded-2xl ${
              msg.sender === 'user' 
                ? 'bg-[#14C89B] text-black rounded-tr-sm' 
                : 'bg-[#A3B8B2]/10 text-[#A3B8B2] rounded-tl-sm'
            }`}>
              {msg.sender === 'agent' ? (
                <div className="text-sm space-y-2 prose prose-invert prose-p:leading-relaxed prose-strong:text-white">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-sm font-medium">{msg.text}</p>
              )}
            </div>
          </div>
        ))}
        {appState === 'tool_execution' && (
           <div className="flex justify-start">
              <div className="max-w-[90%] p-4 rounded-2xl rounded-tl-sm bg-orange-500/10 text-orange-400 border border-orange-500/20 flex items-center gap-3">
                 <div className="w-4 h-4 border-2 border-orange-400 border-t-transparent rounded-full animate-spin"></div>
                 <span className="text-sm font-medium animate-pulse">Running diagnostics & telemetry tools...</span>
              </div>
           </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 bg-black/20 border-t border-[#A3B8B2]/10">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask the campus assistant..."
            className="w-full bg-[#0A0A0A] border border-[#A3B8B2]/20 rounded-xl py-3 px-4 pr-12 text-sm text-white placeholder-[#A3B8B2]/50 focus:outline-none focus:border-[#14C89B] transition-colors"
          />
          <button 
            type="submit"
            disabled={!input.trim() || appState === 'tool_execution'}
            className="absolute right-2 p-2 text-[#14C89B] disabled:opacity-50 hover:bg-[#14C89B]/10 rounded-lg transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
          </button>
        </div>
      </form>
    </div>
  );
}