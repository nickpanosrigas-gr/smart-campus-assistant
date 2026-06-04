import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Cpu, Trash2, ChevronDown, CheckCircle2 } from 'lucide-react';
import { SENSOR_COLORS, ROOM_COLORS } from '@/components/map/constants';

interface ChatPanelProps {
  appState: "idle" | "routing" | "tool_execution" | "resolved";
  llmStatus?: { state: string; message: string; tool_name?: string } | null;
  onSendMessage: (msg: string) => void;
  activeTools: string[];
  messages: Array<{ sender: "user" | "agent"; text: string }>;
  contextData?: { tokens: number };
  sessionTools?: { tool: string; room: string }[];
  onResetSession: () => void;
}

export default function ChatPanel({ 
  appState, 
  llmStatus, 
  onSendMessage, 
  messages, 
  contextData, 
  sessionTools, 
  onResetSession 
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [isStatusExpanded, setIsStatusExpanded] = useState(false); 
  
  // Track the LLM Status History for the current turn
  const [statusHistory, setStatusHistory] = useState<string[]>([]);
  const [isStatusLogExpanded, setIsStatusLogExpanded] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const LLM_MODEL = process.env.NEXT_PUBLIC_LLM_MODEL || "Qwen3.5 4B";
  const MAX_TOKENS = Number(process.env.NEXT_PUBLIC_LLM_CONTEXT_SIZE) || 8192;

  useEffect(() => {
    if (llmStatus && llmStatus.message) {
      setStatusHistory(prev => {
        if (prev[prev.length - 1] === llmStatus.message) return prev;
        return [...prev, llmStatus.message];
      });
    }
  }, [llmStatus]);

  useEffect(() => {
    // Add a tiny delay to allow the CSS grid animation to start so it scrolls to the true bottom
    const timeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
    return () => clearTimeout(timeout);
  }, [messages, appState, isStatusExpanded, isStatusLogExpanded, statusHistory]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Clear the status history for the NEW query so the old one goes away
    setStatusHistory([]);
    setIsStatusLogExpanded(false);
    
    onSendMessage(input);
    setInput("");
  };

  const handleReset = () => {
    setStatusHistory([]);
    setIsStatusLogExpanded(false);
    onResetSession();
  };

  const tokens = contextData?.tokens || 0;
  const pct = MAX_TOKENS > 0 ? (tokens / MAX_TOKENS) * 100 : 0;
  
  let statusColor = SENSOR_COLORS?.good || "#14C89B"; 
  let statusText = "Good";
  
  if (pct >= 85) {
    statusColor = SENSOR_COLORS?.error || "#ef4444";
    statusText = "Error";
  } else if (pct >= 50) {
    statusColor = SENSOR_COLORS?.warning || "#f97316";
    statusText = "Warning";
  }

  const isWorking = appState === 'routing' || appState === 'tool_execution';

  // --- REUSABLE RENDERER FOR THE STATUS BLOCK ---
  const renderStatusBlock = () => (
    <div className="flex justify-start w-full">
      <div 
        onClick={() => setIsStatusLogExpanded(!isStatusLogExpanded)}
        className="max-w-[90%] p-3 rounded-2xl rounded-tl-sm cursor-pointer transition-all duration-300 flex flex-col shadow-sm border border-transparent"
        style={{ color: SENSOR_COLORS.unavailable, backgroundColor: 'transparent' }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = ROOM_COLORS.unavailable;
          e.currentTarget.style.borderColor = `${SENSOR_COLORS.unavailable}40`;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent';
          e.currentTarget.style.borderColor = 'transparent';
        }}
      >
        <div className="flex items-center gap-3">
          {isWorking ? (
            <div 
              className="w-4 h-4 border-2 rounded-full animate-spin flex-shrink-0" 
              style={{ borderColor: SENSOR_COLORS.unavailable, borderTopColor: 'transparent' }}
            ></div>
          ) : (
            <CheckCircle2 size={16} style={{ color: SENSOR_COLORS.unavailable }} className="flex-shrink-0" />
          )}
          
          <span className={`text-sm font-medium ${isWorking ? 'animate-pulse' : 'opacity-80'}`}>
            {isWorking ? (llmStatus?.message || statusHistory[statusHistory.length - 1]) : "Finished running diagnostics"}
          </span>
          <ChevronDown 
            size={16} 
            className={`transition-transform duration-300 ml-2 ${isStatusLogExpanded ? 'rotate-180' : ''}`} 
          />
        </div>

        {/* Dropdown for Status History. Using standard flow to prevent overlap */}
        <div className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${isStatusLogExpanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
          <div className="overflow-hidden">
            <div className="pt-3 pb-1 text-xs font-mono space-y-2 opacity-80 pl-7">
              {statusHistory.map((status, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: SENSOR_COLORS.unavailable }}></span>
                  <span className="leading-tight">{status}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-full bg-[#0A0A0A] border border-[#A3B8B2]/10 rounded-3xl overflow-hidden shadow-2xl relative">
      
      {/* --- TOP: CONTEXT PILL SECTION --- */}
      <div className="px-4 pt-4 z-20 relative flex-shrink-0">
        <div 
          onClick={() => setIsStatusExpanded(!isStatusExpanded)}
          className="w-full rounded-2xl overflow-hidden transition-all duration-300 ease-in-out cursor-pointer hover:brightness-110 group shadow-md"
          style={{ backgroundColor: `${statusColor}1A` }}
        >
          <div className="p-4 flex flex-col w-full">
            <div className="flex justify-between w-full items-center mb-3">
              <span className="font-semibold text-sm flex items-center gap-2" style={{ color: statusColor }}>
                <Cpu size={18} style={{ color: statusColor }} /> {LLM_MODEL} Context ({statusText})
              </span>
              
              <div className="flex items-center gap-4">
                 <span className="text-xs font-mono" style={{ color: statusColor }}>
                   {tokens.toLocaleString("en-US")} / {MAX_TOKENS.toLocaleString("en-US")} ctx
                 </span>
                 
                 <div 
                   onClick={(e) => {
                     e.stopPropagation();
                     handleReset();
                   }}
                   className="p-1.5 rounded-md hover:brightness-125 transition-all flex items-center justify-center shadow-md cursor-pointer"
                   style={{ backgroundColor: SENSOR_COLORS.error, color: ROOM_COLORS.error }}
                   title="Reset Session Context"
                 >
                   <Trash2 size={16} />
                 </div>
              </div>
            </div>
            
            <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden">
              <div className="h-full transition-all duration-500 ease-out" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: statusColor }} />
            </div>
          </div>
          
          <div className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${isStatusExpanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
            <div className="overflow-hidden">
              <div className="px-4 pb-4 text-xs font-mono space-y-3 max-h-64 overflow-y-auto chat-scrollbar">
                 {(!sessionTools || sessionTools.length === 0) ? (
                     <p className="opacity-50 text-center py-2" style={{ color: statusColor }}>No tools used in this session yet.</p>
                 ) : (
                     sessionTools.reduce((acc, curr) => {
                       const last = acc[acc.length - 1];
                       if (last && last.tool === curr.tool) {
                         if (!last.rooms.includes(curr.room)) last.rooms.push(curr.room);
                       } else {
                         acc.push({ tool: curr.tool, rooms: [curr.room] });
                       }
                       return acc;
                     }, [] as { tool: string; rooms: string[] }[]).map((st, idx) => (
                         <div key={idx} className="flex flex-col gap-1 pb-3 border-b border-black/20 last:border-0 last:pb-0">
                             <span className="font-bold" style={{ color: statusColor }}>➜ {st.tool}</span>
                             <span className="pl-5 opacity-80 border-l border-black/20 ml-1" style={{ color: statusColor }}>Target: {st.rooms.join(", ")}</span>
                         </div>
                     ))
                 )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- MIDDLE: CHAT MESSAGES WITH FADES --- */}
      <div className="flex-1 relative flex flex-col overflow-hidden bg-transparent">
        <div className="absolute top-0 left-0 right-0 h-10 bg-gradient-to-b from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>

        {/* Using gap-4 here ensures elements push each other down naturally and never overlap */}
        <div className="flex-1 overflow-y-auto px-4 py-6 z-0 chat-scrollbar flex flex-col gap-4">
          
          {messages.map((msg, idx) => {
            const isLastAgentMessage = msg.sender === 'agent' && idx === messages.length - 1;

            return (
              <React.Fragment key={idx}>
                {/* 1. Inject Status Block right BEFORE the final agent reply */}
                {isLastAgentMessage && statusHistory.length > 0 && renderStatusBlock()}
                
                {/* 2. Render the actual Message Bubble */}
                <div className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[90%] p-4 rounded-2xl shadow-sm ${
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
              </React.Fragment>
            )
          })}

          {/* 3. Inject Status Block if we are STILL WAITING for the agent to reply */}
          {statusHistory.length > 0 && messages[messages.length - 1]?.sender === 'user' && renderStatusBlock()}

          <div ref={messagesEndRef} className="h-4 w-full flex-shrink-0" />
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-10 bg-gradient-to-t from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>
      </div>

      {/* --- BOTTOM: INPUT FORM PILL --- */}
      <form onSubmit={handleSubmit} className="px-4 pb-4 z-20 relative flex-shrink-0">
        <div 
          className="relative flex items-center rounded-2xl overflow-hidden shadow-md transition-all duration-300 group"
          style={{ backgroundColor: `${statusColor}1A` }} 
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask the campus assistant..."
            className="w-full bg-transparent border-none py-4 px-4 pr-12 text-sm text-[#14C89B] placeholder-[#14C89B]/60 focus:outline-none focus:ring-0 transition-colors"
          />
          
          <button 
            type="submit"
            suppressHydrationWarning
            disabled={input.trim().length === 0 || isWorking}
            className="absolute right-2 p-2 text-[#14C89B] disabled:opacity-50 hover:bg-[#14C89B]/20 rounded-xl transition-all duration-300"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </form>
      
    </div>
  );
}