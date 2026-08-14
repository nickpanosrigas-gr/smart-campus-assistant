// frontend/components/desktop/ChatPanel.tsx
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Trash2, ChevronDown, CheckCircle2, Mic, Square, X, ArrowUp, AlertCircle, AlertTriangle } from 'lucide-react';
import { SENSOR_COLORS } from '@/components/map/constants';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';

const API_BASE_URL = process.env.NODE_ENV === "production" ? "" : "http://localhost:8000";
const ALL_TOGGLES = ["Occupancy", "Climate", "Air Quality", "Doors/Windows", "Lights", "Diagnostics", "Schedule"];

interface ChatPanelProps {
  appState: "idle" | "routing" | "tool_execution" | "resolved";
  llmStatus?: { state: string; message: string; tool_name?: string } | null;
  onSendMessage: (msg: string) => void;
  onSendAudio: (audioBase64: string, sendToLLM: boolean, currentInput: string) => void;
  onStopResponse: () => void;
  activeTools: string[];
  messages: Array<{ sender: "user" | "agent"; text: string }>;
  contextData?: { tokens: number };
  sessionTools?: { tool: string; room: string }[];
  onResetSession: () => void;
  transcribedText?: string | null;
  onClearTranscribedText?: () => void;
  ollamaOnline: boolean;
  whisperOnline: boolean;
  userName?: string;
  activeLevel?: string;
  selectedRooms?: string[];
  timeframe?: string;
}

export default function ChatPanel({ 
  appState, 
  llmStatus, 
  onSendMessage,
  onSendAudio, 
  onStopResponse,
  messages, 
  contextData, 
  onResetSession,
  transcribedText,
  onClearTranscribedText,
  ollamaOnline,
  whisperOnline,
  userName,
  activeLevel,
  selectedRooms,
  timeframe
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [statusHistory, setStatusHistory] = useState<string[]>([]);
  const [isStatusLogExpanded, setIsStatusLogExpanded] = useState(false);
  
  const [selectedPromptTool, setSelectedPromptTool] = useState<string>("Air Quality");
  // Updated interface to separate greeting_time and name, and include templates
  const [welcomeData, setWelcomeData] = useState<{ 
    greeting_time: string, 
    name: string, 
    welcome_message: string, 
    questions: string[],
    templates: string[]
  } | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chunksRef = useRef<Blob[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const MAX_TOKENS = Number(process.env.OLLAMA_NUM_CTX) || 8192;

  // --- NEW: Floor-level tool disablement logic ---
  const isToolDisabled = (tool: string) => {
    const level = activeLevel || "B";
    if (tool === "Schedule" && ["B", "0", "-1", "-2", "-3"].includes(level)) return true;
    if (tool === "Doors/Windows" && ["0", "-2", "-3"].includes(level)) return true;
    return false;
  };

  // 1. Initial tool selection on mount (ensuring we pick an enabled tool)
  useEffect(() => {
    const available = ALL_TOGGLES.filter(t => !isToolDisabled(t));
    if (available.length > 0) {
      setSelectedPromptTool(available[Math.floor(Math.random() * available.length)]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 2. Watch for floor changes to auto-kick disabled tools
  useEffect(() => {
    if (isToolDisabled(selectedPromptTool)) {
      const available = ALL_TOGGLES.filter(t => !isToolDisabled(t));
      if (available.length > 0) {
        setSelectedPromptTool(available[Math.floor(Math.random() * available.length)]);
      }
    }
  }, [activeLevel, selectedPromptTool]);

  useEffect(() => {
    setSelectedPromptTool(ALL_TOGGLES[Math.floor(Math.random() * ALL_TOGGLES.length)]);
  }, []);

  useEffect(() => {
    const fetchPrompts = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/welcome`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include", 
          body: JSON.stringify({ 
            tool: selectedPromptTool,
            floor: activeLevel || "B",
            rooms: selectedRooms || [],
            timeframe: timeframe || "now",
            // Pass previous state to preserve randomness on room clicks
            prev_msg: welcomeData?.welcome_message,
            prev_templates: welcomeData?.templates
          })
        });
        if (res.ok) {
          const data = await res.json();
          setWelcomeData(data);
        }
      } catch (error) {
        console.error("Failed to fetch welcome prompts", error);
      }
    };
    
    if (messages.length === 0) {
      fetchPrompts();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPromptTool, activeLevel, selectedRooms, timeframe, messages.length]);

  useEffect(() => {
    if (transcribedText) {
      setInput(prev => prev + (prev ? " " : "") + transcribedText);
      onClearTranscribedText?.();
    }
  }, [transcribedText, onClearTranscribedText]);

  useEffect(() => {
    if (llmStatus && llmStatus.message) {
      setStatusHistory(prev => {
        if (prev[prev.length - 1] === llmStatus.message) return prev;
        return [...prev, llmStatus.message];
      });
    }
  }, [llmStatus]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
    return () => clearTimeout(timeout);
  }, [messages, appState, isStatusLogExpanded, statusHistory]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isRecording) return;
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

  // --- AUDIO RECORDING & VISUALIZER LOGIC ---
  const cleanupAudio = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    if (audioContextRef.current?.state !== 'closed') audioContextRef.current?.close();
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    audioContextRef.current = null;
    streamRef.current = null;
    analyserRef.current = null;
    dataArrayRef.current = null;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      let mimeType = 'audio/webm';
      if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'audio/mp4'; 
      
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const actx = new AudioCtx();
      audioContextRef.current = actx;
      const src = actx.createMediaStreamSource(stream);
      const analyser = actx.createAnalyser();
      
      analyser.fftSize = 512; 
      analyser.smoothingTimeConstant = 0.9; 
      
      src.connect(analyser);
      analyserRef.current = analyser;
      
      const bufferLength = analyser.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);

      const drawVisualizer = () => {
        const canvas = canvasRef.current;
        const analyser = analyserRef.current;
        const dataArray = dataArrayRef.current;

        if (!canvas || !analyser || !dataArray) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;

        analyser.getByteFrequencyData(dataArray as any);
        ctx.clearRect(0, 0, WIDTH, HEIGHT);

        const uniqueLines = 96;
        const totalLines = uniqueLines * 2;
        const barWidth = WIDTH / totalLines;
        let x = 0;

        for (let i = 0; i < totalLines; i++) {
          const dataIndex = i < uniqueLines ? (uniqueLines - 1 - i) : (i - uniqueLines);
          const normalized = dataArray[dataIndex] / 255;
          const barHeight = Math.min(normalized * HEIGHT * 1.5, HEIGHT); 
          
          ctx.fillStyle = SENSOR_COLORS.good || '#14C89B';
          const finalHeight = Math.max(barHeight, 2);
          const y = (HEIGHT - finalHeight) / 2;
          
          ctx.fillRect(x, y, barWidth - 0.5, finalHeight);
          x += barWidth;
        }
        
        animationRef.current = requestAnimationFrame(drawVisualizer);
      };

      recorder.start();
      setIsRecording(true);
      
      setTimeout(() => {
        drawVisualizer();
      }, 50);

    } catch (err) {
      console.error("Failed to start recording", err);
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    cleanupAudio();
    setIsRecording(false);
  };

  const stopRecording = (sendToLLM: boolean) => {
    if (sendToLLM) {
      setStatusHistory([]);
      setIsStatusLogExpanded(false);
    }
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mediaRecorderRef.current?.mimeType });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = (reader.result as string).split(',')[1];
          onSendAudio(base64data, sendToLLM, input);
          if (sendToLLM) setInput(""); 
        };
        reader.readAsDataURL(blob);
        cleanupAudio();
      };
      mediaRecorderRef.current.stop();
    } else {
      cleanupAudio();
    }
    setIsRecording(false);
  };

  // ------------------------------------------

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
  const isSendDisabled = input.trim().length === 0 || isWorking || !ollamaOnline;

  const renderWelcomeScreen = () => {
    if (!welcomeData) return <div className="flex-1" />;

    // --- NEW: Split and sort tools for the bottom pills ---
    const unselectedTools = ALL_TOGGLES.filter(t => t !== selectedPromptTool);
    const enabledUnavailable = unselectedTools.filter(t => !isToolDisabled(t));
    const disabledUnavailable = unselectedTools.filter(t => isToolDisabled(t));
    const sortedUnavailable = [...enabledUnavailable, ...disabledUnavailable];

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0, y: -20, filter: "blur(5px)" }}
        className="flex flex-col h-full px-2 w-full"
      >
        
        {/* --- TOP SECTION: Typography matched identically --- */}
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-1.5">
          <h2 className="text-[clamp(1.25rem,1.5vw,1.5rem)] font-normal text-[#14C89B] tracking-wide">
            {welcomeData.greeting_time}, <span className="font-bold">{welcomeData.name}</span>!
          </h2>
          <p className="text-[clamp(1.25rem,1.5vw,1.5rem)] font-normal text-[#14C89B]">
            {welcomeData.welcome_message}
          </p>
        </div>

        {/* --- BOTTOM SECTION: Questions & Tools --- */}
        <div className="flex flex-col w-full pb-0">
          <LayoutGroup>
            <div className="relative w-full flex flex-col mb-6">
              
              <AnimatePresence mode="wait">
                <motion.div
                  key={`cloud-${selectedPromptTool}`}
                  initial={{ clipPath: "inset(100% 0 0 0)" }}
                  animate={{ clipPath: "inset(0% 0 0 0)" }}
                  exit={{ opacity: 0, transition: { duration: 0.15 } }}
                  transition={{ duration: 0.4, delay: 0.3, ease: "easeOut" }}
                  className="w-full flex flex-col"
                >
                  <div className="w-full bg-[#0A664F] rounded-t-3xl pt-8 px-5 pb-2 relative z-10">
                    <div className="flex flex-col w-full gap-2">
                      {welcomeData.questions.map((q, idx) => {
                        const staggerDelay = 0.35 + ((welcomeData.questions.length - 1 - idx) * 0.1);
                        return (
                          <motion.button
                            key={`${selectedPromptTool}-q-${idx}`}
                            initial={{ opacity: 0, y: 15 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, delay: staggerDelay, ease: "easeOut" }}
                            onClick={() => {
                              setStatusHistory([]);
                              setIsStatusLogExpanded(false);
                              onSendMessage(q);
                            }}
                            className="w-full text-left p-3.5 rounded-2xl bg-black/20 text-[#14C89B] font-semibold text-sm hover:bg-[#14C89B] hover:text-[#0A0A0A] transition-colors shadow-sm"
                          >
                            {q}
                          </motion.button>
                        );
                      })}
                    </div>
                  </div>
                  
                  <div className="w-full flex items-start z-10 relative -mt-[1px]">
                    <div className="flex-1 h-7 bg-[#0A664F] rounded-bl-3xl -mr-[1px]"></div>
                    <div className="bg-[#0A664F] flex flex-col items-center justify-center px-[clamp(1.5rem,2.5vw,2rem)] pt-2 pb-5 rounded-b-3xl relative z-10">
                      <div className="opacity-0 text-[clamp(0.75rem,0.85vw,0.875rem)] font-bold whitespace-nowrap">
                        {selectedPromptTool}
                      </div>
                    </div>
                    <div className="flex-1 h-7 bg-[#0A664F] rounded-br-3xl -ml-[1px]"></div>
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Absolute Flying Text Overlay */}
              <div className="absolute inset-0 flex flex-col items-center justify-end pb-5 pointer-events-none z-20">
                <motion.div
                  key={`selected-${selectedPromptTool}`}
                  layoutId={`tool-${selectedPromptTool}`}
                  initial={{ color: "#A3B8B2" }}
                  animate={{ color: "#0A0A0A" }}
                  transition={{ 
                    layout: { type: "spring", stiffness: 280, damping: 25 },
                    color: { delay: 0.3, duration: 0.15, ease: "easeIn" }
                  }}
                  className="text-[clamp(0.75rem,0.85vw,0.875rem)] font-bold whitespace-nowrap"
                >
                  {selectedPromptTool}
                </motion.div>
              </div>

            </div>

            {/* Grid for Unselected Tools */}
            <div className="flex justify-center w-full z-0">
              <div className="flex flex-wrap justify-center gap-2 bg-[#1A1A1A]/80 border border-[#333333] rounded-3xl p-2.5 shadow-inner w-full">
                {sortedUnavailable.map(toggle => {
                  const isDisabled = isToolDisabled(toggle);
                  return (
                    <motion.button
                      layoutId={`tool-${toggle}`}
                      key={toggle}
                      onClick={() => { if (!isDisabled) setSelectedPromptTool(toggle); }}
                      disabled={isDisabled}
                      className={`px-[clamp(0.6rem,1vw,1.25rem)] py-[clamp(0.35rem,0.6vh,0.625rem)] rounded-full text-[clamp(0.75rem,0.85vw,0.875rem)] font-medium whitespace-nowrap bg-transparent transition-colors ${
                        isDisabled
                          ? "text-[#A3B8B2]/20 cursor-not-allowed"
                          : "text-[#A3B8B2]/50 hover:text-[#A3B8B2] hover:bg-[#2A2A2A]"
                      }`}
                    >
                      {toggle}
                    </motion.button>
                  );
                })}
              </div>
            </div>

          </LayoutGroup>
        </div>
      </motion.div>
    );
  };

  const renderStatusBlock = () => (
    <div className="flex justify-start w-full">
      <div 
        onClick={() => setIsStatusLogExpanded(!isStatusLogExpanded)}
        className="max-w-[90%] p-3.5 rounded-2xl rounded-tl-sm cursor-pointer transition-all duration-300 flex flex-col shadow-sm border border-white/5 bg-white/5 hover:bg-white/10 text-[#14C89B] hover:text-[#14C89B]"
      >
        <div className="flex items-center gap-3">
          {isWorking ? (
            <div className="w-4 h-4 border-2 border-[#14C89B] border-t-transparent rounded-full animate-spin flex-shrink-0" />
          ) : (
            <CheckCircle2 size={16} className="text-[#14C89B] flex-shrink-0" />
          )}
          
          <span className={`text-sm font-medium ${isWorking ? 'animate-pulse text-[#14C89B]' : 'text-[#14C89B]'}`}>
            {isWorking ? (llmStatus?.message || statusHistory[statusHistory.length - 1]) : "Finished running diagnostics"}
          </span>
          <ChevronDown 
            size={16} 
            className={`transition-transform duration-300 ml-2 ${isStatusLogExpanded ? 'rotate-180' : ''}`} 
          />
        </div>

        <div className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${isStatusLogExpanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
          <div className="overflow-hidden">
            <div className="pt-3 pb-1 text-xs font-mono space-y-2 text-[#14C89B] pl-7">
              {statusHistory.map((status, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#14C89B] flex-shrink-0"></span>
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
    <div className="flex flex-col h-full bg-[#0A0A0A] border-2 border-[#0A664F] border-b-0 rounded-t-3xl rounded-b-none overflow-hidden shadow-[0_-10px_30px_rgba(0,0,0,0.5)] relative">
      
      {/* --- TOP: STATIC CONTEXT PILL SECTION --- */}
      <div className="px-4 pt-4 z-20 relative flex-shrink-0">
        <div className="w-full bg-[#0A664F] border border-[#14C89B]/20 rounded-2xl overflow-hidden shadow-[0_10px_30px_rgba(0,0,0,0.3)]">
          <div className="p-4 flex flex-col w-full">
            <div className="flex justify-between w-full items-center mb-3.5">
              
              <div className="flex items-center gap-3">
                <img src="/icon.png" alt="HUAssistant Logo" className="w-8 h-8 rounded-xl shrink-0 object-contain" />
                <span className="font-bold text-[#14C89B] text-base tracking-wide">
                  HUAssistant
                </span>
              </div>
              
              <div className="flex items-center gap-3">
                 <span 
                   className="text-xs font-mono font-bold bg-black/20 px-3 py-2 rounded-full shadow-inner"
                   style={{ color: statusColor }}
                 >
                   {pct.toFixed(1)}% ctx
                 </span>
                 
                 <button 
                   type="button"
                   onClick={(e) => {
                     e.stopPropagation();
                     handleReset();
                   }}
                   className="w-10 h-10 rounded-2xl bg-[#8E2F3E] text-white flex items-center justify-center shrink-0 transition-all hover:bg-[#C84B5E] hover:text-[#0A0A0A] shadow-md cursor-pointer"
                   title="Reset Session Context"
                 >
                   <Trash2 size={18} />
                 </button>
              </div>
            </div>
            
            <div className="w-full h-2 bg-black/30 rounded-full overflow-hidden p-0.5 shadow-inner">
              <div className="h-full rounded-full transition-all duration-500 ease-out" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: statusColor }} />
            </div>
          </div>
        </div>
      </div>

      {/* --- MIDDLE: CHAT MESSAGES / WELCOME SCREEN --- */}
      <div className="flex-1 relative flex flex-col overflow-hidden bg-transparent">
        <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>

        <div className="flex-1 overflow-y-auto px-5 py-6 z-0 chat-scrollbar flex flex-col gap-4">
          
          {!ollamaOnline && (
            <div className="mx-auto w-fit max-w-[90%] px-4 py-2.5 rounded-full bg-white/5 border border-white/5 shadow-md flex items-center gap-3 mb-2 shrink-0">
               <AlertCircle size={16} className="text-[#ef4444] shrink-0" />
               <span className="text-sm font-medium text-[#ef4444]">LLM is currently offline. Chat is disabled.</span>
            </div>
          )}
          {ollamaOnline && !whisperOnline && (
            <div className="mx-auto w-fit max-w-[90%] px-4 py-2.5 rounded-full bg-white/5 border border-white/5 shadow-md flex items-center gap-3 mb-2 shrink-0">
               <AlertTriangle size={16} className="text-[#f97316] shrink-0" />
               <span className="text-sm font-medium text-[#f97316]">Speech-to-text is currently offline.</span>
            </div>
          )}

          <AnimatePresence mode="wait">
            {messages.length === 0 ? (
              renderWelcomeScreen()
            ) : (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col gap-4 w-full h-full justify-start"
              >
                {messages.map((msg, idx) => {
                  const isLastMessage = idx === messages.length - 1;
                  const isLastAgentMessage = isLastMessage && msg.sender === 'agent';
                  const isTranscribing = llmStatus?.state === 'transcribing';
                  
                  const showBeforeThisAgentMsg = isLastAgentMessage && statusHistory.length > 0 && !isTranscribing;

                  return (
                    <React.Fragment key={idx}>
                      {showBeforeThisAgentMsg && renderStatusBlock()}
                      
                      <div className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[90%] p-4 rounded-3xl shadow-md transition-all duration-200 ${
                          msg.sender === 'user' 
                            ? 'bg-[#0A664F] text-[#0A0A0A] rounded-tr-sm font-bold' 
                            : 'bg-white/5 text-[#14C89B] border border-white/5 rounded-tl-sm font-normal'
                        }`}>
                          {msg.sender === 'agent' ? (
                            <div className="text-sm space-y-2 text-[#14C89B] prose prose-p:text-[#14C89B] prose-strong:text-[#14C89B] prose-li:text-[#14C89B] prose-headings:text-[#14C89B] prose-a:text-[#14C89B] prose-code:text-[#14C89B] prose-td:text-[#14C89B] prose-th:text-[#14C89B] prose-p:leading-relaxed prose-td:border-gray-700 prose-th:border-gray-700 max-w-none">
                              <ReactMarkdown 
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  th: ({node, ...props}) => <th {...props} style={{ textAlign: 'center' }} />,
                                  td: ({node, ...props}) => <td {...props} style={{ textAlign: 'center' }} />
                                }}
                              >
                                {msg.text}
                              </ReactMarkdown>
                            </div>
                          ) : (
                            <p className="text-sm font-bold leading-relaxed">{msg.text}</p>
                          )}
                        </div>
                      </div>

                      {isLastMessage && msg.sender === 'user' && statusHistory.length > 0 && renderStatusBlock()}
                    </React.Fragment>
                  )
                })}

                {((llmStatus?.state === 'transcribing' || messages.length === 0) && statusHistory.length > 0) && renderStatusBlock()}
                <div ref={messagesEndRef} className="h-4 w-full flex-shrink-0" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>
      </div>

      {/* --- BOTTOM: INPUT FORM PILL --- */}
      <div className="px-4 pb-4 z-20 relative flex-shrink-0">
        <div className="relative flex items-center rounded-3xl overflow-hidden shadow-[0_-10px_30px_rgba(0,0,0,0.3)] transition-all duration-300 bg-[#0A664F] border border-[#14C89B]/20 h-[64px] p-1.5">
          {!isRecording ? (
            <form onSubmit={handleSubmit} className="flex w-full items-center h-full">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={!ollamaOnline ? "Connection lost..." : "Ask HUAssistant..."}
                disabled={isWorking || !ollamaOnline}
                className={`w-full bg-transparent border-none py-3 px-4 pr-28 text-sm text-[#0A0A0A] placeholder-[#0A0A0A] focus:outline-none focus:ring-0 transition-colors font-normal h-full ${
                  !ollamaOnline ? "opacity-50 cursor-not-allowed" : ""
                }`}
              />
              <div className="absolute right-1.5 flex items-center gap-1.5">
                <button 
                  type="button" 
                  onClick={startRecording}
                  disabled={isWorking || !whisperOnline || !ollamaOnline}
                  className={`w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] flex items-center justify-center transition-all duration-300 shadow-sm shrink-0 ${
                    (isWorking || !whisperOnline || !ollamaOnline) ? "opacity-40 cursor-not-allowed" : "hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                  }`}
                  title={!whisperOnline ? "Speech-to-text offline" : "Record Voice"}
                >
                  <Mic size={18} />
                </button>
                
                {isWorking ? (
                  <button 
                    type="button"
                    onClick={onStopResponse}
                    className="w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] hover:bg-[#14C89B] hover:text-[#0A0A0A] flex items-center justify-center transition-all duration-300 shadow-sm shrink-0"
                    title="Stop Response"
                  >
                    <Square size={16} className="fill-current" />
                  </button>
                ) : (
                  <button 
                    type="submit"
                    disabled={isSendDisabled}
                    className={`w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] flex items-center justify-center transition-all duration-300 shadow-sm font-normal shrink-0 ${
                      isSendDisabled 
                        ? "opacity-40 cursor-not-allowed" 
                        : "hover:bg-[#14C89B] hover:text-[#0A0A0A]"
                    }`}
                    title="Send Message"
                  >
                    <ArrowUp size={20} />
                  </button>
                )}
              </div>
            </form>
          ) : (
            <div className="flex w-full items-center gap-1.5 h-full">
              <button 
                type="button" 
                onClick={cancelRecording}
                className="w-10 h-10 rounded-2xl bg-[#8E2F3E] text-white hover:bg-[#C84B5E] hover:text-[#0A0A0A] flex items-center justify-center transition-all shadow-md shrink-0"
                title="Cancel Recording"
              >
                <X size={18} />
              </button>
              
              <div className="flex-1 flex items-center justify-center overflow-hidden h-10 bg-black/20 rounded-2xl px-3 shadow-inner">
                <canvas ref={canvasRef} className="w-full h-7" width={300} height={28} />
              </div>
              
              <button 
                type="button" 
                onClick={() => stopRecording(false)}
                className="w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] hover:bg-[#14C89B] hover:text-[#0A0A0A] flex items-center justify-center transition-all shadow-sm shrink-0"
                title="Stop & Transcribe Only"
              >
                <Square size={16} className="fill-current" />
              </button>
              <button 
                type="button" 
                onClick={() => stopRecording(true)}
                className="w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] hover:bg-[#14C89B] hover:text-[#0A0A0A] flex items-center justify-center transition-all shadow-sm shrink-0"
                title="Transcribe & Send to LLM"
              >
                <ArrowUp size={20} />
              </button>
            </div>
          )}
        </div>
      </div>
      
    </div>
  );
}