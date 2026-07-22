import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Cpu, Trash2, ChevronDown, CheckCircle2, Mic, Square, X, Send } from 'lucide-react';
import { SENSOR_COLORS, ROOM_COLORS } from '@/components/map/constants';
import remarkGfm from 'remark-gfm';

interface ChatPanelProps {
  appState: "idle" | "routing" | "tool_execution" | "resolved";
  llmStatus?: { state: string; message: string; tool_name?: string } | null;
  onSendMessage: (msg: string) => void;
  onSendAudio: (audioBase64: string, sendToLLM: boolean, currentInput: string) => void;
  activeTools: string[];
  messages: Array<{ sender: "user" | "agent"; text: string }>;
  contextData?: { tokens: number };
  sessionTools?: { tool: string; room: string }[];
  onResetSession: () => void;
  transcribedText?: string | null;
  onClearTranscribedText?: () => void;
}

export default function ChatPanel({ 
  appState, 
  llmStatus, 
  onSendMessage,
  onSendAudio, 
  messages, 
  contextData, 
  sessionTools, 
  onResetSession,
  transcribedText,
  onClearTranscribedText
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [isStatusExpanded, setIsStatusExpanded] = useState(false); 
  const [statusHistory, setStatusHistory] = useState<string[]>([]);
  const [isStatusLogExpanded, setIsStatusLogExpanded] = useState(false);
  
  // Audio Recording States
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

  const LLM_MODEL = process.env.OLLAMA_MODEL || "Qwen3.5 4B";
  const MAX_TOKENS = Number(process.env.OLLAMA_NUM_CTX) || 8192;

  // Listen for newly transcribed text and append it to the input field
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
  }, [messages, appState, isStatusExpanded, isStatusLogExpanded, statusHistory]);

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

      // Set up Audio Context for Visualizer
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const actx = new AudioCtx();
      audioContextRef.current = actx;
      const src = actx.createMediaStreamSource(stream);
      const analyser = actx.createAnalyser();
      
      analyser.fftSize = 512; 
      // 1. BUTTERY SMOOTHING: Increase from the default 0.8 to 0.9 to remove jitter
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

        // 2. MIRROR MATH: 96 unique data points, drawn twice = 192 lines total
        const uniqueLines = 96;
        const totalLines = uniqueLines * 2;
        const barWidth = WIDTH / totalLines;
        let x = 0;

        for (let i = 0; i < totalLines; i++) {
          // Left half reads backwards (95 down to 0), Right half reads forwards (0 up to 95)
          // This puts index 0 (the most reactive voice frequencies) dead in the center!
          const dataIndex = i < uniqueLines ? (uniqueLines - 1 - i) : (i - uniqueLines);
          
          const normalized = dataArray[dataIndex] / 255;
          const barHeight = Math.min(normalized * HEIGHT * 1.5, HEIGHT); 
          
          ctx.fillStyle = SENSOR_COLORS.good || '#14C89B';
          
          // Minimum height of 2px so there is a solid, visible line when quiet
          const finalHeight = Math.max(barHeight, 2);
          
          // 3. VERTICAL CENTERING: Calculate Y so the bar expands equally up and down
          const y = (HEIGHT - finalHeight) / 2;
          
          ctx.fillRect(x, y, barWidth - 0.5, finalHeight);
          
          x += barWidth;
        }
        
        animationRef.current = requestAnimationFrame(drawVisualizer);
      };

      recorder.start();
      setIsRecording(true);
      
      // Delay the visualizer slightly so React has time to mount the <canvas> element
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

      {/* --- MIDDLE: CHAT MESSAGES --- */}
      <div className="flex-1 relative flex flex-col overflow-hidden bg-transparent">
        <div className="absolute top-0 left-0 right-0 h-10 bg-gradient-to-b from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>

        <div className="flex-1 overflow-y-auto px-4 py-6 z-0 chat-scrollbar flex flex-col gap-4">
          {messages.map((msg, idx) => {
            const isLastMessage = idx === messages.length - 1;
            const isLastAgentMessage = isLastMessage && msg.sender === 'agent';
            const isTranscribing = llmStatus?.state === 'transcribing';
            
            // Render BEFORE the last agent message, UNLESS we are currently waiting on an audio transcription 
            // (which means the last agent message is an old one from the previous turn)
            const showBeforeThisAgentMsg = isLastAgentMessage && statusHistory.length > 0 && !isTranscribing;

            return (
              <React.Fragment key={idx}>
                {showBeforeThisAgentMsg && renderStatusBlock()}
                
                <div className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[90%] p-4 rounded-2xl shadow-sm ${
                    msg.sender === 'user' 
                      ? 'bg-[#14C89B] text-black rounded-tr-sm' 
                      : 'bg-[#A3B8B2]/10 text-[#A3B8B2] rounded-tl-sm'
                  }`}>
                    {msg.sender === 'agent' ? (
                      <div className="text-sm space-y-2 prose prose-invert prose-p:leading-relaxed prose-strong:text-white prose-td:border-gray-700 prose-th:border-gray-700">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // Intercept table headers and force them to center
                            th: ({node, ...props}) => <th {...props} style={{ textAlign: 'center' }} />,
                            // Intercept table data cells and force them to center
                            td: ({node, ...props}) => <td {...props} style={{ textAlign: 'center' }} />
                          }}
                        >
                          {msg.text}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm font-medium">{msg.text}</p>
                    )}
                  </div>
                </div>

                {/* Render AFTER the user message while we wait for the LLM to start typing */}
                {isLastMessage && msg.sender === 'user' && statusHistory.length > 0 && renderStatusBlock()}
              </React.Fragment>
            )
          })}

          {/* Absolute Fallback: Render at the very bottom if the chat is completely empty, 
              OR if we are currently recording/transcribing and the user text hasn't appeared yet */}
          {((llmStatus?.state === 'transcribing' || messages.length === 0) && statusHistory.length > 0) && renderStatusBlock()}

          <div ref={messagesEndRef} className="h-4 w-full flex-shrink-0" />
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-10 bg-gradient-to-t from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>
      </div>

      {/* --- BOTTOM: INPUT FORM PILL --- */}
      <div className="px-4 pb-4 z-20 relative flex-shrink-0">
        <div 
          className="relative flex items-center rounded-2xl overflow-hidden shadow-md transition-all duration-300 group min-h-[56px]"
          style={{ backgroundColor: `${statusColor}1A` }} 
        >
          {!isRecording ? (
            <form onSubmit={handleSubmit} className="flex w-full items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask the campus assistant..."
                className="w-full bg-transparent border-none py-4 px-4 pr-24 text-sm text-[#14C89B] placeholder-[#14C89B]/60 focus:outline-none focus:ring-0 transition-colors"
                disabled={isWorking}
              />
              <div className="absolute right-2 flex items-center gap-1">
                <button 
                  type="button" 
                  onClick={startRecording}
                  disabled={isWorking}
                  className="p-2 text-[#14C89B] disabled:opacity-50 hover:bg-[#14C89B]/20 rounded-xl transition-all duration-300"
                  title="Record Voice"
                >
                  <Mic size={20} />
                </button>
                <button 
                  type="submit"
                  disabled={input.trim().length === 0 || isWorking}
                  className="p-2 text-[#14C89B] disabled:opacity-50 hover:bg-[#14C89B]/20 rounded-xl transition-all duration-300"
                  title="Send Message"
                >
                  <Send size={20} className="ml-0.5" />
                </button>
              </div>
            </form>
          ) : (
            <div className="flex w-full items-center px-2 py-2 gap-2 h-[56px]">
              <button 
                type="button" 
                onClick={cancelRecording}
                className="p-2 text-red-400 hover:bg-red-400/20 rounded-xl transition-all"
                title="Cancel Recording"
              >
                <X size={20} />
              </button>
              
              <div className="flex-1 flex items-center justify-center overflow-hidden h-full">
                <canvas ref={canvasRef} className="w-full h-8" width={300} height={32} />
              </div>
              
              <button 
                type="button" 
                onClick={() => stopRecording(false)}
                className="p-2 text-[#14C89B] hover:bg-[#14C89B]/20 rounded-xl transition-all"
                title="Stop & Transcribe Only"
              >
                <Square size={20} className="fill-current" />
              </button>
              <button 
                type="button" 
                onClick={() => stopRecording(true)}
                className="p-2 text-[#14C89B] hover:bg-[#14C89B]/20 rounded-xl transition-all"
                title="Transcribe & Send to LLM"
              >
                <Send size={20} className="ml-0.5" />
              </button>
            </div>
          )}
        </div>
      </div>
      
    </div>
  );
}