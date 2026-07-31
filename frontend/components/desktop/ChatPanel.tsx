// frontend/components/desktop/ChatPanel.tsx
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Trash2, ChevronDown, CheckCircle2, Mic, Square, X, ArrowUp } from 'lucide-react';
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
  onResetSession,
  transcribedText,
  onClearTranscribedText
}: ChatPanelProps) {
  const [input, setInput] = useState("");
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

  const MAX_TOKENS = Number(process.env.OLLAMA_NUM_CTX) || 8192;

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
  const isSendDisabled = input.trim().length === 0 || isWorking;

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
              
              {/* Header Logo & Name */}
              <div className="flex items-center gap-3">
                <img src="/icon.png" alt="HUAssistant Logo" className="w-8 h-8 rounded-xl shrink-0 object-contain" />
                <span className="font-bold text-[#14C89B] text-base tracking-wide">
                  HUAssistant
                </span>
                <span 
                  className="text-xs font-bold px-2.5 py-1 rounded-full bg-black/20 shadow-inner" 
                  style={{ color: statusColor }}
                >
                  {statusText}
                </span>
              </div>
              
              {/* Token Counter & Reset Session Squircle */}
              <div className="flex items-center gap-3">
                 <span className="text-xs font-mono text-[#14C89B] bg-black/20 px-3 py-2 rounded-full shadow-inner">
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
            
            {/* Progress Bar */}
            <div className="w-full h-2 bg-black/30 rounded-full overflow-hidden p-0.5 shadow-inner">
              <div className="h-full rounded-full transition-all duration-500 ease-out" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: statusColor }} />
            </div>
          </div>
        </div>
      </div>

      {/* --- MIDDLE: CHAT MESSAGES --- */}
      <div className="flex-1 relative flex flex-col overflow-hidden bg-transparent">
        <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-[#0A0A0A] to-transparent z-10 pointer-events-none"></div>

        <div className="flex-1 overflow-y-auto px-5 py-6 z-0 chat-scrollbar flex flex-col gap-4">
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
                placeholder="Ask HUAssistant..."
                className="w-full bg-transparent border-none py-3 px-4 pr-28 text-sm text-[#0A0A0A] placeholder-[#0A0A0A] focus:outline-none focus:ring-0 transition-colors font-normal h-full"
                disabled={isWorking}
              />
              <div className="absolute right-1.5 flex items-center gap-1.5">
                <button 
                  type="button" 
                  onClick={startRecording}
                  disabled={isWorking}
                  className="w-10 h-10 rounded-2xl bg-black/20 text-[#14C89B] disabled:opacity-40 hover:bg-[#14C89B] hover:text-[#0A0A0A] flex items-center justify-center transition-all duration-300 shadow-sm shrink-0"
                  title="Record Voice"
                >
                  <Mic size={18} />
                </button>
                
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