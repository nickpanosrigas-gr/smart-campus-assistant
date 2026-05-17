"use client";
import { useState } from "react";
import { motion, useAnimation } from "framer-motion";
import { Send, Mic, Wrench } from "lucide-react";

interface BottomSheetChatProps {
  onSendMessage: (msg: string) => void;
  activeTools: string[]; // E.g., ['Air Quality', 'Occupancy']
  isThinking: boolean;
}

export default function BottomSheetChat({ onSendMessage, activeTools, isThinking }: BottomSheetChatProps) {
  const [input, setInput] = useState("");
  const controls = useAnimation();

  // Snap points for the bottom sheet (percentages of screen height)
  const snapPoints = {
    min: "15%",
    mid: "40%",
    full: "95%",
  };

  const handleDragEnd = (event: any, info: any) => {
    // Basic logic to snap to nearest point based on drag velocity/offset
    if (info.offset.y < -100) {
      controls.start({ height: snapPoints.mid });
    } else if (info.offset.y > 100) {
      controls.start({ height: snapPoints.min });
    }
  };

  return (
    <motion.div
      drag="y"
      dragConstraints={{ top: 0, bottom: 0 }}
      dragElastic={0.2}
      onDragEnd={handleDragEnd}
      animate={controls}
      initial={{ height: snapPoints.min }}
      className="absolute bottom-0 left-0 right-0 bg-[#0A0A0A]/80 backdrop-blur-xl border-t border-[#14C89B]/20 rounded-t-3xl flex flex-col z-50 shadow-[0_-10px_40px_rgba(20,200,155,0.1)]"
    >
      {/* Drag Handle */}
      <div className="w-full flex justify-center py-3 cursor-grab active:cursor-grabbing">
        <div className="w-12 h-1.5 bg-[#A3B8B2]/30 rounded-full" />
      </div>

      {/* Tool Tracker (Appears when thinking) */}
      {isThinking && activeTools.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }} 
          animate={{ opacity: 1, y: 0 }}
          className="px-6 pb-2 flex items-center gap-2 text-xs font-medium text-[#14C89B]"
        >
          <Wrench size={14} className="animate-pulse" />
          <span>Using Tool: {activeTools[activeTools.length - 1]}...</span>
        </motion.div>
      )}

      {/* Main Input Area */}
      <div className="px-4 pb-6 flex items-center gap-3">
        <button className="p-3 rounded-full bg-[#14C89B]/10 text-[#14C89B] hover:bg-[#14C89B]/20 transition-colors">
          <Mic size={20} />
        </button>
        
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the campus..."
          className="flex-1 bg-transparent border border-[#A3B8B2]/20 rounded-full px-5 py-3 text-[#A3B8B2] placeholder-[#A3B8B2]/50 focus:outline-none focus:border-[#14C89B] transition-colors"
          onKeyDown={(e) => {
            if (e.key === "Enter" && input) {
              onSendMessage(input);
              setInput("");
            }
          }}
        />

        <button 
          onClick={() => { if(input) { onSendMessage(input); setInput(""); } }}
          className="p-3 rounded-full bg-[#14C89B] text-black hover:bg-[#14C89B]/80 transition-colors shadow-[0_0_15px_rgba(20,200,155,0.4)]"
        >
          <Send size={20} className="ml-1" />
        </button>
      </div>

      {/* Chat History Area (Expands when dragged up) */}
      <div className="flex-1 overflow-y-auto px-6 opacity-50">
         {/* We will map the chat messages here when the sheet is expanded */}
      </div>
    </motion.div>
  );
}