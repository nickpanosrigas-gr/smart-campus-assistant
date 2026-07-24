"use client";
import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Activity } from "lucide-react";

export default function DiagnosticsGraph({ artifact }: { artifact: any }) {
  if (!artifact?.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/60 p-8 text-center bg-black/20 rounded-3xl border border-white/5">
        <Activity size={36} className="mb-3 text-[#14C89B]/40 animate-pulse" />
        <p className="text-sm font-semibold">No Diagnostics History Recorded</p>
      </div>
    );
  }

  const formattedData = artifact.series.map((pt: any) => {
    const date = new Date(pt.timestamp);
    const isLongTerm = ["30d", "90d"].includes(artifact.timeframe);
    return {
      ...pt,
      timeLabel: isLongTerm
        ? date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })
    };
  });

  const sensorKeys = Object.keys(artifact.series[0] || {}).filter(k => k !== "timestamp" && k !== "timeLabel" && k !== "total");

  return (
    <div className="w-full h-full flex flex-col bg-[#061C16]/60 border border-[#0A664F]/80 rounded-3xl p-6 shadow-2xl backdrop-blur-md overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#14C89B]/10 shrink-0">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-[#14C89B]" />
            Room {artifact.room_id?.toUpperCase()} Health & Battery Diagnostics
            <span className="text-xs font-mono uppercase bg-[#14C89B]/20 text-[#14C89B] px-2.5 py-0.5 rounded-full border border-[#14C89B]/30">
              {artifact.timeframe}
            </span>
          </h3>
          <p className="text-xs text-[#A3B8B2]/70 mt-0.5">Sensor battery depletion and connectivity audits over time</p>
        </div>
      </div>

      <div className="flex-1 w-full pt-6 min-h-[220px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formattedData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#14C89B" strokeOpacity={0.1} vertical={false} />
            <XAxis dataKey="timeLabel" stroke="#A3B8B2" fontSize={11} tickLine={false} axisLine={{ stroke: "#0A664F" }} dy={10} />
            <YAxis stroke="#A3B8B2" fontSize={11} tickLine={false} axisLine={false} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0A0A0A", borderColor: "#0A664F", borderRadius: "16px", color: "#fff", fontSize: "12px" }}
              formatter={(value: any, name: any) => [value, name]}
              labelStyle={{ color: "#14C89B", fontWeight: "bold", marginBottom: "4px" }}
            />
            {sensorKeys.map((key, i) => (
              <Line key={key} type="monotone" dataKey={key} name={key} stroke={i === 0 ? "#14C89B" : i === 1 ? "#38bdf8" : "#f59e0b"} strokeWidth={3} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}