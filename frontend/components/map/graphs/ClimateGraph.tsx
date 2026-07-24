"use client";
import React, { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Thermometer } from "lucide-react";

export default function ClimateGraph({ artifact }: { artifact: any }) {
  const [view, setView] = useState<"temp" | "hum">("temp");

  if (!artifact?.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/60 p-8 text-center bg-black/20 rounded-3xl border border-white/5">
        <Thermometer size={36} className="mb-3 text-[#14C89B]/40 animate-pulse" />
        <p className="text-sm font-semibold">No Climate Telemetry Recorded</p>
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

  return (
    <div className="w-full h-full flex flex-col bg-[#061C16]/60 border border-[#0A664F]/80 rounded-3xl p-6 shadow-2xl backdrop-blur-md overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#14C89B]/10 shrink-0">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Thermometer size={20} className="text-[#14C89B]" />
            Room {artifact.room_id?.toUpperCase()} Climate
            <span className="text-xs font-mono uppercase bg-[#14C89B]/20 text-[#14C89B] px-2.5 py-0.5 rounded-full border border-[#14C89B]/30">
              {artifact.timeframe}
            </span>
          </h3>
          <p className="text-xs text-[#A3B8B2]/70 mt-0.5">Indoor stability vs. outdoor weather correlation</p>
        </div>

        <div className="flex items-center gap-1.5 bg-black/30 p-1 rounded-2xl border border-white/5">
          <button
            onClick={() => setView("temp")}
            className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${view === "temp" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/70 hover:text-white"}`}
          >
            Temperature (°C)
          </button>
          <button
            onClick={() => setView("hum")}
            className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${view === "hum" ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/70 hover:text-white"}`}
          >
            Humidity (%)
          </button>
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
            {view === "temp" ? (
              <>
                <Line type="monotone" dataKey="temperature" name="Indoor Temp (°C)" stroke="#14C89B" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="air_temperature" name="Outdoor Temp (°C)" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              </>
            ) : (
              <>
                <Line type="monotone" dataKey="humidity" name="Indoor Humidity (%)" stroke="#38bdf8" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="relative_humidity" name="Outdoor Humidity (%)" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}