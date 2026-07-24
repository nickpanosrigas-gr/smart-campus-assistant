"use client";
import React, { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";
import { Wind, Wifi, WifiOff } from "lucide-react";

export default function AirQualityGraph({ artifact }: { artifact: any }) {
  const [activeMetric, setActiveMetric] = useState<string>("co2");

  if (!artifact?.series || artifact.series.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-[#A3B8B2]/60 p-8 text-center bg-black/20 rounded-3xl border border-white/5">
        <Wind size={36} className="mb-3 text-[#14C89B]/40 animate-pulse" />
        <p className="text-sm font-semibold">No Air Quality Records Found</p>
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

  const metrics = [
    { key: "co2", label: "CO2", color: "#14C89B", unit: "ppm" },
    { key: "pm2_5", label: "PM2.5 (Indoor)", color: "#38bdf8", unit: "µg/m³" },
    { key: "outdoor_pm2_5", label: "PM2.5 (Outdoor)", color: "#94a3b8", unit: "µg/m³" },
    { key: "tvoc", label: "TVOC", color: "#f43f5e", unit: "ppb" }
  ].filter(m => artifact.series.some((pt: any) => pt[m.key] !== undefined));

  return (
    <div className="w-full h-full flex flex-col bg-[#061C16]/60 border border-[#0A664F]/80 rounded-3xl p-6 shadow-2xl backdrop-blur-md overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#14C89B]/10 shrink-0">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Wind size={20} className="text-[#14C89B]" />
            Room {artifact.room_id?.toUpperCase()} Air Quality
            <span className="text-xs font-mono uppercase bg-[#14C89B]/20 text-[#14C89B] px-2.5 py-0.5 rounded-full border border-[#14C89B]/30">
              {artifact.timeframe}
            </span>
          </h3>
          <p className="text-xs text-[#A3B8B2]/70 mt-0.5">Indoor vs. Outdoor contaminant tracking</p>
        </div>

        {/* Metric Selector Pills */}
        <div className="flex items-center gap-1.5 bg-black/30 p-1 rounded-2xl border border-white/5">
          {metrics.map(m => (
            <button
              key={m.key}
              onClick={() => setActiveMetric(m.key)}
              className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${
                activeMetric === m.key ? "bg-[#14C89B] text-[#0A0A0A] shadow-md" : "text-white/70 hover:text-white"
              }`}
            >
              {m.label}
            </button>
          ))}
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
            {metrics
              .filter(m => m.key === activeMetric || activeMetric === "all")
              .map(m => (
                <Line
                  key={m.key}
                  type="monotone"
                  dataKey={m.key}
                  name={`${m.label} (${m.unit})`}
                  stroke={m.color}
                  strokeWidth={3}
                  dot={false}
                  activeDot={{ r: 6, fill: m.color, stroke: "#0A0A0A", strokeWidth: 2 }}
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}