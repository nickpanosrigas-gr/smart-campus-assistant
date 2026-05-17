"use client";
import { motion, Variants } from "framer-motion";
import { Cpu, Monitor, User, DoorOpen, AppWindow } from "lucide-react";
import { SensorType } from "../config/floor2_data";

interface SmartSensorProps {
  id: string;
  type: SensorType;
  x: number;
  y: number;
  animationState: "idle" | "scanning" | "resolved";
  colorHex?: string;
}

// Highly optimized variants (No filters!)
const rippleVariant: Variants = {
  idle: { scale: 1, opacity: 0.8 },
  scanning: {
    scale: [1, 1.4, 1],
    opacity: [1, 0.4, 1],
    transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
  },
  resolved: { scale: 1, opacity: 1 }
};

const breezeVariant: Variants = {
  idle: { x: 0, opacity: 0.8 },
  scanning: {
    x: [-3, 3, -3],
    opacity: [0.6, 1, 0.6],
    transition: { duration: 2, repeat: Infinity, ease: "easeInOut" }
  },
  resolved: { x: 0, opacity: 1 }
};

export default function SmartSensor({ id, type, x, y, animationState, colorHex = "#14C89B" }: SmartSensorProps) {
  
  const getVariant = (): Variants => {
    if (type === "Door" || type === "Window") return breezeVariant;
    return rippleVariant;
  };

  // Maps your sensor types to beautiful, perfectly centered SVG icons
  const renderIcon = () => {
    // We offset x and y by -10 so the 20x20 icon is perfectly centered on the coordinate
    const iconProps = {
      x: -10,
      y: -10,
      width: 20,
      height: 20,
      stroke: animationState === "idle" ? "#555555" : colorHex,
      strokeWidth: 2,
      fill: animationState === "idle" ? "none" : `${colorHex}33`, // Adds a slight tint to the inside when active
    };

    switch (type) {
      case "IAQ":
        // A CPU chip beautifully represents a multi-sensor hub
        return <Cpu {...iconProps} />;
      case "Desk":
        return <Monitor {...iconProps} />;
      case "PeopleCounter":
        return <User {...iconProps} />;
      case "Door":
        return <DoorOpen {...iconProps} />;
      case "Window":
        return <AppWindow {...iconProps} />;
      default:
        return <circle cx="0" cy="0" r="4" fill="gray" />;
    }
  };

  return (
    // Static coordinate wrapper
    <g transform={`translate(${x}, ${y})`}>
      {/* Animated wrapper */}
      <motion.g 
        variants={getVariant()}
        initial="idle"
        animate={animationState}
      >
        {/* Adds a safe, low-opacity glow behind the icon when active */}
        {animationState !== "idle" && type !== "Door" && type !== "Window" && (
          <circle cx="0" cy="0" r="14" fill={colorHex} opacity="0.15" />
        )}
        
        {renderIcon()}
      </motion.g>
    </g>
  );
}