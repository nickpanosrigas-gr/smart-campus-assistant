// Updated to include the 'unavailable' state for unmonitored infrastructure
export type RoomHealth = "good" | "warning" | "critical" | "error" | "unavailable";

// Maps to your "StatePrimary (Accent) Color"
export const SENSOR_COLORS: Record<RoomHealth, string> = {
  good: "#14C89B",       // Green Bright
  warning: "#F2C94C",    // Yellow Bright
  critical: "#E8863A",   // Orange Bright
  error: "#C84B5E",      // Red Bright
  unavailable: "#7A7A7A" // Gray Light (No Sensor)
};

// Maps to your "Updated Background Color"
export const ROOM_COLORS: Record<RoomHealth, string> = {
  good: "#0A664F",       // Green Dark
  warning: "#A38630",    // Yellow Dark
  critical: "#A8651D",   // Orange Dark
  error: "#8E2F3E",      // Red Dark
  unavailable: "#404040" // Gray Dark (No Sensor)
};

export const BUILDING_LEVELS = ["B", "5", "4", "3", "2", "1", "0", "-1", "-2", "-3"];