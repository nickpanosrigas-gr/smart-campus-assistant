export type RoomHealth = "Good" | "Warning" | "Error";

export const SENSOR_COLORS: Record<RoomHealth, string> = {
  Good: "#14C89B",    // Green Bright
  Warning: "#E89A3D", // Orange Bright
  Error: "#C84B5E"    // Red Bright
};

export const ROOM_COLORS: Record<RoomHealth, string> = {
  Good: "#0A664F",    // Green Dark
  Warning: "#A8651D", // Orange Dark
  Error: "#8E2F3E"    // Red Dark
};

// Simulated backend states for the rooms
export const mockRoomStates: Record<string, RoomHealth> = {
  "2.4": "Good",
  "2.3": "Warning",
  "2.2": "Error",
  "2.1": "Good"
};

export const BUILDING_LEVELS = ["B", "5", "4", "3", "2", "1", "0", "-1", "-2", "-3"];