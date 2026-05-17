export type SensorType = "IAQ" | "Door" | "Window" | "PeopleCounter" | "Desk";

export interface SensorNode {
  id: string;
  type: SensorType;
  x: number;
  y: number;
  room: string;
}

// Coordinates calculated from the center points of the elements in your SVG
export const floor2Sensors: SensorNode[] = [
  // IAQ Sensors (Multi-sensor: Climate, Air Quality, Occupancy, Lights)
  { id: "F2_2.4-IAQ-2", type: "IAQ", x: 46, y: 230, room: "2.4" },
  { id: "F2_2.4-IAQ-1", type: "IAQ", x: 223, y: 302, room: "2.4" },
  { id: "F2_2.3-IAQ", type: "IAQ", x: 243, y: 394, room: "2.3" },
  { id: "F2_2.2-IAQ", type: "IAQ", x: 376, y: 437, room: "2.2" },
  { id: "F2_2.1-IAQ-2", type: "IAQ", x: 415, y: 58, room: "2.1" },
  { id: "F2_2.1-IAQ-1", type: "IAQ", x: 354, y: 226, room: "2.1" },

  // Doors & Magnetic Contacts (MC)
  { id: "F2_2.4-DOOR", type: "Door", x: 171, y: 329, room: "2.4" },
  { id: "F2_2.4-MC-3", type: "Window", x: 76, y: 132, room: "2.4" },
  { id: "F2_2.4-MC-2", type: "Window", x: 113, y: 132, room: "2.4" },
  { id: "F2_2.4-MC-1", type: "Door", x: 281, y: 298, room: "2.4" },
  { id: "F2_2.3-MC-5", type: "Window", x: 200, y: 550, room: "2.3" },
  { id: "F2_2.3-MC-4", type: "Window", x: 266, y: 560, room: "2.3" },
  { id: "F2_2.3-MC-3", type: "Window", x: 315, y: 568, room: "2.3" },
  { id: "F2_2.3-MC-2", type: "Door", x: 197, y: 366, room: "2.3" },
  { id: "F2_2.3-MC-1", type: "Door", x: 309, y: 365, room: "2.3" },
  { id: "F2_2.2-DOOR", type: "Door", x: 353, y: 342, room: "2.2" },
  { id: "F2_2.1-WINDOW", type: "Window", x: 500, y: 176, room: "2.1" },
  { id: "F2_2.1-MC-4", type: "Window", x: 502, y: 272, room: "2.1" },
  { id: "F2_2.1-MC-3", type: "Window", x: 502, y: 240, room: "2.1" },
  { id: "F2_2.1-MC-2", type: "Window", x: 499, y: 111, room: "2.1" },
  { id: "F2_2.1-MC-1", type: "Door", x: 332, y: 168, room: "2.1" },

  // Specialized Occupancy Sensors
  { id: "F2_2.4-PC", type: "PeopleCounter", x: 255, y: 267, room: "2.4" },
  { id: "F2_2.3-WO", type: "PeopleCounter", x: 326, y: 458, room: "2.3" },
  { id: "F2_2.2-Desk-4", type: "Desk", x: 443, y: 375, room: "2.2" },
  { id: "F2_2.2-Desk-3", type: "Desk", x: 443, y: 444, room: "2.2" },
  { id: "F2_2.2-Desk-2", type: "Desk", x: 422, y: 444, room: "2.2" },
  { id: "F2_2.2-Desk-1", type: "Desk", x: 422, y: 375, room: "2.2" },
];