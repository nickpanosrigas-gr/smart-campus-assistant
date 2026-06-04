<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->
# Smart Campus Assistant: Map Architecture & Implementation Plan

## 1. Overview
This document outlines the architecture, state management, and visual implementation for the interactive Smart Campus Map. The map visualizes real-time IoT sensor data via SVG layers, syncing seamlessly with an LLM backend to provide context-aware responses and user-driven exploration.

---

## 2. Global State Management (Zustand)
To ensure persistence across floors and maintain a single source of truth for both User UI actions and LLM function calls, the map relies on a global state store.

### State Structure
```typescript
interface MapState {
  // Navigation & Selection
  activeFloor: string; 
  activeDomain: string; // e.g., 'Climate', 'Occupancy'
  
  // Persisted selections per floor: { "2": ["2.3", "2.4"], "3": ["3.9"] }
  selectedRoomsByFloor: Record<string, string[]>; 
  
  // Data Cache: { Domain -> Floor -> Room ID -> Payload }
  roomData: Record<string, Record<string, Record<string, any>>>;

  // Actions
  toggleRoomSelection: (floorId: string, roomId: string) => void;
  setActiveFloor: (floorId: string) => void;
  setActiveDomain: (domain: string) => void;
  updateRoomData: (domain: string, floorId: string, roomId: string, payload: any) => void;
}
```

---

## 3. UI Layout & Navigation
The UI separates navigation from contextual tool actions to maintain a clean workspace.

### Vertical Floor Selector (Left)
Maps through the `BUILDING_LEVELS` array (`["B", "5", "4", "3", "2", "1", "0", "-1", "-2", "-3"]`). Colors indicate data availability and active states:
*   **Active Floor:** Green Bright (`#14C89B`).
*   **Inactive Floor (with cached data for active tool):** Green Dark (`#0A664F`).
*   **Inactive Floor (no data):** Neutral UI Gray.

### Horizontal Tool Bar (Bottom)
Pill-shaped toggles representing the available domains (Climate, Occupancy, etc.). 
*   Only **one toggle** can be active at a time.
*   Selecting a toggle immediately re-renders the map using cached data from `roomData`.
*   Cross-fading opacity transitions should be used when switching tools to prevent jarring SVG snaps.

---

## 4. Visuals, Colors, & Interaction

### Color System
Colors are mapped directly to the `status` string returned in the backend JSON packets.

| Status | Room Fill (Dark) | Sensor Fill (Bright) |
| :--- | :--- | :--- |
| **Good** | `#0A664F` | `#14C89B` |
| **Warning** | `#A38630` | `#F2C94C` |
| **Critical** | `#A8651D` | `#E8863A` |
| **Error** | `#8E2F3E` | `#C84B5E` |
| **Unavailable** | `#404040` | `#7A7A7A` |

### SVG Integration Rules
1.  **Rooms:** `<path>` or `<polygon>` elements must have an `id` matching the backend `room_id` (e.g., `id="2.3"`).
2.  **Sensors:** Sensor groups/icons must have an `id` matching the backend sensor keys (e.g., `id="F2_2.3-IAQ"`).

### Hover & Selection Effects
*   **Unselected Rooms:** Receive a CSS class (`.room-interactive`) that applies a `brightness(1.2)` or `opacity` filter on hover.
*   **Selected Rooms:** Hover effect is disabled. A white stroke (`stroke="#FFF" strokeWidth={2}`) is applied to clearly distinguish the active selection.

---

## 5. Dynamic Data Rendering (Centroids)
To prevent misalignment in irregular rooms, data text is placed using hardcoded center coordinates and the SVG `<foreignObject>` tag.

### Centroid Configuration
A dictionary maps each `room_id` to its perfect visual center on the SVG canvas.
```typescript
export const FLOOR_2_CENTROIDS: Record<string, { x: number, y: number }> = {
  "2.3": { x: 450, y: 320 },
  "2.4": { x: 600, y: 320 },
  // ...
};
```

### Rendering Implementation
```tsx

{FLOOR_2_CENTROIDS[roomId] && (
  <foreignObject 
    x={FLOOR_2_CENTROIDS[roomId].x - 50} 
    y={FLOOR_2_CENTROIDS[roomId].y - 15} 
    width={100} 
    height={30}
    style={{ pointerEvents: 'none' }}
  >
    <div className="flex items-center justify-center w-full h-full text-white text-xs font-bold drop-shadow-md">
      {getRoomDisplayData(activeDomain, roomData.room_aggregates)}
    </div>
  </foreignObject>
)}
```

---

## 6. Domain-Specific Data Formatting
The `getRoomDisplayData` function formats the `room_aggregates` payload based on the active tool.

| Domain | Backend Payload Example | UI Display Format |
| :--- | :--- | :--- |
| **Climate** | `{'temperature': 26.9, 'humidity': 39.5}` | `26.9°C | 39.5%` |
| **Occupancy** | `{'occupancy': 12, 'capacity': 120}` | `12 / 120` |
| **Doors/Windows** | `{'open_count': 2, 'total_count': 5}` | `2 Open` *(or "Secure" if 0)* |
| **Air Quality** | `{'co2': 552.0}` | `CO2: 552` |
| **Lights** | `{'light_level': 1.0}` | `100%` |
| **Diagnostics** | `{'critical': 2, 'warning': 1}` | `2 Critical, 1 Warn` |
| **Schedule** | `{'time_remaining': '0hr 35mins'}` | `In Use (35m left)` |

*Note: Outdoor/Global sensors (e.g., `F5_Roof-WeatherStation` on Floor 5 while viewing Floor 2) will not map to a specific room. These should be caught by the UI and displayed in a dedicated "Outdoor Stats" floating widget.*

---

## 7. LLM Integration & Dual-Triggering
To keep the UI and the LLM perfectly synced, both follow the same event pathway: **The Global Store**.

### Workflow A: User-Driven
1. User clicks a tool (e.g., "Climate").
2. UI dispatches `setActiveDomain('Climate')` to Zustand. Map re-renders with cached data.
3. UI silently passes a context message to the LLM: *"User selected the Climate tool for rooms [2.3, 2.4]"*.

### Workflow B: LLM-Driven
1. LLM executes a tool call (e.g., `get_climate_data(rooms=[2.3])`).
2. Backend returns JSON packet; frontend caches it in `roomData`.
3. Frontend automatically dispatches `setActiveDomain('Climate')` and `setActiveFloor('2')` to the Zustand store.
4. Map re-renders to reflect the LLM's action.