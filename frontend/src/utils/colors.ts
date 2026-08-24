// VERBATIM from docs/specs/INTERFACE.md "COLOR SYSTEM".
// Single source of truth. Never hardcode hex in components.

// Background
export const CANVAS_BG       = '#0A0E1A'
export const SURFACE_BG      = '#0F1629'
export const PANEL_BORDER    = '#1E2D4A'
export const SURFACE_HOVER   = '#162035'

// Text
export const TEXT_PRIMARY    = '#E8EFF8'
export const TEXT_SECONDARY  = '#7A9CC4'
export const TEXT_MONO       = '#5AC8FA'
export const TEXT_ERROR      = '#FF3B30'

// Agent type colors
export const COLOR_EXPLORATION = '#007AFF'  // blue
export const COLOR_CAPABILITY  = '#AF52DE'  // purple
export const COLOR_WORLDMODEL  = '#34C759'  // green
export const COLOR_SIMULATION  = '#5AC8FA'  // light blue
export const COLOR_ROOT        = '#FFFFFF'

// Status colors
export const STATUS_SPAWNING  = '#FFD60A'   // yellow pulse
export const STATUS_COMPLETED = '#34C759'   // green
export const STATUS_FAILED    = '#FF3B30'   // red

// HSL HEATMAP FORMULA — do not change
// errorToColor(0.0) → hsl(120, 85%, 45%) green
// errorToColor(0.5) → hsl(60,  85%, 45%) yellow
// errorToColor(1.0) → hsl(0,   85%, 45%) red
// If 0.5 appears brown: RGB interpolation is being used instead of HSL.
export function errorToColor(error: number): string {
  const e   = Math.max(0, Math.min(1, error))
  const hue = 120 * (1 - e)
  return `hsl(${hue}, 85%, 45%)`
}

// Brighter on hover
export function errorToColorHover(error: number): string {
  const e   = Math.max(0, Math.min(1, error))
  const hue = 120 * (1 - e)
  return `hsl(${hue}, 85%, 65%)`
}

// Map agent_id prefix to type color
export function agentColor(agentId: string): string {
  if (agentId.startsWith('exp_')) return COLOR_EXPLORATION
  if (agentId.startsWith('cap_')) return COLOR_CAPABILITY
  if (agentId === 'root')         return COLOR_ROOT
  if (agentId.startsWith('wm_'))  return COLOR_WORLDMODEL
  return TEXT_SECONDARY
}

// Human-readable region names
export const REGION_LABELS: Record<string, string> = {
  upper_left:              'Upper Left',
  upper_right:             'Upper Right',
  lower_left:              'Lower Left',
  lower_right:             'Lower Right',
  tool_tissue_boundary:    'Tool Zone',
  surgical_target_vicinity:'Target Zone',
}
