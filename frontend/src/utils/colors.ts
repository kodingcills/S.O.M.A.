// SOMA research-interface color tokens.
// Components should communicate meaning with text and shape as well as color.

// Background and surfaces
export const APP_BG           = '#F3F3F0'
export const CANVAS_BG        = '#F8F8F6'
export const SURFACE_BG       = '#FFFFFF'
export const PANEL_BORDER     = '#E2E2DE'
export const BORDER_SUBTLE    = '#ECECE8'
export const SURFACE_HOVER    = '#F2F2EF'
export const SURFACE_SELECTED = '#F0F2FA'

// Text
export const TEXT_PRIMARY    = '#20201E'
export const TEXT_SECONDARY  = '#686863'
export const TEXT_TERTIARY   = '#979791'
export const TEXT_MONO       = '#3E4C70'
export const TEXT_ERROR      = '#A64A44'

// Semantic accents
export const COLOR_EXPLORATION = '#5572A6'
export const COLOR_CAPABILITY  = '#75669A'
export const COLOR_WORLDMODEL  = '#5E7E70'
export const COLOR_SIMULATION  = '#64808A'
export const COLOR_ROOT        = '#777773'
export const COLOR_SELECTION   = '#5368A5'

// Status colors
export const STATUS_SPAWNING  = '#B68135'
export const STATUS_COMPLETED = '#557967'
export const STATUS_FAILED    = '#A64A44'

// Sequential uncertainty scale. Risk/failure red is intentionally reserved.
export function errorToColor(error: number): string {
  const e = Math.max(0, Math.min(1, error))
  const lightness = 92 - e * 45
  const saturation = 22 + e * 24
  return `hsl(218, ${saturation}%, ${lightness}%)`
}

export function errorToColorHover(error: number): string {
  const e = Math.max(0, Math.min(1, error))
  const lightness = 86 - e * 42
  return `hsl(218, 42%, ${lightness}%)`
}

export function agentColor(agentId: string): string {
  if (agentId.startsWith('exp_')) return COLOR_EXPLORATION
  if (agentId.startsWith('cap_')) return COLOR_CAPABILITY
  if (agentId === 'root') return COLOR_ROOT
  if (agentId.startsWith('wm_')) return COLOR_WORLDMODEL
  return COLOR_SIMULATION
}

export const REGION_LABELS: Record<string, string> = {
  upper_left:               'Upper left',
  upper_right:              'Upper right',
  lower_left:               'Lower left',
  lower_right:              'Lower right',
  tool_tissue_boundary:     'Tool–tissue boundary',
  surgical_target_vicinity: 'Surgical target vicinity',
}
