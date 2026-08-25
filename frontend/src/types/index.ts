// VERBATIM from docs/specs/INTERFACE.md "TYPESCRIPT INTERFACES".
// Canonical. Field names must exactly match Python JSON serialization.
// T | null not T | undefined — this matters for JSON deserialization.
// Do not add fields without updating the backend serializer.

export type EventType =
  | 'system_ready' | 'training_started' | 'training_completed'
  | 'simulation_started' | 'simulation_step'
  | 'world_model_updated' | 'belief_snapshot'
  | 'prediction_error_high'
  | 'agent_spawned' | 'agent_step' | 'agent_completed' | 'agent_failed'
  | 'capability_unlocked'

export interface EventLogEntry {
  id:           number
  timestamp:    number             // unix float
  event_type:   EventType
  agent_id:     string | null
  parent_id:    string | null
  sim_id:       string | null
  region:       string | null
  error_before: number | null
  error_after:  number | null
  payload:      Record<string, unknown>
}

export type NodeStatus = 'spawning' | 'active' | 'completed' | 'failed'

export type NodeKind =
  | 'root' | 'exploration' | 'capability' | 'worldmodel' | 'simulation'

export interface SomaNodeData {
  label:       string
  title:       string
  status:      NodeStatus
  agent_id:    string
  kind:        NodeKind
  region:      string | null
  error_before: number | null
  error_after:  number | null
  dof_level:   number | null
  timestamp:   number
  sim_id:      string | null
  event_id:    number
  generation:  number
  samples_collected: number | null
  fine_tune_loss: number | null
  api_tokens_used: number | null
  error_message: string | null
  // Mutable during lifetime — set by useEffect timer:
  activeMs:    number              // ms since spawned (drives spawning→active)
}

export interface SomaEdgeData {
  label:  string
  status: 'active' | 'completed' | 'failed'
}

export interface BeliefStateSnapshot {
  version:            number
  world_model_version?: number   // actual key sent by backend snapshot()
  prediction_error_map: number[][]   // [16][16]
  global_mean_error:  number
  regional_errors:    Record<string, number>  // exactly 6 keys
  active_dof:         number
  episode_count:      number
  timestamp:          number
  error_map:          number[][]    // alias for prediction_error_map
}

export interface SimulationState {
  readonly simulation_id: string
  readonly token_2d: readonly (readonly number[])[]
  readonly vector: readonly number[]
  readonly direction: readonly number[]
  readonly raw_observation: readonly number[]
  readonly achievements_count: number
  readonly stats: {
    readonly health: number
    readonly drink: number
    readonly food: number
    readonly energy: number
    readonly light: number
    readonly is_sleeping: number
    readonly is_resting: number
  }
  readonly step: number
  readonly done: boolean
  readonly reward: number
  readonly active_dof: number
}

export interface DetailResponse {
  simulation_state: SimulationState
  belief_state:     BeliefStateSnapshot
}

export interface HealthResponse {
  status:              'starting' | 'training' | 'ok'
  world_model_version: number
  global_error:        number
  active_agents:       number
  primary_sim_step:    number
  training_progress:   number | null
}
