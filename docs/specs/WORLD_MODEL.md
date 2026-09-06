# WORLD_MODEL.md
# SOMA — World Model Specification
# Authoritative for: BeliefState, PredictionNetwork, WorldModel class,
# initial training, fine-tuning, replay buffer, MPC planning loop,
# W&B integration.
#
# Invariants referenced here are DEFINED in ARCHITECTURE.md.
# Constants duplicated in QUICK_REFERENCE.md for fast lookup.
# Read this before building: belief_state.py, prediction_net.py,
# world_model.py, train.py, mpc_agent.py.
#
# Network input: 818 float32 (806 state + 12 action)
# State source: BeliefState.to_input_vector() → (806,) float32
# This matches env.get_state_vector() exactly. See SIMULATION.md.

---

## TWO-COMPONENT ARCHITECTURE

The world model has two components with distinct roles. Confusing them
is the most common source of architectural errors in this file.

```
COMPONENT A — BeliefState
  What it is: persistent structured representation of observed state
  Source of truth: copied directly from env.get_state_vector() each step
  Contains also: prediction_error_map, confidence_map (computed, not observed)
  Never inferred. Always accurate for the observation fields.

COMPONENT B — PredictionNetwork
  What it is: learned dynamics model f_θ(s_t, a_t) → ŝ_{t+1}
  Source of truth: training data from simulation rollouts
  Can be wrong. Improves through fine-tuning.
  Input: BeliefState.to_input_vector() (806) + action (12) = 818 floats
  Does NOT receive: prediction_error_map or confidence_map
```

**Per-step data flow:**
```
1. s_t = belief.to_input_vector()          # 806 floats, no error map
2. For each candidate action a_i:
     ŝ_{t+1,i} = network(s_t ++ a_i)      # 4 output heads
     v_i = value_function(ŝ_{t+1,i})
3. Execute a* = argmax(v_i)
4. Observe s_{t+1} from env.get_state_vector()
5. error_map update:
     ε[r,c] = (ŝ_{tissue}[r,c] - s_{t+1}.integrity[r,c])²
     error_map = 0.3×ε + 0.7×error_map    # EMA
6. belief.update_from_observation(s_{t+1})
7. replay_buffer.add(s_t, a*, s_{t+1})
```

The error map (step 5) is computed AFTER observing s_{t+1}. It cannot
exist at step 2. See CIRCULAR INPUT INVARIANT in ARCHITECTURE.md.

---

## BELIEF STATE

**File:** `backend/belief_state.py`
**Imports:** `numpy`, `dataclasses`, `time`. Zero imports from backend.

### Fields

```python
@dataclass
class BeliefState:
    # Observation fields — copied from env.get_state_vector() each step
    integrity:        np.ndarray   # (16,16) float32, [0,1]
    vascularity:      np.ndarray   # (16,16) float32, [0,1] — set once at init
    bleeding:         np.ndarray   # (16,16) bool
    ee_position:      np.ndarray   # (3,) float32, workspace coords
    ee_quaternion:    np.ndarray   # (4,) float32, unit quat
    gripper_state:    float        # [0,1]
    target_pos:       np.ndarray   # (2,) float32, grid col/row
    target_reached:   bool
    vessels:          list[VesselBelief]  # up to 5 entries
    active_dof:       int          # [1,6]

    # Computed fields — NOT observation, NOT network input
    prediction_error_map: np.ndarray  # (16,16) float32, [0,1]
    confidence_map:       np.ndarray  # (16,16) float32, = 1 - error_map

    # Metadata
    world_model_version: int  # increments on each fine_tune()
    episode_count:       int  # total steps across all episodes
    last_updated:        float  # unix timestamp
```

### Methods

**`update_from_observation(state_vec: np.ndarray) → None`**

Unpacks `env.get_state_vector()` (806 floats) into structured fields.
Index ranges are exact — see SIMULATION.md state vector table.

```python
def update_from_observation(self, state_vec: np.ndarray) -> None:
    # state_vec must be shape (806,) float32
    self.integrity     = state_vec[0:256].reshape(16, 16).copy()
    self.vascularity   = state_vec[256:512].reshape(16, 16).copy()
    self.bleeding      = state_vec[512:768].reshape(16, 16).astype(bool)
    self.ee_position   = state_vec[768:771].copy()
    self.ee_quaternion = state_vec[771:775].copy()
    self.gripper_state = float(state_vec[775])
    self.target_pos    = state_vec[776:778].copy()
    self.target_reached = bool(state_vec[778] > 0.5)
    # vessel data: [779:799], 5 slots × 4 values each
    self._unpack_vessels(state_vec[779:799])
    self.active_dof = int(np.argmax(state_vec[799:805])) + 1
    # state_vec[805]: padding, ignored
    self.last_updated = time.time()
    self.episode_count += 1
    # Does NOT update: prediction_error_map, confidence_map, world_model_version
```

**`to_input_vector() → np.ndarray`**

See CIRCULAR INPUT INVARIANT in ARCHITECTURE.md. The test for this
invariant must pass before Task 1.3 is marked complete.

Returns the same 806 values that produced this belief state — i.e.,
re-packs the observation fields in the exact order as `get_state_vector()`.

```python
def to_input_vector(self) -> np.ndarray:
    # Re-packs observation fields into 806-element vector.
    # MUST NOT include prediction_error_map or confidence_map.
    # MUST return identical output regardless of prediction_error_map value.
    parts = [
        self.integrity.flatten(),           # 256
        self.vascularity.flatten(),          # 256
        self.bleeding.flatten().astype(np.float32),  # 256
        self.ee_position,                   # 3
        self.ee_quaternion,                 # 4
        np.array([self.gripper_state], dtype=np.float32),  # 1
        self.target_pos,                    # 2
        np.array([float(self.target_reached)], dtype=np.float32),  # 1
        self._pack_vessels(),               # 20 (5 slots × 4)
        self._dof_onehot(),                 # 6
        np.zeros(1, dtype=np.float32),      # 1 padding
    ]
    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape == (806,)  # remove in production if bottleneck
    return vec

def _dof_onehot(self) -> np.ndarray:
    v = np.zeros(6, dtype=np.float32)
    v[self.active_dof - 1] = 1.0  # active_dof ∈ [1,6] → index [0,5]
    return v

def _pack_vessels(self) -> np.ndarray:
    v = np.zeros(20, dtype=np.float32)
    for i, vessel in enumerate(self.vessels[:5]):
        base = i * 4
        v[base]   = vessel.col / 15.0
        v[base+1] = vessel.row / 15.0
        v[base+2] = vessel.radius / 15.0
        v[base+3] = float(vessel.damaged)
    return v
```

**`update_prediction_error(predicted_integrity: np.ndarray, actual_integrity: np.ndarray) → None`**

Both inputs shape `(16,16)` float32. Called by `WorldModel.update()`
after every simulation step.

```python
def update_prediction_error(self, predicted: np.ndarray,
                             actual: np.ndarray) -> None:
    pointwise = (predicted - actual) ** 2          # per-cell squared error
    α = 0.3
    self.prediction_error_map = (
        α * pointwise + (1 - α) * self.prediction_error_map
    ).astype(np.float32)
    self.confidence_map = 1.0 - self.prediction_error_map
```

Why tissue integrity only (not full state): integrity change is the
primary signal for surgical quality. Predicting EE quaternion poorly
doesn't affect spawning decisions or the heatmap.

Why α=0.3: α=0.3 means ~10 steps to respond to a sustained error
change (the "half-life" of the EMA is approximately `1/α` steps).
Lower α = smoother but slower to respond to newly high-error regions.
Higher α = noisy heatmap that thrashes on every step.

**`get_regional_errors() → dict[str, float]`**

Returns exactly these 6 keys. Missing keys cause `KeyError` in the
orchestrator. All values are float in `[0.0, 1.0]`.

```python
REGIONS = {
    'upper_left':              (slice(0,8),   slice(0,8)),
    'upper_right':             (slice(0,8),   slice(8,16)),
    'lower_left':              (slice(8,16),  slice(0,8)),
    'lower_right':             (slice(8,16),  slice(8,16)),
    # Dynamic regions — computed from current state:
    'tool_tissue_boundary':    None,   # 3×3 around EE position
    'surgical_target_vicinity': None,  # 3×3 around target position
}

def get_regional_errors(self) -> dict[str, float]:
    m = self.prediction_error_map
    ee_r = int(np.clip(self.ee_position[1] * 15, 1, 14))
    ee_c = int(np.clip(self.ee_position[0] * 15, 1, 14))
    tgt_r = int(np.clip(self.target_pos[1] * 15, 1, 14))
    tgt_c = int(np.clip(self.target_pos[0] * 15, 1, 14))
    return {
        'upper_left':              float(m[0:8, 0:8].mean()),
        'upper_right':             float(m[0:8, 8:16].mean()),
        'lower_left':              float(m[8:16, 0:8].mean()),
        'lower_right':             float(m[8:16, 8:16].mean()),
        'tool_tissue_boundary':    float(m[ee_r-1:ee_r+2, ee_c-1:ee_c+2].mean()),
        'surgical_target_vicinity':float(m[tgt_r-1:tgt_r+2, tgt_c-1:tgt_c+2].mean()),
    }
```

**`snapshot() → dict`**

Returns JSON-serializable dict for the `belief_snapshot` event payload
and `/detail/{sim_id}` API response.

```python
def snapshot(self) -> dict:
    return {
        'global_mean_error':  float(self.prediction_error_map.mean()),
        'regional_errors':    self.get_regional_errors(),
        'world_model_version': self.world_model_version,
        'active_dof':         self.active_dof,
        'episode_count':      self.episode_count,
        'error_map':          self.prediction_error_map.tolist(),  # [[float]]×16×16
        'timestamp':          self.last_updated,
    }
```

---

## PREDICTION NETWORK

**File:** `backend/prediction_net.py`
**Imports:** `torch`, `torch.nn`, `numpy`. No backend imports.

### Architecture

```
Input: concatenate(state_806, action_12) → (818,)

ENCODER (shared):
  Linear(818, 512) → LayerNorm(512) → ReLU → Dropout(0.1)
  Linear(512, 256) → LayerNorm(256) → ReLU → Dropout(0.1)
  Linear(256, 128) → LayerNorm(128) → ReLU
  → 128-dim shared representation

HEAD 1 — tissue:     Linear(128, 256) → Sigmoid
  Output: predicted tissue_integrity, all 256 cells, range (0,1)
  Reshape to (16,16) for error map comparison.

HEAD 2 — damage:     Linear(128, 1)   → Sigmoid
  Output: P(vessel_damaged_this_step), range (0,1)
  Hard constraint in value function at >0.5.

HEAD 3 — vessel_risk: Linear(128, 5)  → Sigmoid
  Output: P(too_close_to_vessel_i), one per vessel slot, range (0,1)
  Soft penalty in value function.

HEAD 4 — instrument: Linear(128, 4)   → (no activation)
  Output: [pred_ee_x, pred_ee_y, pred_ee_z, pred_orientation]
  All normalized (same normalization as input). Denormalize on use.
  No sigmoid: instrument coordinates can be at grid edges.
```

See LAYERNORM INVARIANT in ARCHITECTURE.md. BatchNorm is prohibited.

**Why Dropout(0.1) in encoder only:** Light regularization on the
shared representation. The four heads should not be individually
regularized — they each receive only 128 inputs and overfit easily
if dropout is applied per-head with small fine-tune batches.

**Why no instrument head activation:** `Sigmoid` would constrain
output to (0,1), but normalized EE coordinates are already in [0,1]
so this seems harmless. The problem: Sigmoid saturates near 0 and 1,
producing near-zero gradients for positions near grid edges. Positions
near the grid boundary (where the robot starts) would be predicted
with high error and near-zero gradient — the head never improves
on those positions. Linear output avoids this.

### PredictedState Dataclass

```python
@dataclass
class PredictedState:
    tissue: np.ndarray          # (16,16) float32, range (0,1)
    damage_prob: float          # scalar, range (0,1)
    vessel_risk: np.ndarray     # (5,) float32, range (0,1)
    pred_ee_norm: np.ndarray    # (4,) float32 — normalized, denorm to use
    # Denormalize: ee_x = pred_ee_norm[0]*0.2-0.1, z = pred_ee_norm[2]*0.3
```

### Forward Pass

```python
class PredictionNetwork(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # x: (batch, 818) float32
        h = self.encoder(x)  # (batch, 128)
        return (
            self.tissue_head(h),      # (batch, 256)
            self.damage_head(h),      # (batch, 1)
            self.vessel_head(h),      # (batch, 5)
            self.instrument_head(h),  # (batch, 4)
        )

    def predict(self,
                state_vec: np.ndarray,    # (806,)
                action_vec: np.ndarray,   # (12,)
               ) -> PredictedState:
        # Synchronous. Called by WorldModel.predict().
        # Sets eval() mode internally. Caller must NOT set mode.
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(
                np.concatenate([state_vec, action_vec])
            ).float().unsqueeze(0).to(self.device)
            tissue, damage, vessel, instrument = self(x)
        return PredictedState(
            tissue=tissue.squeeze().cpu().numpy().reshape(16, 16),
            damage_prob=float(damage.squeeze().cpu()),
            vessel_risk=vessel.squeeze().cpu().numpy(),
            pred_ee_norm=instrument.squeeze().cpu().numpy(),
        )
```

**Device selection:**
```python
self.device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
self.to(self.device)
```

MPS (Apple Silicon) provides 3-5× speedup over CPU for this network
size. Always prefer MPS on M-series Mac. If MPS is unavailable, CPU
is ~3ms per forward pass — acceptable for demo but measure and log.

**Total parameters:** ~615K. Saves/loads in <1 second.

---

## WORLD MODEL CLASS

**File:** `backend/world_model.py`

See WORLD MODEL SINGLETON INVARIANT in ARCHITECTURE.md.
See EVENT LOOP NON-BLOCKING INVARIANT in ARCHITECTURE.md.

```python
class WorldModel:
    def __init__(self, model_path: str):
        self.network = PredictionNetwork()
        self.belief  = BeliefState(...)    # zero-initialized
        self._predict_lock  = asyncio.Lock()
        self._finetune_lock = asyncio.Lock()
        self._replay_buffer: list[tuple] = []  # (state_806, action_12, next_806)
        self._replay_max = 100_000
        self.load(model_path)
```

### `predict(state_vec, action_vec) → PredictedState`

Synchronous. Called by `MPCAgent` directly (Architecture permitted call #1).
Must NOT be called with `await` — it is not a coroutine.

```python
def predict(self,
            state_vec: np.ndarray,   # (806,)
            action_vec: np.ndarray,  # (12,)
           ) -> PredictedState:
    # Caller acquires _predict_lock via asyncio.Lock before calling.
    # This method does not acquire the lock itself.
    return self.network.predict(state_vec, action_vec)
```

### `update(state_vec, action_vec, next_state_vec) → None`

Synchronous. Called by `MPCAgent` after every simulation step
(Architecture permitted call #1 continuation).

```python
def update(self,
           state_vec:      np.ndarray,  # (806,) — state before action
           action_vec:     np.ndarray,  # (12,)
           next_state_vec: np.ndarray,  # (806,) — state after action
          ) -> None:
    predicted = self.network.predict(state_vec, action_vec)
    actual_integrity = next_state_vec[0:256].reshape(16, 16)
    self.belief.update_prediction_error(predicted.tissue, actual_integrity)
    self.belief.update_from_observation(next_state_vec)
    self._add_to_replay(state_vec, action_vec, next_state_vec)
```

### `fine_tune(new_samples) → float` (async)

See EVENT LOOP NON-BLOCKING INVARIANT in ARCHITECTURE.md.

```python
async def fine_tune(self,
                    new_samples: list[tuple],  # [(s_806, a_12, s_next_806)]
                   ) -> float:
    async with self._finetune_lock:
        async with self._predict_lock:  # block predictions during weight swap
            loss = await asyncio.to_thread(
                self._fine_tune_sync,
                new_samples
            )
        # _predict_lock released here; _finetune_lock still held
        self.belief.world_model_version += 1
        wandb.log({
            "fine_tune_loss":    loss,
            "world_model_version": self.belief.world_model_version,
            "global_error":      float(self.belief.prediction_error_map.mean()),
        })
        write_event("world_model_updated",
                    payload_version=self.belief.world_model_version,
                    payload_loss=loss)
        self._save()
    return loss
```

`_predict_lock` is held only during the weight-update step at the end
of `_fine_tune_sync`, not during the entire training loop. This allows
predictions to continue during most of fine-tuning; they pause only
during the final milliseconds when weights are being swapped. The
`_finetune_lock` is held for the entire duration to prevent concurrent
`fine_tune()` calls.

### `_fine_tune_sync(new_samples) → float` (synchronous, runs in thread)

```python
def _fine_tune_sync(self, new_samples: list[tuple]) -> float:
    # Build batch: 80% replay, 20% new
    n_new = len(new_samples)
    n_old = min(n_new * 4, len(self._replay_buffer))  # 4:1 old:new = 80/20
    old_samples = random.sample(self._replay_buffer, n_old)
    batch = new_samples + old_samples
    random.shuffle(batch)

    self.network.train()
    optimizer = self._optimizer  # Adam instance, persisted across calls
    total_loss = 0.0

    for epoch in range(FINE_TUNE_EPOCHS):    # 5 epochs
        for i in range(0, len(batch), FINE_TUNE_BATCH_SIZE):  # 64
            chunk = batch[i:i+FINE_TUNE_BATCH_SIZE]
            states, actions, nexts = zip(*chunk)

            x = torch.from_numpy(
                np.column_stack([np.stack(states), np.stack(actions)])
            ).float().to(self.network.device)
            y_tissue = torch.from_numpy(
                np.stack([n[0:256] for n in nexts])
            ).float().reshape(-1, 256).to(self.network.device)
            y_damage = torch.from_numpy(
                np.array([1.0 if any(n[779+3::4] > 0.5) else 0.0
                          for n in nexts])
            ).float().unsqueeze(1).to(self.network.device)

            tissue_pred, damage_pred, _, _ = self.network(x)
            loss = (
                LOSS_W_TISSUE * F.mse_loss(tissue_pred, y_tissue) +
                LOSS_W_DAMAGE * F.binary_cross_entropy(damage_pred, y_damage)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Swap weights atomically (network is in eval mode for predictions)
    # _predict_lock is acquired by caller before this function returns
    self.network.eval()
    self._add_to_replay(*zip(*new_samples))  # add new samples to buffer
    return total_loss / (FINE_TUNE_EPOCHS * max(1, len(batch) // FINE_TUNE_BATCH_SIZE))
```

**Why persist the Adam optimizer across fine-tune calls:**
Adam maintains per-parameter momentum and adaptive learning rate state.
Discarding the optimizer between fine-tune calls loses accumulated
gradient statistics. The optimizer's "memory" of parameter update
history makes subsequent fine-tune calls more efficient — it already
knows which parameters change frequently (and should be updated
conservatively) vs rarely (and can accept larger steps).

### `load(path) → None`

```python
def load(self, path: str) -> None:
    p = Path(path)
    if p.exists():
        checkpoint = torch.load(str(p), map_location=self.network.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.eval()
    else:
        # No saved model — random weights
        # world_model_version stays 0
        # Write a warning to the event log
        write_event("system_ready",
                    warning="No saved model found — training required before useful predictions")
        self.network.eval()
```

### `_save() → None`

```python
def _save(self) -> None:
    torch.save({
        'model_state_dict':     self.network.state_dict(),
        'optimizer_state_dict': self._optimizer.state_dict(),
        'world_model_version':  self.belief.world_model_version,
    }, "models/prediction_network.pt")
```

### Replay Buffer

```python
def _add_to_replay(self,
                   state_vecs:  np.ndarray | list,
                   action_vecs: np.ndarray | list,
                   next_vecs:   np.ndarray | list) -> None:
    for s, a, n in zip(state_vecs, action_vecs, next_vecs):
        self._replay_buffer.append((s.copy(), a.copy(), n.copy()))
    if len(self._replay_buffer) > self._replay_max:
        # Evict oldest samples (FIFO)
        self._replay_buffer = self._replay_buffer[-self._replay_max:]
```

Buffer starts empty. Initial training samples are added during
`_fine_tune_sync()` when `train.py` calls fine_tune with the full
50,000-sample training set split into batches.

---

## INITIAL TRAINING

**File:** `backend/train.py`

Runs at startup if `models/prediction_network.pt` does not exist.
Called as `await asyncio.to_thread(train_prediction_network)` in
`main.py` lifespan. Never called directly from async context.

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| epochs | 50 (extend to 100) | 50 usually sufficient; extend if targets not met |
| batch_size | 256 | Fills GPU/MPS memory well without OOM on M-series |
| train/val split | 90/10, seed=42 | Reproducible split for comparing checkpoints |
| optimizer | Adam, lr=0.001 | Standard for feedforward networks |
| scheduler | CosineAnnealingLR, T_max=epochs | Smooth LR decay, no manual schedule |
| early_stop patience | 10 epochs | Stop if val tissue MSE doesn't improve |

### Loss Function

```
total_loss = 0.6 × MSE(tissue_pred, tissue_actual)
           + 0.3 × BCE(damage_pred, damage_actual)
           + 0.1 × BCE(vessel_risk_pred, vessel_proximity_actual)
```

**Weight rationale:** Tissue MSE (0.6) is the primary prediction target —
it drives the error map and MPC value function. Damage prediction (0.3)
is the safety signal and has asymmetric cost (missed damage > false alarm).
Vessel risk (0.1) is supplementary — partially captured by damage already.

**`damage_actual` construction (from training sample):**
```python
# From next_state_vec[779:799] — vessel data, 5 slots × 4 values
# Slot i: [col/15, row/15, radius/15, damaged_float]
# damaged_float at index [779 + i*4 + 3]
damage_actual = 1.0 if any(
    next_state_vec[779 + i*4 + 3] > 0.5 for i in range(5)
) else 0.0
```

**`vessel_proximity_actual` construction:**
```python
# vessel_proximity[i] = 1.0 if EE is within (vessel_radius + 2mm) of vessel_i
# Use state_vec to get EE position and vessel positions
# Threshold: vessel_radius_workspace + 0.002 meters
```

### Training Loop Structure

```python
def train_prediction_network():
    write_event("training_started")
    wandb.init(project="soma-surgical",
               name=f"initial-train-{int(time.time())}",
               config={"lr": 0.001, "epochs": 50,
                       "state_dim": 806, "action_dim": 12,
                       "batch_size": 256})

    # Load data
    samples = load_all_samples()  # from training.db, 50K+ rows
    random.seed(42); random.shuffle(samples)
    n_val = int(len(samples) * 0.1)
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    net = PredictionNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=0.0001)
    best_val_mse = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):    # 50 or 100
        net.train()
        train_loss = run_epoch(net, optimizer, train_samples, BATCH_SIZE)

        net.eval()
        val_tissue_mse, val_damage_acc = evaluate(net, val_samples)
        scheduler.step()

        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "val_tissue_mse": val_tissue_mse,
                   "val_damage_acc": val_damage_acc,
                   "lr": scheduler.get_last_lr()[0]})

        print(f"Epoch {epoch}: loss={train_loss:.4f} "
              f"val_mse={val_tissue_mse:.4f} dam_acc={val_damage_acc:.3f}")

        if val_tissue_mse < best_val_mse - 0.001:
            best_val_mse = val_tissue_mse
            patience_counter = 0
            save_checkpoint(net, optimizer, epoch)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        # Check success criteria
        if val_tissue_mse < 0.05 and val_damage_acc > 0.85:
            print("Training targets met — stopping early")
            break

    load_best_checkpoint(net)
    write_event("training_completed",
                payload_val_mse=best_val_mse,
                payload_dam_acc=val_damage_acc)
    wandb.finish()
```

### Success Criteria

| Metric | Target | Action if not met |
|---|---|---|
| `val_tissue_mse` | < 0.05 | Extend to 100 epochs, lr=0.0001 |
| `val_damage_accuracy` | > 0.85 | Extend; check class balance |
| Both still not met at 100 epochs | — | Proceed with best checkpoint |

Higher initial error = more dramatic agent spawning during demo.
A model that doesn't perfectly meet targets is not a failure —
it produces more visible self-improvement during the pitch.

---

## FINE-TUNING

**File:** `backend/world_model.py` (`_fine_tune_sync` method)

Triggered by `ExplorationAgent` after completing its episode:
```python
# In exploration.py:
final_loss = await world_model.fine_tune(agent.samples)
```

### Parameters

| Parameter | Value | vs Initial Training | Reason |
|---|---|---|---|
| epochs | 5 | 50 | Targeted update, not full retraining |
| batch_size | 64 | 256 | Fine-tune batches are smaller total |
| learning_rate | 0.0001 | 0.001 | 10× smaller — avoid overwriting prior knowledge |
| optimizer | Same Adam instance | New Adam | Preserves momentum state across calls |

### Replay Ratio

**80% old (from replay buffer) / 20% new (from agent).**

Implementation: `n_old = len(new_samples) * 4` old samples drawn
randomly from buffer. If buffer has fewer than `n_old` samples: use
all available old samples (only occurs in the first few fine-tunes).

Why 80/20 and not more new samples: the exploration agent ran 100-500
steps focused on one specific region. Training predominantly on these
samples overwrites weights encoding all other regions learned during
initial training. The model improves on the target region but
regresses everywhere else (catastrophic forgetting). The 80% old
sample injection forces the optimizer to maintain performance on
previously learned regions while absorbing the new data.

Why the same Adam instance: see `_fine_tune_sync` rationale above.

### DOF-Expansion Fine-Tuning (special case)

When `CapabilityAgent` unlocks a new DOF:
```
1. Collect 1,000 steps of DOF-focused data
   (actions that actively use the new DOF)
2. Call world_model.fine_tune(samples)
   but override: epochs=10, n_old=4000
   Reason: new DOF is completely unseen —
   needs more adaptation than regional fine-tuning.
3. Only then: update primary_sim.config.dof and comparison_sim.config.dof
```

If MPC planning is enabled for the new DOF before fine-tuning, the
network predicts zero-movement for that DOF (correct during training)
and the MPC agent never uses it (the value function scores DOF-movement
actions at near-zero since predicted position doesn't change).

---

## MPC PLANNING AGENT

**File:** `backend/mpc_agent.py`

### Candidate Generation

8 candidates per step. All positions clamped to workspace bounds.

```python
def _generate_candidates(self, state_vec: np.ndarray) -> list[np.ndarray]:
    ee = state_vec[768:771]    # normalized EE position
    tgt = state_vec[776:778]   # normalized target position
    # Compute direction toward target (in normalized space)
    diff = tgt - ee[:2]
    norm = np.linalg.norm(diff) + 1e-8
    d = diff / norm            # unit vector toward target

    # Step size in normalized space ≈ 1.5 grid cells / 15 = 0.1
    STEP = 0.1
    candidates = []

    def make_move(dx, dy, dz=0.0, mode=0, magnitude=0.7):
        a = np.zeros(12, dtype=np.float32)
        a[0], a[1], a[2] = dx, dy, dz
        a[7 + mode] = 1.0  # MOVE=0, GRASP=1, CAUTERIZE=2, WAIT=3
        a[11] = magnitude
        return a

    candidates.append(make_move(d[0]*STEP, d[1]*STEP))           # direct
    candidates.append(make_move(d[0]*STEP, d[1]*STEP, magnitude=0.4))  # gentle
    # ±45° flanking moves
    cos45, sin45 = 0.707, 0.707
    candidates.append(make_move(
        d[0]*cos45 - d[1]*sin45, d[0]*sin45 + d[1]*cos45) * STEP)
    candidates.append(make_move(
        d[0]*cos45 + d[1]*sin45, -d[0]*sin45 + d[1]*cos45) * STEP)
    # Cauterize if any bleeding in state
    bleeding_vec = state_vec[512:768]
    if bleeding_vec.max() > 0.5:
        # Find nearest bleeding cell (grid), convert to workspace delta
        candidates.append(make_move(0, 0, mode=2, magnitude=1.0))
    else:
        candidates.append(make_move(d[0]*STEP, d[1]*STEP, dz=0.05))
    # Exploration moves
    candidates.append(make_move(*np.random.uniform(-STEP, STEP, 2)))
    candidates.append(make_move(0, 0, dz=STEP))         # depth probe
    candidates.append(make_move(0, 0, mode=3))           # WAIT
    return candidates[:8]
```

### Value Function

```python
def _compute_value(self,
                   current_state_vec: np.ndarray,
                   predicted: PredictedState) -> float:

    # Progress: how much closer to target does this action get us?
    curr_ee = current_state_vec[768:770]    # normalized xy
    tgt     = current_state_vec[776:778]    # normalized xy
    pred_ee = predicted.pred_ee_norm[:2]    # normalized xy (instrument head)
    curr_dist = float(np.linalg.norm(curr_ee - tgt))
    pred_dist = float(np.linalg.norm(pred_ee - tgt))
    progress  = max(0.0, curr_dist - pred_dist)   # positive = closer
    # Normalize: max single-step = STEP/15 in normalized space ≈ 0.1
    progress_score = progress / 0.1

    # Tissue damage: mean integrity decrease
    curr_mean_integrity = float(current_state_vec[0:256].mean())
    pred_mean_integrity = float(predicted.tissue.mean())
    damage_penalty = max(0.0, curr_mean_integrity - pred_mean_integrity)

    # Vessel risk: mean over all 5 slots
    vessel_penalty = float(predicted.vessel_risk.mean())

    # Hard constraint: any action with >50% damage prob is disqualified
    hard_penalty = 1.0 if predicted.damage_prob > 0.5 else 0.0

    value = (
        0.4 * progress_score
      - 0.3 * damage_penalty
      - 0.2 * vessel_penalty
      - 100.0 * hard_penalty   # dominates all other terms
    )
    return value
```

### MPC Run Loop

```python
async def run(self) -> None:
    while True:
        state_vec = self._env.get_state_vector()
        belief_vec = self._world_model.belief.to_input_vector()

        # Select action (prediction is CPU-bound)
        async with self._world_model._predict_lock:
            candidates = self._generate_candidates(state_vec)
            best_action, best_value = None, float('-inf')
            for action_vec in candidates:
                predicted = await asyncio.to_thread(
                    self._world_model.network.predict, belief_vec, action_vec
                )
                v = self._compute_value(state_vec, predicted)
                if v > best_value:
                    best_value, best_action = v, action_vec

        # Execute best action
        surrol_action = action_to_surrol(best_action, self._world_model.belief.active_dof)
        await asyncio.to_thread(self._env.step, surrol_action)

        # Update world model
        next_state_vec = self._env.get_state_vector()
        self._world_model.update(state_vec, best_action, next_state_vec)

        write_event('simulation_step', sim_id='primary',
                    step=self._step_count,
                    reward=float(self._env._last_reward),
                    tissue_mean=float(next_state_vec[0:256].mean()),
                    vessel_damaged=bool(next_state_vec[779+3::4].max() > 0.5),
                    target_reached=bool(next_state_vec[778] > 0.5),
                    task_failed=bool(self._env._task_failed),
                    active_dof=self._world_model.belief.active_dof,
                    ee_pos=next_state_vec[768:771].tolist())

        if self._env._task_complete or self._env._task_failed:
            await asyncio.to_thread(self._env.reset)
            self._step_count = 0
        else:
            self._step_count += 1

        await asyncio.sleep(0.05)  # yield event loop; 20 steps/sec max
        # DO NOT remove this sleep. Without it, the MPC loop starves
        # WebSocket broadcasts and orchestrator cycles.
```

---

## W&B INTEGRATION

**Install:** `uv add wandb`

Initialize once in `train.py` and once in `main.py`:

```python
# train.py — initial training run
wandb.init(project="soma-surgical",
           name=f"train-{int(time.time())}",
           config={"lr": 0.001, "epochs": 50,
                   "state_dim": 806, "action_dim": 12})

# main.py — operational run (after training)
wandb.init(project="soma-surgical",
           name=f"demo-{int(time.time())}",
           resume="allow")
```

**Log during initial training (every epoch):**
```python
wandb.log({"val_tissue_mse": mse, "val_damage_acc": acc, "epoch": e})
```

**Log during fine-tuning (every fine_tune() call):**
```python
wandb.log({"fine_tune_loss": loss,
           "world_model_version": version,
           "global_error": float(belief.prediction_error_map.mean()),
           "active_agents": len(active_agents)})
```

**Log on capability unlock:**
```python
wandb.log({"active_dof": new_dof, "unlock_error": trigger_error})
```

**W&B URL:** `wandb.run.url` — log to console on startup. Open in a
browser tab during the demo. Judges can see "live training dashboard"
at a public URL. This is independent evidence that learning is
occurring, not just a claim in the pitch.

**If W&B login fails:** wrap all `wandb.*` calls in try/except. Log
a warning but do not crash. The system operates correctly without W&B.

---

## PITFALLS — SYMPTOM FIRST

---

**SYMPTOM: Heatmap shows near-uniform error everywhere, never changes
regardless of agent activity. `wandb` shows decreasing training loss
but the error map stays red.**

CAUSE 1 (most likely): CIRCULAR INPUT INVARIANT violated.
`to_input_vector()` includes `prediction_error_map`. The network
learns to predict its own previous error signal rather than tissue
dynamics. Training loss decreases because this is easy to memorize.
Actual tissue prediction improves 0%.
CONFIRM: Run `test:belief:circular_input`. If it fails, this is the cause.
FIX: Remove all references to `prediction_error_map` and
`confidence_map` from `to_input_vector()`. Retrain from scratch.

CAUSE 2: `update_prediction_error()` called with wrong arguments —
both `predicted` and `actual` are from the network output (no actual
observation). Or called with reversed argument order.
CONFIRM: Add `assert not np.array_equal(predicted, actual)` at the
top of `update_prediction_error`. If it fires: the caller is passing
the same array twice.
FIX: In `WorldModel.update()`, ensure `actual_integrity` is extracted
from `next_state_vec[0:256]` (the observed next state), not from
the network's prediction.

CAUSE 3: World model version never increments — `fine_tune()` is not
completing. Check if `_finetune_lock` is held indefinitely.
CONFIRM: Log `world_model_version` after each agent completes. If it
never increases: fine_tune() is not being called or is deadlocking.
FIX: Check that ExplorationAgent awaits `world_model.fine_tune()` and
handles exceptions without swallowing them.

---

**SYMPTOM: WebSocket clients disconnect during agent completion.
Canvas stops updating for 2-3 minutes then resumes.**

CAUSE: `fine_tune()` running synchronously on the event loop.
The `asyncio.to_thread()` call is missing or incorrectly applied.
CONFIRM: Check `world_model.py` — `_fine_tune_sync` must be called
as `await asyncio.to_thread(self._fine_tune_sync, samples)`, not
as `self._fine_tune_sync(samples)` or `await self._fine_tune_sync(samples)`.
FIX: Wrap `_fine_tune_sync` call in `asyncio.to_thread()`.
`_fine_tune_sync` must be a regular `def`, not `async def`.
See EVENT LOOP NON-BLOCKING INVARIANT in ARCHITECTURE.md.
Test: `test:world_model:nonblocking`

---

**SYMPTOM: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
on first `world_model.predict()` call.**

CAUSE: Input to network is wrong size. Expected 818, got something else.
CONFIRM: `print(np.concatenate([state_vec, action_vec]).shape)`.
Must be `(818,)`. If state_vec is wrong size, see STATE VECTOR
DIMENSION INVARIANT and `test:simulation:state_vector_dim`.
FIX: Check `BeliefState.to_input_vector()` returns exactly 806.
Check `action_vec` is exactly 12. Concatenation must produce 818.

---

**SYMPTOM: After fine-tuning, world model performs WORSE on previously
learned regions while improving on the target region.**

CAUSE: Replay ratio is wrong. Too many new samples, not enough old.
The optimizer overwrites weights encoding past knowledge.
CONFIRM: Print `n_new` and `n_old` in `_fine_tune_sync`.
Should be approximately 4:1 old:new ratio.
FIX: Ensure `n_old = len(new_samples) * 4` and buffer has enough
old samples. If buffer is too small: the initial training samples
must be added to the replay buffer during `train.py`.

---

**SYMPTOM: `RuntimeError: no running event loop` in thread pool
worker when `fine_tune()` is called.**

CAUSE: `_fine_tune_sync` contains an `await` statement. Thread pool
workers run in threads without an event loop. Any coroutine call
inside the thread raises this error.
CONFIRM: Search `_fine_tune_sync` for any `await` keyword.
FIX: Remove all `await` from `_fine_tune_sync`. Any async operation
needed inside training (e.g., writing an event) must be done in the
async caller after `asyncio.to_thread()` returns.

---

**SYMPTOM: W&B `wandb.init()` hangs for >30 seconds at startup.**

CAUSE: W&B attempting to authenticate to cloud service, slow network.
FIX: Either pre-authenticate (`wandb login` before starting) or
set `WANDB_MODE=offline` in `.env`. Offline mode saves logs locally
and syncs later. Offline mode is acceptable for demo — you can sync
after. The dashboard URL still works from a previous run.

---

## TESTING PROTOCOL

Run only these after Tasks 1.3–1.5. Do not rerun full suite.

```bash
# BeliefState
pytest tests/test_belief_state.py -v
pytest tests/test_belief_state.py::test_circular_input -v         # MUST PASS
pytest tests/test_belief_state.py::test_to_input_vector_shape -v
pytest tests/test_belief_state.py::test_ema_update -v
pytest tests/test_belief_state.py::test_regional_errors_keys -v

# PredictionNetwork
pytest tests/test_prediction_net.py -v
pytest tests/test_prediction_net.py::test_layernorm -v            # MUST PASS
pytest tests/test_prediction_net.py::test_forward_shapes -v
pytest tests/test_prediction_net.py::test_inference_time -v

# WorldModel
pytest tests/test_world_model.py -v
pytest tests/test_world_model.py::test_singleton -v               # MUST PASS
pytest tests/test_world_model.py::test_nonblocking -v             # MUST PASS
pytest tests/test_world_model.py::test_replay_ratio -v
pytest tests/test_world_model.py::test_version_increments -v
```

### Minimal Set for Task Completion

**Task 1.3 (belief_state.py)** — these three must pass:
```python
# test:belief:circular_input — see ARCHITECTURE.md CIRCULAR INPUT INVARIANT
# test:belief:to_input_vector_shape
belief = BeliefState(); v = belief.to_input_vector()
assert v.shape == (806,) and v.dtype == np.float32

# test:belief:regional_errors_keys
errors = belief.get_regional_errors()
assert set(errors.keys()) == {
    'upper_left','upper_right','lower_left',
    'lower_right','tool_tissue_boundary','surgical_target_vicinity'
}
assert all(0.0 <= v <= 1.0 for v in errors.values())
```

**Task 1.4 (prediction_net.py)** — these two must pass:
```python
# test:prediction_net:layernorm — see ARCHITECTURE.md LAYERNORM INVARIANT
# test:prediction_net:forward_shapes
net = PredictionNetwork()
x = torch.randn(4, 818)  # batch of 4
tissue, damage, vessel, instrument = net(x)
assert tissue.shape     == (4, 256)
assert damage.shape     == (4, 1)
assert vessel.shape     == (4, 5)
assert instrument.shape == (4, 4)
assert torch.all(tissue >= 0) and torch.all(tissue <= 1)   # sigmoid
assert torch.all(damage >= 0) and torch.all(damage <= 1)   # sigmoid
assert torch.all(vessel >= 0) and torch.all(vessel <= 1)   # sigmoid
```

**Task 1.5 (world_model.py)** — these two must pass:
```python
# test:world_model:singleton
# grep -r "WorldModel(" backend/ must return exactly one hit (main.py)

# test:world_model:nonblocking
# Run the asyncio timing test from ARCHITECTURE.md EVENT LOOP INVARIANT
```

---

## VERIFICATION CHECKLIST

Manual checks before each task is marked complete.

```
BELIEF STATE (Task 1.3):
  □ to_input_vector() returns (806,) float32
  □ to_input_vector() output identical before/after update_prediction_error()
  □ update_prediction_error(ones, zeros): after 10 calls, error_map > 0.5
  □ update_prediction_error(actual=pred): error_map stays near 0
  □ get_regional_errors() returns dict with exactly 6 keys, all in [0,1]
  □ snapshot() returns JSON-serializable dict with 'error_map' as list of lists

PREDICTION NETWORK (Task 1.4):
  □ No BatchNorm1d in model.modules()
  □ Forward pass with (1, 818) input runs without error
  □ All four output head shapes correct
  □ tissue + vessel + damage outputs in (0,1) — sigmoid applied
  □ Inference time < 10ms on target device (check with timeit)
  □ Model saves and loads: weights identical after save→load cycle

WORLD MODEL (Task 1.5):
  □ WorldModel() instantiated exactly once (grep test passes)
  □ predict() returns PredictedState without error
  □ update() updates belief.prediction_error_map (non-zero after mismatch)
  □ fine_tune() completes without blocking event loop (timing test)
  □ world_model_version increments after fine_tune()
  □ Checkpoint saved to models/prediction_network.pt after fine_tune()
  □ wandb.log() called in fine_tune() (check console for W&B output)
```
