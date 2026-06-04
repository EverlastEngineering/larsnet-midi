# MTL Quad-Head Implementation Plan

> **STATUS: ABANDONED** — This four-head architecture (Gatekeeper / Groupings /
> Precision / Velocity) was attempted in spring 2026 and did not converge to a
> useful model. The work lives on branch `mtl-four-head-failed-approach`
> (commit `90fedcb mtl failed approach, aborting`). A duplicate higher-level
> sketch of this same plan also existed at `model-training/MTL Triple Head Plan.md`
> (mislabeled — internally titled "Quad-Head"); it was deleted during the
> rescue cleanup as redundant with this more detailed version.
>
> What shipped instead was the simpler **dual-head** approach (10 onset
> classification channels + 10 velocity regression channels) — see
> `multi-task-velocity.SHIPPED.plan.md` and commit
> `4a60cc5 successful mtl with velocity`.
>
> Preserved here as a record of one direction explored. A future iteration
> may want to reconsider whether splitting into 3 or 4 heads (per the
> "Gatekeeper veto" logic below) would help with cross-talk between
> frequency-distinct drum groups.

---

## Overview
Transform the single-head `DrumTranscriber` into a multi-task learning architecture with 4 specialized heads:
- Head 1: Gatekeeper (3-class instrument family)
- Head 2: Groupings (10-class drum category)
- Head 3: Precision (24-channel per-Roland-pitch trigger)
- Head 4: Velocity (24-channel per-Roland-pitch dynamics)

Note: 24 unique Roland TD-17 pitches mapped in MAPPING.

---

## Phase 1: Architecture Changes (`model.py`)

### 1.1 Add Head Classes
```python
class GatekeeperHead(nn.Module):
    """Head 1: 3-class instrument family classification (Kick/Snare/Cymbal)"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 3)  # 3 families
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # [B, T, 3]

class GroupingsHead(nn.Module):
    """Head 2: 10-class instrument refinement"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # [B, T, 10]

class PrecisionHead(nn.Module):
    """Head 3: 10-class precise trigger detection"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # [B, T, 10]

class VelocityHead(nn.Module):
    """Head 4: 10-channel velocity regression (0.3-1.0 range)"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
    
    def forward(self, x):
        # No sigmoid - raw regression output
        # Will be scaled to 0.3-1.0 range during loss computation
        return torch.relu(self.fc(x))  # [B, T, 10]
```

### 1.2 Modify DrumTranscriber.__init__
```python
self.gatekeeper = GatekeeperHead(256)
self.groupings = GroupingsHead(256)
self.precision = PrecisionHead(256)
self.velocity = VelocityHead(256)
```

### 1.3 Modify DrumTranscriber.forward
```python
def forward(self, x):
    x = self.conv(x)
    x = x.permute(0, 3, 1, 2).flatten(2)  # [B, T, 2048]
    x, _ = self.rnn(x)  # [B, T, 256]
    
    return {
        'gatekeeper': self.gatekeeper(x),
        'groupings': self.groupings(x),
        'precision': self.precision(x),
        'velocity': self.velocity(x)
    }
```

**Test**: Run model forward pass, verify all 4 dict keys return correct shapes.

---

## Phase 2: Label Encoder Changes (`label_encoder.py`)

### 2.1 Define Category Mapping
```python
# 3 categories for Gatekeeper
CATEGORY_MAP = {
    0: [36, 35],   # Kick
    1: [38, 40, 37, 39, 45, 47, 43, 58, 48, 50],  # Snare/Toms
    2: [42, 44, 22, 46, 26, 49, 55, 52, 57, 51, 53, 59]  # Cymbals
}
```

### 2.2 New Function: `midi_to_multitarget_arrays`
```python
def midi_to_multitarget_arrays(midi_notes, total_frames, hop_length=512, sr=44100):
    """
    Returns dict with 4 targets:
      - 'gatekeeper': [3, T] one-hot category labels
      - 'groupings': [10, T] binary labels (smear)
      - 'precision': [10, T] binary labels (smear)
      - 'velocity': [10, T] velocity-scaled labels (0.3-1.0)
    """
    gatekeeper_labels = torch.zeros((3, total_frames))
    groupings_labels = torch.zeros((10, total_frames))
    precision_labels = torch.zeros((10, total_frames))
    velocity_labels = torch.zeros((10, total_frames))
    
    seconds_per_frame = hop_length / sr
    
    for note in midi_notes:
        if note.pitch in MAPPING:
            hit_frame = int(note.start_time / seconds_per_frame)
            idx = MAPPING[note.pitch]
            
            # Category (Head 1)
            for cat_idx, pitches in enumerate(CATEGORY_MAP.values()):
                if note.pitch in pitches:
                    if 0 <= hit_frame < total_frames:
                        gatekeeper_labels[cat_idx, hit_frame] = 1.0
                    break
            
            # Smear for groupings and precision (causal: 1.0, 0.8, 0.5, 0.2)
            if 0 <= hit_frame < total_frames:
                for offset, val in enumerate([1.0, 0.8, 0.5, 0.2]):
                    f = hit_frame + offset
                    if f < total_frames:
                        groupings_labels[idx, f] = val
                        precision_labels[idx, f] = val
            
            # Velocity-scaled labels (Head 4)
            # Scale: 0.3 (vel=1) to 1.0 (vel=127)
            vel_scale = 0.3 + (note.velocity / 127.0) * 0.7
            if 0 <= hit_frame < total_frames:
                velocity_labels[idx, hit_frame] = vel_scale
```

**Test**: Call with sample MIDI, verify all 4 arrays have correct shapes and velocity values vary.

---

## Phase 3: Training Changes (`smoke_test.py`)

### 3.1 Loss Computation
```python
def compute_mtl_loss(outputs, targets, device):
    """
    Combined loss:
      - BCE for gatekeeper, groupings, precision
      - Masked MSE for velocity (only where precision ground truth is 1.0)
    """
    gatekeeper_loss = nn.BCELoss()(outputs['gatekeeper'], targets['gatekeeper'])
    groupings_loss = nn.BCELoss()(outputs['groupings'], targets['groupings'])
    precision_loss = nn.BCELoss()(outputs['precision'], targets['precision'])
    
    # Masked MSE: only compute where precision target > 0.5
    precision_mask = (targets['precision'] > 0.5).float()
    velocity_loss = nn.MSELoss(reduction='none')(
        outputs['velocity'], targets['velocity']
    )
    masked_velocity_loss = (velocity_loss * precision_mask).sum() / (precision_mask.sum() + 1e-8)
    
    total_loss = gatekeeper_loss + groupings_loss + precision_loss + masked_velocity_loss
    return total_loss, {
        'gatekeeper': gatekeeper_loss.item(),
        'groupings': groupings_loss.item(),
        'precision': precision_loss.item(),
        'velocity': masked_velocity_loss.item()
    }
```

### 3.2 Update Training Loop
- Use `midi_to_multitarget_arrays()` instead of `midi_to_frame_array()`
- Call `compute_mtl_loss()` instead of single BCE

**Test**: Train 1 epoch, verify all 4 loss components are printed.

---

## Phase 4: Inference Changes (`inference.py`)

### 4.1 Update `heatmap_to_notes`
```python
def heatmap_to_notes(prediction: dict, threshold: float = 0.75) -> list:
    """
    prediction: dict with keys 'gatekeeper', 'groupings', 'precision', 'velocity'
    Returns list of (time_seconds, midi_note, velocity) tuples
    """
    gatekeeper = prediction['gatekeeper'][0].cpu().detach().numpy()  # [T, 3]
    precision = prediction['precision'][0].cpu().detach().numpy()      # [T, 10]
    velocity = prediction['velocity'][0].cpu().detach().numpy()       # [T, 10]
    
    # Category index to pitch mapping (simplified)
    category_pitch_map = {
        0: [36],  # Kick
        1: [38],  # Snare
        2: [42, 46, 49, 51]  # Cymbals first
    }
    
    notes = []
    for class_idx in range(10):
        probs = precision[:, class_idx]
        peaks = find_peaks_with_onset_snap(probs, threshold, min_distance=1)
        
        for frame, prob in peaks:
            # Veto: check gatekeeper category probability
            midi_note = INDEX_TO_MIDI[class_idx]
            cat_idx = get_category_for_midi(midi_note)  # helper
            cat_prob = gatekeeper[frame, cat_idx]
            
            if cat_prob < 0.4:  # Gatekeeper veto
                continue
            
            # Get velocity from velocity head
            raw_vel = velocity[frame, class_idx]
            midi_vel = max(1, int(raw_vel * 127))
            
            time_seconds = frame * SECONDS_PER_FRAME
            notes.append((time_seconds, midi_note, midi_vel))
    
    notes.sort(key=lambda x: x[0])
    return notes
```

### 4.2 Update `run_inference`
- Pass `threshold=0.75` to `heatmap_to_notes`
- Velocity now comes from Head 4, not probability × 127

**Test**: Run inference, verify velocity values differ based on audio dynamics.

---

## Phase 5: Integration Testing

### 5.1 Smoke Test
```bash
conda run -n drumtomidi python smoke_test.py --audio dl-1.wav --midi dl-1.mid --epochs 1
```

### 5.2 Verify Outputs
- Model outputs 4 heads
- Loss components decrease over training
- Inference produces notes with varied velocities (not uniform × prob)

### 5.3 Full Training Run
```bash
conda run -n drumtomidi python smoke_test.py --list batch_training_list.txt --epochs 100
```

---

## Success Criteria

1. **Architecture**: Model produces 4-head output (gatekeeper/groupings/precision/velocity)
2. **Loss**: All 4 components computed each epoch, velocity loss is masked
3. **Training**: Loss converges, model learns both detection and velocity
4. **Inference**: Velocity varies with audio dynamics, not uniform
5. **Metrics**: F1 score maintained while velocity accuracy improves

---

## Risk Mitigation

- **Catastrophic forgetting**: Monitor each head's loss independently
- **Velocity collapse**: If all velocities converge to same value, increase learning rate or add dropout to velocity head
- **Category confusion**: Start with higher gatekeeper threshold (0.5) and lower if too many false negatives
