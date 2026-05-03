# Plan: Velocity-1 Proofing MIDI Workflow

*(Extracted from plan-interactiveTuning.prompt.md Step 6)*

## Objective

Add a "Proofing Export" button that generates a MIDI file with configured events at normal velocity + all sensitive-only events at velocity 1. User edits in DAW (delete noise, promote good hits), re-imports. Store the proofed MIDI alongside the original analysis.json as a labeled training pair (`{stem}.proofed.mid` + `{stem}.analysis.json`). Build a simple schema for this dataset.

## Training Data Format

Each proofed pair (analysis.json + proofed.mid) needs a standard schema: audio hash (for dedup), stem type, config snapshot, original event count, proofed event count, per-event label (true_positive / false_positive / false_negative). This should be defined early so community contributions are consistent.
