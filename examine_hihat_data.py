#!/usr/bin/env python
"""Examine hihat detection data from analysis JSON"""
import json
import sys

# Load analysis file
with open('user_files/14 - AC_DC_Thunderstruck_Drums/midi/AC_DC_Thunderstruck_Drums.analysis.json', 'r') as f:
    data = json.load(f)

print('=== Analysis File Overview ===')
print(f"Version: {data.get('version', 'unknown')}")
print(f"Tempo: {data.get('tempo_bpm', 'unknown')} BPM")

# Get hihat events
hihat_data = data.get('stems', {}).get('hihat', {})
events = hihat_data.get('events', [])
logic = hihat_data.get('logic', {})

print(f"Total hihat events: {len(events)}")
print(f"Geomean threshold: {logic.get('geomean_threshold', 'unknown')}")
print(f"Min sustain (ms): {logic.get('min_sustain_ms', 'unknown')}")

print(f"\n=== First 20 Events ===")
for i, evt in enumerate(events[:20]):
    time = evt.get('time', 0)
    status = evt.get('status', 'unknown')
    body = evt.get('body_energy', 0)
    sizzle = evt.get('sizzle_energy', 0)
    geomean = (body * sizzle) ** 0.5 if body > 0 and sizzle > 0 else 0
    sustain = evt.get('sustain_ms', 0)
    velocity = evt.get('velocity', 0)
    note = evt.get('note', 0)
    print(f"{i+1:3d}. t={time:7.3f}s  {status:8s}  note={note:2d}  vel={velocity:3d}  geomean={geomean:6.1f}  body={body:.1f}  sizzle={sizzle:.1f}  sustain={sustain:.1f}ms")

# Count by status
kept = [e for e in events if e.get('status') == 'KEPT']
rejected = [e for e in events if e.get('status') == 'REJECTED']

print(f"\n=== Status Summary ===")
print(f"Total detected: {len(events)}")
print(f"Kept: {len(kept)} ({100*len(kept)/len(events) if events else 0:.1f}%)")
print(f"Rejected: {len(rejected)} ({100*len(rejected)/len(events) if events else 0:.1f}%)")

# Analyze note distribution for kept events
note_counts = {}
for evt in kept:
    note = evt.get('note', 0)
    note_counts[note] = note_counts.get(note, 0) + 1

print(f"\n=== Note Distribution (Kept Events) ===")
note_names = {42: 'Closed HH', 46: 'Open HH', 44: 'Pedal HH'}
for note in sorted(note_counts.keys()):
    count = note_counts[note]
    name = note_names.get(note, f'Note {note}')
    print(f"{name:15s} (note {note:2d}): {count:3d} events ({100*count/len(kept) if kept else 0:.1f}%)")

# Show some rejected events
if rejected:
    print(f"\n=== Sample of Rejected Events (first 15) ===")
    for i, evt in enumerate(rejected[:15]):
        time = evt.get('time', 0)
        body = evt.get('body_energy', 0)
        sizzle = evt.get('sizzle_energy', 0)
        geomean = (body * sizzle) ** 0.5 if body > 0 and sizzle > 0 else 0
        sustain = evt.get('sustain_ms', 0)
        print(f"{i+1:3d}. t={time:7.3f}s  geomean={geomean:6.1f}  body={body:.1f}  sizzle={sizzle:.1f}  sustain={sustain:.1f}ms")
    
    rejected_geomeans = [(e.get('body_energy', 0) * e.get('sizzle_energy', 0)) ** 0.5 for e in rejected]
    rejected_geomeans = [g for g in rejected_geomeans if g > 0]
    if rejected_geomeans:
        print(f"\nRejected geomean range: {min(rejected_geomeans):.1f} - {max(rejected_geomeans):.1f}")
