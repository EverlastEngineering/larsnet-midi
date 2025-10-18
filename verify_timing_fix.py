#!/usr/bin/env python3
"""Quick verification that timing is now correct"""

import mido

midi_path = "/app/midi/The Fate Of Ophelia.mid"
midi_file = mido.MidiFile(midi_path)

print("🎵 Quick Timing Verification")
print("=" * 60)

# Build tempo map (same as fixed render code)
tempo_map = []
for track in midi_file.tracks:
    absolute_time = 0.0
    current_tempo = 500000
    
    for msg in track:
        if msg.time > 0:
            absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
        
        if msg.type == 'set_tempo':
            tempo_map.append((absolute_time, msg.tempo))
            current_tempo = msg.tempo
            print(f"📍 Tempo at {absolute_time:.3f}s: {mido.tempo2bpm(msg.tempo):.2f} BPM")

tempo_map.sort()
if not tempo_map:
    tempo_map = [(0.0, 500000)]

# Parse notes with tempo map
print(f"\n🎯 Notes around 11.6s:")
for track in midi_file.tracks:
    absolute_time = 0.0
    tempo_idx = 0
    current_tempo = tempo_map[0][1]
    
    for msg in track:
        while (tempo_idx + 1 < len(tempo_map) and 
               absolute_time >= tempo_map[tempo_idx + 1][0] - 0.001):
            tempo_idx += 1
            current_tempo = tempo_map[tempo_idx][1]
        
        if msg.time > 0:
            absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
        
        if msg.type == 'note_on' and msg.velocity > 0:
            if 10.8 <= absolute_time <= 12.5:
                print(f"  {absolute_time:7.3f}s - Note {msg.note:3d} (vel={msg.velocity:3d})")

print(f"\n✅ Using tempo: {mido.tempo2bpm(tempo_map[0][1]):.2f} BPM")
print(f"🎬 Expected at 11.6s in video: Notes should appear ~11.505-11.994s")
