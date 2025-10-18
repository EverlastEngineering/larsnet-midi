#!/usr/bin/env python3
"""
MIDI Timing Diagnostic Tool

Analyzes MIDI file timing and compares with video renderer calculations
to identify timing drift issues.

Usage:
    python diagnose_midi_timing.py midi/The\ Fate\ Of\ Ophelia.mid
"""

import argparse
import mido
from collections import defaultdict


def analyze_midi_timing(midi_path: str, focus_time: float = 11.6):
    """Analyze MIDI file timing in detail"""
    
    print(f"🔍 Analyzing: {midi_path}")
    print("=" * 80)
    
    midi_file = mido.MidiFile(midi_path)
    
    print(f"\n📊 MIDI File Properties:")
    print(f"  Type: {midi_file.type}")
    print(f"  Ticks per beat: {midi_file.ticks_per_beat}")
    print(f"  Number of tracks: {len(midi_file.tracks)}")
    print(f"  Total length: {midi_file.length:.2f} seconds")
    
    # Track all tempo changes
    tempo_changes = []
    
    # Analyze each track separately (important for Type 1 MIDI files)
    all_notes = []
    
    for track_idx, track in enumerate(midi_file.tracks):
        print(f"\n🎵 Track {track_idx}: {track.name}")
        
        current_time_ticks = 0
        current_tempo = 500000  # Default 120 BPM
        absolute_time = 0.0
        
        track_notes = []
        track_tempos = []
        
        for msg in track:
            current_time_ticks += msg.time
            
            if msg.type == 'set_tempo':
                # Calculate absolute time at this point with OLD tempo
                if msg.time > 0:
                    absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
                    current_time_ticks = 0  # Reset for new tempo
                
                bpm = mido.tempo2bpm(msg.tempo)
                track_tempos.append({
                    'time': absolute_time,
                    'tempo': msg.tempo,
                    'bpm': bpm
                })
                print(f"  ⏱️  Tempo change at {absolute_time:.3f}s: {bpm:.2f} BPM (tempo={msg.tempo})")
                current_tempo = msg.tempo
                tempo_changes.append((absolute_time, msg.tempo, bpm))
                
            elif msg.type == 'note_on' and msg.velocity > 0:
                # Calculate time with current tempo
                note_time = absolute_time + mido.tick2second(current_time_ticks, midi_file.ticks_per_beat, current_tempo)
                
                track_notes.append({
                    'time': note_time,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'track': track_idx
                })
                
                # Show notes near focus time
                if abs(note_time - focus_time) < 1.0:
                    print(f"  🎵 Note {msg.note} at {note_time:.3f}s (vel={msg.velocity})")
        
        all_notes.extend(track_notes)
    
    # Sort all notes by time
    all_notes.sort(key=lambda x: x['time'])
    
    print(f"\n\n🎯 Notes around {focus_time}s (±1.0s):")
    print("-" * 80)
    print(f"{'Time (s)':<12} {'Note':<6} {'Velocity':<10} {'Track':<8} {'Delta from focus'}")
    print("-" * 80)
    
    for note in all_notes:
        if abs(note['time'] - focus_time) < 1.0:
            delta = note['time'] - focus_time
            print(f"{note['time']:<12.3f} {note['note']:<6} {note['velocity']:<10} {note['track']:<8} {delta:+.3f}s")
    
    print("\n\n🔬 Tempo Analysis:")
    print("-" * 80)
    if tempo_changes:
        for time, tempo, bpm in tempo_changes:
            print(f"  {time:.3f}s: {bpm:.2f} BPM (tempo={tempo} µs/beat)")
    else:
        print("  No tempo changes found (using default 120 BPM)")
    
    # Test the renderer's calculation method
    print("\n\n🧪 Testing Renderer's Calculation Method:")
    print("-" * 80)
    print("Method 1: Per-track tempo (CURRENT - potentially buggy)")
    
    for track_idx, track in enumerate(midi_file.tracks):
        current_time = 0.0
        tempo = 500000
        
        for msg in track:
            current_time += msg.time
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                time_seconds = mido.tick2second(current_time, midi_file.ticks_per_beat, tempo)
                if abs(time_seconds - focus_time) < 0.5:
                    print(f"  Track {track_idx}: Note {msg.note} at {time_seconds:.3f}s")
    
    print("\nMethod 2: Global tempo map (RECOMMENDED)")
    
    # Build global tempo map
    tempo_map = [(0.0, 500000)]  # Start with default
    current_time_ticks = 0
    absolute_time = 0.0
    current_tempo = 500000
    
    # First pass: build tempo map from ALL tracks
    for track in midi_file.tracks:
        track_time_ticks = 0
        track_absolute_time = 0.0
        track_tempo = 500000
        
        for msg in track:
            track_time_ticks += msg.time
            
            if msg.type == 'set_tempo':
                if msg.time > 0:
                    track_absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, track_tempo)
                    track_time_ticks = 0
                track_tempo = msg.tempo
                # Add to global tempo map if not already there
                if not any(abs(t - track_absolute_time) < 0.001 for t, _ in tempo_map):
                    tempo_map.append((track_absolute_time, track_tempo))
    
    tempo_map.sort()
    print(f"\n  Found {len(tempo_map)} tempo changes:")
    for t, tempo in tempo_map:
        print(f"    {t:.3f}s: {mido.tempo2bpm(tempo):.2f} BPM")
    
    # Second pass: calculate note times with global tempo map
    def get_tempo_at_time(time_seconds):
        """Get the tempo that should be active at a given time"""
        active_tempo = 500000
        for t, tempo in tempo_map:
            if t <= time_seconds:
                active_tempo = tempo
            else:
                break
        return active_tempo
    
    print("\n  Notes calculated with global tempo map:")
    for track_idx, track in enumerate(midi_file.tracks):
        current_time_ticks = 0
        absolute_time = 0.0
        tempo_idx = 0
        current_tempo = tempo_map[0][1]
        
        for msg in track:
            current_time_ticks += msg.time
            
            # Check if we've passed a tempo change
            while tempo_idx + 1 < len(tempo_map) and absolute_time >= tempo_map[tempo_idx + 1][0]:
                tempo_idx += 1
                current_tempo = tempo_map[tempo_idx][1]
            
            if msg.type == 'note_on' and msg.velocity > 0:
                time_seconds = absolute_time + mido.tick2second(current_time_ticks, midi_file.ticks_per_beat, current_tempo)
                if abs(time_seconds - focus_time) < 0.5:
                    print(f"    Track {track_idx}: Note {msg.note} at {time_seconds:.3f}s (tempo={mido.tempo2bpm(current_tempo):.1f} BPM)")
            
            # Update absolute time
            if msg.time > 0:
                absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
                current_time_ticks = 0
    
    return all_notes, tempo_changes


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose MIDI timing issues'
    )
    parser.add_argument('midi_file', help='Input MIDI file path')
    parser.add_argument('--focus-time', '-t', type=float, default=11.6,
                       help='Time to focus analysis on (default: 11.6)')
    
    args = parser.parse_args()
    
    analyze_midi_timing(args.midi_file, args.focus_time)


if __name__ == '__main__':
    main()
