#!/usr/bin/env python3
"""
MIDI to Rock Band-Style Video Renderer

Creates falling notes visualization videos from MIDI drum files, 
perfect for learning to play drums Rock Band style.

Usage:
    python render_midi_to_video.py input.mid --output output.mp4 --fps 60 --width 1920 --height 1080
"""

import argparse
import mido
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os


@dataclass
class DrumNote:
    """Represents a single drum note to be rendered"""
    midi_note: int
    time: float  # seconds
    velocity: int
    lane: int
    color: Tuple[int, int, int]  # BGR color


# Standard GM Drum Map - adjust based on your MIDI files
DRUM_MAP = {
    42: {"name": "Hi-Hat Closed", "lane": 0, "color": (255, 255, 0)},  # Cyan
    44: {"name": "Hi-Hat Pedal", "lane": 1, "color": (200, 200, 0)},   # Dark Cyan
    46: {"name": "Hi-Hat Open", "lane": 2, "color": (255, 200, 0)},    # Light Blue
    38: {"name": "Snare", "lane": 3, "color": (0, 0, 255)},       # Red
    40: {"name": "Snare Rim", "lane": 4, "color": (0, 0, 200)},   # Dark Red
    36: {"name": "Kick", "lane": 5, "color": (0, 255, 255)},      # Yellow
    47: {"name": "Tom 1", "lane": 6, "color": (0, 255, 0)},       # Green
    48: {"name": "Tom 2", "lane": 7, "color": (0, 200, 0)},       # Dark Green
    50: {"name": "Tom 3", "lane": 8, "color": (255, 0, 255)},     # Magenta
    49: {"name": "Left Cymbal", "lane": 9, "color": (255, 80, 0)},     # Dark Orange
    57: {"name": "Right Cymbal", "lane": 10, "color": (255, 100, 0)},     # Orange
    54: {"name": "Ride", "lane": 11, "color": (255, 150, 100)},    # Light Orange
}


class MidiVideoRenderer:
    """Renders MIDI drum files to Rock Band-style falling notes videos"""
    
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 60):
        self.width = width
        self.height = height
        self.fps = fps
        self.num_lanes = len(set(info["lane"] for info in DRUM_MAP.values()))  
        self.note_width = width // self.num_lanes
        self.strike_line_y = int(height * 0.85)  # Where notes are "hit"
        self.note_height = 30  # Height of each note rectangle
        self.pixels_per_second = height * 0.5  # How fast notes fall
        
    def parse_midi(self, midi_path: str) -> Tuple[List[DrumNote], float]:
        """Parse MIDI file and extract drum notes with timing"""
        midi_file = mido.MidiFile(midi_path)
        notes = []
        total_duration = 0.0
        
        # Default tempo (120 BPM = 500000 microseconds per beat)
        current_tempo = 500000
        
        for track in midi_file.tracks:
            current_time = 0.0
            track_tempo = current_tempo  # Each track starts with default tempo
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'set_tempo':
                    track_tempo = msg.tempo
                    
                elif msg.type == 'note_on' and msg.velocity > 0:
                    # Use the current tempo for this track
                    time_seconds = mido.tick2second(current_time, midi_file.ticks_per_beat, track_tempo)
                    
                    if msg.note in DRUM_MAP:
                        drum_info = DRUM_MAP[msg.note]
                        note = DrumNote(
                            midi_note=msg.note,
                            time=time_seconds,
                            velocity=msg.velocity,
                            lane=drum_info["lane"],
                            color=drum_info["color"]
                        )
                        notes.append(note)
                        total_duration = max(total_duration, time_seconds)
        
        # Sort by time
        notes.sort(key=lambda n: n.time)
        
        # Add 3 seconds at the end
        total_duration += 3.0
        
        return notes, total_duration
    
    def draw_lane(self, frame: np.ndarray, lane: int, highlight: bool = False):
        """Draw a single note lane"""
        x = lane * self.note_width + self.note_width // 2
        color = (100, 100, 100) if not highlight else (200, 200, 200)
        cv2.line(frame, (x, 0), (x, self.height), color, 2)
    
    def draw_strike_line(self, frame: np.ndarray):
        """Draw the line where notes are 'hit'"""
        cv2.line(frame, (0, self.strike_line_y), (self.width, self.strike_line_y), 
                 (255, 255, 255), 4)
        
        # Draw lane markers at strike line
        for lane in range(self.num_lanes):
            x = lane * self.note_width + self.note_width // 2
            cv2.circle(frame, (x, self.strike_line_y), 20, (200, 200, 200), 2)
    
    def draw_note(self, frame: np.ndarray, note: DrumNote, current_time: float):
        """Draw a single falling note"""
        time_until_hit = note.time - current_time
        
        if time_until_hit < -0.5:  # Note already hit and passed
            return False
            
        # Calculate position using float math, only round at the end for pixel coordinates
        y_pos_float = self.strike_line_y - (time_until_hit * self.pixels_per_second)
        y_pos = int(round(y_pos_float))
        
        if y_pos < -self.note_height:  # Note not visible yet
            return True
        
        # Calculate alpha based on proximity to strike line
        if abs(time_until_hit) < 0.1:
            alpha = 1.0  # Bright when close
        else:
            alpha = 0.7
        
        # Draw note rectangle
        x = note.lane * self.note_width + 10
        width = self.note_width - 20
        
        # Draw note with velocity-based brightness
        brightness = note.velocity / 127.0
        color = tuple(int(c * brightness) for c in note.color)
        
        cv2.rectangle(frame, 
                     (x, y_pos - self.note_height), 
                     (x + width, y_pos),
                     color, -1)
        
        # Draw outline
        cv2.rectangle(frame, 
                     (x, y_pos - self.note_height), 
                     (x + width, y_pos),
                     (255, 255, 255), 2)
        
        # Highlight when at strike line
        if abs(time_until_hit) < 0.05:
            cv2.circle(frame, 
                      (x + width // 2, self.strike_line_y), 
                      40, color, -1)
            cv2.circle(frame, 
                      (x + width // 2, self.strike_line_y), 
                      40, (255, 255, 255), 3)
        
        return True
    
    def draw_ui(self, frame: np.ndarray, current_time: float, total_time: float):
        """Draw UI elements like time and progress"""
        # Time display
        time_str = f"{current_time:.2f}s / {total_time:.2f}s"
        cv2.putText(frame, time_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Progress bar
        progress = current_time / total_time
        bar_width = self.width - 40
        bar_filled = int(bar_width * progress)
        cv2.rectangle(frame, (20, self.height - 40), 
                     (20 + bar_width, self.height - 20),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (20, self.height - 40), 
                     (20 + bar_filled, self.height - 20),
                     (0, 255, 0), -1)
        
        # Legend
        legend_y = 60
        for note_num, info in sorted(DRUM_MAP.items(), key=lambda x: x[1]["lane"]):
            text = f"{info['name']}"
            cv2.putText(frame, text, (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, info["color"], 2)
            legend_y += 25
    
    def render(self, midi_path: str, output_path: str, show_preview: bool = False):
        """Render MIDI file to video"""
        print(f"Parsing MIDI file: {midi_path}")
        notes, total_duration = self.parse_midi(midi_path)
        print(f"Found {len(notes)} notes, duration: {total_duration:.2f}s")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        total_frames = int(total_duration * self.fps)
        print(f"Rendering {total_frames} frames at {self.fps} FPS...")
        
        # Pre-calculate time step to avoid floating point accumulation errors
        time_step = 1.0 / self.fps
        lookahead_time = 3.0  # Show notes up to 3 seconds in advance
        note_index = 0  # Track which notes we need to check
        
        for frame_num in range(total_frames):
            # Use precise time calculation to avoid drift
            current_time = frame_num * time_step
            
            # Create black frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw lanes
            for lane in range(self.num_lanes):
                self.draw_lane(frame, lane)
            
            # Draw strike line
            self.draw_strike_line(frame)
            
            # Draw visible notes - only check notes in the visible time window
            # Start from first note that hasn't passed completely
            visible_start = note_index
            for i in range(visible_start, len(notes)):
                note = notes[i]
                time_until_hit = note.time - current_time
                
                # Note is too far in the future
                if time_until_hit > lookahead_time:
                    break
                
                # Note has passed - update start index for next frame
                if time_until_hit < -0.5 and i == note_index:
                    note_index = i + 1
                    continue
                    
                self.draw_note(frame, note, current_time)
            
            # Draw UI
            self.draw_ui(frame, current_time, total_duration)
            
            # Write frame
            out.write(frame)
            
            # Show preview
            if show_preview and frame_num % 10 == 0:
                cv2.imshow('Preview', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_num % (self.fps * 5) == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"✅ Video saved to: {output_path}")
        print(f"\nTo add audio, run:")
        print(f"ffmpeg -i {output_path} -i input/your_audio.wav -c:v copy -c:a aac -shortest output_with_audio.mp4")


def main():
    parser = argparse.ArgumentParser(
        description='Render MIDI drum files to Rock Band-style falling notes videos'
    )
    parser.add_argument('midi_file', help='Input MIDI file path')
    parser.add_argument('--output', '-o', default='drums_video.mp4',
                       help='Output video file path (default: drums_video.mp4)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height (default: 1080)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--preview', action='store_true',
                       help='Show live preview while rendering')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.midi_file):
        print(f"❌ Error: MIDI file not found: {args.midi_file}")
        return 1
    
    renderer = MidiVideoRenderer(width=args.width, height=args.height, fps=args.fps)
    renderer.render(args.midi_file, args.output, show_preview=args.preview)
    
    return 0


if __name__ == '__main__':
    exit(main())
