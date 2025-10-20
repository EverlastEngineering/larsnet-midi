#!/usr/bin/env python3
"""
MIDI to Rock Band-Style Video Renderer

Creates falling notes visualization videos from MIDI drum files, 
perfect for learning to play drums Rock Band style.

Uses project-based workflow: automatically detects projects with MIDI files
and renders videos to the project/video/ directory.

Usage:
    python render_midi_to_video.py              # Auto-detect project
    python render_midi_to_video.py 1            # Render specific project
    python render_midi_to_video.py --fps 60     # Custom settings
"""

import argparse
import mido
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import sys
from pathlib import Path

# Import project manager
from project_manager import (
    discover_projects,
    select_project,
    get_project_by_number,
    update_project_metadata,
    USER_FILES_DIR
)


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
        
        # Build global tempo map from ALL tracks (critical for Type 1 MIDI files)
        # In Type 1 MIDI, Track 0 usually contains tempo, but notes are in other tracks
        tempo_map = []  # List of (absolute_time_seconds, tempo) tuples
        
        for track in midi_file.tracks:
            current_time_ticks = 0
            absolute_time = 0.0
            current_tempo = 500000  # Default 120 BPM
            
            for msg in track:
                # Update absolute time BEFORE processing the message
                if msg.time > 0:
                    absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
                
                if msg.type == 'set_tempo':
                    # Record this tempo change
                    tempo_map.append((absolute_time, msg.tempo))
                    current_tempo = msg.tempo
        
        # Sort tempo map by time and remove duplicates
        tempo_map.sort()
        if not tempo_map:
            tempo_map = [(0.0, 500000)]  # Default to 120 BPM if no tempo found
        
        # Remove duplicate tempo changes at same time (keep last one)
        unique_tempo_map = []
        for i, (time, tempo) in enumerate(tempo_map):
            if i == 0 or abs(time - tempo_map[i-1][0]) > 0.001:
                unique_tempo_map.append((time, tempo))
            else:
                # Replace previous if at same time
                unique_tempo_map[-1] = (time, tempo)
        tempo_map = unique_tempo_map
        
        # Now parse notes using the global tempo map
        for track in midi_file.tracks:
            absolute_time = 0.0
            tempo_idx = 0
            current_tempo = tempo_map[0][1]
            
            for msg in track:
                # Check if we need to advance to next tempo change
                while (tempo_idx + 1 < len(tempo_map) and 
                       absolute_time >= tempo_map[tempo_idx + 1][0] - 0.001):
                    tempo_idx += 1
                    current_tempo = tempo_map[tempo_idx][1]
                
                # Calculate time delta and add to absolute time
                if msg.time > 0:
                    absolute_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, current_tempo)
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    if msg.note in DRUM_MAP:
                        drum_info = DRUM_MAP[msg.note]
                        note = DrumNote(
                            midi_note=msg.note,
                            time=absolute_time,
                            velocity=msg.velocity,
                            lane=drum_info["lane"],
                            color=drum_info["color"]
                        )
                        notes.append(note)
                        total_duration = max(total_duration, absolute_time)
        
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


def render_project_video(
    project: dict,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    preview: bool = False
):
    """
    Render MIDI to video for a specific project.
    
    Args:
        project: Project info dictionary from project_manager
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        preview: Show live preview while rendering
    """
    project_dir = project["path"]
    
    print(f"\n{'='*60}")
    print(f"Rendering Video - Project {project['number']}: {project['name']}")
    print(f"{'='*60}\n")
    
    # Find MIDI files in project/midi/ directory
    midi_dir = project_dir / "midi"
    if not midi_dir.exists():
        print(f"ERROR: No midi/ directory found in project.")
        print("Run stems_to_midi.py first!")
        sys.exit(1)
    
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"ERROR: No MIDI files found in {midi_dir}")
        print("Run stems_to_midi.py first!")
        sys.exit(1)
    
    # Use first MIDI file (or could prompt user if multiple)
    midi_file = midi_files[0]
    if len(midi_files) > 1:
        print(f"Found {len(midi_files)} MIDI files, using: {midi_file.name}")
    else:
        print(f"Using MIDI file: {midi_file.name}")
    
    # Output to project/video/ directory
    video_dir = project_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = video_dir / f"{midi_file.stem}.mp4"
    
    print(f"Rendering video to: {output_file}")
    print(f"Settings: {width}x{height} @ {fps}fps")
    if preview:
        print("Preview mode enabled")
    print()
    
    # Render video
    renderer = MidiVideoRenderer(width=width, height=height, fps=fps)
    renderer.render(str(midi_file), str(output_file), show_preview=preview)
    
    # Update project metadata
    update_project_metadata(project_dir, {
        "status": {
            "separated": project["metadata"]["status"].get("separated", False) if project["metadata"] else False,
            "cleaned": project["metadata"]["status"].get("cleaned", False) if project["metadata"] else False,
            "midi_generated": project["metadata"]["status"].get("midi_generated", False) if project["metadata"] else False,
            "video_rendered": True
        }
    })
    
    print(f"\n✓ Video rendering complete!")
    print(f"  Video saved to: {output_file}")
    print(f"  Project status updated\n")


def main():
    parser = argparse.ArgumentParser(
        description='Render MIDI drum files to Rock Band-style falling notes videos',
        epilog="""
Examples:
  python render_midi_to_video.py              # Auto-detect project
  python render_midi_to_video.py 1            # Render specific project
  python render_midi_to_video.py --fps 60     # 60 FPS rendering
  python render_midi_to_video.py --preview    # Show live preview
        """
    )
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                       help='Project number to process (optional)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Video height (default: 1080)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--preview', action='store_true',
                       help='Show live preview while rendering')
    
    args = parser.parse_args()
    
    # Select project
    if args.project_number is not None:
        project = get_project_by_number(args.project_number, USER_FILES_DIR)
        if project is None:
            print(f"ERROR: Project {args.project_number} not found")
            sys.exit(1)
    else:
        # Auto-select project
        project = select_project(None, USER_FILES_DIR, allow_interactive=True)
        if project is None:
            print("\nNo projects found in user-files/")
            print("Run separate.py and stems_to_midi.py first!")
            sys.exit(0)
    
    # Check that project has MIDI files
    has_midi = (project["path"] / "midi").exists()
    
    if not has_midi:
        print(f"\nERROR: Project {project['number']} has no MIDI files.")
        print("Run stems_to_midi.py first!")
        sys.exit(1)
    
    # Render video
    render_project_video(
        project=project,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=args.preview
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
