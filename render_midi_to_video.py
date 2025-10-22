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
import mido # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageFilter # type: ignore
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Import project manager
from project_manager import (
    discover_projects,
    select_project,
    get_project_by_number,
    update_project_metadata,
    USER_FILES_DIR
)


# ============================================================================
# FUNCTIONAL CORE: Pure functions for calculations
# ============================================================================

def calculate_note_alpha(time_until_hit: float, y_pos: float, strike_line_y: float, height: float) -> float:
    """Pure function to calculate note transparency based on timing and position.
    
    Args:
        time_until_hit: Seconds until note reaches strike line (negative = after hit)
        y_pos: Current y position of note (pixels)
        strike_line_y: Y position of strike line (pixels)
        height: Total screen height (pixels)
    
    Returns:
        Alpha multiplier from 0.0 to 1.0
    """
    # Before strike line: always fully opaque
    if time_until_hit >= 0:
        return 1.0
    
    # After strike line: fade from 100% to 20% as note travels to bottom
    distance_after_strike = y_pos - strike_line_y
    max_distance = height - strike_line_y
    fade_progress = min(distance_after_strike / max_distance, 1.0)
    return 1.0 - (0.8 * fade_progress)  # 1.0 → 0.2


def calculate_brightness(velocity: int) -> float:
    """Pure function: Convert MIDI velocity to brightness factor"""
    return velocity / 127.0


def apply_brightness_to_color(color: Tuple[int, int, int], brightness: float) -> Tuple[int, int, int]:
    """Pure function: Apply brightness factor to RGB color"""
    return tuple(int(c * brightness) for c in color)


def get_brighter_outline_color(base_color: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int, int]:
    """Pure function: Calculate brighter outline color from base color"""
    # Brighten each channel by adding 80% of remaining headroom to 255
    bright_color = tuple(min(255, int(c + (255 - c) * 0.8)) for c in base_color)
    return (*bright_color, alpha)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGBA) to OpenCV array (BGR) with proper alpha compositing"""
    # If image has alpha channel, composite it onto black background
    if pil_image.mode == 'RGBA':
        # Create black background with alpha
        background = Image.new('RGBA', pil_image.size, (0, 0, 0, 255))
        # Composite using alpha blending
        composited = Image.alpha_composite(background, pil_image)
        # Convert to RGB for OpenCV
        pil_image = composited.convert('RGB')
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV array (BGR) to PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def draw_rounded_rectangle(draw: ImageDraw.ImageDraw, 
                           xy: Tuple[int, int, int, int], 
                           radius: int,
                           fill: Optional[Tuple[int, int, int, int]] = None,
                           outline: Optional[Tuple[int, int, int, int]] = None,
                           width: int = 1):
    """Draw anti-aliased rounded rectangle using PIL"""
    if radius <= 0:
        if fill:
            draw.rectangle(xy, fill=fill)
        if outline:
            draw.rectangle(xy, outline=outline, width=width)
        return
    
    # Use PIL's built-in rounded rectangle for better anti-aliasing
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


@dataclass
class DrumNote:
    """Represents a single drum note to be rendered"""
    midi_note: int
    time: float  # seconds
    velocity: int
    lane: int
    color: Tuple[int, int, int]  # BGR color


# Standard GM Drum Map - adjust based on your MIDI files
# Each MIDI note maps to a list of lane definitions (most have 1, but some can have multiple)
# Kick drum (36) uses lane -1 to indicate it's drawn as a screen-wide bar
DRUM_MAP = {
    42: [{"name": "Hi-Hat Closed", "lane": 0, "color": (255, 255, 0)}],  # Cyan
    46: [{"name": "Hi-Hat Open", "lane": 1, "color": (80, 255, 30)}],     # Light Blue 
    38: [{"name": "Snare", "lane": 2, "color": (0, 0, 255)}],       # Red
    40: [{"name": "Snare Rim", "lane": 2, "color": (255, 0, 255)}],   # Dark Red
    39: [{"name": "Clap", "lane": 3, "color": (128, 128, 255)}],
    49: [{"name": "Left Cymbal", "lane": 4, "color": (255, 80, 0)}],     # Dark Orange
    47: [{"name": "Tom 1", "lane": 5, "color": (0, 255, 0)}],       # Green
    48: [{"name": "Tom 2", "lane": 6, "color": (0, 200, 0)}],       # Dark Green
    50: [{"name": "Tom 3", "lane": 7, "color": (255, 0, 255)}],     # Magenta
    36: [{"name": "Kick", "lane": -1, "color": (0, 255, 255)}],     # Yellow - Special: screen-wide bar
    57: [{"name": "Right Cymbal", "lane": 8, "color": (255, 100, 0)}],     # Orange
    54: [{"name": "Ride", "lane": 9, "color": (255, 150, 100)}],    # Light Orange
}


class MidiVideoRenderer:
    """Renders MIDI drum files to Rock Band-style falling notes videos"""
    
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 60, fall_speed_multiplier: float = 1.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.fall_speed_multiplier = fall_speed_multiplier
        # Count lanes excluding kick drum (lane -1)
        self.num_lanes = len(set(info["lane"] for lane_list in DRUM_MAP.values() for info in lane_list if info["lane"] >= 0))  
        self.note_width = width // self.num_lanes
        self.strike_line_y = int(height * 0.85)  # Where notes are "hit"
        self.note_height = 30  # Height of each note rectangle
        self.kick_bar_height = 15  # Height of kick drum bar
        self.pixels_per_second = height * 0.2 * fall_speed_multiplier  # How fast notes fall
        self.corner_radius = 8  # Rounded corners for anti-aliasing
        self.motion_blur_strength = 2  # Pixels of motion blur
        
        # Load font for UI
        self.font = self._load_font(24)
        self.font_small = self._load_font(16)
    
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load system font or fall back to default"""
        font_paths = [
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
            'C:\\Windows\\Fonts\\arialbd.ttf',  # Windows
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        return ImageFont.load_default()
        
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
                        # Create a note for each lane definition (most notes have 1, some have multiple)
                        for drum_info in DRUM_MAP[msg.note]:
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
    
    def draw_note(self, draw: ImageDraw.ImageDraw, note: DrumNote, current_time: float, draw_kick_only: bool = False, skip_highlight: bool = False, first_kick_frame: set = None) -> bool:
        """Draw a single falling note with anti-aliasing and motion blur
        
        Args:
            draw: ImageDraw instance to draw on
            note: Note to draw
            current_time: Current time in seconds
            draw_kick_only: If True, only draw kick drum notes; if False, skip kick drum notes
            first_kick_frame: Set tracking which kick notes are showing for first time at strike line
        """
        time_until_hit = note.time - current_time
        
        # Calculate position using float math, only round at the end for pixel coordinates
        y_pos_float = self.strike_line_y - (time_until_hit * self.pixels_per_second)
        y_pos = int(round(y_pos_float))
        
        # Calculate alpha and brightness using functional core
        alpha_factor = calculate_note_alpha(time_until_hit, y_pos, self.strike_line_y, self.height)
        brightness = calculate_brightness(note.velocity)
        base_color = apply_brightness_to_color(note.color, brightness)
        alpha = int(255 * alpha_factor)
        outline_color = get_brighter_outline_color(base_color, alpha)
        
        # Kick drum (lane -1) is drawn as screen-wide bar
        if note.lane == -1:
            # Skip if we're not drawing kick drums
            if not draw_kick_only:
                return True
            # Note has passed off the bottom of the screen
            if y_pos > self.height + self.kick_bar_height:
                return False
            
            if y_pos < -self.kick_bar_height:  # Note not visible yet
                return True
            
            # Draw kick with slight motion blur
            for i in range(self.motion_blur_strength):
                blur_alpha = int(alpha * (1.0 - i / (self.motion_blur_strength + 1)))
                y_offset = i * 2
                
                draw_rounded_rectangle(draw,
                    (0, y_pos - self.kick_bar_height + y_offset, 
                     self.width, y_pos + y_offset),
                    3,
                    fill=(*base_color, blur_alpha))
            
            # Main bar with rounded corners
            draw_rounded_rectangle(draw,
                (0, y_pos - self.kick_bar_height, self.width, y_pos),
                3,
                fill=(*base_color, alpha),
                outline=outline_color,
                width=2)
            
            # Highlight when at strike line
            if abs(time_until_hit) < 0.05:
                # Check if this is the first frame of this kick at strike line
                note_id = (note.time, note.lane)
                is_first_frame = False
                if first_kick_frame is not None:
                    is_first_frame = note_id not in first_kick_frame
                    if is_first_frame:
                        first_kick_frame.add(note_id)
                
                # Use white on first frame, normal color after
                if is_first_frame:
                    highlight_color = (255, 255, 255, alpha)
                else:
                    highlight_color = (*base_color, alpha)
                
                # Draw thicker bar when hitting
                bright_outline = get_brighter_outline_color(base_color, 255)
                draw_rounded_rectangle(draw,
                    (0, self.strike_line_y - self.kick_bar_height - 5,
                     self.width, self.strike_line_y + 5),
                    5,
                    fill=highlight_color,
                    outline=bright_outline,
                    width=3)
        
            return True
        
        # Skip kick drums if we're drawing kick only
        if draw_kick_only:
            return True
        
        # Regular lane notes
        # Note has passed off the bottom of the screen
        if y_pos > self.height + self.note_height:
            return False
        
        if y_pos < -self.note_height:  # Note not visible yet
            return True
        
        # Calculate note position
        x = note.lane * self.note_width + 10
        width = self.note_width - 20
        
        # Clip the portion of note that would be behind strike line
        note_top = y_pos - self.note_height
        note_bottom = y_pos
        
        # Hide note when it's within the highlight zone (pixel-based, independent of fall speed)
        # Highlight zone is 1.0x note height centered on strike line
        highlight_zone_height = int(self.note_height * 1.0)
        highlight_zone_start = self.strike_line_y - highlight_zone_height // 2
        highlight_zone_end = self.strike_line_y + highlight_zone_height // 2
        
        # Hide note when its center is within the highlight zone
        note_center_y = (note_top + note_bottom) // 2
        if highlight_zone_start <= note_center_y <= highlight_zone_end:
            return True
        
        # Draw motion blur trail
        for i in range(self.motion_blur_strength):
            blur_alpha = int(alpha * 0.3 * (1.0 - i / (self.motion_blur_strength + 1)))
            y_offset = i * 3
            
            draw_rounded_rectangle(draw,
                (x, note_top + y_offset, x + width, note_bottom + y_offset),
                self.corner_radius,
                fill=(*base_color, blur_alpha))
        
        # Outer glow for outline (wider and more visible)
        glow_color = get_brighter_outline_color(base_color, int(alpha * 0.6))
        draw_rounded_rectangle(draw,
            (x - 2, note_top - 2, x + width + 2, note_bottom + 2),
            self.corner_radius + 2,
            outline=glow_color,
            width=4)
        
        # Main note with rounded corners and anti-aliasing
        draw_rounded_rectangle(draw,
            (x, note_top, x + width, note_bottom),
            self.corner_radius,
            fill=(*base_color, alpha),
            outline=outline_color,
            width=2)
        
        return True
    
    def should_draw_highlight(self, note: DrumNote, current_time: float) -> bool:
        """Check if highlight circle should be drawn for this note (pixel-based)"""
        time_until_hit = note.time - current_time
        y_pos = int(round(self.strike_line_y - (time_until_hit * self.pixels_per_second)))
        note_top = y_pos - self.note_height
        note_bottom = y_pos
        note_center_y = (note_top + note_bottom) // 2
        
        # Show highlight when note center is within 1.0x note height of strike line
        highlight_zone_height = int(self.note_height * 1.0)
        highlight_zone_start = self.strike_line_y - highlight_zone_height // 2
        highlight_zone_end = self.strike_line_y + highlight_zone_height // 2
        
        return highlight_zone_start <= note_center_y <= highlight_zone_end
    
    def draw_highlight_circle(self, draw: ImageDraw.ImageDraw, note: DrumNote, current_time: float, first_highlight_frame: set):
        """Draw the strike line highlight circle for a note
        
        Args:
            draw: PIL ImageDraw object
            note: DrumNote to draw
            current_time: Current time in seconds
            first_highlight_frame: Set tracking which notes are showing highlight for first time
        """
        if note.lane == -1:  # Skip kick drums
            return
        
        time_until_hit = note.time - current_time
        alpha_factor = calculate_note_alpha(time_until_hit, self.strike_line_y, self.strike_line_y, self.height)
        brightness = calculate_brightness(note.velocity)
        base_color = apply_brightness_to_color(note.color, brightness)
        alpha = int(255 * alpha_factor)
        
        x = note.lane * self.note_width + 10
        width = self.note_width - 20
        center_x = x + width // 2
        
        # Create unique ID for this note (time + lane should be unique enough)
        note_id = (note.time, note.lane)
        
        # Check if this is the first frame this highlight appears
        is_first_frame = note_id not in first_highlight_frame
        if is_first_frame:
            first_highlight_frame.add(note_id)
            fill_color = (255, 255, 255, alpha)  # White flash on first frame
        else:
            fill_color = (*base_color, alpha)  # Normal color after first frame
        
        bright_outline = get_brighter_outline_color(base_color, 255)
        
        draw.ellipse(
            [center_x - 40, self.strike_line_y - 40,
             center_x + 40, self.strike_line_y + 40],
            fill=fill_color,
            outline=bright_outline,
            width=3)
    
    def draw_ui(self, draw: ImageDraw.ImageDraw, current_time: float, total_time: float):
        """Draw UI elements with glassy transparent progress bar at top and legend at bottom"""
        
        # === Progress bar at top with glassy effect ===
        progress = current_time / total_time
        bar_height = 50
        bar_margin = 20
        
        # Glassy background (60% transparent)
        draw_rounded_rectangle(draw,
            (bar_margin, bar_margin, 
             self.width - bar_margin, bar_margin + bar_height),
            15,
            fill=(20, 20, 20, 153))
        
        # Progress fill with gradient effect
        bar_filled_width = int((self.width - 2 * bar_margin - 20) * progress)
        if bar_filled_width > 0:
            # Main progress bar
            draw_rounded_rectangle(draw,
                (bar_margin + 10, bar_margin + 10,
                 bar_margin + 10 + bar_filled_width, bar_margin + bar_height - 10),
                10,
                fill=(0, 200, 255, 153))
            
            # Highlight on top for glassy effect
            draw_rounded_rectangle(draw,
                (bar_margin + 10, bar_margin + 10,
                 bar_margin + 10 + bar_filled_width, bar_margin + 20),
                8,
                fill=(100, 220, 255, 100))
        
        # === Legend at bottom with glassy background ===
        legend_height = 60
        legend_y = self.height - legend_height - 10
        
        # Glassy background for legend
        draw_rounded_rectangle(draw,
            (10, legend_y, self.width - 10, self.height - 10),
            15,
            fill=(20, 20, 20, 180))
        
        # Draw legend items horizontally across the bottom
        sorted_drums = sorted(DRUM_MAP.items(), key=lambda x: x[1][0]["lane"])
        item_width = (self.width - 40) // len(sorted_drums)
        
        for idx, (note_num, lane_list) in enumerate(sorted_drums):
            info = lane_list[0]
            x_pos = 20 + (idx * item_width)
            y_pos = legend_y + 15
            
            # Color indicator circle
            circle_size = 12
            draw.ellipse(
                [x_pos, y_pos, x_pos + circle_size, y_pos + circle_size],
                fill=(*info["color"], 255),
                outline=(255, 255, 255, 255),
                width=2)
            
            # Instrument name
            text = info['name']
            # Shorten long names
            if len(text) > 12:
                text = text[:10] + "."
            
            # Text shadow
            draw.text((x_pos + circle_size + 7, y_pos - 1), text,
                     font=self.font_small, fill=(0, 0, 0, 200))
            # Text
            draw.text((x_pos + circle_size + 5, y_pos - 2), text,
                     font=self.font_small, fill=(255, 255, 255, 255))
    
    def render(self, midi_path: str, output_path: str, show_preview: bool = False, audio_path: Optional[str] = None):
        """Render MIDI file to video
        
        Args:
            midi_path: Path to MIDI file
            output_path: Path to output video file
            show_preview: Show live preview window
            audio_path: Optional path to audio file to include in video
        """
        print(f"Status Update: Rendering Video")
        print(f"Parsing MIDI file: {midi_path}")
        notes, total_duration = self.parse_midi(midi_path)
        print(f"Found {len(notes)} notes, duration: {total_duration:.2f}s")
        
        total_frames = int(total_duration * self.fps)
        print(f"Rendering {total_frames} frames at {self.fps} FPS...")
        
        # Setup FFmpeg pipe for H.264 encoding with web optimization
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Read video from stdin
        ]
        
        # Add audio input if provided
        if audio_path:
            print(f"Including audio from: {audio_path}")
            ffmpeg_cmd.extend(['-i', audio_path])
            # Map video from stdin and audio from file
            ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
            # Use shortest stream (in case audio is longer/shorter than video)
            ffmpeg_cmd.append('-shortest')
        else:
            ffmpeg_cmd.append('-an')  # No audio
        
        # Video encoding settings
        ffmpeg_cmd.extend([
            '-vcodec', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
        ])
        
        # Audio encoding settings (if audio is included)
        if audio_path:
            ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        
        # Web optimization
        ffmpeg_cmd.extend(['-movflags', '+faststart', output_path])
        
        print(f"Starting FFmpeg encoder for H.264 output...")
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                         stdout=subprocess.DEVNULL, 
                                         stderr=subprocess.PIPE)
        
        # Pre-calculate time step to avoid floating point accumulation errors
        time_step = 1.0 / self.fps
        # Calculate lookahead time based on screen height and fall speed
        # Notes should appear at top of screen (y=0) and fall to strike line
        # Distance to travel = strike_line_y pixels
        # Time needed = distance / speed
        lookahead_time = self.strike_line_y / self.pixels_per_second
        note_index = 0  # Track which notes we need to check
        first_highlight_frame = set()  # Track which notes are showing highlight for first time
        
        for frame_num in range(total_frames):
            # Use precise time calculation to avoid drift
            current_time = frame_num * time_step
            
            # Create base layer with opaque black background
            base_layer = Image.new('RGB', (self.width, self.height), (0, 0, 0))
            
            # Create kick drum layer (rendered first, below everything)
            kick_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            kick_draw = ImageDraw.Draw(kick_layer, 'RGBA')
            
            # Create notes layer with transparency
            notes_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            notes_draw = ImageDraw.Draw(notes_layer, 'RGBA')
            
            # Draw lanes on notes layer
            for lane in range(self.num_lanes):
                x = lane * self.note_width + self.note_width // 2
                notes_draw.line([(x, 0), (x, self.height)], fill=(80, 80, 80, 255), width=1)
            
            # Draw visible notes - only check notes in the visible time window
            # Start from first note that hasn't passed completely
            # Calculate time needed for note to pass off bottom of screen
            passthrough_time = (self.height - self.strike_line_y + self.note_height) / self.pixels_per_second
            
            visible_start = note_index
            for i in range(visible_start, len(notes)):
                note = notes[i]
                time_until_hit = note.time - current_time
                
                # Note is too far in the future
                if time_until_hit > lookahead_time:
                    break
                
                # Note has passed off bottom of screen - update start index for next frame
                if time_until_hit < -passthrough_time and i == note_index:
                    note_index = i + 1
                    continue
                
                # Draw kick drums on kick layer, regular notes on notes layer
                if note.lane == -1:
                    self.draw_note(kick_draw, note, current_time, draw_kick_only=True, first_kick_frame=first_highlight_frame)
                else:
                    self.draw_note(notes_draw, note, current_time, draw_kick_only=False)
            
            # Create strike line layer (rendered on top of everything)
            strike_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            strike_draw = ImageDraw.Draw(strike_layer, 'RGBA')
            
            # Draw highlight circles for notes at strike line (before strike line itself)
            for i in range(visible_start, len(notes)):
                note = notes[i]
                time_until_hit = note.time - current_time
                
                if time_until_hit > lookahead_time:
                    break
                
                if self.should_draw_highlight(note, current_time):
                    self.draw_highlight_circle(strike_draw, note, current_time, first_highlight_frame)
            
            # Draw strike line
            strike_draw.line([(0, self.strike_line_y), (self.width, self.strike_line_y)], 
                     fill=(255, 255, 255, 255), width=4)
            
            # Draw lane markers at strike line
            for lane in range(self.num_lanes):
                x = lane * self.note_width + self.note_width // 2
                strike_draw.ellipse([x - 20, self.strike_line_y - 20, x + 20, self.strike_line_y + 20],
                           outline=(200, 200, 200, 255), width=2)
            
            # Create UI layer with transparency
            ui_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            ui_draw = ImageDraw.Draw(ui_layer, 'RGBA')
            self.draw_ui(ui_draw, current_time, total_duration)
            
            # Composite layers: base -> kick -> notes -> UI -> strike line (on top)
            base_layer.paste(kick_layer, (0, 0), kick_layer)
            base_layer.paste(notes_layer, (0, 0), notes_layer)
            base_layer.paste(ui_layer, (0, 0), ui_layer)
            base_layer.paste(strike_layer, (0, 0), strike_layer)
            
            # Convert to OpenCV for FFmpeg
            frame = pil_to_cv2(base_layer)
            
            # Write frame to FFmpeg
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg process terminated unexpectedly")
                break
            
            # Show preview
            if show_preview and frame_num % 10 == 0:
                cv2.imshow('Preview', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_num % 50 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Close FFmpeg stdin and wait for completion
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        if show_preview:
            cv2.destroyAllWindows()
        
        if ffmpeg_process.returncode == 0:
            print(f"✅ Video saved to: {output_path}")
            if audio_path:
                print(f"   Encoded with H.264 video codec and AAC audio codec")
            else:
                print(f"   Encoded with H.264 video codec (no audio)")
            print(f"   Optimized for web streaming")
            print(f"   Note: If video player shows old version, hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)")
        else:
            stderr_output = ffmpeg_process.stderr.read().decode('utf-8') if ffmpeg_process.stderr else ''
            print(f"⚠️  FFmpeg encoding completed with return code {ffmpeg_process.returncode}")
            if stderr_output:
                print(f"   Error details: {stderr_output[-500:]}")  # Last 500 chars


def render_project_video(
    project: dict,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    preview: bool = False,
    audio_source: Optional[str] = 'original',
    include_audio: Optional[bool] = None,  # Deprecated: kept for backward compatibility
    fall_speed_multiplier: float = 1.0
):
    """
    Render MIDI to video for a specific project.
    
    Args:
        project: Project info dictionary from project_manager
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        preview: Show live preview while rendering
        audio_source: Audio source selection - None (no audio), 'original', or 'alternate_mix/{filename}'
        include_audio: DEPRECATED - use audio_source instead. If True, uses 'original'
        fall_speed_multiplier: Speed multiplier for falling notes (1.0 = default, 0.5 = half speed, 2.0 = double speed)
    """
    project_dir = project["path"]
    
    print(f"\n{'='*60}")
    print(f"Rendering Video - Project {project['number']}: {project['name']}")
    print(f"{'='*60}\n")
    
    # Handle backward compatibility with include_audio
    if include_audio is not None and audio_source is None:
        audio_source = 'original' if include_audio else None
        print("Note: include_audio parameter is deprecated, use audio_source instead")
    
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
    
    # Resolve audio file path based on audio_source
    audio_file = None
    if audio_source:
        if audio_source == 'original':
            # Look for original audio file in project root
            audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif']
            for ext in audio_extensions:
                potential_audio = project_dir / f"{project['name']}{ext}"
                if potential_audio.exists():
                    audio_file = str(potential_audio)
                    break
            
            if not audio_file:
                print("WARNING: Original audio requested but not found in project root")
                print(f"Looked for: {project['name']}{{.wav,.mp3,.flac,.aiff,.aif}}")
        
        elif audio_source.startswith('alternate_mix/'):
            # Use alternate audio file
            alternate_audio_path = project_dir / audio_source
            if alternate_audio_path.exists():
                audio_file = str(alternate_audio_path)
            else:
                print(f"WARNING: Alternate audio '{audio_source}' not found")
        else:
            print(f"WARNING: Unknown audio_source value: {audio_source}")
    
    print(f"Rendering video to: {output_file}")
    print(f"Settings: {width}x{height} @ {fps}fps")
    print(f"Note fall speed: {fall_speed_multiplier}x")
    if audio_file:
        print(f"Including audio: {Path(audio_file).name}")
    else:
        print("Audio: None")
    if preview:
        print("Preview mode enabled")
    print()
    
    # Render video
    renderer = MidiVideoRenderer(width=width, height=height, fps=fps, fall_speed_multiplier=fall_speed_multiplier)
    renderer.render(str(midi_file), str(output_file), show_preview=preview, audio_path=audio_file)
    
    # Update project metadata
    update_project_metadata(project_dir, {
        "status": {
            "separated": project["metadata"]["status"].get("separated", False) if project["metadata"] else False,
            "cleaned": project["metadata"]["status"].get("cleaned", False) if project["metadata"] else False,
            "midi_generated": project["metadata"]["status"].get("midi_generated", False) if project["metadata"] else False,
            "video_rendered": True
        }
    })
    
    print(f"Status Update: Video rendering complete")
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
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio in the video (audio included by default)')
    parser.add_argument('--fall-speed', type=float, default=1.0,
                       help='Note fall speed multiplier (default: 1.0, range: 0.5-2.0)')
    
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
            print("\nNo projects found in user_files/")
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
        preview=args.preview,
        audio_source=None if args.no_audio else 'original',
        fall_speed_multiplier=args.fall_speed
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
