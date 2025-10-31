#!/usr/bin/env python3
"""
Test pure OpenCV rendering speed without PIL conversions.

This skips format conversions to isolate OpenCV performance.
"""
import time
import numpy as np
import cv2
from render_midi_to_video import (
    create_cv2_canvas,
    cv2_draw_rounded_rectangle,
    cv2_draw_highlight_circle,
    MidiVideoRenderer
)

def test_pure_opencv_rendering(project_id: int = 12, num_frames: int = 300):
    """Time OpenCV drawing operations without PIL conversion overhead"""
    from pathlib import Path
    
    # Setup renderer
    renderer = MidiVideoRenderer(use_opencv=True)
    
    # Find MIDI file
    project_dir = Path(f"user_files/{project_id} - sdrums")
    midi_file = list(project_dir.glob("midi/*.mid"))[0]
    
    # Parse MIDI
    notes, total_duration = renderer.parse_midi(str(midi_file))
    total_frames = min(num_frames, int(total_duration * renderer.fps))
    
    print(f"\n{'='*60}")
    print(f"Pure OpenCV Rendering Speed Test")
    print(f"{'='*60}")
    print(f"Project: {project_id}")
    print(f"Notes: {len(notes)}")
    print(f"Frames to render: {total_frames}")
    print(f"Resolution: {renderer.width}x{renderer.height}")
    print(f"{'='*60}\n")
    
    # Time just the OpenCV drawing operations
    start_time = time.time()
    
    time_step = 1.0 / renderer.fps
    lookahead_time = renderer.strike_line_y / renderer.pixels_per_second
    note_index = 0
    
    for frame_num in range(total_frames):
        current_time = frame_num * time_step
        
        # Create OpenCV canvases (this is the cost we pay)
        strike_layer = create_cv2_canvas(renderer.width, renderer.height, channels=4)
        
        # Draw strike line with OpenCV
        cv2.line(strike_layer, (0, renderer.strike_line_y), 
                 (renderer.width, renderer.strike_line_y),
                 (255, 255, 255, 255), 4, cv2.LINE_AA)
        
        # Draw lane markers
        for lane in range(renderer.num_lanes):
            x = lane * renderer.note_width + renderer.note_width // 2
            cv2.circle(strike_layer, (x, renderer.strike_line_y), 20,
                      (200, 200, 200, 255), 2, cv2.LINE_AA)
        
        # Draw highlight circles for visible notes
        visible_start = note_index
        for i in range(visible_start, len(notes)):
            note = notes[i]
            time_until_hit = note.time - current_time
            
            if time_until_hit > lookahead_time:
                break
            
            if renderer.should_draw_highlight(note, current_time):
                # Simplified highlight drawing
                if note.lane >= 0:
                    x = note.lane * renderer.note_width + 10
                    width = renderer.note_width - 20
                    center_x = x + width // 2
                    
                    progress = renderer.calculate_strike_animation_progress(note, current_time)
                    pulse = abs(np.sin(progress * np.pi))
                    max_size = 50 + 20 * pulse
                    
                    cv2_draw_highlight_circle(strike_layer, center_x, renderer.strike_line_y,
                                             max_size, note.color, 220, pulse)
        
        # Progress
        if frame_num % 50 == 0:
            elapsed = time.time() - start_time
            fps = (frame_num + 1) / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_num}/{total_frames} ({fps:.1f} fps)")
    
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time
    
    print(f"\n{'='*60}")
    print(f"Pure OpenCV Results")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Time per frame: {(total_time/total_frames)*1000:.2f}ms")
    print(f"{'='*60}\n")
    
    return total_time, avg_fps


def test_pure_pil_rendering(project_id: int = 12, num_frames: int = 300):
    """Time PIL drawing operations for comparison"""
    from pathlib import Path
    from PIL import Image, ImageDraw
    
    # Setup renderer
    renderer = MidiVideoRenderer(use_opencv=False)
    
    # Find MIDI file
    project_dir = Path(f"user_files/{project_id} - sdrums")
    midi_file = list(project_dir.glob("midi/*.mid"))[0]
    
    # Parse MIDI
    notes, total_duration = renderer.parse_midi(str(midi_file))
    total_frames = min(num_frames, int(total_duration * renderer.fps))
    
    print(f"\n{'='*60}")
    print(f"Pure PIL Rendering Speed Test")
    print(f"{'='*60}")
    print(f"Project: {project_id}")
    print(f"Frames to render: {total_frames}")
    print(f"{'='*60}\n")
    
    # Time just the PIL drawing operations
    start_time = time.time()
    
    time_step = 1.0 / renderer.fps
    lookahead_time = renderer.strike_line_y / renderer.pixels_per_second
    note_index = 0
    first_highlight_frame = set()
    
    for frame_num in range(total_frames):
        current_time = frame_num * time_step
        
        # Create PIL canvas
        strike_layer = Image.new('RGBA', (renderer.width, renderer.height), (0, 0, 0, 0))
        strike_draw = ImageDraw.Draw(strike_layer, 'RGBA')
        
        # Draw strike line with PIL
        strike_draw.line([(0, renderer.strike_line_y), (renderer.width, renderer.strike_line_y)],
                        fill=(255, 255, 255, 255), width=4)
        
        # Draw lane markers
        for lane in range(renderer.num_lanes):
            x = lane * renderer.note_width + renderer.note_width // 2
            strike_draw.ellipse([x - 20, renderer.strike_line_y - 20, 
                                x + 20, renderer.strike_line_y + 20],
                               outline=(200, 200, 200, 255), width=2)
        
        # Draw highlight circles
        visible_start = note_index
        for i in range(visible_start, len(notes)):
            note = notes[i]
            time_until_hit = note.time - current_time
            
            if time_until_hit > lookahead_time:
                break
            
            if renderer.should_draw_highlight(note, current_time):
                renderer.draw_highlight_circle(strike_draw, note, current_time, first_highlight_frame)
        
        # Progress
        if frame_num % 50 == 0:
            elapsed = time.time() - start_time
            fps = (frame_num + 1) / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_num}/{total_frames} ({fps:.1f} fps)")
    
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time
    
    print(f"\n{'='*60}")
    print(f"Pure PIL Results")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Time per frame: {(total_time/total_frames)*1000:.2f}ms")
    print(f"{'='*60}\n")
    
    return total_time, avg_fps


if __name__ == '__main__':
    import sys
    project_id = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    # Test PIL
    pil_time, pil_fps = test_pure_pil_rendering(project_id, num_frames)
    
    # Test OpenCV
    cv2_time, cv2_fps = test_pure_opencv_rendering(project_id, num_frames)
    
    # Compare
    print(f"\n{'='*60}")
    print(f"COMPARISON (Pure Rendering Only)")
    print(f"{'='*60}")
    print(f"PIL:    {pil_time:.2f}s ({pil_fps:.1f} fps)")
    print(f"OpenCV: {cv2_time:.2f}s ({cv2_fps:.1f} fps)")
    
    if cv2_time < pil_time:
        speedup = pil_time / cv2_time
        print(f"\n✓ OpenCV is {speedup:.2f}x FASTER for pure rendering")
    else:
        slowdown = cv2_time / pil_time
        print(f"\n✗ OpenCV is {slowdown:.2f}x SLOWER for pure rendering")
    
    print(f"{'='*60}\n")
