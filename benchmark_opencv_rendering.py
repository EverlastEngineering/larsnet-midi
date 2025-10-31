#!/usr/bin/env python3
"""
Benchmark script for comparing PIL vs OpenCV rendering performance.

Actually renders videos to measure real-world performance.
"""
import time
import sys
from pathlib import Path
from render_midi_to_video import render_project_video

def benchmark_render(project_id: int, use_opencv: bool):
    """Benchmark rendering with PIL or OpenCV
    
    Args:
        project_id: Project ID to render
        use_opencv: True for OpenCV, False for PIL
    
    Returns:
        Total render time in seconds
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {'OpenCV' if use_opencv else 'PIL'}")
    print(f"Project: {project_id}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # Use the actual render function with use_opencv parameter
        # We'll need to pass it through - for now just time it
        render_project_video(
            project_id,
            preview=False,
            use_opencv=use_opencv
        )
    except TypeError:
        # render_project_video doesn't have use_opencv param yet
        # We need to modify it temporarily
        print("Error: render_project_video needs use_opencv parameter")
        print("Let's render manually instead...")
        
        from render_midi_to_video import MidiVideoRenderer
        from project_manager import find_project_midi
        
        project_dir = Path(f"user_files/{project_id} - sdrums")
        midi_file = find_project_midi(project_id)
        output_file = project_dir / "video" / f"benchmark_{'opencv' if use_opencv else 'pil'}.mp4"
        output_file.parent.mkdir(exist_ok=True)
        
        # Find audio file
        audio_file = None
        for ext in ['mp3', 'wav', 'ogg', 'flac']:
            audio_path = project_dir / f"{project_dir.name}.{ext}"
            if audio_path.exists():
                audio_file = audio_path
                break
        
        renderer = MidiVideoRenderer(use_opencv=use_opencv)
        renderer.render(str(midi_file), str(output_file), 
                       show_preview=False, audio_path=str(audio_file) if audio_file else None)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {'OpenCV' if use_opencv else 'PIL'}")
    print(f"{'='*60}")
    print(f"Total render time: {total_time:.2f}s")
    print(f"{'='*60}\n")
    
    return total_time


def main():
    project_id = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    
    print("\n" + "="*60)
    print("OpenCV vs PIL Rendering Benchmark")
    print("="*60)
    
    # Test PIL
    pil_time = benchmark_render(project_id, use_opencv=False)
    
    # Test OpenCV
    opencv_time = benchmark_render(project_id, use_opencv=True)
    
    # Compare
    if pil_time and opencv_time:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"PIL:    {pil_time:.2f}s total")
        print(f"OpenCV: {opencv_time:.2f}s total")
        
        speedup = pil_time / opencv_time if opencv_time > 0 else 0
        time_saved = pil_time - opencv_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.2f}s ({(time_saved/pil_time)*100:.1f}%)")
        
        if speedup > 1.0:
            print(f"✓ OpenCV is {speedup:.2f}x FASTER")
        elif speedup < 1.0:
            print(f"✗ OpenCV is {1/speedup:.2f}x SLOWER")
        else:
            print("= Same performance")
        print("="*60)


if __name__ == '__main__':
    main()
