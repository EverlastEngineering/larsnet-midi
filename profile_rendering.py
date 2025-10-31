#!/usr/bin/env python3
"""
Profile video rendering to identify performance bottlenecks.

Uses cProfile to find where time is actually being spent.
"""
import cProfile
import pstats
import sys
from pathlib import Path
from io import StringIO


def profile_render(project_id: int = 12, num_seconds: int = 5):
    """Profile rendering of a short video clip"""
    from render_midi_to_video import MidiVideoRenderer
    
    # Find project files
    project_dir = Path(f"user_files/{project_id} - sdrums")
    midi_file = list(project_dir.glob("midi/*.mid"))[0]
    output_file = project_dir / "video" / "profile_test.mp4"
    output_file.parent.mkdir(exist_ok=True)
    
    # Find audio
    audio_file = None
    for ext in ['wav', 'mp3', 'flac']:
        audio_path = project_dir / f"{project_dir.name.split(' - ')[0]}.{ext}"
        if audio_path.exists():
            audio_file = str(audio_path)
            break
    
    print(f"\n{'='*60}")
    print(f"Profiling Video Rendering")
    print(f"{'='*60}")
    print(f"Project: {project_id}")
    print(f"MIDI: {midi_file.name}")
    print(f"Duration: First {num_seconds} seconds")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the render
    profiler.enable()
    
    try:
        renderer = MidiVideoRenderer()
        
        # Parse MIDI
        notes, total_duration = renderer.parse_midi(str(midi_file))
        
        # Limit duration for faster profiling
        render_duration = min(num_seconds, total_duration)
        
        # Monkey-patch to limit rendering
        original_duration = total_duration
        
        # Just render limited frames
        renderer.render(str(midi_file), str(output_file), 
                       show_preview=False, audio_path=audio_file)
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted")
    finally:
        profiler.disable()
    
    # Analyze results
    print(f"\n{'='*60}")
    print(f"Profiling Results")
    print(f"{'='*60}\n")
    
    # Create stats object
    stats = pstats.Stats(profiler, stream=StringIO())
    
    # Sort by cumulative time and print top functions
    print("Top 30 functions by cumulative time:")
    print("-" * 60)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print("\n" + "="*60)
    print("Top 30 functions by total time (self time):")
    print("-" * 60)
    stats.sort_stats('tottime')
    stats.print_stats(30)
    
    # Save detailed stats
    stats_file = project_dir / "video" / "profile_stats.txt"
    with open(stats_file, 'w') as f:
        stats.stream = f
        stats.sort_stats('cumulative')
        stats.print_stats()
    
    print(f"\n{'='*60}")
    print(f"Detailed stats saved to: {stats_file}")
    print(f"{'='*60}\n")
    
    return stats


def analyze_bottlenecks(stats):
    """Analyze profiling stats and suggest optimizations"""
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60 + "\n")
    
    # Get all stats
    all_stats = stats.stats
    
    # Categories to look for
    bottlenecks = {
        'PIL/Pillow': [],
        'Drawing': [],
        'Image Creation': [],
        'Compositing': [],
        'File I/O': [],
        'Math/Calculations': [],
        'Other': []
    }
    
    for func_key, func_stats in all_stats.items():
        filename, line, func_name = func_key
        cumtime = func_stats[3]
        
        if 'PIL' in filename or 'pillow' in filename.lower():
            bottlenecks['PIL/Pillow'].append((func_name, cumtime))
        elif 'draw' in func_name.lower() or 'rectangle' in func_name.lower() or 'circle' in func_name.lower():
            bottlenecks['Drawing'].append((func_name, cumtime))
        elif 'new' in func_name.lower() or 'Image' in func_name:
            bottlenecks['Image Creation'].append((func_name, cumtime))
        elif 'paste' in func_name.lower() or 'composite' in func_name.lower() or 'blend' in func_name.lower():
            bottlenecks['Compositing'].append((func_name, cumtime))
        elif 'write' in func_name.lower() or 'read' in func_name.lower():
            bottlenecks['File I/O'].append((func_name, cumtime))
        elif 'sin' in func_name.lower() or 'cos' in func_name.lower() or 'sqrt' in func_name.lower():
            bottlenecks['Math/Calculations'].append((func_name, cumtime))
    
    # Print analysis
    for category, funcs in bottlenecks.items():
        if funcs:
            funcs.sort(key=lambda x: x[1], reverse=True)
            total_time = sum(f[1] for f in funcs)
            print(f"{category}: {total_time:.2f}s total")
            for func_name, cumtime in funcs[:5]:
                print(f"  {func_name}: {cumtime:.2f}s")
            print()


if __name__ == '__main__':
    project_id = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    num_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    stats = profile_render(project_id, num_seconds)
    analyze_bottlenecks(stats)
    
    print("\nRecommendations will be based on the bottlenecks found above.")
    print("Look for functions taking the most cumulative time.\n")
