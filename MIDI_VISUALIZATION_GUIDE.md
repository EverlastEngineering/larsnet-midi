# MIDI Drum Visualization Guide 🎬🥁

This guide shows you how to create Rock Band-style falling notes videos from your MIDI drum files, perfect for learning to play drums!

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_visualization.txt
```

This installs:
- `mido` - MIDI file parsing
- `opencv-python` - Video rendering
- `numpy` - Numerical operations

### 2. Render Your First Video

```bash
python render_midi_to_video.py learn_midi_output/The\ Fate\ Of\ Ophelia.mid --output drums_video.mp4
```

That's it! You'll get a video with falling notes in Rock Band style.

---

## Features

✅ **Rock Band-style falling notes** - Notes fall from top to bottom  
✅ **Color-coded drum pieces** - Each drum has its own color  
✅ **Strike line indicator** - Shows exactly when to hit each note  
✅ **Velocity visualization** - Note brightness reflects how hard to hit  
✅ **Time display** - Current time and progress bar  
✅ **Multi-lane layout** - 8 lanes for different drum pieces  
✅ **High-quality output** - Supports up to 4K resolution at 60 FPS  

---

## Drum Color Mapping

The script uses standard General MIDI drum mapping with these colors:

| Drum Piece | MIDI Note | Color | Lane |
|------------|-----------|-------|------|
| Kick | 36 | Yellow | 0 |
| Snare | 38, 40 | Red | 1 |
| Hi-Hat Closed | 42, 44 | Cyan | 2 |
| Hi-Hat Open | 46 | Light Blue | 3 |
| Tom 1/2 | 47, 48 | Green | 4 |
| Tom 3 | 50 | Magenta | 5 |
| Crash/Splash | 49, 55 | Orange | 6 |
| Ride | 51 | Light Orange | 7 |

---

## Usage Examples

### Basic Usage

```bash
# Simple 1080p video at 60 FPS
python render_midi_to_video.py learn_midi_output/drums_edited.mid --output my_drums.mp4
```

### Custom Resolution & FPS

```bash
# 720p for faster rendering
python render_midi_to_video.py input.mid --output output.mp4 --width 1280 --height 720 --fps 30

# 4K for maximum quality
python render_midi_to_video.py input.mid --output output_4k.mp4 --width 3840 --height 2160 --fps 60
```

### With Live Preview

```bash
# Shows preview window while rendering (press 'q' to quit)
python render_midi_to_video.py input.mid --output output.mp4 --preview
```

### All Options

```bash
python render_midi_to_video.py input.mid \
  --output output.mp4 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --preview
```

---

## Adding Audio to Your Video

The rendered video only contains the visual notes. To add the original audio track:

### Option 1: Using FFmpeg (Recommended)

```bash
# Install FFmpeg if needed
brew install ffmpeg

# Combine video and audio
ffmpeg -i drums_video.mp4 -i input/The\ Fate\ Of\ Ophelia.wav \
  -c:v copy -c:a aac -shortest drums_with_audio.mp4
```

### Option 2: Using Your DAW or Video Editor

1. Import the MP4 video into your DAW (Logic, Ableton, etc.) or video editor
2. Add the audio track
3. Export the combined video

---

## Workflow for Learning

### Step 1: Generate MIDI from Audio

```bash
# Use your stems-to-midi pipeline
python stems_to_midi.py -i input/song.wav -o midi_output/
```

### Step 2: (Optional) Edit MIDI in DAW

Load the MIDI in your DAW and refine the notes if needed.

### Step 3: Create Visual Learning Video

```bash
python render_midi_to_video.py midi_output/drums.mid --output learning_video.mp4
```

### Step 4: Add Audio

```bash
ffmpeg -i learning_video.mp4 -i input/song.wav \
  -c:v copy -c:a aac -shortest learning_video_with_audio.mp4
```

### Step 5: Practice!

Play the video and follow along with your drum kit or practice pad. The falling notes will show you exactly when to hit each drum piece.

---

## Customization

### Changing Colors

Edit the `DRUM_MAP` dictionary in `render_midi_to_video.py`:

```python
DRUM_MAP = {
    36: {"name": "Kick", "lane": 0, "color": (0, 255, 255)},  # BGR format
    # ... add or modify as needed
}
```

### Adjusting Note Speed

Change `pixels_per_second` in the `MidiVideoRenderer.__init__()` method:

```python
self.pixels_per_second = height * 0.7  # Lower = slower falling notes
```

### Changing Strike Line Position

Modify `strike_line_y`:

```python
self.strike_line_y = int(height * 0.85)  # Higher = lower on screen
```

---

## Troubleshooting

### "Module not found" errors

```bash
pip install -r requirements_visualization.txt
```

### Video is too fast/slow

Adjust the `--fps` parameter or modify `pixels_per_second` in the code.

### Colors don't match my MIDI file

Your MIDI file might use different note numbers. Check which MIDI notes are used:

```python
import mido
midi = mido.MidiFile('your_file.mid')
for track in midi.tracks:
    for msg in track:
        if msg.type == 'note_on':
            print(f"Note: {msg.note}")
```

Then update the `DRUM_MAP` accordingly.

### Video file is huge

- Use lower resolution: `--width 1280 --height 720`
- Use lower FPS: `--fps 30`
- Re-encode with better compression:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium output.mp4
  ```

---

## Performance Tips

- **Faster rendering**: Lower resolution (720p) and FPS (30)
- **Better quality**: Higher resolution (4K) and FPS (60)
- **Balance**: 1080p at 60 FPS is the sweet spot

| Resolution | FPS | Render Time (3 min song) | Quality |
|------------|-----|--------------------------|---------|
| 1280x720 | 30 | ~2 minutes | Good |
| 1920x1080 | 60 | ~5 minutes | Great |
| 3840x2160 | 60 | ~15 minutes | Excellent |

---

## Alternative Tools

If you want more features or professional-grade output:

### MIDIVisualizer (Free, Open Source)

**Download**: https://github.com/kosua20/MIDIVisualizer/releases

**Features**:
- GPU accelerated
- More visual effects
- Real-time preview
- Custom color schemes
- Particle effects

**macOS Installation**:
```bash
brew install --cask midivis
```

### Synthesia (Commercial, $29)

**Website**: https://synthesiagame.com/

**Features**:
- Interactive learning mode
- Follows along with MIDI input
- Shows finger positions
- Great for practice

---

## Examples

### Process All MIDI Files in a Directory

```bash
#!/bin/bash
for midi in learn_midi_output/*.mid; do
    basename=$(basename "$midi" .mid)
    python render_midi_to_video.py "$midi" --output "midi_videos/${basename}.mp4"
done
```

### Batch Process with Audio

```bash
#!/bin/bash
for midi in learn_midi_output/*.mid; do
    basename=$(basename "$midi" .mid)
    
    # Render video
    python render_midi_to_video.py "$midi" --output "midi_videos/${basename}_video.mp4"
    
    # Add audio if exists
    if [ -f "input/${basename}.wav" ]; then
        ffmpeg -i "midi_videos/${basename}_video.mp4" \
               -i "input/${basename}.wav" \
               -c:v copy -c:a aac -shortest \
               "midi_videos/${basename}_final.mp4"
        rm "midi_videos/${basename}_video.mp4"
    fi
done
```

---

## Contributing

Want to improve the visualization? Here are some ideas:

- [ ] Add different visual themes (neon, retro, minimal)
- [ ] Support for multiple note shapes
- [ ] Add combo counter for consecutive hits
- [ ] Show hand/foot indicators (L/R hand, foot pedal)
- [ ] Add difficulty rating
- [ ] Export to other formats (GIF, WebM)
- [ ] Add audio waveform visualization

---

## Credits

Built on top of:
- **LarsNet** - Deep drum source separation
- **mido** - MIDI file parsing
- **OpenCV** - Video rendering

---

## Support

Having issues? Check:

1. This guide's troubleshooting section
2. The main README.md for project setup
3. STEMS_TO_MIDI_GUIDE.md for MIDI generation help

Happy drumming! 🥁🎵
