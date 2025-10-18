# MIDI Visualization for Drum Learning

## Option 1: MIDIVisualizer (Recommended)

**MIDIVisualizer** is a professional open-source tool that creates Rock Band-style falling notes videos.

### Installation (macOS):
```bash
# Install via Homebrew
brew install --cask midivis

# Or download from: https://github.com/kosua20/MIDIVisualizer/releases
```

### Usage:
1. Open MIDIVisualizer
2. Load your MIDI file (e.g., `learn_midi_output/The Fate Of Ophelia.mid`)
3. Configure the visualization:
   - Set quality/resolution
   - Choose color scheme
   - Adjust note speed
   - Select which tracks to show
4. Export to video (MP4/MOV)

### Features:
- ✅ Professional quality output
- ✅ Customizable colors per drum piece
- ✅ Real-time preview
- ✅ Multiple export formats
- ✅ GPU accelerated

---

## Option 2: Python Script (Custom Solution)

Create custom visualizations with full control over the appearance.

### Installation:
```bash
pip install -r requirements_visualization.txt
```

### Usage:
```bash
# Basic usage
python render_midi_to_video.py learn_midi_output/The\ Fate\ Of\ Ophelia.mid --output ophelia_drums.mp4

# With custom resolution and FPS
python render_midi_to_video.py learn_midi_output/drums_edited.mid --output drums_1080p.mp4 --width 1920 --height 1080 --fps 60

# With live preview (press 'q' to quit)
python render_midi_to_video.py learn_midi_output/ophilia.mid --output ophilia_video.mp4 --preview

# 4K high quality
python render_midi_to_video.py learn_midi_output/The\ Fate\ Of\ Ophelia.mid --output ophelia_4k.mp4 --width 3840 --height 2160 --fps 60
```

See `render_midi_to_video.py` for the implementation.

---

## Option 3: Synthesia-style with Python

For a more traditional piano roll view (like Synthesia for drums).

---

## Recommended Workflow:

1. **For Quick Results**: Use MIDIVisualizer (Option 1)
2. **For Custom Look**: Use the Python script (Option 2)
3. **Combine with Audio**: Use FFmpeg to overlay the original audio

```bash
# Add audio to the video
ffmpeg -i drums_video.mp4 -i input/The\ Fate\ Of\ Ophelia.wav \
  -c:v copy -c:a aac -shortest drums_with_audio.mp4
```
