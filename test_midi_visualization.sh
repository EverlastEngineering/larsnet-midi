#!/bin/bash
# Test script for MIDI visualization

set -e

echo "🎵 Testing MIDI Visualization Tool"
echo "=================================="
echo ""

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import mido; import cv2; import numpy" 2>/dev/null || {
    echo "❌ Dependencies not installed. Installing..."
    pip install -r requirements_visualization.txt
}

echo "✅ Dependencies OK"
echo ""

# Check if MIDI files exist
MIDI_DIR="learn_midi_output"
if [ ! -d "$MIDI_DIR" ]; then
    echo "❌ Error: $MIDI_DIR directory not found"
    exit 1
fi

# Find first MIDI file
MIDI_FILE=$(find "$MIDI_DIR" -name "*.mid" | head -n 1)

if [ -z "$MIDI_FILE" ]; then
    echo "❌ Error: No MIDI files found in $MIDI_DIR"
    exit 1
fi

echo "🎼 Found MIDI file: $MIDI_FILE"
echo ""

# Create output directory
OUTPUT_DIR="midi_videos"
mkdir -p "$OUTPUT_DIR"

# Generate video
OUTPUT_FILE="$OUTPUT_DIR/test_drums_video.mp4"
echo "🎬 Rendering video to: $OUTPUT_FILE"
echo ""

python3 render_midi_to_video.py "$MIDI_FILE" --output "$OUTPUT_FILE" --width 1280 --height 720 --fps 30

echo ""
echo "✅ Video rendering complete!"
echo ""
echo "📹 Output file: $OUTPUT_FILE"
echo ""
echo "To add audio, run:"
echo "ffmpeg -i $OUTPUT_FILE -i input/The\\ Fate\\ Of\\ Ophelia.wav -c:v copy -c:a aac -shortest ${OUTPUT_FILE%.mp4}_with_audio.mp4"
