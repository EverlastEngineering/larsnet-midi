#!/bin/bash
# Render all MIDI files to Rock Band-style videos
# Usage: ./render_all_midi.sh [--with-audio]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎬 MIDI to Video Renderer${NC}"
echo "=================================="
echo ""

# Configuration
MIDI_DIR="learn_midi_output"
OUTPUT_DIR="midi_videos"
WIDTH=1920
HEIGHT=1080
FPS=60
ADD_AUDIO=false

# Check for --with-audio flag
if [[ "$1" == "--with-audio" ]]; then
    ADD_AUDIO=true
    echo -e "${YELLOW}Audio will be added to videos${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all MIDI files
MIDI_FILES=($(find "$MIDI_DIR" -name "*.mid" 2>/dev/null))

if [ ${#MIDI_FILES[@]} -eq 0 ]; then
    echo -e "❌ No MIDI files found in $MIDI_DIR"
    exit 1
fi

echo -e "${GREEN}Found ${#MIDI_FILES[@]} MIDI files${NC}"
echo ""

# Render each MIDI file
for midi_file in "${MIDI_FILES[@]}"; do
    basename=$(basename "$midi_file" .mid)
    video_file="$OUTPUT_DIR/${basename}_video.mp4"
    
    echo -e "${BLUE}🎵 Processing: $basename${NC}"
    
    # Render video in Docker
    docker exec -it larsnet-midi python /app/render_midi_to_video.py \
        "/app/$midi_file" \
        --output "/app/$video_file" \
        --width $WIDTH \
        --height $HEIGHT \
        --fps $FPS
    
    # Add audio if requested and available
    if [ "$ADD_AUDIO" = true ]; then
        # Try to find matching audio file
        audio_file=""
        for ext in wav mp3 flac m4a; do
            if [ -f "input/${basename}.$ext" ]; then
                audio_file="input/${basename}.$ext"
                break
            fi
            # Also try with spaces converted
            test_name=$(echo "$basename" | tr '_' ' ')
            if [ -f "input/${test_name}.$ext" ]; then
                audio_file="input/${test_name}.$ext"
                break
            fi
        done
        
        if [ -n "$audio_file" ]; then
            final_file="$OUTPUT_DIR/${basename}_final.mp4"
            echo -e "${YELLOW}🎵 Adding audio from: $audio_file${NC}"
            
            docker exec -it larsnet-midi ffmpeg -y \
                -i "/app/$video_file" \
                -i "/app/$audio_file" \
                -c:v copy -c:a aac -shortest \
                "/app/$final_file" \
                -loglevel warning
            
            # Remove video-only file
            rm "$video_file"
            echo -e "${GREEN}✅ Created: $final_file (with audio)${NC}"
        else
            echo -e "${YELLOW}⚠️  No matching audio file found for $basename${NC}"
            echo -e "${GREEN}✅ Created: $video_file (video only)${NC}"
        fi
    else
        echo -e "${GREEN}✅ Created: $video_file${NC}"
    fi
    
    echo ""
done

echo -e "${GREEN}🎉 All done!${NC}"
echo ""
echo "Videos saved to: $OUTPUT_DIR/"
echo ""

if [ "$ADD_AUDIO" = false ]; then
    echo "To add audio, run:"
    echo "  ./render_all_midi.sh --with-audio"
    echo ""
    echo "Or manually with ffmpeg:"
    echo "  ffmpeg -i $OUTPUT_DIR/video.mp4 -i input/audio.wav -c:v copy -c:a aac -shortest output.mp4"
fi
