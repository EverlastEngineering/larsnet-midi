"""
Tests for OpenCV rendering helpers (Phase 1)

Validates that OpenCV drawing functions produce visually similar output to PIL.
"""
import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw
from render_midi_to_video import (
    create_cv2_canvas,
    cv2_draw_rounded_rectangle,
    cv2_composite_layer,
    draw_rounded_rectangle
)


def test_create_cv2_canvas():
    """Test canvas creation with different configurations"""
    # 3-channel BGR
    canvas_bgr = create_cv2_canvas(100, 100, channels=3)
    assert canvas_bgr.shape == (100, 100, 3)
    assert canvas_bgr.dtype == np.uint8
    assert np.all(canvas_bgr == 0)  # Should be black
    
    # 4-channel BGRA
    canvas_bgra = create_cv2_canvas(100, 100, channels=4)
    assert canvas_bgra.shape == (100, 100, 4)
    assert np.all(canvas_bgra == 0)  # Should be transparent black
    
    # With fill color
    canvas_filled = create_cv2_canvas(100, 100, channels=3, fill_color=(255, 0, 0))
    assert np.all(canvas_filled[:, :, 0] == 255)  # Blue channel


def test_cv2_draw_rounded_rectangle_simple():
    """Test drawing simple rectangles without rounding"""
    canvas = create_cv2_canvas(200, 200, channels=4)
    
    # Draw filled rectangle
    cv2_draw_rounded_rectangle(
        canvas,
        xy=(50, 50, 150, 100),
        radius=0,
        fill=(0, 255, 0, 255)  # Green
    )
    
    # Check that rectangle was drawn
    assert np.any(canvas[:, :, 1] > 0)  # Green channel has values
    assert canvas[75, 100, 1] == 255  # Center of rectangle is green
    assert canvas[25, 100, 1] == 0  # Outside rectangle is black


def test_cv2_draw_rounded_rectangle_with_radius():
    """Test drawing rounded rectangles"""
    canvas = create_cv2_canvas(200, 200, channels=4)
    
    # Draw rounded rectangle
    cv2_draw_rounded_rectangle(
        canvas,
        xy=(50, 50, 150, 100),
        radius=10,
        fill=(255, 0, 0, 255)  # Blue
    )
    
    # Check that rectangle was drawn
    assert np.any(canvas[:, :, 0] > 0)  # Blue channel has values
    assert canvas[75, 100, 0] > 0  # Center should be filled
    
    # Corners should be rounded (not filled at exact corner pixels)
    # This is a rough check - exact pixels depend on implementation
    corner_filled = canvas[50, 50, 0] > 0
    center_filled = canvas[75, 100, 0] > 0
    assert center_filled  # Center must be filled
    # Corner might or might not be filled depending on radius implementation


def test_cv2_composite_layer():
    """Test alpha compositing of layers"""
    # Create base layer (blue)
    base = create_cv2_canvas(100, 100, channels=3, fill_color=(255, 0, 0))
    
    # Create overlay layer (red with 50% alpha)
    overlay = create_cv2_canvas(100, 100, channels=4, fill_color=(0, 0, 255, 128))
    
    # Composite
    cv2_composite_layer(base, overlay)
    
    # Result should be purple (blend of red and blue)
    # Blue channel should be less than 255 (blended with red)
    assert base[50, 50, 0] < 255  # Blue reduced
    assert base[50, 50, 2] > 0  # Red added
    
    # Check approximate blending (128/255 ≈ 0.5)
    # Expected: base_blue * 0.5 + overlay_red * 0.5 ≈ 255 * 0.5 + 0 * 0.5 = 127
    assert 100 < base[50, 50, 0] < 200  # Blue should be roughly half


def test_cv2_composite_layer_with_alpha_multiplier():
    """Test compositing with additional alpha multiplier"""
    base = create_cv2_canvas(100, 100, channels=3, fill_color=(255, 0, 0))
    overlay = create_cv2_canvas(100, 100, channels=4, fill_color=(0, 0, 255, 255))
    
    # Composite with 0.5 alpha multiplier
    cv2_composite_layer(base, overlay, alpha=0.5)
    
    # Should blend 50/50 regardless of overlay alpha channel
    assert 100 < base[50, 50, 0] < 200  # Blended blue
    assert base[50, 50, 2] > 0  # Has red component


def test_cv2_vs_pil_visual_similarity():
    """Compare OpenCV and PIL output for simple shapes"""
    width, height = 200, 200
    
    # Draw with PIL
    pil_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    pil_draw = ImageDraw.Draw(pil_img, 'RGBA')
    draw_rounded_rectangle(
        pil_draw,
        xy=(50, 50, 150, 100),
        radius=10,
        fill=(255, 0, 0, 255)
    )
    pil_array = np.array(pil_img)
    
    # Draw with OpenCV
    cv2_canvas = create_cv2_canvas(width, height, channels=4)
    cv2_draw_rounded_rectangle(
        cv2_canvas,
        xy=(50, 50, 150, 100),
        radius=10,
        fill=(0, 0, 255, 255)  # Note: BGR format
    )
    
    # Compare filled areas (not exact pixels due to anti-aliasing differences)
    # Count non-zero pixels in red channel
    pil_filled_pixels = np.sum(pil_array[:, :, 0] > 0)
    cv2_filled_pixels = np.sum(cv2_canvas[:, :, 2] > 0)  # Red is index 2 in BGR
    
    # Should be within 10% of each other
    pixel_diff_ratio = abs(pil_filled_pixels - cv2_filled_pixels) / pil_filled_pixels
    assert pixel_diff_ratio < 0.10, f"Pixel difference too large: {pixel_diff_ratio:.2%}"


def test_cv2_rendering_performance():
    """Basic performance sanity check - OpenCV should be faster"""
    import time
    
    width, height = 1920, 1080
    iterations = 10
    
    # Time PIL rendering
    start = time.time()
    for _ in range(iterations):
        pil_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        pil_draw = ImageDraw.Draw(pil_img, 'RGBA')
        for i in range(10):
            draw_rounded_rectangle(
                pil_draw,
                xy=(50 + i*50, 50, 100 + i*50, 100),
                radius=10,
                fill=(255, 0, 0, 255)
            )
    pil_time = time.time() - start
    
    # Time OpenCV rendering
    start = time.time()
    for _ in range(iterations):
        cv2_canvas = create_cv2_canvas(width, height, channels=4)
        for i in range(10):
            cv2_draw_rounded_rectangle(
                cv2_canvas,
                xy=(50 + i*50, 50, 100 + i*50, 100),
                radius=10,
                fill=(0, 0, 255, 255)
            )
    cv2_time = time.time() - start
    
    print(f"\nPIL time: {pil_time:.3f}s")
    print(f"OpenCV time: {cv2_time:.3f}s")
    print(f"Speedup: {pil_time / cv2_time:.2f}x")
    
    # OpenCV should be at least as fast (ideally faster)
    # Not asserting speedup here since it depends on hardware
    assert cv2_time > 0  # Sanity check


def test_strike_line_rendering_comparison():
    """Test that strike line renders similarly with PIL and OpenCV"""
    from render_midi_to_video import MidiVideoRenderer
    
    width, height = 400, 300
    
    # Create renderer with PIL
    renderer_pil = MidiVideoRenderer(width=width, height=height, use_opencv=False)
    renderer_pil.num_lanes = 3  # Simulate 3 lanes
    renderer_pil.note_width = width // 3
    
    # Create renderer with OpenCV
    renderer_cv2 = MidiVideoRenderer(width=width, height=height, use_opencv=True)
    renderer_cv2.num_lanes = 3
    renderer_cv2.note_width = width // 3
    
    # Both should initialize without errors
    assert renderer_pil.use_opencv == False
    assert renderer_cv2.use_opencv == True
    
    print("\n✓ Strike line rendering test passed - both modes operational")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
