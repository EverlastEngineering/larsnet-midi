# Video Renderer Visual Enhancement Plan

## Objective
Transform the MIDI video renderer from basic OpenCV rectangles to a visually appealing Rock Band-style experience with modern graphics effects.

## Current State
- Basic OpenCV drawing (rectangles, lines, circles)
- Flat colors, no anti-aliasing
- Hard edges, no visual effects
- Simple velocity-based brightness
- Basic text rendering

## Target State
- Smooth anti-aliased graphics
- Rounded corners on note boxes
- Fade in/out at screen extremes
- 3D perspective effect (Rock Band style)
- Glow effects on strike line hits
- Enhanced typography with shadows/outlines
- Gradient backgrounds
- Smooth visual polish

## Technical Approach
Use Pillow (PIL) for enhanced drawing, compose to numpy arrays, feed to OpenCV/FFmpeg pipeline.

## Phases

### Phase 1: Infrastructure Setup
- Add Pillow dependency to environment.yml
- Create PIL/numpy conversion utilities
- Refactor frame creation to use PIL Image objects
- Maintain backward compatibility with FFmpeg pipeline

### Phase 2: Enhanced Note Rendering
- Implement rounded rectangle drawing
- Add anti-aliasing to note edges
- Implement fade-in at top (alpha blend based on distance from top)
- Implement fade-out at bottom (alpha blend based on distance past strike line)
- Add gradient fill to notes (lighter at top, darker at bottom for depth)

### Phase 3: 3D Perspective Effect
- Calculate perspective transformation matrix
- Apply warp to simulate notes coming toward viewer
- Adjust note width/scale based on y-position
- Fine-tune perspective parameters for natural look

### Phase 4: Strike Line Effects
- Implement glow effect when notes cross strike line
- Add radial gradient glow
- Consider particle burst effect (optional)
- Enhance circle highlight with smooth alpha

### Phase 5: UI Polish
- Enhanced typography with shadows/outlines
- Better color scheme for UI elements
- Smooth gradient background (subtle)
- Improved legend with better spacing and shadows
- Better progress bar with gradient

## Risks & Mitigations
- **Performance**: Pillow may be slower than OpenCV
  - Mitigation: Profile and optimize hot paths, use caching where possible
- **Dependency**: Adding Pillow increases environment complexity
  - Mitigation: Pillow is lightweight and widely used
- **Complexity**: More code to maintain
  - Mitigation: Keep functions modular and well-documented

## Success Criteria
- Video output maintains 60fps rendering capability
- Visuals are noticeably more polished and professional
- Code remains maintainable with clear function separation
- All existing functionality preserved
- No regression in video encoding quality/compatibility

## Configuration
Add visual quality settings to allow users to adjust:
- Perspective strength
- Glow intensity
- Fade distances
- Enable/disable individual effects

## Testing Strategy
- Render test video before/after
- Verify timing accuracy unchanged
- Check performance benchmarks
- Visual inspection of effects
