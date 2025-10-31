# Video Renderer OpenCV Migration Plan

## Problem Statement
Video rendering is too slow due to PIL/Pillow being CPU-bound and creating multiple layers per frame. Need to migrate to OpenCV for 5-10x performance improvement.

## Current Architecture
- **PIL/Pillow**: All drawing operations (rounded rectangles, motion blur, layers)
- **4-5 layers per frame**: base, kick, notes, strike, UI - then composited with `.paste()`
- **Multiple draws per note**: Motion blur (2x), glow, outline, main body
- **OpenCV usage**: Only color conversion (PIL→cv2) and preview display

## Incremental Migration Strategy

### Phase 1: Infrastructure (Minimal Risk)
**Goal**: Set up OpenCV drawing infrastructure alongside PIL without breaking existing renderer

**Tasks**:
1. Create helper function `create_cv2_canvas(width, height, channels=3)` → returns NumPy array
2. Create `cv2_draw_rounded_rectangle()` using cv2.rectangle + rounded corners via polylines
3. Create `cv2_composite_layer()` for alpha blending NumPy arrays
4. Add flag `use_opencv=False` to renderer init for gradual testing
5. Write unit tests comparing PIL vs OpenCV output (visual diff threshold)

**Success Criteria**:
- New functions exist and pass tests
- No changes to existing render path
- Can render test frame with both methods

**Risk**: Low - only adds new code

---

### Phase 2: Single Layer Migration (Medium Risk)
**Goal**: Convert one simple layer (UI or strike line) to OpenCV

**Tasks**:
1. Choose simplest layer (strike line - just lines and circles)
2. Create `draw_strike_line_cv2()` parallel to PIL version
3. Add A/B testing: render same frame with both, compare output
4. Switch strike line to OpenCV when `use_opencv=True`
5. Benchmark: measure frame render time improvement

**Success Criteria**:
- Strike line renders identically (< 5% pixel diff)
- 10-20% overall frame time improvement
- No visual artifacts

**Risk**: Medium - first real migration

---

### Phase 3: Notes Layer Migration (High Impact)
**Goal**: Convert main notes drawing to OpenCV (biggest bottleneck)

**Tasks**:
1. Rewrite `draw_note()` to accept cv2 canvas instead of PIL draw object
2. Implement rounded rectangles in cv2 (or use regular rectangles initially)
3. Simplify/remove motion blur (may not be working anyway)
4. Keep glow effect but reduce to 1 pass instead of multiple
5. Test with actual MIDI files for visual correctness

**Success Criteria**:
- All note types render correctly (kick, snare, hihat, toms, cymbals)
- 40-60% overall frame time improvement
- Acceptable visual quality (can iterate on prettiness later)

**Risk**: High - most complex layer, most visual impact

---

### Phase 4: Eliminate Layer Compositing (High Impact)
**Goal**: Draw everything directly to single buffer instead of separate layers

**Tasks**:
1. Change render loop to use single cv2 canvas
2. Draw in correct Z-order: kick → lanes → notes → strike line → UI
3. Implement basic transparency for highlights (alpha blend manually)
4. Remove all `.paste()` operations
5. Remove PIL→cv2 conversion at end (already cv2)

**Success Criteria**:
- 60-80% overall frame time improvement vs original
- Single canvas throughout render loop
- Memory usage reduced (no multiple layers)

**Risk**: Medium - significant architectural change

---

### Phase 5: Optimizations (Performance Tuning)
**Goal**: Further optimize the OpenCV path

**Tasks**:
1. Pre-calculate reusable shapes (rounded rect masks)
2. Batch similar draw operations
3. Use cv2.addWeighted() for efficient alpha blending
4. Profile and optimize hot paths
5. Consider GPU acceleration (cv2.cuda if available)

**Success Criteria**:
- 80-90% improvement vs original
- < 5 seconds per minute of video on typical hardware
- Smooth rendering without stuttering

**Risk**: Low - pure optimization, no functional changes

---

### Phase 6: Polish & Cleanup (Low Risk)
**Goal**: Remove PIL completely, improve visual quality

**Tasks**:
1. Remove all PIL imports and helper functions
2. Add back better rounded corners using cv2.ellipse
3. Improve anti-aliasing with cv2.LINE_AA
4. Add subtle blur effects using cv2.GaussianBlur
5. Update documentation

**Success Criteria**:
- Zero PIL dependencies
- Visual quality equal or better than original
- Clean, maintainable code

**Risk**: Low - cleanup phase

---

## Rollback Strategy
- Each phase keeps `use_opencv` flag for instant rollback
- Commit after each phase passes tests
- Can revert to last working commit
- Original PIL code remains until Phase 6

## Success Metrics
- **Performance**: 5-10x faster frame rendering
- **Quality**: Visual parity with original (< 5% pixel diff on test frames)
- **Stability**: No crashes, all MIDI files render correctly
- **Maintainability**: Cleaner code, fewer dependencies

## Estimated Timeline
- Phase 1: 2-3 hours
- Phase 2: 3-4 hours
- Phase 3: 6-8 hours (most complex)
- Phase 4: 4-5 hours
- Phase 5: 3-4 hours
- Phase 6: 2-3 hours

**Total**: 20-27 hours of focused work, spread over multiple sessions

## Open Questions
1. Should we keep motion blur or remove it? (May not be working)
2. Target visual quality: Pixel-perfect or "good enough"?
3. GPU acceleration: Worth the complexity? (macOS Metal via cv2.dnn?)
4. Font rendering: PIL has good font support, cv2 is more limited - acceptable tradeoff?
