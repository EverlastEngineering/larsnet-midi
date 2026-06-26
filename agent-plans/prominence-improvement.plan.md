# Plan: Improve PGA Prominence Calculation

**Date**: 2026-06-22
**Status**: Draft / pre-implementation
**Owner**: TBD (depends on which task scope picks this up)

---

## Problem statement

The PGA detector's `prominence` value, used by the `pga_min_prominence` filter,
underestimates the true depth of a percussive strike when the strike is part of
a double-trigger (two envelope peaks within ~12 ms of each other). The
leading-edge peak's `prominence` collapses to ~47 (essentially the small dip
between the two peaks) while the second peak's `prominence` is ~11356 (the
full depth of the strike from the deep valley on either side).

Observed on project 4 (hihat stem), time ~73.12s:

```json
// event A — first peak of the strike (FILTERED, prominence ≈ 47)
{"time": 73.1218, "frame": 12594, "envelope_value": 11341.68,
 "prominence": 46.8807, "right_base_minus_peak_frames": 1, ...}
// event B — second peak of the strike (KEPT, prominence ≈ 11356)
{"time": 73.1339, "frame": 12596, "envelope_value": 11418.95,
 "prominence": 11356.441, "right_base_minus_peak_frames": 155, ...}
```

The current filter still catches the strike via event B, so the MIDI output
is unaffected. But the rejection of event A produces a misleading sidecar
tooltip ("below pga_min_prominence (47 < 800)" on a real strike) and a
cascading null `duration_ms` on event A (because `_find_attack_peak` merges
event A into event B's peak).

## Goals

1. **Primary**: replace the local-prominence calculation with one that
   reflects the strike's depth relative to the broader envelope context, so
   double-trigger leading edges get a prominence value consistent with the
   strike's actual depth.

2. **Secondary**: ensure `duration_ms` and `attack_rise_ms` on double-trigger
   leading edges are computed against the event's *own* peak, not the
   neighboring peak's height.

3. **Tertiary**: preserve all current calibration. The replacement must not
   regress real-strike detection on the existing test set (project 4, project
   6, the Taylor Swift toms run, the calibration suite).

## Non-goals

- Changing the `pga_min_prominence` filter threshold defaults. The threshold
  resolution order stays the same.
- Removing the existing `decay_col_min_median_db` filter or `duration_ms`
  diagnostic. The layered filter chain is by design.
- Touching the snap/ring detector (`spectral_transient_core.py`). The issue is
  specific to the PGA detector.

## Constraints

- **Calibration data is limited.** The current calibration set is projects
  4 and 6 plus a few test fixtures. We have not yet collected enough
  double-trigger examples to confidently validate a new prominence metric.
- **Backward compatibility required.** Existing analysis sidecar files
  contain `prominence` values and downstream consumers (the WebUI filter
  evaluation, `rebuild_core`) read them. Changing the calculation must
  either (a) be a forward-compatible additive change, or (b) include a
  migration path for old sidecars.
- **The fix must not regress the existing layered filter chain.** Adding
  prominence as a new feature alongside the existing one (rather than
  replacing it) is the safest path.

## Approach options (to be evaluated)

### Option A: Regional prominence with fixed radius

Replace scipy's left/right base search with a fixed-radius search (e.g.,
±200 ms or ±20 frames on each side). The contour line is `min(envelope)`
in the window. This is straightforward to implement (one helper function
replacing scipy's algorithm) and gives a much more robust measure for
double-trigger scenarios.

**Pros**:
- Captures the strike's depth in a wider context.
- Cheap to compute (~100-element min over a fixed window).
- Easy to reason about and document.

**Cons**:
- A fixed radius is a magic number. Too small and we get the same
  double-trigger problem. Too large and a quiet strike in a sustained-ringing
  region gets a low prominence (the min of the window is the previous strike's
  sustained ring).
- Per-stem tuning would help (the toms ring can be 800 ms+, the snare ring
  is typically < 300 ms), but adds another config knob.

### Option B: Peak-relative-to-IQR prominence

Use the IQR threshold (`q3 + 2.5 * IQR` of the envelope, already computed at
peak-pick time) as the baseline. Prominence = `peak_height - iqr_threshold`.
This is mathematically well-defined, captures the strike's depth relative to
the song's bulk-envelope level, and is invariant to local neighbor topology.

**Pros**:
- Already have the IQR threshold on every event (no new computation).
- Conceptually simple: "how far above the song's typical envelope is this
  peak?"
- Robust to double-trigger (the IQR is a song-wide statistic, not a
  local-to-neighbor statistic).

**Cons**:
- The IQR threshold is dominated by the bulk distribution, which may be
  skewed by sustained ringing (a long ring raises q3 and IQR, lifting
  the threshold).
- Less directly comparable to scipy's prominence (filter sliders tuned to
  the current scipy values would need recalibration).

### Option C: Hybrid prominence (local-or-regional, take the max)

Compute both scipy's local prominence AND a regional prominence
(option A or B). Store both on the event (`prominence_local`,
`prominence_regional`). The filter uses `max(prominence_local,
prominence_regional)`. The filter reason template records which one
won.

**Pros**:
- Minimal risk: existing calibration (local prominence) is preserved as
  one of the two signals.
- The "real" strikes that already have high local prominence continue to
  be caught correctly.
- The double-trigger leading edge gets a high regional prominence and
  passes the filter.
- Easy to test against existing calibration: every event with high local
  prominence should still pass; events with low local prominence get a
  second chance.

**Cons**:
- Adds two new fields to the sidecar (modest storage impact).
- The filter logic is slightly more complex (max of two values).
- May allow FPs that previously failed (a sustained-ringing tail might
  have high regional prominence if it sits at a peak relative to the
  IQR threshold but low local prominence because the surrounding ringing
  is at similar level). Needs calibration to confirm.

### Option D: Adjust scipy's prominence by widening the search window

Modify the prominence calculation to use a wider search window than
scipy's default. scipy uses the entire signal by default; the issue is
not the window size but the algorithm's choice of max(left, right). So
this option is essentially option A repackaged — skip.

## Recommended approach (Option C)

The hybrid approach is the safest. Existing calibration is preserved
because `prominence_local` continues to behave exactly as it does today.
`prominence_regional` adds a second signal that captures double-trigger
leading edges. The filter takes the max, so any event that would have
passed before still passes.

The regional metric (option A vs B) needs a sub-decision:
- A is more local and easier to reason about
- B is more global and uses an existing computation
- A 3-month calibration window should decide between them

### Implementation plan

**Phase 1: data gathering (1-2 weeks)**

1. Collect a representative set of double-trigger examples across stems.
   Start with project 4 hihat (where the issue was first observed) and
   expand to snare, toms, cymbals.
2. For each example, compute both regional prominence variants (option A
   and B) and record them in a separate diagnostic file (not the main
   sidecar yet).
3. Compare against human-judged "should this event have been KEPT?"
   labels. Build a confusion matrix for each variant.

**Phase 2: implementation (1 week)**

4. Add `prominence_regional` computation to
   `_detect_percentile_gated_broad_attacks_impl` (the detector function).
   Choose between option A and B based on Phase 1 results.
5. Add the value to the per-event dict in `pga_event_builder.detect_pga_events`
   next to the existing `prominence` field.
6. Update `midi.py` serializer to include `prominence_regional` in the
   sidecar JSON.
7. Update `filter_registry.json` to add a new filter
   `pga_min_prominence_hybrid` that uses `max(prominence_local,
   prominence_regional) >= threshold`. Mark the existing
   `pga_min_prominence` filter as deprecated-but-supported.

**Phase 3: validation (1-2 weeks)**

8. Re-run the calibration suite. Compare MIDI output between old and new
   filters on the same audio. Verify the layered filter chain (prominence +
   decay_col_min + duration) still catches the FPs it used to.
9. Spot-check 20-30 known FPs and 20-30 known real strikes. Verify the
   new filter doesn't introduce regressions.
10. Compare event counts in the MIDI output before and after. A small
    increase in KEPT events (the leading-edge double-trigger cases) is
    expected; a large increase would indicate the regional metric is
    over-permissive.

**Phase 4: rollout**

11. If validation passes: update the WebUI to read the new hybrid filter.
    Keep the legacy `pga_min_prominence` slider visible-but-deprecated for
    one release cycle so users can compare.
12. Document in [`docs/prominence.md`](../docs/prominence.md) (the new file
    created alongside this plan).

## Risks

- **Risk**: the regional prominence variant is too permissive, allowing
  sustained-ringing FPs that the local prominence correctly rejected.
  **Mitigation**: Phase 1 calibration data is critical. Run both option A
  and option B variants; pick the one with the best FP rejection. The
  hybrid `max(...)` filter still requires the regional value to exceed
  the threshold, so a regional variant that lets noise through will be
  caught during Phase 3 validation.

- **Risk**: existing analysis sidecar files don't have
  `prominence_regional`. The legacy `prominence` value still works, so
  the WebUI can fall back to the legacy filter for old files.
  **Mitigation**: do not make `prominence_regional` required. Old sidecars
  work as-is. New runs compute both.

- **Risk**: the `duration_ms` and `attack_rise_ms` cascading bug (where
  `_find_attack_peak` merges event A into event B) is a separate issue
  that the prominence fix doesn't fully resolve. Even if event A passes
  the prominence filter, its duration_ms will still be null.
  **Mitigation**: a separate Phase 2 sub-task is to bound
  `_find_attack_peak` to the event's own reported peak (or to within a
  small window around the reported frame, not 50 ms forward).

## Open questions

1. What is the right window size for option A? 100 ms? 200 ms?
   Calibration data should decide. A toms ring can be 800 ms+, but a
   100 ms window should still capture the "deep valley before the
   attack" we want to see.

2. Does option B work for snare (where the IQR can be small because
   the dynamic range is narrow)? The IQR-auto threshold already has
   problems on snare; using it as the prominence baseline may inherit
   those problems.

3. Should the hybrid filter be a separate WebUI slider, or should it
   replace the existing prominence slider silently? A separate slider
   gives more control but adds UI complexity.

## Decision log

- **2026-06-22**: Plan created. Approach: Option C (hybrid max of local
  and regional). Regional metric to be decided after Phase 1 calibration.
  No code changes yet — data gathering first.
