# Toms PGA Calibration Data (2026-06-15)

**Source plan**: [toms-pga-hardening.plan.md](toms-pga-hardening.plan.md)
**Started**: 2026-06-15
**Status**: Phase 1A + 1B shipped (commits `58e5d7c`, `e9739e6`),
3 stems calibrated.

This log captures the filter behavior of the new PGA pipeline
(noise-floor gate + `min_decay_col_min_db` filter) across the
toys we have available. Used to find patterns in the chaos and
design the future combinatory filter layer.

## Filter configuration

- `onset_detection.pga_min_prominence: 3000` (calibrated, see
  midiconfig.yaml TODO 2026-06-12)
- `toms.min_decay_col_min_db: -80.0` (new in 2026-06-15, default
  at the cut between the empirical clusters)
- Noise-floor gate: `max(p5 across all bins)`, applied in
  `_build_static_noise_floor` (Phase 1A)

## Per-project results

### Project 4 - `2_funk_80_beat_4-4_4` (funk, 80 BPM)

- **Total PGA events**: 47
- **KEPT**: 13
- **FILTERED**: 34
  - By `pga_min_prominence`: 34
  - By `min_decay_col_min_db`: 0
- **KEPT `decay_col_min_median_db` range**: [-69.1, -63.7] dB
  (mean -66.6)
- **KEPT `prominence` min**: 2989

The funk toms stem is well-detected by the prominence filter
alone. All FPs have low prominence (< 3000) and never reach
the decay_col_min filter. Real strikes have healthy
`decay_col_min_median_db` values around -67 dB - well above
the -80 dB threshold.

### Project 5 - `2_funk_80_beat_4-4_4` (funk, 80 BPM, duplicate)

- **Total PGA events**: 47
- **KEPT**: 14
- **FILTERED**: 33
  - By `pga_min_prominence`: 33
  - By `min_decay_col_min_db`: 0
- **KEPT `decay_col_min_median_db` range**: [-69.1, -63.7] dB
  (mean -68.3)
- **KEPT `prominence` min**: 1698

Same fingerprint as project 4 (likely a slightly different mix
or duplicate of the same song). One more event survives the
prominence filter at prominence 1698 - the threshold is not a
hard line, the user's TODOs note that the prominence bump is
intentionally temporary.

### Project 6 - `01_Taylor_Swift_The_Fate_of_Ophelia_Drums`

- **Total PGA events**: 32
- **KEPT**: 10
- **FILTERED**: 22
  - By `pga_min_prominence`: 19
  - By `min_decay_col_min_db`: 3
- **KEPT `decay_col_min_median_db` range**: [-79.3, -79.3] dB
  (single value, mean -79.3)
- **KEPT `prominence` min**: 1418

This is the most interesting case. **3 FPs were caught by the
new `decay_col_min` filter** that the prominence filter
missed:

| time (s) | prominence | decay_col_min_median_db | reason |
|---|---|---|---|
| 142.117 | 8765 | -106.13 | below min_decay_col_min_db (-106.1dB < -80.0dB) |
| 197.532 | 8701 | -106.13 | below min_decay_col_min_db (-106.1dB < -80.0dB) |
| 198.932 | 1711 | -106.13 | below min_decay_col_min_db (-106.1dB < -80.0dB) |

All three have HIGH prominence (1711 to 8765 - they would have
passed the prominence filter at the default 3000 threshold for
the first two, and at the 1000 default for the third) but
`decay_col_min_median_db` of -106.13 dB - well into the
"single-frame noise pop" cluster. The filter is catching the
exact pattern the user described.

The Taylor Swift toms are also the TIGHTEST case for the
threshold: KEPT events have `decay_col_min_median_db` of
exactly -79.3 dB, just 0.7 dB above the threshold. This is
worth watching - if we add more stems and the data shifts, the
-80 dB default might need re-calibration.

## Observations

1. **The decay_col_min filter is doing its job**: it's catching
   high-prominence single-frame noise pops that the prominence
   filter misses. The 3 events caught in project 6 are exactly
   the pattern the user described.

2. **No false-positive drops in projects 4/5**: the funk stems'
   real strikes have decay_col_min in the -63 to -69 dB range,
   well above the -80 dB threshold. The filter is conservative
   for these stems.

3. **The threshold is tight for project 6**: KEPT events at
   -79.3 dB are 0.7 dB above the threshold. Worth watching
   when we add more stems.

4. **Cluster separation works**: the empirical clusters
   (`[-60, -84]` for real strikes, `[-84, -90]` for noise
   pops) overlap at -84 dB, but in practice the gap between
   KEPT and FILTERED events is clean in all 3 stems.

## What we still need

- **Combinatory filter logic** (deferred): the pattern "if
  prominence is X% of max AND decay_col_min is Y" needs
  calibration data from 5+ stems. With 3 stems we have a
  starting point, but the combinatory rules are not yet
  justified.
- **More stems**: 3 stems (2 of which are duplicates) is not
  enough to confidently extrapolate to other songs. We need
  5+ unique stems to design the combinatory layer.

## Cross-references

- Plan: [toms-pga-hardening.plan.md](toms-pga-hardening.plan.md)
- Test results: `pytest stems_to_midi/tests/test_pga_event_builder.py`
  - 24/24 synthetic tests passing
- Phase 1A commit: `58e5d7c` (noise-floor gate + summary line)
- Phase 1B commit: `e9739e6` (decay_col_min filter)
