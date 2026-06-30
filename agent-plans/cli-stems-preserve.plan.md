# Plan: Preserve Other Stems When `--stems <subset>` Is Used

## Problem

`python stems_to_midi_cli.py 6 --stems snare` re-processes ONLY snare but **overwrites** the entire `.analysis.json` sidecar with just the snare stem — wiping kick/hihat/toms/cymbals. The MIDI is then rebuilt from the now-stem-poor sidecar, so the `.mid` file also loses those stems. The user must rerun a full 5-stem conversion every time they tweak one stem.

### Reproduction (verified 2026-06-30)

1. Full conversion: `python stems_to_midi_cli.py 6` → sidecar has 5 stems (kick/snare/toms/hihat/cymbals)
2. Edit `snare.pga_min_prominence: 2400` in midiconfig.yaml
3. `python stems_to_midi_cli.py 6 --stems snare` → sidecar has ONLY snare (other 4 erased), MIDI has only snare

### Root Cause

`stems_to_midi_cli.py::stems_to_midi_for_project` (the orchestration function, lines ~210-320):

1. Loops over `stems_to_process` (just `snare` if `--stems snare`)
2. Builds `events_by_stem` and `analysis_by_stem` dicts containing ONLY those stems
3. Calls `save_analysis_sidecar(events_by_stem, midi_path, analysis_by_stem=analysis_by_stem)` which iterates `events_by_stem.items()` and writes `sidecar_data['stems']` from scratch
4. Calls `rebuild_events_from_analysis(analysis_data=...)` which uses only the stems in the (now-stem-poor) sidecar → MIDI loses those stems

`save_analysis_sidecar` is a pure-write function with no merge logic — that's correct. The bug is in the CLI which doesn't consult the existing sidecar before saving.

## Approach

Load-merge-save at the CLI orchestration layer. The CLI is the only place that knows the user invoked `--stems`. The merge preserves the existing sidecar's data for stems not in `--stems`, and overwrites only the stems that were re-processed.

### Phase 1: Implement load-merge-save in CLI

**1a. New helper: `_deserialize_sidecar_stems_for_merge`**

Pure function. Takes a sidecar data dict (output of `load_analysis_sidecar`) and a list of stems NOT in `stems_to_process`. Returns two dicts in the shapes `save_analysis_sidecar` expects:

- `midi_events_by_stem`: `{stem: [{time, note, velocity, duration, hihat_state}, ...]}` — built by filtering `events_pga` to `status='KEPT'` and mapping fields:
  - `time` → `time`
  - `note` → `note` (already set by classify_notes at original process time)
  - `midi_velocity` → `velocity`
  - `duration_ms` / 1000 → `duration` (clamped to existing `max_note_duration` config)
  - `hihat_state` → `hihat_state` (only hihat has this)
- `analysis_data_by_stem`: `{stem: {'pga_onset_data': events_pga, ...}}` — wraps the existing `events_pga` as `pga_onset_data` so `save_analysis_sidecar`'s existing serialization path runs unchanged.

**Signature**:
```python
def _deserialize_sidecar_stems_for_merge(
    existing_sidecar: Dict,
    stems_to_preserve: List[str],
    config: Optional[Dict] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    ...
```

Returns `(midi_events_by_stem, analysis_by_stem)`.

**1b. Modify `stems_to_midi_for_project`**

After the per-stem loop (lines ~290) and BEFORE `save_analysis_sidecar` (line ~303):

```python
# 2026-06-30: preserve other stems' data when --stems is a subset.
# Without this, save_analysis_sidecar overwrites the sidecar
# completely and the MIDI loses non-reprocessed stems.
stems_to_preserve = [
    s for s in existing_sidecar.get('stems', {}).keys()
    if s not in stems_to_process
]
if stems_to_preserve:
    print(f"  Preserving {len(stems_to_preserve)} non-reprocessed stem(s) from existing sidecar: {stems_to_preserve}")
    preserved_midi, preserved_analysis = _deserialize_sidecar_stems_for_merge(
        existing_sidecar, stems_to_preserve, config=config,
    )
    events_by_stem.update(preserved_midi)
    analysis_by_stem.update(preserved_analysis)
```

Where `existing_sidecar` is loaded once at the top of `stems_to_midi_for_project`:

```python
# 2026-06-30: load existing sidecar to preserve non-reprocessed stems.
existing_sidecar_path = midi_dir / f"{base_name}.analysis.json"
existing_sidecar = (
    load_analysis_sidecar(existing_sidecar_path) or {}
)
```

If no existing sidecar (first run on this project), `existing_sidecar` is `{}` and the merge is a no-op — current behavior preserved.

### Phase 2: Tests

**2a. New test file: `tests/test_stems_subset_preservation.py`**

Pure-function tests on `_deserialize_sidecar_stems_for_merge`:

1. `test_empty_sidecar_returns_empty_dicts`: empty input → empty output, no error
2. `test_preserves_ke_events_only`: sidecar has 100 events (90 KEPT, 10 FILTERED) → output has 90 MIDI events
3. `test_extracts_midi_event_fields_correctly`: KEPT event with note=38, velocity=112, duration_ms=150 → MIDI event has note=38, velocity=112, duration=0.15
4. `test_hihat_state_preserved`: hihat KEPT event with hihat_state='open' → MIDI event has hihat_state='open'
5. `test_stems_to_preserve_filters_correctly`: sidecar has 5 stems, stems_to_preserve=[kick, hihat] → only those 2 stems appear in output
6. `test_clamps_duration_to_max_note_duration`: KEPT event with duration_ms=2000 → MIDI duration is min(2.0, config.max_note_duration)

**2b. E2E test on project 6**

End-to-end: run full conversion, capture sidecar, then run `--stems snare`, verify other stems are intact:

```python
def test_cli_subset_preserves_other_stems(tmp_path, project_6_dir):
    """Run full conversion, then --stems snare, verify other stems preserved."""
    # ... full conversion ...
    full_sidecar = load_analysis_sidecar(midi_path)
    assert set(full_sidecar['stems'].keys()) == {'kick', 'snare', 'toms', 'hihat', 'cymbals'}

    # ... --stems snare ...
    subset_sidecar = load_analysis_sidecar(midi_path)

    # Other stems must have identical events_pga counts
    for stem in ('kick', 'toms', 'hihat', 'cymbals'):
        assert len(subset_sidecar['stems'][stem]['events_pga']) == \
               len(full_sidecar['stems'][stem]['events_pga'])

    # Snare events_pga must differ (re-processed)
    assert len(subset_sidecar['stems']['snare']['events_pga']) != \
           len(full_sidecar['stems']['snare']['events_pga'])
```

Run as: `python stems_to_midi_cli.py 6` (full), then `python stems_to_midi_cli.py 6 --stems snare` (subset), then verify sidecar.

### Phase 3: Edge cases & polish

1. **Learning mode**: `--learn` with `--stems` — currently learning mode processes all stems. Don't merge when `learning_mode=True` — the merge would preserve old events that don't match the new learning-mode velocity=1 annotation. Add an early return.

2. **Override files**: `event_overrides.json` is read at rebuild time, not written by this path. No interaction with the merge.

3. **Sidecar schema version bump**: The sidecar has a `version: '3.0'` field. Don't bump it — the merge produces the same v3.0 sidecar, just with more stems. If we ever add a v3.1 field, we can bump then.

4. **WebUI's "Reconvert" path**: Likely has the same bug (single-stem changes wiping others). Out of scope for this fix — file a follow-up issue. Verify by inspecting `webui/api/reconvert.py`.

5. **Atomic write**: `save_analysis_sidecar` should write the merged sidecar atomically (write to temp file, rename) to avoid corruption if the write is interrupted. Check existing implementation; add if missing.

## Files Changed

1. `stems_to_midi_cli.py`:
   - Add `_deserialize_sidecar_stems_for_merge` helper (~40 lines)
   - Modify `stems_to_midi_for_project` to load existing sidecar and merge (~15 lines)

2. `stems_to_midi/tests/test_stems_subset_preservation.py` (NEW):
   - 6 unit tests on the merge helper (~120 lines)

3. `agent-plans/cli-stems-preserve.results.md` (NEW):
   - Tracking file (mutable)

## Risks

- **Sidecar size growth**: Running `--stems snare` 5 times in a row produces the same merged sidecar each time (deterministic merge). No growth.
- **MIDI byte-identical for non-reprocessed stems**: After merge, the sidecar's events_pga for kick/toms/hihat/cymbals is byte-identical (same JSON values). The MIDI rebuild produces the same MIDI events for those stems. The MIDI file's overall byte content may differ by timestamps/headers (because the rebuild step re-encodes the whole file), but the note events for non-reprocessed stems are identical.
- **Sidecar schema drift**: If a future code change modifies what `events_pga` looks like (adds a field, renames one), the merge of OLD sidecar data into NEW code path could lose fields. The `_deserialize_sidecar_stems_for_merge` helper is the natural seam to add schema-version handling later.
- **Override file interaction**: None — overrides are read at rebuild time, not by this path.

## Success Criteria

1. Run `python stems_to_midi_cli.py 6` — sidecar has all 5 stems.
2. Run `python stems_to_midi_cli.py 6 --stems snare` — sidecar STILL has all 5 stems; only snare's `events_pga` differs from the first run (because the threshold may have changed).
3. The MIDI file's note events for kick/toms/hihat/cymbals are byte-identical between the two runs (timestamps and headers may differ; notes don't).
4. New tests pass; full pytest run shows no regressions beyond the 4 pre-existing failures.
5. The CLI prints "Preserving N non-reprocessed stem(s) from existing sidecar: [...]" so the user can see what's happening.

## Estimated Effort

- **Code**: ~55 lines (40 new helper + 15 modification)
- **Tests**: ~120 lines (6 unit tests + 1 E2E)
- **Plan/Results**: ~75 lines
- **Single commit**, ~5-10 minutes to write + test + commit
- **Post-fix verification**: 2 CLI runs + 1 sidecar diff = ~30 seconds