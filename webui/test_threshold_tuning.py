"""
Tests for Threshold Tuning Feature (Step 5)

Validates:
  - HTML structure includes tuning panel and controls
  - threshold-tuning.js is loaded
  - Client-side filtering logic correctness (Python mirror of JS logic)
  - Waveform.js tuning state variables exist

Run with: pytest webui/test_threshold_tuning.py
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.app import create_app


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    """Create test Flask app."""
    return create_app('testing')


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


# ─── HTML Structure Tests ─────────────────────────────────────────────────

class TestTuningPanelHTML:
    """Verify the tuning panel markup is present in the rendered page."""

    def test_tuning_toggle_button_exists(self, client):
        """The Tune button should exist in the analysis section."""
        response = client.get('/')
        html = response.data.decode()
        assert 'id="tuning-toggle-btn"' in html

    def test_tuning_panel_exists(self, client):
        """The tuning panel container should exist."""
        response = client.get('/')
        html = response.data.decode()
        assert 'id="tuning-panel"' in html

    def test_tuning_panel_hidden_by_default(self, client):
        """The tuning panel should be hidden by default."""
        response = client.get('/')
        html = response.data.decode()
        # Find the tuning-panel div and verify it has 'hidden' class
        idx = html.index('id="tuning-panel"')
        div_start = html.rfind('<div', 0, idx)
        div_snippet = html[div_start:idx + 50]
        assert 'hidden' in div_snippet

    def test_tuning_sliders_container_exists(self, client):
        """The slider container should exist."""
        response = client.get('/')
        html = response.data.decode()
        assert 'id="tuning-sliders"' in html

    def test_tuning_event_counts_exists(self, client):
        """The event count display should exist."""
        response = client.get('/')
        html = response.data.decode()
        assert 'id="tuning-event-counts"' in html

    def test_tuning_reset_button_exists(self, client):
        """The reset button should exist."""
        response = client.get('/')
        html = response.data.decode()
        assert 'resetTuningSliders()' in html

    def test_threshold_tuning_js_loaded(self, client):
        """threshold-tuning.js should be included in the page."""
        response = client.get('/')
        html = response.data.decode()
        assert 'threshold-tuning.js' in html


class TestTuningPanelCSS:
    """Verify tuning-specific CSS is present."""

    def test_tuning_range_style(self, client):
        """Custom range slider styles should be defined."""
        response = client.get('/')
        html = response.data.decode()
        assert '.tuning-range' in html

    def test_tuning_btn_active_style(self, client):
        """Active state for the Tune button should be defined."""
        response = client.get('/')
        html = response.data.decode()
        assert '.tuning-btn-active' in html

    def test_tuning_indicator_animation(self, client):
        """Pulsing indicator animation should be defined."""
        response = client.get('/')
        html = response.data.decode()
        assert 'tuning-pulse' in html


# ─── JavaScript File Tests ────────────────────────────────────────────────

class TestThresholdTuningJS:
    """Verify threshold-tuning.js has expected structure."""

    @pytest.fixture
    def js_content(self):
        js_path = Path(__file__).parent / 'static' / 'js' / 'threshold-tuning.js'
        return js_path.read_text()

    def test_file_exists(self):
        js_path = Path(__file__).parent / 'static' / 'js' / 'threshold-tuning.js'
        assert js_path.exists(), 'threshold-tuning.js should exist'

    def test_has_toggle_function(self, js_content):
        assert 'function toggleTuningPanel()' in js_content

    def test_has_stem_changed_callback(self, js_content):
        assert 'function onTuningStemChanged(' in js_content

    def test_has_reset_function(self, js_content):
        assert 'function resetTuningSliders()' in js_content

    def test_has_spectral_filter(self, js_content):
        assert 'function applySpectralFilter(' in js_content

    def test_has_reverb_continuation_filter(self, js_content):
        assert 'function applyReverbContinuationFilter(' in js_content

    def test_has_slider_configs_for_all_stems(self, js_content):
        for stem in ['kick', 'snare', 'toms', 'hihat', 'cymbals']:
            assert f"'{stem}':" in js_content or f'{stem}:' in js_content, \
                f'Slider config for {stem} should exist'

    def test_has_filter_modes(self, js_content):
        assert 'geomean_only' in js_content
        assert 'require_both' in js_content

    def test_has_debounced_input_handler(self, js_content):
        assert 'requestAnimationFrame' in js_content

    def test_has_event_count_updater(self, js_content):
        assert 'function updateEventCounts(' in js_content


class TestWaveformTuningIntegration:
    """Verify waveform.js has tuning state variables."""

    @pytest.fixture
    def waveform_js(self):
        js_path = Path(__file__).parent / 'static' / 'js' / 'waveform.js'
        return js_path.read_text()

    def test_tuning_events_state(self, waveform_js):
        assert 'waveformTuningEvents' in waveform_js

    def test_tuning_active_state(self, waveform_js):
        assert 'waveformTuningActive' in waveform_js

    def test_tuning_indicator_referenced(self, waveform_js):
        """Tuning label element is toggled by waveform.js."""
        assert 'waveform-tuning-label' in waveform_js

    def test_stem_change_notifies_tuning(self, waveform_js):
        assert 'onTuningStemChanged' in waveform_js


# ─── Filtering Logic Tests (Python Mirror) ────────────────────────────────

def apply_spectral_filter_py(events, params, filter_mode):
    """
    Python mirror of the JS applySpectralFilter function.
    Used to verify the logic is correct.

    Note (2026-06-09, bug fix): spectral events (method='spectral')
    are EXEMPT from the geomean / sustain / strength filters. Those
    signals are properties of the energy detector; spectral events
    have band_powers / band_max_ratio / band_delta / snap_delta
    instead. The previous behavior filtered them whenever geomean
    was null (which it always is for spectral events), which
    silently destroyed all magenta events when the user dragged
    the geomean slider.
    """
    geomean_threshold = params.get('geomean_threshold')
    min_sustain_ms = params.get('min_sustain_ms')
    min_strength = params.get('min_strength_threshold')

    for event in events:
        event['status'] = 'KEPT'

        # Spectral events are exempt — see JS applySpectralFilter
        # for the rationale.
        if event.get('method') == 'spectral':
            continue

        # Strength gate
        if min_strength is not None and event.get('strength') is not None:
            if event['strength'] < min_strength:
                event['status'] = 'FILTERED'
                continue

        # No thresholds → keep
        if geomean_threshold is None and min_sustain_ms is None:
            continue

        if filter_mode == 'require_both':
            if geomean_threshold is not None and min_sustain_ms is not None:
                pass_geomean = event.get('geomean') is not None and event['geomean'] > geomean_threshold
                pass_sustain = event.get('sustain_ms') is not None and event['sustain_ms'] >= min_sustain_ms
                if not pass_geomean or not pass_sustain:
                    event['status'] = 'FILTERED'
            elif min_sustain_ms is not None:
                if event.get('sustain_ms') is None or event['sustain_ms'] < min_sustain_ms:
                    event['status'] = 'FILTERED'
            elif geomean_threshold is not None:
                if event.get('geomean') is None or event['geomean'] <= geomean_threshold:
                    event['status'] = 'FILTERED'
        else:
            # geomean_only
            if geomean_threshold is not None:
                if event.get('geomean') is None or event['geomean'] <= geomean_threshold:
                    event['status'] = 'FILTERED'


def apply_reverb_continuation_filter_py(events, attack_threshold):
    """Python mirror of JS applyReverbContinuationFilter."""
    TIME_MARGIN = 0.005
    AMP_MARGIN = 0.001

    events.sort(key=lambda e: e.get('time', 0))

    for i in range(1, len(events)):
        curr = events[i]
        prev = events[i - 1]

        if curr['status'] != 'KEPT':
            continue
        if prev['status'] not in ('KEPT', 'REVERB_CONTINUATION'):
            continue

        if prev.get('duration_sec') is None or prev.get('amplitude_at_end') is None or \
           curr.get('amplitude_at_start') is None:
            continue

        prev_end_time = prev['time'] + prev['duration_sec']
        gap = curr['time'] - prev_end_time
        is_adjacent = abs(gap) <= TIME_MARGIN

        amp_diff = abs(curr['amplitude_at_start'] - prev['amplitude_at_end'])
        is_amplitude_continuous = amp_diff <= AMP_MARGIN

        is_smooth = curr.get('attack_sharpness') is not None and \
                    curr['attack_sharpness'] < attack_threshold

        if is_adjacent and is_amplitude_continuous and is_smooth:
            curr['status'] = 'REVERB_CONTINUATION'

