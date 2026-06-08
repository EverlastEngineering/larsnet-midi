"""
Tests for the ``detection_method`` WebUI form integration.

Covers:
  * the form template contains the ``detection_method`` <select>
  * the settings API exposes ``detection_method`` (schema, /setting/...)
  * the value is persisted to the project's midiconfig.yaml via
    ``POST /api/config/<id>/midiconfig`` (the channel the form
    submission would use to actually save the dropdown selection).

The form is hand-rolled (not schema-driven) — it lives in
``webui/templates/index.html`` and its values flow through the
localStorage-backed ``SettingsManager`` to either
``/api/stems-to-midi`` (per-run) or ``/api/config/<id>/midiconfig``
(persistent save). The test for template presence uses
plain string search on the file (no JS runtime).
"""
import json
import re
from pathlib import Path

import pytest

from webui.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / 'webui' / 'templates' / 'index.html'


@pytest.fixture
def app():
    app = create_app('testing')
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def tmp_project(tmp_path):
    """Create a tmp project dir with a minimal midiconfig.yaml so
    the persistent-save tests don't touch the user's real config."""
    import yaml
    from unittest.mock import patch

    project_dir = tmp_path / '1 - test_dm_project'
    project_dir.mkdir()
    config_path = project_dir / 'midiconfig.yaml'
    config_path.write_text(yaml.safe_dump({
        'onset_detection': {
            'threshold': 0.3,
            'delta': 0.01,
            'wait': 3,
            'hop_length': 512,
        },
    }))

    fake_project = {
        'number': 1,
        'name': 'test_dm_project',
        'path': project_dir,
    }
    p = patch('project_manager.get_project_by_number', return_value=fake_project)
    p.start()
    yield project_dir, config_path
    p.stop()


# ---------------------------------------------------------------------------
# 1. Template: form contains the dropdown
# ---------------------------------------------------------------------------

class TestDetectionMethodFormTemplate:
    """The hand-rolled WebUI form must render a <select> for
    ``detection_method`` with the three required options, defaulting
    to "Both (recommended)"."""

    def test_template_has_detection_method_select(self):
        assert INDEX_HTML.exists(), f"missing {INDEX_HTML}"
        text = INDEX_HTML.read_text()
        assert 'id="setting-detection-method"' in text, (
            "WebUI form must include a <select id='setting-detection-method'> "
            "in the MIDI settings panel so the SettingsManager wires it up"
        )

    def test_template_has_three_required_options(self):
        text = INDEX_HTML.read_text()
        # The dropdown must offer the three choice values from the schema.
        # (The <option> tags should live inside the select, but a substring
        # search is sufficient for this contract.)
        assert re.search(
            r'<select[^>]*id="setting-detection-method".*?</select>',
            text, re.DOTALL,
        ), "select wrapper must contain all three options"

        select_block = re.search(
            r'<select[^>]*id="setting-detection-method".*?</select>',
            text, re.DOTALL,
        ).group(0)
        assert 'value="energy"' in select_block
        assert 'value="spectral"' in select_block
        assert 'value="both"' in select_block

    def test_template_default_is_both_recommended(self):
        """The 'Both (recommended)' option must be selected by default
        so new projects start in the recommended state."""
        text = INDEX_HTML.read_text()
        select_block = re.search(
            r'<select[^>]*id="setting-detection-method".*?</select>',
            text, re.DOTALL,
        ).group(0)
        assert 'selected' in select_block, (
            "the 'both' option must carry the 'selected' attribute so "
            "new projects start with detection_method='both'"
        )
        # And the 'both' option must be the one marked selected.
        both_option = re.search(
            r'<option[^>]*value="both"[^>]*>[^<]*</option>',
            select_block,
        )
        assert both_option is not None, "missing <option value='both'>"
        assert 'selected' in both_option.group(0), (
            "the <option value='both'> must be selected by default"
        )
        assert 'Both (recommended)' in both_option.group(0)

    def test_template_in_midi_settings_panel(self):
        """The dropdown must live in the MIDI settings panel — that's
        where ONSET_DETECTION settings are surfaced in this form."""
        text = INDEX_HTML.read_text()
        # The select must be inside the #settings-midi panel.
        midi_panel = re.search(
            r'<div id="settings-midi".*?</div>\s*</div>\s*</div>',
            text, re.DOTALL,
        )
        assert midi_panel is not None, (
            "could not locate #settings-midi panel"
        )
        assert 'id="setting-detection-method"' in midi_panel.group(0), (
            "the detection_method dropdown must live inside the "
            "#settings-midi panel"
        )


# ---------------------------------------------------------------------------
# 2. Settings API: GET exposes detection_method
# ---------------------------------------------------------------------------

class TestDetectionMethodSettingsApi:
    """The settings API (schema + per-setting endpoints) must
    expose detection_method so the JS form can be built from the
    schema in the future (and so the unit-level test confirms the
    schema is wired)."""

    def test_get_schema_includes_detection_method(self, client):
        response = client.get('/api/settings/schema')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'settings' in data
        assert 'detection_method' in data['settings']
        entry = data['settings']['detection_method']
        assert entry['key'] == 'detection_method'
        assert entry['type'] == 'choice'
        assert entry['default'] == 'both'
        assert set(entry['allowed_values']) == {'energy', 'spectral', 'both'}

    def test_get_schema_lists_detection_method_in_onset_category(self, client):
        response = client.get('/api/settings/schema')
        assert response.status_code == 200
        data = json.loads(response.data)
        cat = data['categories']['onset_detection']
        assert 'detection_method' in cat['settings']

    def test_get_single_setting_returns_detection_method(self, client):
        """The /api/settings/setting/<key> endpoint returns the
        single-setting dict — the form would use this to render
        the dropdown if it were schema-driven."""
        response = client.get('/api/settings/setting/detection_method')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['key'] == 'detection_method'
        assert data['default'] == 'both'
        assert set(data['allowed_values']) == {'energy', 'spectral', 'both'}

    def test_get_single_setting_404_for_unknown(self, client):
        response = client.get('/api/settings/setting/nonexistent_key')
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 3. Persistence: POST persists detection_method to midiconfig.yaml
# ---------------------------------------------------------------------------

class TestDetectionMethodPersistence:
    """The form's save path is ``POST /api/config/<id>/midiconfig``
    with a dotted-path updates list. The dropdown value must land
    in ``onset_detection.detection_method`` in the YAML."""

    def test_post_spectral_persists_to_yaml(self, client, tmp_project):
        project_dir, config_path = tmp_project
        response = client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['onset_detection', 'detection_method'],
                 'value': 'spectral'},
            ]},
        )
        assert response.status_code == 200, response.data
        data = json.loads(response.data)
        assert data['success'] is True

        # Reload the YAML from disk and verify the value was written.
        import yaml
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved['onset_detection']['detection_method'] == 'spectral'

    def test_post_energy_persists_to_yaml(self, client, tmp_project):
        project_dir, config_path = tmp_project
        response = client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['onset_detection', 'detection_method'],
                 'value': 'energy'},
            ]},
        )
        assert response.status_code == 200, response.data
        import yaml
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved['onset_detection']['detection_method'] == 'energy'

    def test_post_both_persists_to_yaml(self, client, tmp_project):
        project_dir, config_path = tmp_project
        response = client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['onset_detection', 'detection_method'],
                 'value': 'both'},
            ]},
        )
        assert response.status_code == 200, response.data
        import yaml
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved['onset_detection']['detection_method'] == 'both'

    def test_get_config_returns_detection_method_after_save(
        self, client, tmp_project,
    ):
        """End-to-end: save via POST, then GET the config and confirm
        the value comes back. This is the round-trip the WebUI
        depends on when the user reopens the settings dialog."""
        project_dir, config_path = tmp_project

        # Save spectral
        client.post(
            '/api/config/1/midiconfig',
            json={'updates': [
                {'path': ['onset_detection', 'detection_method'],
                 'value': 'spectral'},
            ]},
        )

        # Read back
        response = client.get('/api/config/1/midiconfig')
        assert response.status_code == 200
        data = json.loads(response.data)
        # Walk sections to find onset_detection.detection_method
        flat = {}
        for section in data['sections']:
            for field in section['fields']:
                flat[field['path']] = field.get('value')
        assert 'onset_detection.detection_method' in flat, (
            f"expected onset_detection.detection_method in {list(flat)}"
        )
        assert flat['onset_detection.detection_method'] == 'spectral'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
