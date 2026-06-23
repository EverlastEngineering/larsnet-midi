"""
Tests for the filter registry API (2026-06-15).

The /api/filters/* endpoints serve the JSON registry to the
WebUI. Python and JS both consume the SAME registry file,
so any drift is a bug — these tests lock the contract.
"""
import pytest

from webui.app import create_app


@pytest.fixture
def client():
    """Flask test client."""
    app = create_app('testing')
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestFilterSchemaEndpoint:
    """GET /api/filters/schema — full registry."""

    def test_returns_200(self, client):
        resp = client.get('/api/filters/schema')
        assert resp.status_code == 200

    def test_returns_valid_json(self, client):
        resp = client.get('/api/filters/schema')
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_has_version(self, client):
        data = client.get('/api/filters/schema').get_json()
        assert 'version' in data

    def test_has_kinds_enum(self, client):
        data = client.get('/api/filters/schema').get_json()
        assert 'kinds' in data
        assert isinstance(data['kinds'], list)
        assert 'min_value' in data['kinds']

    def test_has_filters_list(self, client):
        data = client.get('/api/filters/schema').get_json()
        assert 'filters' in data
        assert isinstance(data['filters'], list)
        # 2026-06-22: the registry has 4 entries. The Python
        # pipeline applies all four as a layered PGA chain
        # (envelope_value -> prominence -> decay_col_min ->
        # attack_rise_max). All four are WebUI-exposed as of
        # 2026-06-22 (envelope_value was added to the WebUI
        # for all 5 stems).
        assert len(data['filters']) == 4

    def test_pga_min_prominence_in_registry(self, client):
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'pga_min_prominence' in ids

    def test_decay_col_min_still_in_registry_but_hidden(self, client):
        # 2026-06-19: min_decay_col_min_db is still in the
        # registry (the Python pipeline still applies it as
        # part of the layered PGA chain) but is no longer
        # exposed in the WebUI. The entry is marked
        # `expose_in_webui: false` so the WebUI consumer
        # filters it out.
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'min_decay_col_min_db' in ids
        for f in data['filters']:
            if f['id'] == 'min_decay_col_min_db':
                assert f.get('expose_in_webui') is False

    def test_attack_rise_max_ms_still_in_registry_but_hidden(self, client):
        # 2026-06-19: attack_rise_max_ms is still in the
        # registry (Python still applies it) but is no
        # longer exposed in the WebUI.
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'attack_rise_max_ms' in ids
        for f in data['filters']:
            if f['id'] == 'attack_rise_max_ms':
                assert f.get('expose_in_webui') is False


class TestFilterByIdEndpoint:
    """GET /api/filters/<id> — single filter."""

    def test_existing_filter_returns_200(self, client):
        resp = client.get('/api/filters/pga_min_prominence')
        assert resp.status_code == 200

    def test_existing_filter_returns_full_spec(self, client):
        data = client.get('/api/filters/pga_min_prominence').get_json()
        assert data['id'] == 'pga_min_prominence'
        assert data['filter']['kind'] == 'min_value'
        assert data['filter']['field'] == 'prominence'
        assert 'yaml_paths' in data
        assert 'reason_template' in data['filter']

    def test_missing_filter_returns_404(self, client):
        resp = client.get('/api/filters/does_not_exist')
        assert resp.status_code == 404

    def test_missing_filter_error_message(self, client):
        data = client.get('/api/filters/does_not_exist').get_json()
        assert 'error' in data
        assert 'does_not_exist' in data['error']


class TestFiltersForStemEndpoint:
    """GET /api/filters/stem/<stem_type> — per-stem filter list.

    2026-06-19: pga_min_prominence is now the single WebUI filter
    and is exposed for ALL five stems (toms, snare, hihat, kick,
    cymbals). The previous decay_col_min and attack_rise_max_ms
    filters were removed from the WebUI; the previous
    geomean_threshold / min_sustain_ms / min_strength_threshold /
    reverb_continuation_attack_threshold energy-based filters
    were already removed in the snare→toms migration (2026-06-18).
    """

    def test_toms_returns_pga_min_prominence(self, client):
        data = client.get('/api/filters/stem/toms').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids

    def test_snare_returns_pga_min_prominence(self, client):
        # 2026-06-18: snare adopted the toms PGA-only slideout.
        data = client.get('/api/filters/stem/snare').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids

    def test_hihat_returns_pga_min_prominence(self, client):
        # 2026-06-19: hihat slideout is now PGA-only; the
        # previous energy-based filters (geomean_threshold,
        # min_sustain_ms, min_strength_threshold,
        # reverb_continuation_attack_threshold) are gone. The
        # open_geomean_min / open_sustain_ms classification
        # controls are kept in the JS hard-coded fallback
        # (STEM_SLIDER_CONFIGS) but are NOT filters.
        data = client.get('/api/filters/stem/hihat').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids

    def test_kick_returns_pga_min_prominence(self, client):
        # 2026-06-19: kick slideout is now PGA-only; the
        # previous geomean_threshold /
        # reverb_continuation_attack_threshold sliders are
        # gone. The kick stem has no classification controls.
        data = client.get('/api/filters/stem/kick').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids

    def test_cymbals_returns_pga_min_prominence(self, client):
        # 2026-06-19: cymbals slideout is now PGA-only; the
        # previous energy-based filters are gone. The
        # expected_clusters classification control is kept in
        # the JS hard-coded fallback.
        data = client.get('/api/filters/stem/cymbals').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids

    def test_snare_has_no_geomean_filter(self, client):
        # 2026-06-18: snare's previous geomean_threshold filter
        # is gone — it adopts the toms PGA-only slideout.
        data = client.get('/api/filters/stem/snare').get_json()
        ids = [f['id'] for f in data]
        assert 'geomean_threshold' not in ids

    def test_geomean_threshold_not_in_registry(self, client):
        # 2026-06-18: geomean_threshold was snare-only and is
        # removed in the snare→toms slideout migration.
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'geomean_threshold' not in ids

    def test_pga_min_prominence_and_envelope_value_per_stem(self, client):
        # 2026-06-22: the WebUI-exposed filters for any stem
        # are pga_min_prominence and pga_min_envelope_value
        # (the latter was added for all 5 stems 2026-06-22).
        # min_decay_col_min_db and attack_rise_max_ms are
        # hidden in the WebUI (expose_in_webui: false).
        # Each per-stem endpoint returns both PGA filters
        # in registry order (envelope_value first, then
        # prominence — matches the WebUI panel order).
        for stem in ('toms', 'snare', 'hihat', 'kick', 'cymbals'):
            data = client.get(f'/api/filters/stem/{stem}').get_json()
            ids = [f['id'] for f in data]
            assert ids == ['pga_min_envelope_value', 'pga_min_prominence'], (
                f"stem {stem} returned {ids}, expected "
                f"['pga_min_envelope_value', 'pga_min_prominence']"
            )

    def test_returns_list_not_dict(self, client):
        data = client.get('/api/filters/stem/toms').get_json()
        assert isinstance(data, list)
