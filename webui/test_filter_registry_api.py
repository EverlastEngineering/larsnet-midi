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
        assert len(data['filters']) >= 2

    def test_pga_min_prominence_in_registry(self, client):
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'pga_min_prominence' in ids

    def test_decay_col_min_in_registry(self, client):
        data = client.get('/api/filters/schema').get_json()
        ids = [f['id'] for f in data['filters']]
        assert 'min_decay_col_min_db' in ids


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
    """GET /api/filters/stem/<stem_type> — per-stem filter list."""

    def test_toms_returns_both_pga_filters(self, client):
        data = client.get('/api/filters/stem/toms').get_json()
        ids = [f['id'] for f in data]
        assert 'pga_min_prominence' in ids
        assert 'min_decay_col_min_db' in ids

    def test_other_stem_returns_empty(self, client):
        # The 2 PGA filters are toms-only. Other stems have
        # no entries in this PR — that's expected; the other
        # 5 filters migrate in Phase 6.
        data = client.get('/api/filters/stem/kick').get_json()
        assert data == []

    def test_returns_list_not_dict(self, client):
        data = client.get('/api/filters/stem/toms').get_json()
        assert isinstance(data, list)
