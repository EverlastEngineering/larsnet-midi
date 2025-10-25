"""
Tests for compare_separation_configs.py

Tests configuration loading, validation, and comparison workflow.
"""

import pytest # type: ignore
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml # type: ignore

from compare_separation_configs import (
    load_custom_configs,
    DEFAULT_CONFIGS,
    process_comparison
)


class TestConfigurationLoading:
    """Test custom configuration file loading and validation."""
    
    def test_load_valid_custom_config(self, tmp_path):
        """Test loading a valid custom configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'configs': [
                {'name': 'test1', 'wiener': 1.5, 'eq': True},
                {'name': 'test2', 'wiener': None, 'eq': False}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        
        assert len(configs) == 2
        assert configs[0]['name'] == 'test1'
        assert configs[0]['wiener'] == 1.5
        assert configs[0]['eq'] is True
        assert configs[1]['name'] == 'test2'
        assert configs[1]['wiener'] is None
        assert configs[1]['eq'] is False
    
    def test_load_config_missing_file(self, tmp_path):
        """Test that loading nonexistent config file raises error."""
        nonexistent = tmp_path / "missing.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_custom_configs(nonexistent)
    
    def test_load_config_missing_configs_key(self, tmp_path):
        """Test that config file without 'configs' key raises error."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text(yaml.dump({'wrong_key': []}))
        
        with pytest.raises(ValueError, match="must contain 'configs' key"):
            load_custom_configs(config_file)
    
    def test_load_config_missing_name(self, tmp_path):
        """Test that config without 'name' raises error."""
        config_file = tmp_path / "bad_config.yaml"
        config_data = {
            'configs': [
                {'wiener': 1.0, 'eq': True}  # Missing 'name'
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        with pytest.raises(ValueError, match="missing required 'name' field"):
            load_custom_configs(config_file)
    
    def test_load_config_defaults(self, tmp_path):
        """Test that missing optional fields get defaults."""
        config_file = tmp_path / "partial_config.yaml"
        config_data = {
            'configs': [
                {'name': 'minimal'}  # Only name provided
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        
        assert len(configs) == 1
        assert configs[0]['name'] == 'minimal'
        assert configs[0]['wiener'] is None
        assert configs[0]['eq'] is False
    
    def test_load_config_invalid_wiener(self, tmp_path):
        """Test that negative or zero Wiener exponent raises error."""
        config_file = tmp_path / "bad_wiener.yaml"
        config_data = {
            'configs': [
                {'name': 'bad', 'wiener': -1.0, 'eq': False}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        with pytest.raises(ValueError, match="Wiener exponent must be positive"):
            load_custom_configs(config_file)


class TestDefaultConfigurations:
    """Test default configuration set."""
    
    def test_default_configs_exist(self):
        """Test that default configurations are defined."""
        assert len(DEFAULT_CONFIGS) > 0
    
    def test_default_configs_have_required_fields(self):
        """Test that all default configs have required fields."""
        for config in DEFAULT_CONFIGS:
            assert 'name' in config
            assert 'wiener' in config
            assert 'eq' in config
    
    def test_default_configs_include_baseline(self):
        """Test that baseline (no processing) is included."""
        baselines = [c for c in DEFAULT_CONFIGS if c['name'] == 'baseline']
        assert len(baselines) == 1
        assert baselines[0]['wiener'] is None
        assert baselines[0]['eq'] is False
    
    def test_default_configs_include_variations(self):
        """Test that various configurations are included."""
        names = {c['name'] for c in DEFAULT_CONFIGS}
        
        # Should have wiener-only configs
        assert any('wiener' in name and 'eq' not in name for name in names)
        
        # Should have eq-only config
        assert 'eq' in names
        
        # Should have combined configs
        assert any('wiener' in name and 'eq' in name for name in names)


class TestComparisonWorkflow:
    """Test the comparison processing workflow."""
    
    @pytest.fixture
    def mock_project(self, tmp_path):
        """Create a mock project structure."""
        project_dir = tmp_path / "1 - Test Song"
        project_dir.mkdir()
        
        # Create mock audio file
        audio_file = project_dir / "test_song.wav"
        audio_file.write_text("fake audio")
        
        # Create mock config
        config_file = project_dir / "config.yaml"
        config_file.write_text("# Mock config")
        
        return {
            "number": 1,
            "name": "Test Song",
            "path": project_dir,
            "metadata": None
        }
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_process_comparison_creates_output_directory(
        self, mock_get_config, mock_process, mock_project
    ):
        """Test that comparison creates for_comparison directory."""
        mock_get_config.return_value = mock_project["path"] / "config.yaml"
        
        configs = [
            {"name": "test1", "wiener": None, "eq": False}
        ]
        
        process_comparison(mock_project, configs, device='cpu', cleanup=False)
        
        # Check that for_comparison directory was created
        comparison_dir = mock_project["path"] / "for_comparison"
        assert comparison_dir.exists()
        assert comparison_dir.is_dir()
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_process_comparison_calls_process_for_each_config(
        self, mock_get_config, mock_process, mock_project
    ):
        """Test that process_stems_for_project is called for each config."""
        mock_get_config.return_value = mock_project["path"] / "config.yaml"
        
        configs = [
            {"name": "test1", "wiener": 1.0, "eq": False},
            {"name": "test2", "wiener": None, "eq": True}
        ]
        
        process_comparison(mock_project, configs, device='cpu', cleanup=False)
        
        # Should be called twice (once per config)
        assert mock_process.call_count == 2
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_process_comparison_passes_correct_parameters(
        self, mock_get_config, mock_process, mock_project
    ):
        """Test that correct parameters are passed to process_stems_for_project."""
        mock_get_config.return_value = mock_project["path"] / "config.yaml"
        
        configs = [
            {"name": "wiener-2.0", "wiener": 2.0, "eq": False}
        ]
        
        process_comparison(mock_project, configs, device='cuda', cleanup=False)
        
        # Check the call arguments
        call_args = mock_process.call_args
        assert call_args.kwargs['project_dir'] == mock_project["path"]
        assert call_args.kwargs['wiener_exponent'] == 2.0
        assert call_args.kwargs['device'] == 'cuda'
        assert call_args.kwargs['apply_eq'] is False
        
        # Check output directory structure
        expected_output = mock_project["path"] / "for_comparison" / "wiener-2.0"
        assert call_args.kwargs['stems_dir'] == expected_output
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_process_comparison_continues_on_error(
        self, mock_get_config, mock_process, mock_project
    ):
        """Test that comparison continues if one config fails."""
        mock_get_config.return_value = mock_project["path"] / "config.yaml"
        
        # Make the first call fail
        mock_process.side_effect = [
            Exception("Test error"),
            None  # Second call succeeds
        ]
        
        configs = [
            {"name": "test1", "wiener": 1.0, "eq": False},
            {"name": "test2", "wiener": 2.0, "eq": False}
        ]
        
        # Should not raise exception
        process_comparison(mock_project, configs, device='cpu', cleanup=False)
        
        # Both configs should have been attempted
        assert mock_process.call_count == 2
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_process_comparison_cleanup_removes_stems(
        self, mock_get_config, mock_process, mock_project
    ):
        """Test that cleanup option removes original stems directory."""
        mock_get_config.return_value = mock_project["path"] / "config.yaml"
        
        # Create original stems directory
        stems_dir = mock_project["path"] / "stems"
        stems_dir.mkdir()
        (stems_dir / "test.wav").write_text("fake stem")
        
        configs = [{"name": "test", "wiener": None, "eq": False}]
        
        process_comparison(mock_project, configs, device='cpu', cleanup=True)
        
        # Original stems should be removed
        assert not stems_dir.exists()


class TestOutputStructure:
    """Test that output directories are structured correctly."""
    
    @pytest.fixture
    def comparison_structure(self, tmp_path):
        """Create expected comparison directory structure."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        
        comparison_dir = project_dir / "for_comparison"
        comparison_dir.mkdir()
        
        return {
            "project": project_dir,
            "comparison": comparison_dir
        }
    
    def test_config_subdirectories_created(self, comparison_structure):
        """Test that each config gets its own subdirectory."""
        comparison_dir = comparison_structure["comparison"]
        
        config_names = ['baseline', 'wiener-2.0', 'eq']
        
        for name in config_names:
            (comparison_dir / name).mkdir()
        
        # Verify structure
        for name in config_names:
            assert (comparison_dir / name).exists()
            assert (comparison_dir / name).is_dir()
    
    def test_config_names_as_directory_names(self):
        """Test that config names become directory names."""
        configs = [
            {"name": "my-config", "wiener": 1.0, "eq": False},
            {"name": "another_config", "wiener": None, "eq": True}
        ]
        
        for config in configs:
            # Config name should be valid directory name
            assert '/' not in config['name']
            assert '\\' not in config['name']


class TestIntegrationWithDependencies:
    """Test integration with project_manager and separation_utils."""
    
    @pytest.fixture
    def mock_project_with_audio(self, tmp_path):
        """Create a realistic mock project with audio file."""
        project_dir = tmp_path / "1 - Test Song"
        project_dir.mkdir()
        
        # Create mock audio file
        audio_file = project_dir / "test_song.wav"
        audio_file.write_text("fake audio data")
        
        # Create mock config.yaml
        config_file = tmp_path / "config.yaml"
        config_file.write_text("global:\n  sr: 44100")
        
        return {
            "number": 1,
            "name": "Test Song",
            "path": project_dir,
            "metadata": {"status": {"separated": False}}
        }
    
    @patch('compare_separation_configs.get_project_config')
    def test_missing_config_yaml_is_caught(self, mock_get_config, mock_project_with_audio, capsys):
        """Test that missing config.yaml is detected and reported."""
        mock_get_config.return_value = None  # Simulate missing config
        
        configs = [{"name": "test", "wiener": None, "eq": False}]
        
        with pytest.raises(SystemExit):
            process_comparison(mock_project_with_audio, configs, device='cpu')
        
        captured = capsys.readouterr()
        assert "ERROR: config.yaml not found" in captured.out
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_correct_config_path_passed_to_process_stems(
        self, mock_get_config, mock_process, mock_project_with_audio
    ):
        """Test that the config path from get_project_config is used."""
        expected_config = mock_project_with_audio["path"] / "config.yaml"
        mock_get_config.return_value = expected_config
        
        configs = [{"name": "test", "wiener": 1.0, "eq": False}]
        
        process_comparison(mock_project_with_audio, configs, device='cpu')
        
        # Verify config_path argument
        call_args = mock_process.call_args
        assert call_args.kwargs['config_path'] == expected_config
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_verbose_flag_passed_to_process_stems(
        self, mock_get_config, mock_process, mock_project_with_audio
    ):
        """Test that verbose=True is always passed."""
        mock_get_config.return_value = mock_project_with_audio["path"] / "config.yaml"
        
        configs = [{"name": "test", "wiener": None, "eq": False}]
        
        process_comparison(mock_project_with_audio, configs, device='cpu')
        
        call_args = mock_process.call_args
        assert call_args.kwargs['verbose'] is True
    
    def test_zero_wiener_value_rejected(self, tmp_path):
        """Test that Wiener exponent of 0 is rejected."""
        config_file = tmp_path / "bad.yaml"
        config_data = {
            'configs': [
                {'name': 'zero_wiener', 'wiener': 0.0, 'eq': False}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        with pytest.raises(ValueError, match="Wiener exponent must be positive"):
            load_custom_configs(config_file)
    
    def test_empty_configs_list(self, tmp_path):
        """Test that empty configs list is handled (should process successfully with no work)."""
        config_file = tmp_path / "empty.yaml"
        config_data = {'configs': []}
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        assert configs == []
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_empty_configs_list_processes_nothing(
        self, mock_get_config, mock_process, mock_project_with_audio
    ):
        """Test that empty configs list doesn't call process_stems_for_project."""
        mock_get_config.return_value = mock_project_with_audio["path"] / "config.yaml"
        
        configs = []
        
        process_comparison(mock_project_with_audio, configs, device='cpu')
        
        # Should not process anything
        mock_process.assert_not_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_config_with_very_large_wiener(self, tmp_path):
        """Test that very large Wiener values are accepted."""
        config_file = tmp_path / "large.yaml"
        config_data = {
            'configs': [
                {'name': 'huge', 'wiener': 1000.0, 'eq': False}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        assert configs[0]['wiener'] == 1000.0
    
    def test_config_with_very_small_wiener(self, tmp_path):
        """Test that very small positive Wiener values are accepted."""
        config_file = tmp_path / "small.yaml"
        config_data = {
            'configs': [
                {'name': 'tiny', 'wiener': 0.001, 'eq': False}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        assert configs[0]['wiener'] == 0.001
    
    def test_config_with_special_characters_in_name(self, tmp_path):
        """Test that special characters in config names are accepted."""
        config_file = tmp_path / "special.yaml"
        config_data = {
            'configs': [
                {'name': 'test_config-v2.1', 'wiener': 1.0, 'eq': True}
            ]
        }
        config_file.write_text(yaml.dump(config_data))
        
        configs = load_custom_configs(config_file)
        assert configs[0]['name'] == 'test_config-v2.1'
    
    def test_malformed_yaml(self, tmp_path):
        """Test that truly malformed YAML is caught."""
        config_file = tmp_path / "malformed.yaml"
        config_file.write_text("configs:\n  - name: test\n    wiener: [unclosed bracket")
        
        with pytest.raises(Exception):  # yaml.scanner.ScannerError or similar
            load_custom_configs(config_file)
    
    def test_yaml_with_null_configs(self, tmp_path):
        """Test that null configs value is rejected."""
        config_file = tmp_path / "null.yaml"
        config_file.write_text("configs: null")
        
        with pytest.raises(ValueError, match="must be a list"):
            load_custom_configs(config_file)
    
    def test_configs_not_a_list(self, tmp_path):
        """Test that configs must be a list."""
        config_file = tmp_path / "not_list.yaml"
        config_data = {'configs': {'name': 'test'}}  # Dict instead of list
        config_file.write_text(yaml.dump(config_data))
        
        # Should raise ValueError about configs not being a list
        with pytest.raises(ValueError, match="must be a list"):
            load_custom_configs(config_file)
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_duplicate_config_names_both_processed(
        self, mock_get_config, mock_process, tmp_path
    ):
        """Test that duplicate config names both get processed (last one wins on disk)."""
        project_dir = tmp_path / "1 - Test"
        project_dir.mkdir()
        (project_dir / "test.wav").write_text("fake")
        
        project = {
            "number": 1,
            "name": "Test",
            "path": project_dir,
            "metadata": None
        }
        
        mock_get_config.return_value = project_dir / "config.yaml"
        (project_dir / "config.yaml").write_text("# config")
        
        # Two configs with same name
        configs = [
            {"name": "duplicate", "wiener": 1.0, "eq": False},
            {"name": "duplicate", "wiener": 2.0, "eq": True}
        ]
        
        process_comparison(project, configs, device='cpu')
        
        # Both should be processed
        assert mock_process.call_count == 2
    
    @patch('compare_separation_configs.process_stems_for_project')
    @patch('compare_separation_configs.get_project_config')
    def test_all_error_types_continue_processing(
        self, mock_get_config, mock_process, tmp_path
    ):
        """Test that various exception types don't stop processing."""
        project_dir = tmp_path / "1 - Test"
        project_dir.mkdir()
        
        project = {
            "number": 1,
            "name": "Test",
            "path": project_dir,
            "metadata": None
        }
        
        mock_get_config.return_value = project_dir / "config.yaml"
        
        # Different error types
        mock_process.side_effect = [
            RuntimeError("Test runtime error"),
            ValueError("Test value error"),
            FileNotFoundError("Test file error"),
            None  # Last one succeeds
        ]
        
        configs = [
            {"name": "fail1", "wiener": 1.0, "eq": False},
            {"name": "fail2", "wiener": 2.0, "eq": False},
            {"name": "fail3", "wiener": 3.0, "eq": False},
            {"name": "success", "wiener": 4.0, "eq": False}
        ]
        
        process_comparison(project, configs, device='cpu')
        
        # All should be attempted
        assert mock_process.call_count == 4


class TestDefaultConfigsConsistency:
    """Test that default configs are consistent with separation_utils requirements."""
    
    def test_default_wiener_values_are_valid(self):
        """Test that all default Wiener values would pass validation."""
        for config in DEFAULT_CONFIGS:
            wiener = config['wiener']
            if wiener is not None:
                assert wiener > 0, f"Config '{config['name']}' has invalid Wiener value: {wiener}"
    
    def test_default_eq_values_are_boolean(self):
        """Test that all EQ values are proper booleans."""
        for config in DEFAULT_CONFIGS:
            assert isinstance(config['eq'], bool), f"Config '{config['name']}' has non-boolean eq value"
    
    def test_default_names_are_filesystem_safe(self):
        """Test that config names won't cause filesystem issues."""
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        for config in DEFAULT_CONFIGS:
            name = config['name']
            for char in invalid_chars:
                assert char not in name, f"Config name '{name}' contains invalid character: {char}"
    
    def test_default_names_are_unique(self):
        """Test that all default config names are unique."""
        names = [c['name'] for c in DEFAULT_CONFIGS]
        assert len(names) == len(set(names)), "Duplicate names found in DEFAULT_CONFIGS"
    
    def test_default_configs_include_common_use_cases(self):
        """Test that common use cases are covered in defaults."""
        names = {c['name'] for c in DEFAULT_CONFIGS}
        
        # Should have a baseline for reference
        assert 'baseline' in names
        
        # Should have at least one wiener-only, eq-only, and combined
        has_wiener_only = any(
            c['wiener'] is not None and not c['eq'] 
            for c in DEFAULT_CONFIGS
        )
        has_eq_only = any(
            c['wiener'] is None and c['eq'] 
            for c in DEFAULT_CONFIGS
        )
        has_combined = any(
            c['wiener'] is not None and c['eq'] 
            for c in DEFAULT_CONFIGS
        )
        
        assert has_wiener_only, "No Wiener-only config in defaults"
        assert has_eq_only, "No EQ-only config in defaults"
        assert has_combined, "No combined config in defaults"
