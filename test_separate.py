"""
Tests for separate.py with project-based workflow.

Tests the integration between separate.py and project_manager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from project_manager import create_project, USER_FILES_DIR


class TestSeparateIntegration:
    """Test separate.py integration with project manager."""
    
    @pytest.fixture
    def temp_env(self, tmp_path):
        """Create a complete test environment."""
        user_files = tmp_path / "user-files"
        user_files.mkdir()
        
        # Create root configs
        (tmp_path / "config.yaml").write_text("# Mock LarsNet config")
        (tmp_path / "midiconfig.yaml").write_text("# Mock MIDI config")
        
        return {
            "root": tmp_path,
            "user_files": user_files
        }
    
    def test_project_creation_from_audio_file(self, temp_env):
        """Test that a new audio file triggers project creation."""
        # Create an audio file in user-files root
        audio_file = temp_env["user_files"] / "test_song.wav"
        audio_file.write_text("fake audio data")
        
        # Create project
        project = create_project(audio_file, temp_env["user_files"], temp_env["root"])
        
        # Verify project structure
        assert project["number"] == 1
        assert project["name"] == "test_song"
        assert (project["path"] / "test_song.wav").exists()
        assert (project["path"] / "config.yaml").exists()
        assert (project["path"] / "stems").is_dir()
        
    def test_separation_workflow_preparation(self, temp_env):
        """Test that separation workflow properly prepares project structure."""
        # Create project
        audio_file = temp_env["user_files"] / "drums.wav"
        audio_file.write_text("fake audio data")
        
        project = create_project(audio_file, temp_env["user_files"], temp_env["root"])
        
        # Verify stems directory exists
        stems_dir = project["path"] / "stems"
        assert stems_dir.exists()
        assert stems_dir.is_dir()
        
        # Verify config is accessible
        config = project["path"] / "config.yaml"
        assert config.exists()
        assert "Mock LarsNet config" in config.read_text()
