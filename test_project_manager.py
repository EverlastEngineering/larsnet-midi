"""
Tests for project_manager.py

Tests the functional core and imperative shell of project management.
Uses temporary directories for file system tests.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from project_manager import (
    # Pure functions
    parse_project_number,
    extract_song_name,
    generate_project_folder_name,
    next_project_number,
    is_audio_file,
    validate_project_structure,
    create_project_metadata,
    update_status_field,
    # Imperative shell
    discover_projects,
    find_loose_files,
    get_project_by_number,
    copy_configs_to_project,
    create_project,
    update_project_metadata,
    get_project_config,
    select_project,
    PROJECT_METADATA_FILE,
)


# ============================================================================
# TESTS FOR PURE FUNCTIONS (Functional Core)
# ============================================================================

class TestPureFunctions:
    """Test pure functions without file I/O."""
    
    def test_parse_project_number_valid(self):
        assert parse_project_number("1 - My Song") == 1
        assert parse_project_number("42 - Another Track") == 42
        assert parse_project_number("999 - Test") == 999
        assert parse_project_number("1-NoSpace") == 1
        assert parse_project_number("1  -  Extra Spaces") == 1
    
    def test_parse_project_number_invalid(self):
        assert parse_project_number("invalid") is None
        assert parse_project_number("not - a - number") is None
        assert parse_project_number("- Missing Number") is None
        assert parse_project_number("") is None
        assert parse_project_number("1") is None  # Missing separator and name
    
    def test_extract_song_name_valid(self):
        assert extract_song_name("1 - My Song") == "My Song"
        assert extract_song_name("42 - Another Track") == "Another Track"
        assert extract_song_name("1  -  Extra Spaces") == "Extra Spaces"
        assert extract_song_name("1-NoSpace") == "NoSpace"
    
    def test_extract_song_name_invalid(self):
        assert extract_song_name("invalid") is None
        assert extract_song_name("- No Number") is None
        assert extract_song_name("") is None
    
    def test_generate_project_folder_name(self):
        assert generate_project_folder_name(1, "My Song") == "1 - My Song"
        assert generate_project_folder_name(42, "test.wav") == "42 - test.wav"
        assert generate_project_folder_name(999, "Track") == "999 - Track"
    
    def test_next_project_number(self):
        assert next_project_number([1, 2, 3]) == 4
        assert next_project_number([1, 3, 5]) == 6
        assert next_project_number([]) == 1
        assert next_project_number([42]) == 43
        assert next_project_number([1, 2, 10, 5]) == 11
    
    def test_is_audio_file(self):
        assert is_audio_file(Path("song.wav")) is True
        assert is_audio_file(Path("song.WAV")) is True
        assert is_audio_file(Path("song.mp3")) is True
        assert is_audio_file(Path("song.flac")) is True
        assert is_audio_file(Path("song.aiff")) is True
        assert is_audio_file(Path("song.aif")) is True
        
        assert is_audio_file(Path("readme.txt")) is False
        assert is_audio_file(Path("config.yaml")) is False
        assert is_audio_file(Path("song.mp4")) is False
    
    def test_create_project_metadata(self):
        metadata = create_project_metadata("Test Song", 1, "test.wav")
        
        assert metadata["project_name"] == "Test Song"
        assert metadata["project_number"] == 1
        assert metadata["original_file"] == "test.wav"
        assert "created" in metadata
        assert "last_modified" in metadata
        assert metadata["status"]["separated"] is False
        assert metadata["status"]["cleaned"] is False
        assert metadata["status"]["midi_generated"] is False
        assert metadata["status"]["video_rendered"] is False
    
    def test_update_status_field(self):
        original = create_project_metadata("Test", 1, "test.wav")
        original_modified = original["last_modified"]
        
        # Update separated status
        updated = update_status_field(original, "separated", True)
        
        assert updated["status"]["separated"] is True
        assert updated["status"]["cleaned"] is False  # Others unchanged
        assert updated["last_modified"] != original_modified
        
        # Original unchanged (immutable)
        assert original["status"]["separated"] is False
        assert original["last_modified"] == original_modified


# ============================================================================
# TESTS FOR IMPERATIVE SHELL (File I/O)
# ============================================================================

class TestImperativeShell:
    """Test functions with file I/O using temporary directories."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        tmp = tempfile.mkdtemp()
        yield Path(tmp)
        shutil.rmtree(tmp)
    
    @pytest.fixture
    def user_files_dir(self, temp_dir):
        """Create a user_files directory."""
        user_dir = temp_dir / "user_files"
        user_dir.mkdir()
        return user_dir
    
    @pytest.fixture
    def root_configs(self, temp_dir):
        """Create mock root config files."""
        (temp_dir / "config.yaml").write_text("# Mock config")
        (temp_dir / "midiconfig.yaml").write_text("# Mock MIDI config")
        return temp_dir
    
    def test_discover_projects_empty(self, user_files_dir):
        projects = discover_projects(user_files_dir)
        assert projects == []
    
    def test_discover_projects_with_valid_projects(self, user_files_dir):
        # Create valid project folders
        proj1 = user_files_dir / "1 - Song One"
        proj2 = user_files_dir / "2 - Song Two"
        proj1.mkdir()
        proj2.mkdir()
        
        # Add metadata to first project
        metadata1 = create_project_metadata("Song One", 1, "song1.wav")
        (proj1 / PROJECT_METADATA_FILE).write_text(json.dumps(metadata1))
        
        projects = discover_projects(user_files_dir)
        
        assert len(projects) == 2
        assert projects[0]["number"] == 1
        assert projects[0]["name"] == "Song One"
        assert projects[1]["number"] == 2
        assert projects[1]["name"] == "Song Two"
    
    def test_discover_projects_ignores_invalid_folders(self, user_files_dir):
        # Create invalid folders
        (user_files_dir / "not a project").mkdir()
        (user_files_dir / "- Missing Number").mkdir()
        
        # Create valid project
        proj1 = user_files_dir / "1 - Valid"
        proj1.mkdir()
        
        projects = discover_projects(user_files_dir)
        
        assert len(projects) == 1
        assert projects[0]["number"] == 1
    
    def test_find_loose_files(self, user_files_dir):
        # Create loose audio files
        (user_files_dir / "song1.wav").write_text("fake audio")
        (user_files_dir / "song2.mp3").write_text("fake audio")
        (user_files_dir / "readme.txt").write_text("not audio")
        
        # Create a project folder with audio (should be ignored)
        proj = user_files_dir / "1 - Project"
        proj.mkdir()
        (proj / "song3.wav").write_text("fake audio")
        
        loose = find_loose_files(user_files_dir)
        
        assert len(loose) == 2
        assert any("song1.wav" in str(f) for f in loose)
        assert any("song2.mp3" in str(f) for f in loose)
        assert not any("song3.wav" in str(f) for f in loose)
    
    def test_get_project_by_number(self, user_files_dir):
        # Create projects
        proj1 = user_files_dir / "1 - First"
        proj2 = user_files_dir / "2 - Second"
        proj1.mkdir()
        proj2.mkdir()
        
        # Get existing project
        project = get_project_by_number(2, user_files_dir)
        assert project is not None
        assert project["number"] == 2
        
        # Get non-existent project
        project = get_project_by_number(999, user_files_dir)
        assert project is None
    
    def test_copy_configs_to_project(self, temp_dir, root_configs):
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        results = copy_configs_to_project(project_dir, root_configs)
        
        assert results["config.yaml"] is True
        assert results["midiconfig.yaml"] is True
        
        assert (project_dir / "config.yaml").exists()
        assert (project_dir / "midiconfig.yaml").exists()
    
    def test_copy_configs_skips_existing(self, temp_dir, root_configs):
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create existing config
        (project_dir / "config.yaml").write_text("# Existing")
        
        results = copy_configs_to_project(project_dir, root_configs)
        
        # Should still report success for existing file
        assert results["config.yaml"] is True
        assert (project_dir / "config.yaml").read_text() == "# Existing"
    
    def test_create_project_success(self, user_files_dir, temp_dir, root_configs):
        # Create audio file in user_files root
        audio_file = user_files_dir / "test_song.wav"
        audio_file.write_text("fake audio data")
        
        # Create project
        project = create_project(audio_file, user_files_dir, root_configs)
        
        assert project["number"] == 1
        assert project["name"] == "test_song"
        assert project["path"].exists()
        
        # Check audio file moved
        assert not audio_file.exists()
        assert (project["path"] / "test_song.wav").exists()
        
        # Check metadata created
        assert (project["path"] / PROJECT_METADATA_FILE).exists()
        
        # Check configs copied
        assert (project["path"] / "config.yaml").exists()
        assert (project["path"] / "midiconfig.yaml").exists()
        
        # Check subdirectories created
        assert (project["path"] / "stems").is_dir()
        assert (project["path"] / "cleaned").is_dir()
        assert (project["path"] / "midi").is_dir()
        assert (project["path"] / "video").is_dir()
    
    def test_create_project_increments_number(self, user_files_dir, temp_dir, root_configs):
        # Create existing project
        existing = user_files_dir / "1 - Existing"
        existing.mkdir()
        
        # Create new project
        audio_file = user_files_dir / "new_song.wav"
        audio_file.write_text("fake audio")
        
        project = create_project(audio_file, user_files_dir, root_configs)
        
        assert project["number"] == 2
    
    def test_create_project_invalid_file(self, user_files_dir, root_configs):
        # File doesn't exist
        with pytest.raises(FileNotFoundError):
            create_project(user_files_dir / "nonexistent.wav", user_files_dir, root_configs)
        
        # Not an audio file
        text_file = user_files_dir / "readme.txt"
        text_file.write_text("not audio")
        
        with pytest.raises(ValueError):
            create_project(text_file, user_files_dir, root_configs)
    
    def test_create_project_file_not_in_root(self, user_files_dir, root_configs):
        # Create file in subdirectory
        subdir = user_files_dir / "subdir"
        subdir.mkdir()
        audio_file = subdir / "song.wav"
        audio_file.write_text("fake audio")
        
        with pytest.raises(ValueError):
            create_project(audio_file, user_files_dir, root_configs)
    
    def test_update_project_metadata(self, user_files_dir):
        # Create project with metadata
        proj_dir = user_files_dir / "1 - Test"
        proj_dir.mkdir()
        
        metadata = create_project_metadata("Test", 1, "test.wav")
        metadata_path = proj_dir / PROJECT_METADATA_FILE
        metadata_path.write_text(json.dumps(metadata))
        
        # Update status
        update_project_metadata(proj_dir, {
            "status": {
                "separated": True,
                "cleaned": False,
                "midi_generated": False,
                "video_rendered": False
            }
        })
        
        # Check update
        updated = json.loads(metadata_path.read_text())
        assert updated["status"]["separated"] is True
        assert updated["last_modified"] != metadata["last_modified"]
    
    def test_get_project_config_prefers_project(self, temp_dir):
        # Create root and project configs
        root_config = temp_dir / "config.yaml"
        root_config.write_text("# Root config")
        
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        project_config = project_dir / "config.yaml"
        project_config.write_text("# Project config")
        
        # Should prefer project config
        config_path = get_project_config(project_dir, "config.yaml", temp_dir)
        assert config_path == project_config
        assert config_path.read_text() == "# Project config"
    
    def test_get_project_config_falls_back_to_root(self, temp_dir):
        # Only root config exists
        root_config = temp_dir / "config.yaml"
        root_config.write_text("# Root config")
        
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Should fall back to root
        config_path = get_project_config(project_dir, "config.yaml", temp_dir)
        assert config_path == root_config
    
    def test_get_project_config_not_found(self, temp_dir):
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Neither exists
        config_path = get_project_config(project_dir, "nonexistent.yaml", temp_dir)
        assert config_path is None
    
    def test_select_project_by_number(self, user_files_dir):
        # Create projects
        (user_files_dir / "1 - First").mkdir()
        (user_files_dir / "2 - Second").mkdir()
        
        project = select_project(2, user_files_dir, allow_interactive=False)
        
        assert project is not None
        assert project["number"] == 2
    
    def test_select_project_auto_single(self, user_files_dir):
        # Only one project - should auto-select
        (user_files_dir / "1 - Only").mkdir()
        
        project = select_project(None, user_files_dir, allow_interactive=False)
        
        assert project is not None
        assert project["number"] == 1
    
    def test_select_project_no_projects(self, user_files_dir):
        project = select_project(None, user_files_dir, allow_interactive=False)
        assert project is None
    
    def test_select_project_multiple_no_interaction(self, user_files_dir):
        # Multiple projects, no interaction allowed
        (user_files_dir / "1 - First").mkdir()
        (user_files_dir / "2 - Second").mkdir()
        
        project = select_project(None, user_files_dir, allow_interactive=False)
        assert project is None
    
    def test_validate_project_structure(self, user_files_dir):
        proj_dir = user_files_dir / "1 - Test"
        proj_dir.mkdir()
        
        # Empty project
        validation = validate_project_structure(proj_dir)
        assert validation["has_metadata"] is False
        assert validation["has_stems"] is False
        
        # Add metadata
        (proj_dir / PROJECT_METADATA_FILE).write_text("{}")
        validation = validate_project_structure(proj_dir)
        assert validation["has_metadata"] is True
        
        # Add stems directory
        (proj_dir / "stems").mkdir()
        validation = validate_project_structure(proj_dir)
        assert validation["has_stems"] is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test complete workflows."""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Set up complete test environment."""
        user_files = tmp_path / "user_files"
        user_files.mkdir()
        
        # Create root configs
        (tmp_path / "config.yaml").write_text("# Config")
        (tmp_path / "midiconfig.yaml").write_text("# MIDI config")
        
        return {
            "root": tmp_path,
            "user_files": user_files
        }
    
    def test_complete_project_workflow(self, setup_environment):
        """Test complete workflow from audio file to project."""
        env = setup_environment
        
        # 1. Drop audio file in user_files
        audio_file = env["user_files"] / "my_song.wav"
        audio_file.write_text("fake audio data")
        
        # 2. Discover loose files
        loose = find_loose_files(env["user_files"])
        assert len(loose) == 1
        
        # 3. Create project
        project = create_project(loose[0], env["user_files"], env["root"])
        assert project["number"] == 1
        assert project["name"] == "my_song"
        
        # 4. Verify project structure
        assert (project["path"] / "my_song.wav").exists()
        assert (project["path"] / "config.yaml").exists()
        assert (project["path"] / "stems").is_dir()
        
        # 5. No more loose files
        loose = find_loose_files(env["user_files"])
        assert len(loose) == 0
        
        # 6. Can discover project
        projects = discover_projects(env["user_files"])
        assert len(projects) == 1
        assert projects[0]["number"] == 1
        
        # 7. Update project status
        update_project_metadata(project["path"], {
            "status": {"separated": True, "cleaned": False, "midi_generated": False, "video_rendered": False}
        })
        
        # 8. Verify update
        updated_project = get_project_by_number(1, env["user_files"])
        assert updated_project["metadata"]["status"]["separated"] is True
