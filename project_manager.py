"""
Project Manager - Functional Core for LarsNet Project Management

Manages user projects in the user-files/ directory with auto-numbering,
metadata tracking, and per-project configuration files.

Architecture: Functional Core
- Pure functions for project discovery, validation, and data transformation
- No side effects (file I/O) except in clearly marked functions
- All logic testable without touching filesystem

Project Structure:
    user-files/
    └── 1 - song name/
        ├── .larsnet_project.json    # Metadata
        ├── config.yaml              # Project-specific LarsNet config
        ├── midiconfig.yaml          # Project-specific MIDI config
        ├── eq.yaml                  # Project-specific EQ config (optional)
        ├── song name.wav            # Original audio
        ├── stems/                   # Separated stems
        ├── cleaned/                 # Cleaned stems
        ├── midi/                    # Generated MIDI
        └── video/                   # Rendered videos
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
import re
from datetime import datetime
import shutil


# ============================================================================
# CONSTANTS
# ============================================================================

USER_FILES_DIR = Path("user-files")
PROJECT_METADATA_FILE = ".larsnet_project.json"

# Root config files to copy to projects
ROOT_CONFIGS = {
    "config.yaml": "config.yaml",
    "midiconfig.yaml": "midiconfig.yaml",
    "eq.yaml": "eq.yaml"  # Optional
}


# ============================================================================
# PURE FUNCTIONS (Functional Core - No Side Effects)
# ============================================================================

def parse_project_number(folder_name: str) -> Optional[int]:
    """
    Extract project number from folder name.
    
    Expected format: "N - song name" where N is an integer.
    
    Args:
        folder_name: Name of the folder
        
    Returns:
        Project number if valid format, None otherwise
        
    Examples:
        >>> parse_project_number("1 - My Song")
        1
        >>> parse_project_number("42 - Another Track")
        42
        >>> parse_project_number("invalid")
        None
        >>> parse_project_number("not - a - number")
        None
    """
    match = re.match(r'^(\d+)\s*-\s*.+', folder_name)
    return int(match.group(1)) if match else None


def extract_song_name(folder_name: str) -> Optional[str]:
    """
    Extract song name from project folder name.
    
    Args:
        folder_name: Name of the project folder
        
    Returns:
        Song name if valid format, None otherwise
        
    Examples:
        >>> extract_song_name("1 - My Song")
        'My Song'
        >>> extract_song_name("42 - Another Track")
        'Another Track'
        >>> extract_song_name("invalid")
        None
    """
    match = re.match(r'^\d+\s*-\s*(.+)', folder_name)
    return match.group(1).strip() if match else None


def generate_project_folder_name(project_number: int, base_name: str) -> str:
    """
    Generate standardized project folder name.
    
    Args:
        project_number: Sequential project number
        base_name: Base name for the project (usually filename without extension)
        
    Returns:
        Formatted folder name
        
    Examples:
        >>> generate_project_folder_name(1, "My Song")
        '1 - My Song'
        >>> generate_project_folder_name(42, "test.wav")
        '42 - test.wav'
    """
    return f"{project_number} - {base_name}"


def next_project_number(existing_numbers: List[int]) -> int:
    """
    Calculate next available project number.
    
    Args:
        existing_numbers: List of currently used project numbers
        
    Returns:
        Next sequential number
        
    Examples:
        >>> next_project_number([1, 2, 3])
        4
        >>> next_project_number([1, 3, 5])
        6
        >>> next_project_number([])
        1
    """
    return max(existing_numbers, default=0) + 1


def is_audio_file(file_path: Path) -> bool:
    """
    Check if file is a supported audio format.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if supported audio file
        
    Examples:
        >>> is_audio_file(Path("song.wav"))
        True
        >>> is_audio_file(Path("song.mp3"))
        True
        >>> is_audio_file(Path("readme.txt"))
        False
    """
    return file_path.suffix.lower() in {'.wav', '.mp3', '.flac', '.aiff', '.aif'}


def validate_project_structure(project_dir: Path) -> Dict[str, bool]:
    """
    Validate project directory structure (pure check, no modifications).
    
    Args:
        project_dir: Path to project directory
        
    Returns:
        Dictionary with validation results for each component
    """
    return {
        "has_metadata": (project_dir / PROJECT_METADATA_FILE).exists(),
        "has_config": (project_dir / "config.yaml").exists(),
        "has_midiconfig": (project_dir / "midiconfig.yaml").exists(),
        "has_audio": any(is_audio_file(f) for f in project_dir.iterdir() if f.is_file()),
        "has_stems": (project_dir / "stems").exists(),
        "has_cleaned": (project_dir / "cleaned").exists(),
        "has_midi": (project_dir / "midi").exists(),
        "has_video": (project_dir / "video").exists(),
    }


def create_project_metadata(
    project_name: str,
    project_number: int,
    original_file: str
) -> Dict:
    """
    Create initial project metadata structure (pure function).
    
    Args:
        project_name: Name of the project
        project_number: Sequential project number
        original_file: Name of the original audio file
        
    Returns:
        Metadata dictionary ready for JSON serialization
    """
    now = datetime.now().isoformat()
    return {
        "project_name": project_name,
        "project_number": project_number,
        "created": now,
        "last_modified": now,
        "original_file": original_file,
        "status": {
            "separated": False,
            "cleaned": False,
            "midi_generated": False,
            "video_rendered": False
        }
    }


def update_status_field(metadata: Dict, field: str, value: bool) -> Dict:
    """
    Update a status field in metadata (pure function, returns new dict).
    
    Args:
        metadata: Existing metadata dictionary
        field: Status field to update
        value: New value
        
    Returns:
        New metadata dictionary with updated field and timestamp
    """
    updated = metadata.copy()
    updated["status"] = metadata["status"].copy()
    updated["status"][field] = value
    updated["last_modified"] = datetime.now().isoformat()
    return updated


# ============================================================================
# IMPERATIVE SHELL (Side Effects - File I/O)
# ============================================================================

def discover_projects(user_files_dir: Path = USER_FILES_DIR) -> List[Dict]:
    """
    Discover all valid projects in user-files directory.
    
    Args:
        user_files_dir: Path to user-files directory
        
    Returns:
        List of project info dictionaries, sorted by project number
    """
    if not user_files_dir.exists():
        return []
    
    projects = []
    
    for item in user_files_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Parse project number
        project_num = parse_project_number(item.name)
        if project_num is None:
            continue
        
        # Try to load metadata
        metadata_path = item / PROJECT_METADATA_FILE
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Corrupted metadata, will be recreated if needed
        
        # Get validation info
        validation = validate_project_structure(item)
        
        projects.append({
            "number": project_num,
            "name": extract_song_name(item.name),
            "path": item,
            "metadata": metadata,
            "validation": validation,
            "last_modified": metadata.get("last_modified") if metadata else None
        })
    
    return sorted(projects, key=lambda p: p["number"])


def find_loose_files(user_files_dir: Path = USER_FILES_DIR) -> List[Path]:
    """
    Find audio files in root of user-files directory (not in project folders).
    
    Args:
        user_files_dir: Path to user-files directory
        
    Returns:
        List of audio file paths
    """
    if not user_files_dir.exists():
        return []
    
    loose_files = []
    for item in user_files_dir.iterdir():
        if item.is_file() and is_audio_file(item):
            loose_files.append(item)
    
    return sorted(loose_files)


def get_project_by_number(
    project_number: int,
    user_files_dir: Path = USER_FILES_DIR
) -> Optional[Dict]:
    """
    Retrieve project information by number.
    
    Args:
        project_number: Project number to find
        user_files_dir: Path to user-files directory
        
    Returns:
        Project info dictionary or None if not found
    """
    projects = discover_projects(user_files_dir)
    for project in projects:
        if project["number"] == project_number:
            return project
    return None


def copy_configs_to_project(
    project_dir: Path,
    root_dir: Path = Path(".")
) -> Dict[str, bool]:
    """
    Copy root configuration files to project directory.
    
    Args:
        project_dir: Path to project directory
        root_dir: Path to root directory containing config files
        
    Returns:
        Dictionary mapping config names to success status
    """
    results = {}
    
    for config_name, config_file in ROOT_CONFIGS.items():
        source = root_dir / config_file
        dest = project_dir / config_file
        
        if source.exists() and not dest.exists():
            try:
                shutil.copy2(source, dest)
                results[config_name] = True
            except IOError:
                results[config_name] = False
        else:
            # Skip if source doesn't exist or dest already exists
            results[config_name] = dest.exists()
    
    return results


def create_project(
    audio_file: Path,
    user_files_dir: Path = USER_FILES_DIR,
    root_dir: Path = Path(".")
) -> Dict:
    """
    Create a new project from an audio file.
    
    This function:
    1. Determines next project number
    2. Creates project folder
    3. Moves audio file into project
    4. Copies configuration files
    5. Creates metadata file
    6. Creates subdirectories
    
    Args:
        audio_file: Path to audio file (should be in user_files_dir root)
        user_files_dir: Path to user-files directory
        root_dir: Path to root directory containing config files
        
    Returns:
        Project info dictionary for newly created project
        
    Raises:
        ValueError: If audio_file is not in user_files_dir root
        IOError: If project creation fails
    """
    # Validate input
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if not is_audio_file(audio_file):
        raise ValueError(f"Not a supported audio file: {audio_file}")
    
    if audio_file.parent != user_files_dir:
        raise ValueError(f"Audio file must be in {user_files_dir} root")
    
    # Determine project number and name
    existing_projects = discover_projects(user_files_dir)
    existing_numbers = [p["number"] for p in existing_projects]
    project_num = next_project_number(existing_numbers)
    
    # Use filename without extension as project name
    base_name = audio_file.stem
    folder_name = generate_project_folder_name(project_num, base_name)
    project_dir = user_files_dir / folder_name
    
    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=False)
    
    try:
        # Move audio file into project
        new_audio_path = project_dir / audio_file.name
        audio_file.rename(new_audio_path)
        
        # Copy configuration files
        copy_configs_to_project(project_dir, root_dir)
        
        # Create metadata
        metadata = create_project_metadata(base_name, project_num, audio_file.name)
        metadata_path = project_dir / PROJECT_METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create subdirectories
        (project_dir / "stems").mkdir(exist_ok=True)
        (project_dir / "cleaned").mkdir(exist_ok=True)
        (project_dir / "midi").mkdir(exist_ok=True)
        (project_dir / "video").mkdir(exist_ok=True)
        
        # Return project info
        return {
            "number": project_num,
            "name": base_name,
            "path": project_dir,
            "metadata": metadata,
            "validation": validate_project_structure(project_dir)
        }
        
    except Exception as e:
        # Cleanup on failure
        if project_dir.exists():
            shutil.rmtree(project_dir)
        raise IOError(f"Failed to create project: {e}")


def update_project_metadata(
    project_dir: Path,
    updates: Dict
) -> None:
    """
    Update project metadata file.
    
    Args:
        project_dir: Path to project directory
        updates: Dictionary of updates to apply to metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        IOError: If update fails
    """
    metadata_path = project_dir / PROJECT_METADATA_FILE
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Apply updates
    metadata.update(updates)
    metadata["last_modified"] = datetime.now().isoformat()
    
    # Write back
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_project_config(
    project_dir: Path,
    config_name: str,
    root_dir: Path = Path(".")
) -> Optional[Path]:
    """
    Get path to configuration file, checking project first, then root.
    
    Args:
        project_dir: Path to project directory
        config_name: Name of config file (e.g., "config.yaml")
        root_dir: Path to root directory
        
    Returns:
        Path to config file, or None if not found
    """
    # Check project directory first
    project_config = project_dir / config_name
    if project_config.exists():
        return project_config
    
    # Fall back to root
    root_config = root_dir / config_name
    if root_config.exists():
        return root_config
    
    return None


def select_project(
    project_number: Optional[int] = None,
    user_files_dir: Path = USER_FILES_DIR,
    allow_interactive: bool = True
) -> Optional[Dict]:
    """
    Select a project by number or interactively.
    
    Selection logic:
    1. If project_number provided, use that
    2. If only one project exists, auto-select it
    3. If multiple projects and allow_interactive, prompt user
    4. Otherwise return None
    
    Args:
        project_number: Specific project number to select
        user_files_dir: Path to user-files directory
        allow_interactive: Whether to prompt user if multiple projects
        
    Returns:
        Selected project info dictionary or None
    """
    projects = discover_projects(user_files_dir)
    
    if not projects:
        return None
    
    # Specific number requested
    if project_number is not None:
        for project in projects:
            if project["number"] == project_number:
                return project
        return None
    
    # Only one project - auto-select
    if len(projects) == 1:
        return projects[0]
    
    # Multiple projects
    if allow_interactive:
        print("\nMultiple projects found:")
        for p in projects:
            last_mod = p.get("last_modified", "unknown")
            print(f"  {p['number']} - {p['name']} (last modified: {last_mod})")
        
        try:
            choice = input(f"\nSelect project [1]: ").strip()
            if not choice:
                choice = "1"
            selected_num = int(choice)
            
            for project in projects:
                if project["number"] == selected_num:
                    return project
        except (ValueError, KeyboardInterrupt):
            return None
    
    return None
