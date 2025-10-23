# Bug Tracking

## Bug: Upload file box appears in project window
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Upload new file box shows in project window where it doesn't make sense - can't replace file in existing project
- **Expected Behavior**: Upload should only show in main area when no projects exist. Once project exists, upload drag-and-drop box should appear in projects left area
- **Actual Behavior**: Upload box appears inside project window
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: Upload box now only shows in main area when no project selected. When project is selected, a new compact upload box appears in sidebar and main upload box is hidden.

## Bug: Active jobs box placement
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Active jobs box should be at top of page for better visibility
- **Expected Behavior**: Active jobs appear at top; auto-remove after 10 seconds when complete
- **Actual Behavior**: Active jobs box not at top of page
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: Active jobs section moved to top of content area, above all other content. Cards fade out smoothly and auto-remove 10 seconds after completion.

## Bug: Job complete X button behavior
- **Status**: Fixed
- **Priority**: Low
- **Description**: X button should close completed job box, not cancel job
- **Expected Behavior**: X closes the completed job notification box
- **Actual Behavior**: X cancels the job (appropriate behavior while running, wrong for completed jobs)
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: X button behavior changes when job completes - removes inline onclick attribute and sets new handler to close notification instead of canceling job.

## Bug: Downloads section placement
- **Status**: Fixed
- **Priority**: Low
- **Description**: Downloads should appear above operations section for better UX
- **Expected Behavior**: Downloads section above operations
- **Actual Behavior**: Downloads below operations
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: Downloads section moved above operations section in HTML template.

## Bug: Downloads/icons don't update until refresh
- **Status**: Fixed
- **Priority**: High
- **Description**: Download buttons and project icons don't appear until page refresh after job completion
- **Expected Behavior**: UI updates automatically when job completes
- **Actual Behavior**: Requires manual page refresh to see downloads and updated icons
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: Created `refreshCurrentProjectFiles()` function that updates only file-dependent UI elements without reloading jobs section, preventing completed job cards from being hidden prematurely.

## Bug: YAML config corruption on save from webui
- **Status**: Fixed
- **Priority**: High
- **Description**: When saving config from webui, dictionary values (like `toms` and `midi`) were being replaced with primitive values (integers), corrupting the YAML structure and causing AttributeError when loading config
- **Steps to Reproduce**: 
  1. Open webui settings
  2. Modify a nested setting (e.g., midi.min_velocity)
  3. Save configuration
  4. Resulting YAML had `toms: 600` or `midi: 60` instead of nested dicts
- **Expected Behavior**: Only leaf values should be updated, preserving dictionary structure
- **Actual Behavior**: Entire dictionary was replaced with primitive value, breaking config schema
- **Root Causes**: 
  1. **Frontend path bug**: `_parse_dict()` passed wrong path to ConfigField - used parent `path` instead of `current_path`, causing fields to have incomplete paths (e.g., `['midi']` instead of `['midi', 'min_velocity']`)
  2. **No validation**: `update_value()` didn't validate that new value matched expected type (dict vs primitive)
- **Impact**: Caused AttributeError in stems_to_midi conversion: `'int' object has no attribute 'get'`
- **Solution**: 
  - **Fixed path bug** in `config_engine.py` line 307: changed `path=path` to `path=current_path`
  - Created schema validation system in `webui/config_schema.py` defining structure for each config type
  - Updated `YAMLConfigEngine.update_value()` to reject dict→primitive replacements
  - Added structure validation in `YAMLConfigEngine.save()` to catch corruption before writing
  - Changed `update_value()` return type to tuple (success, error_message) for better error reporting
  - Updated `onset_detection` schema to match current YAML structure (dict not primitive)
  - Added comprehensive tests in `webui/test_config_schema_validation.py` (9 tests)
  - Added frontend-focused tests in `webui/test_config_api_frontend.py` (9 tests, 3 passing - path validation)
  - All 12 new protection tests passing

## Bug: Job status case sensitivity
- **Status**: Fixed
- **Priority**: High
- **Description**: JavaScript checked for lowercase status values ('completed') but API returns capitalized values ('Completed'), causing job completion logic to never trigger
- **Expected Behavior**: Job cards update correctly when status changes
- **Actual Behavior**: Completed jobs never triggered special handling (button change, auto-remove timer)
- **Fixed in Commit**: WebUI QoL improvements
- **Solution**: Added `statusLower = job.status.toLowerCase()` conversion before all status comparisons throughout operations.js and projects.js. 

## Bug: Video player doesn't conform to screen size limits
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Video player allows portrait videos to overflow viewport vertically
- **Steps to Reproduce**: Play a portrait video (e.g., 1080×1920) in the web UI
- **Expected Behavior**: Video should scale to fit within viewport with no scrolling required
- **Actual Behavior**: Top and bottom of video player extend beyond visible page area
- **Fixed in Commit**: fix: Constrain video player to viewport for portrait videos
- **Solution**: Added `max-height: 85vh` and `object-fit: contain` CSS to video element, centered container with flexbox. Videos now scale responsively while maintaining aspect ratio.

## Bug: if you reprocess files and it fails, the old file is overwriteen
only remove old file, say, on video render, whn the new file is done