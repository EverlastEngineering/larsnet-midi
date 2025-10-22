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