---
applyTo: "**/*.js"
---

# JavaScript Style Guide

Requirements for JavaScript code in this project.

## Required

### JSDoc Comments

All functions should have JSDoc comments:

```javascript
/**
 * Brief one-liner description.
 *
 * @param {string} endpoint - Description of parameter
 * @returns {Promise<Object>} Description of return value
 */
async function get(endpoint) {
    ...
}
```

### Module Documentation

Add file-level JSDoc:

```javascript
/**
 * API Client for DrumToMIDI Web UI
 *
 * Provides interface to backend API endpoints.
 */

/**
 * API client class
 */
class LarsNetAPI {
    ...
}
```

## Recommended

### Function Descriptions

Keep descriptions brief and action-oriented:

```javascript
/**
 * Fetch project list from API.
 *
 * @returns {Promise<Array>} List of projects
 */
```

### Error Handling

Always handle errors in API calls:

```javascript
async function get(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API error: ${endpoint}`, error);
        throw error;
    }
}
```

## File Organization

```
webui/static/js/
├── api.js          # API client
├── app.js          # Main application
├── projects.js     # Projects UI
├── settings.js    # Settings UI
├── operations.js  # Operations UI
├── waveform.js    # Waveform visualization
└── threshold-tuning.js  # Threshold tuning UI
```

## Testing

JavaScript in this project is tested via:
- Manual browser testing
- Integration with Python API tests

## ESLint (Future)

Consider adding ESLint for automated linting. Currently:
- No build tooling
- Plain JavaScript (no TypeScript)
- Served directly as static files
