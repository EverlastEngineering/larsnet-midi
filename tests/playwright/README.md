# Playwright E2E tests for larsnet

End-to-end smoke + regression tests for the larsnet webui. This
directory is the durable artifact that survives across the toms
threshold-tuning refactor — every step extends the same spec file
(`specs/01-toms-tuning-smoke.spec.ts`) rather than replacing it.

## Quick start

```bash
# 1. Install JS deps
npm install

# 2. Install the chromium browser binary (one-time, ~170 MB)
npx playwright install chromium

# 3. Make sure the webui dev server is up
#    Default: Flask on http://localhost:4915 (see webui/app.py:142)
#    Override the port:  LARSNET_WEBUI_PORT=5050 npm test
#
#    Easiest way to launch it:
./start_webui.sh
#    (or, in another terminal:)
#    conda run -n drumtomidi python -m webui.app

# 4. Run the suite
npm test            # headless, default chromium
npm run test:ui     # Playwright UI mode (interactive)
```

The first test run downloads chromium; subsequent runs are fast.

## What this suite is

This is the **regression baseline** for the toms threshold-tuning
refactor. The current spec asserts only that:

1. The webui loads and the project list is rendered.
2. The fixture project (`user_files/4 - 2_funk_80_beat_4-4_4/`,
   project `#4`, the funk-80 stem-separated loop) can be selected.
3. The analysis section reveals the toms stem tab.
4. The threshold-tuning panel can be opened (via the visible
   "Tune" button, or by calling `toggleTuningPanel()` directly when
   the button is hidden — see "Tune button visibility" below).
5. The panel renders the "Threshold Tuning" header and the
   `#tuning-sliders` container is present (the container is empty
   on 2026-06-13 because the toms slider config is `[]`; subsequent
   refactor steps will populate it).

### Tune button visibility — current-state quirk

The "Tune" button (`#tuning-toggle-btn`) is hidden by the client when
no stem in the project has `events_sensitive` populated
(`webui/static/js/waveform.js:236-242`). On 2026-06-13 the toms
pipeline is PGA-only and the toms stem reports
`events_sensitive: []` for every project (including the funk-80
fixture). The button is therefore hidden when the toms stem is the
only one.

The smoke spec handles this gracefully: it tries the visible path
first (`await tuneButton.click()`) and falls back to
`page.evaluate(() => toggleTuningPanel())` when the button is
hidden. Both paths exercise the same end-state (panel becomes
visible, header renders), and either path is a valid regression
signal. When the toms refactor lands the PGA slider, `events_sensitive`
will start populating for toms and the visible-button path will
become the default — at which point the fallback can be removed.

Subsequent refactor steps will extend the same spec file with
assertions on slider presence, slider value debounce, the
reclassify API round-trip, and the Save & Reconvert flow. **They
must not delete or weaken the assertions above.**

## Snapshot policy

| Event                       | Destination                                                |
|-----------------------------|------------------------------------------------------------|
| Test passes                 | `__snapshots__/<spec-name>/<test-name>.png`                |
| Test fails                  | Same dir, plus `*-failure.png` and a `.webm` video trace   |
| HTML report                 | `__snapshots__/html-report/index.html`                     |

`__snapshots__/` is auto-created on first run. It is intentionally
gitignored — these are review artifacts, not source.

## Port + fixture

- **Port:** 4915 by default. Override with the `LARSNET_WEBUI_PORT`
  env var. The webui dev server hardcodes 4915 in
  `webui/app.py:142`; if you launch the server with a different
  port (e.g. `flask --app app run --port 5050`), set
  `LARSNET_WEBUI_PORT=5050` when running the tests.
- **Fixture project:** `user_files/4 - 2_funk_80_beat_4-4_4/`.
  Must have a toms stem and a `midi/*.analysis.json` sidecar (it
  does, as of 2026-06-13). The spec locates it by name fragment
  (`2_funk_80_beat_4-4_4`) rather than by project number, so
  renumbering the directory does not break the test.

## Why a separate `package.json`?

The webui is a Flask app — no `node_modules` at the repo root. This
subdirectory keeps the JS toolchain self-contained and lets the
suite install `playwright` without touching Python dependencies.
The root `.gitignore` excludes `tests/playwright/node_modules/`.

## CI integration (TODO)

The suite is designed to be the foundation for a CI job:

```yaml
- name: Install Playwright
  run: |
    cd tests/playwright
    npm ci
    npx playwright install --with-deps chromium
- name: Start webui
  run: |
    conda run -n drumtomidi python -m webui.app &
    # wait for /health
- name: Run E2E tests
  run: cd tests/playwright && npm test
```

Wiring this up in the project's CI is out of scope for the
scaffold step; this README documents the contract.
