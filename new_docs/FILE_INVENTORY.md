# File Inventory

Generated: 2026-02-15

This file is auto-generated. Run `python docs/generate_inventory.py` to update.

## Summary

| Root Python | 59 files | 15163 lines |
| stems_to_midi Package | 55 files | 32921 lines |
| webui Package | 48 files | 16776 lines |
| moderngl_renderer Package | 34 files | 13496 lines |
| JavaScript | 52 files | 19013 lines |
| **Total** | **248** files | **97369** lines |

## Files

### Root Python

| File | Lines |
|------|-------|
| `analyze_clustering_results.py` | 269 |
| `analyze_reverb_continuations.py` | 194 |
| `compare_backtracking.py` | 41 |
| `compare_detection_methods.py` | 114 |
| `compare_peak_vs_backtracked.py` | 43 |
| `count_events.py` | 7 |
| `device_shell.py` | 233 |
| `diagnose_missing_events.py` | 66 |
| `examine_audio_channels.py` | 59 |
| `examine_hihat_data.py` | 71 |
| `export_clustering_table.py` | 145 |
| `export_energy_detection_data.py` | 196 |
| `export_raw_lr_data.py` | 176 |
| `generate_cymbal_midi_new_detection.py` | 116 |
| `mdx23c_optimized.py` | 609 |
| `mdx23c_utils.py` | 357 |
| `midi_core.py` | 346 |
| `midi_parser.py` | 40 |
| `midi_render_core.py` | 446 |
| `midi_shell.py` | 141 |
| `midi_types.py` | 429 |
| `new_docs/docs/generate_api_docs.py` | 81 |
| `new_docs/docs/generate_inventory.py` | 159 |
| `normalize_yaml.py` | 83 |
| `project_manager.py` | 656 |
| `quick_comparison.py` | 108 |
| `render_midi_video_shell.py` | 1211 |
| `render_video_core.py` | 294 |
| `separate.py` | 174 |
| `separation_shell.py` | 302 |
| `sidechain_core.py` | 205 |
| `sidechain_shell.py` | 403 |
| `site/docs/generate_api_docs.py` | 81 |
| `site/docs/generate_inventory.py` | 159 |
| `test_compare_renderers.py` | 71 |
| `test_coordinate_system.py` | 209 |
| `test_cpu_threading.py` | 106 |
| `test_cv2_rendering.py` | 211 |
| `test_dual_sensitivity.py` | 421 |
| `test_energy_detection_integration.py` | 142 |
| `test_gpu_coordinate_debug.py` | 79 |
| `test_integration.py` | 670 |
| `test_mdx23c_utils.py` | 162 |
| `test_mdx_performance.py` | 387 |
| `test_midi_core.py` | 161 |
| `test_midi_parser.py` | 200 |
| `test_midi_render_core.py` | 515 |
| `test_midi_shell.py` | 125 |
| `test_midi_types.py` | 607 |
| `test_normalization.py` | 44 |
| `test_note_classification_core.py` | 1003 |
| `test_optimization_real_audio.py` | 113 |
| `test_project_manager.py` | 476 |
| `test_pure_opencv_speed.py` | 225 |
| `test_render_video_core.py` | 372 |
| `test_separate.py` | 112 |
| `test_sidechain_core.py` | 456 |
| `test_stem_comparison.py` | 226 |
| `test_threshold_sweep.py` | 56 |

### stems_to_midi Package

| File | Lines |
|------|-------|
| `stems_to_midi/__init__.py` | 28 |
| `stems_to_midi/__init__.py` | 28 |
| `stems_to_midi/analysis_core.py` | 2631 |
| `stems_to_midi/analysis_core.py` | 2631 |
| `stems_to_midi/clustering_core.py` | 310 |
| `stems_to_midi/clustering_core.py` | 310 |
| `stems_to_midi/config.py` | 108 |
| `stems_to_midi/config.py` | 108 |
| `stems_to_midi/detection_shell.py` | 505 |
| `stems_to_midi/detection_shell.py` | 505 |
| `stems_to_midi/energy_detection_core.py` | 713 |
| `stems_to_midi/energy_detection_core.py` | 713 |
| `stems_to_midi/energy_detection_shell.py` | 147 |
| `stems_to_midi/energy_detection_shell.py` | 147 |
| `stems_to_midi/learning.py` | 325 |
| `stems_to_midi/learning.py` | 325 |
| `stems_to_midi/midi.py` | 456 |
| `stems_to_midi/midi.py` | 456 |
| `stems_to_midi/note_classification_core.py` | 757 |
| `stems_to_midi/note_classification_core.py` | 757 |
| `stems_to_midi/optimization/__init__.py` | 12 |
| `stems_to_midi/optimization/__init__.py` | 12 |
| `stems_to_midi/optimization/extract_features.py` | 404 |
| `stems_to_midi/optimization/extract_features.py` | 404 |
| `stems_to_midi/optimization/optimize.py` | 298 |
| `stems_to_midi/optimization/optimize.py` | 298 |
| `stems_to_midi/optimization_core.py` | 465 |
| `stems_to_midi/optimization_core.py` | 465 |
| `stems_to_midi/processing_shell.py` | 1204 |
| `stems_to_midi/processing_shell.py` | 1204 |
| `stems_to_midi/rebuild_core.py` | 639 |
| `stems_to_midi/rebuild_core.py` | 639 |
| `stems_to_midi/rebuild_shell.py` | 169 |
| `stems_to_midi/rebuild_shell.py` | 169 |
| `stems_to_midi/stereo_core.py` | 558 |
| `stems_to_midi/stereo_core.py` | 558 |
| `stems_to_midi/test_analysis_core.py` | 1815 |
| `stems_to_midi/test_analysis_core.py` | 1815 |
| `stems_to_midi/test_analysis_core_features.py` | 269 |
| `stems_to_midi/test_analysis_core_features.py` | 269 |
| `stems_to_midi/test_clustering_core.py` | 421 |
| `stems_to_midi/test_clustering_core.py` | 421 |
| `stems_to_midi/test_detection_shell.py` | 497 |
| `stems_to_midi/test_detection_shell.py` | 497 |
| `stems_to_midi/test_learning.py` | 726 |
| `stems_to_midi/test_learning.py` | 726 |
| `stems_to_midi/test_optimization_core.py` | 402 |
| `stems_to_midi/test_optimization_core.py` | 402 |
| `stems_to_midi/test_rebuild_core.py` | 752 |
| `stems_to_midi/test_rebuild_core.py` | 752 |
| `stems_to_midi/test_stems_to_midi.py` | 965 |
| `stems_to_midi/test_stems_to_midi.py` | 965 |
| `stems_to_midi/test_stereo_core.py` | 660 |
| `stems_to_midi/test_stereo_core.py` | 660 |
| `stems_to_midi_cli.py` | 449 |

### webui Package

| File | Lines |
|------|-------|
| `webui/__init__.py` | 8 |
| `webui/__init__.py` | 8 |
| `webui/api/__init__.py` | 14 |
| `webui/api/__init__.py` | 14 |
| `webui/api/config.py` | 268 |
| `webui/api/config.py` | 268 |
| `webui/api/downloads.py` | 258 |
| `webui/api/downloads.py` | 258 |
| `webui/api/job_status.py` | 436 |
| `webui/api/job_status.py` | 436 |
| `webui/api/operations.py` | 658 |
| `webui/api/operations.py` | 658 |
| `webui/api/projects.py` | 777 |
| `webui/api/projects.py` | 777 |
| `webui/api/settings.py` | 226 |
| `webui/api/settings.py` | 226 |
| `webui/api/upload.py` | 103 |
| `webui/api/upload.py` | 103 |
| `webui/app.py` | 150 |
| `webui/app.py` | 150 |
| `webui/config.py` | 86 |
| `webui/config.py` | 86 |
| `webui/config_schema.py` | 101 |
| `webui/config_schema.py` | 101 |
| `webui/jobs.py` | 409 |
| `webui/jobs.py` | 409 |
| `webui/settings_schema.py` | 1103 |
| `webui/settings_schema.py` | 1103 |
| `webui/test_analysis_api.py` | 278 |
| `webui/test_analysis_api.py` | 278 |
| `webui/test_api.py` | 556 |
| `webui/test_api.py` | 556 |
| `webui/test_config_api.py` | 243 |
| `webui/test_config_api.py` | 243 |
| `webui/test_config_api_frontend.py` | 347 |
| `webui/test_config_api_frontend.py` | 347 |
| `webui/test_config_schema_validation.py` | 197 |
| `webui/test_config_schema_validation.py` | 197 |
| `webui/test_reclassify_api.py` | 558 |
| `webui/test_reclassify_api.py` | 558 |
| `webui/test_settings_schema.py` | 265 |
| `webui/test_settings_schema.py` | 265 |
| `webui/test_threshold_tuning.py` | 452 |
| `webui/test_threshold_tuning.py` | 452 |
| `webui/test_yaml_config_core.py` | 413 |
| `webui/test_yaml_config_core.py` | 413 |
| `webui/yaml_config_core.py` | 482 |
| `webui/yaml_config_core.py` | 482 |

### moderngl_renderer Package

| File | Lines |
|------|-------|
| `moderngl_renderer/__init__.py` | 105 |
| `moderngl_renderer/__init__.py` | 105 |
| `moderngl_renderer/animation.py` | 311 |
| `moderngl_renderer/animation.py` | 311 |
| `moderngl_renderer/core.py` | 558 |
| `moderngl_renderer/core.py` | 558 |
| `moderngl_renderer/midi_animation.py` | 336 |
| `moderngl_renderer/midi_animation.py` | 336 |
| `moderngl_renderer/midi_video_core.py` | 447 |
| `moderngl_renderer/midi_video_core.py` | 447 |
| `moderngl_renderer/midi_video_shell.py` | 457 |
| `moderngl_renderer/midi_video_shell.py` | 457 |
| `moderngl_renderer/shell.py` | 1344 |
| `moderngl_renderer/shell.py` | 1344 |
| `moderngl_renderer/test_animation.py` | 274 |
| `moderngl_renderer/test_animation.py` | 274 |
| `moderngl_renderer/test_core.py` | 560 |
| `moderngl_renderer/test_core.py` | 560 |
| `moderngl_renderer/test_fade_logic.py` | 25 |
| `moderngl_renderer/test_fade_logic.py` | 25 |
| `moderngl_renderer/test_midi_animation.py` | 306 |
| `moderngl_renderer/test_midi_animation.py` | 306 |
| `moderngl_renderer/test_midi_render_simple.py` | 183 |
| `moderngl_renderer/test_midi_render_simple.py` | 183 |
| `moderngl_renderer/test_midi_video_core.py` | 427 |
| `moderngl_renderer/test_midi_video_core.py` | 427 |
| `moderngl_renderer/test_midi_video_moderngl.py` | 428 |
| `moderngl_renderer/test_midi_video_moderngl.py` | 428 |
| `moderngl_renderer/test_shell.py` | 608 |
| `moderngl_renderer/test_shell.py` | 608 |
| `moderngl_renderer/test_visual_quality.py` | 196 |
| `moderngl_renderer/test_visual_quality.py` | 196 |
| `moderngl_renderer/text_overlay_shell.py` | 183 |
| `moderngl_renderer/text_overlay_shell.py` | 183 |

### JavaScript

| File | Lines |
|------|-------|
| `site/assets/javascripts/bundle.79ae519e.min.js` | 16 |
| `site/assets/javascripts/lunr/min/lunr.ar.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.da.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.de.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.du.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.el.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.es.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.fi.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.fr.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.he.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.hi.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.hu.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.hy.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.it.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.ja.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.jp.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.kn.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.ko.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.multi.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.nl.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.no.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.pt.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.ro.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.ru.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.sa.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.stemmer.support.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.sv.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.ta.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.te.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.th.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.tr.min.js` | 18 |
| `site/assets/javascripts/lunr/min/lunr.vi.min.js` | 1 |
| `site/assets/javascripts/lunr/min/lunr.zh.min.js` | 1 |
| `site/assets/javascripts/lunr/tinyseg.js` | 206 |
| `site/assets/javascripts/lunr/wordcut.js` | 6708 |
| `site/assets/javascripts/workers/search.2c215733.min.js` | 42 |
| `webui/static/js/advanced-midi.js` | 450 |
| `webui/static/js/advanced-midi.js` | 450 |
| `webui/static/js/api.js` | 350 |
| `webui/static/js/api.js` | 350 |
| `webui/static/js/app.js` | 480 |
| `webui/static/js/app.js` | 480 |
| `webui/static/js/operations.js` | 733 |
| `webui/static/js/operations.js` | 733 |
| `webui/static/js/projects.js` | 976 |
| `webui/static/js/projects.js` | 976 |
| `webui/static/js/settings.js` | 304 |
| `webui/static/js/settings.js` | 304 |
| `webui/static/js/threshold-tuning.js` | 1048 |
| `webui/static/js/threshold-tuning.js` | 1048 |
| `webui/static/js/waveform.js` | 1536 |
| `webui/static/js/waveform.js` | 1536 |
