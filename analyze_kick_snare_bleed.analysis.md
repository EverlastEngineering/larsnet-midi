Loading audio files...
Detecting onsets...
  Kick onsets:  194
  Snare onsets: 103
  Snare-only hits (no coinciding kick): 94

Analyzing spectral content...

================================================================================
SNARE BLEED ANALYSIS
================================================================================

QUESTION: Is there snare energy in the kick track when snares hit?

Analyzing 94 snare-only hits (excluding kick+snare coincidences)

================================================================================
1. ENERGY ANALYSIS
================================================================================

Kick track RMS at snare-only times:
  Mean:   0.195861
  Median: 0.224133
  Max:    0.369686

Kick track RMS at kick times (baseline):
  Mean:   0.332018
  Median: 0.346985

  Energy ratio (snare-times / kick-times): 0.59
  ⚠️  SIGNIFICANT BLEED: Kick track has 59.0% of normal energy at snare times!

================================================================================
2. FREQUENCY CONTENT ANALYSIS
================================================================================

Kick track at snare-only times:
  Low (0-200 Hz, kick fundamental):     1.45e+05  (96.0%)
  Low-mid (200-500 Hz, kick body):      4.88e+03  (2.4%)
  Mid (500-2000 Hz, snare body):        7.29e+02  (0.4%)
  High-mid (2-5 kHz, snare crack):      1.15e+02  (0.9%)
  High (5+ kHz, snare sizzle):          6.10e+01  (0.3%)
  Spectral centroid: 1245.5 Hz
  Zero crossing rate: 0.0106

Kick track at kick times (baseline):
  Low (0-200 Hz, kick fundamental):     3.59e+05  (95.9%)
  Low-mid (200-500 Hz, kick body):      3.42e+03  (1.0%)
  Mid (500-2000 Hz, snare body):        1.46e+03  (0.5%)
  High-mid (2-5 kHz, snare crack):      6.72e+03  (2.1%)
  High (5+ kHz, snare sizzle):          1.52e+03  (0.5%)
  Spectral centroid: 2058.9 Hz
  Zero crossing rate: 0.0573

Actual snare track at snare times (reference):
  Low (0-200 Hz, kick fundamental):     2.87e+04  (38.2%)
  Low-mid (200-500 Hz, kick body):      1.21e+04  (13.3%)
  Mid (500-2000 Hz, snare body):        2.81e+03  (3.8%)
  High-mid (2-5 kHz, snare crack):      1.53e+04  (23.4%)
  High (5+ kHz, snare sizzle):          1.38e+04  (21.2%)
  Spectral centroid: 4896.4 Hz
  Zero crossing rate: 0.1414

================================================================================
3. SNARE SIGNATURE DETECTION
================================================================================

Mid/Low frequency ratio in kick track:
  At snare times: 0.0050
  At kick times:  0.0041
  Ratio:          1.24x

Zero crossing rate in kick track:
  At snare times: 0.0106
  At kick times:  0.0573

Spectral centroid in kick track:
  At snare times: 1245.5 Hz
  At kick times:  2058.9 Hz
  ⚠️  LOWER centroid at snare times - unexpected! May indicate issue with detection

================================================================================
CONCLUSION
================================================================================
✓ Significant energy present at snare times (59.0%)

Bleed indicators found: 1/3

⚠️  MODERATE EVIDENCE OF SNARE BLEED
Recommendation: Consider light sidechain compression

✓ Analysis complete!