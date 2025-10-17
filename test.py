import librosa
import numpy as np

# 1. Load the audio file
# Replace 'audio.wav' with the actual path to your audio file.
y, sr = librosa.load('./separated_stems/The Fate Of Ophelia/The Fate Of Ophelia-kick.wav')

# 2. Compute the onset strength envelope
# Parameters like hop_length can be adjusted to influence the envelope.
# A smaller hop_length results in a more detailed envelope.
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)

# 3. Use librosa.util.peak_pick with adjusted parameters to find the peaks
# Tune these parameters to capture all your events.
# Here, `delta` is reduced for higher sensitivity, and `wait` is reduced
# to allow for closely spaced peaks.
# onset_frames = librosa.onset.onset_detect(
#     onset_env, 
#     pre_max=3, 
#     post_max=3, 
#     pre_avg=3, 
#     post_avg=5, 
#     delta=0.05, 
#     wait=5
# )

# # 4. Convert the peak frame indices to timestamps in seconds
# onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

# # 5. Print the detected onset times to the console
# print("Detected onset times (in seconds):", onset_times)


onset_times = librosa.onset.onset_detect(
    y=y,
    sr=sr,
    units='time',
    # Optional parameters for tuning detection:
    # pre_max=3,  # Window size before a peak to find the local max
    # post_max=3, # Window size after a peak to find the local max
    # delta=0.05, # Amplitude threshold above the local average
    # wait=5      # Minimum time (in frames) between detected onsets
)

# 4. Print the detected onset times.
print("Detected onset times (in seconds):")
for time in onset_times:
    print(f"{time:.3f}")