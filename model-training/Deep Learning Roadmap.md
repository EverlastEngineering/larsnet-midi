# **DrumToMIDI: Deep Learning Architecture & Training Plan**

This document serves as the master blueprint for transitioning a heuristic DSP engine into a supervised deep-learning system for inferring MIDI from "dirty" (stem-split) drum audio.

## **1\. Feature Engineering: 3-Channel Input Generator**

This step transforms raw 1D audio into a 2D feature set. We use a 3-channel approach to preserve spatial information (Stereo Width) which is critical for differentiating centered Snares from wide Claps/Reverbs.

**Dimensions:** \[3, 128, Time\_Steps\]

* **Temporal Resolution:** \~11.6ms per frame (512 hop length @ 44.1kHz).  
* **Frequency Resolution:** 128 Mel bins (Log-scaled energy).

import torch  
import torchaudio  
import torchaudio.transforms as T

def get\_input\_tensor(audio\_path, sample\_rate=44100):  
    """  
    Converts a stereo wav file into a 3-channel Mel-Spectrogram.  
    Channels: \[Left, Right, Stereo-Width-Ratio\]  
    """  
    \# Load audio: waveform shape is \[channels, samples\]  
    waveform, \_ \= torchaudio.load(audio\_path)  
      
    \# Initialize the MelSpectrogram transform  
    mel\_transform \= T.MelSpectrogram(  
        sample\_rate=sample\_rate,   
        n\_mels=128,   
        n\_fft=2048,   
        hop\_length=512  
    )  
      
    \# Extract L/R and convert to Decibels for neural network stability  
    spec\_l \= T.AmplitudeToDB()(mel\_transform(waveform\[0:1\]))  
    spec\_r \= T.AmplitudeToDB()(mel\_transform(waveform\[1:2\]))  
      
    \# Feature 3: Stereo Width (The "Clap vs Snare" engine)  
    \# Calculated as the energy of the 'Side' signal (L-R)  
    side \= waveform\[0:1\] \- waveform\[1:2\]  
    spec\_width \= T.AmplitudeToDB()(mel\_transform(side))  
      
    \# Stack into a single 3D tensor: \[Channels, Freq, Time\]  
    return torch.cat(\[spec\_l, spec\_r, spec\_width\], dim=0)

## **2\. Label Mapping: Hierarchical MIDI Grouping**

We map 128 MIDI notes into 11 probability channels. This "Hierarchy" ensures the model focuses on broad timbral categories while retaining specific instrument identity.

| Index | Label | MIDI Notes (GM) | Logic / Reasoning |
| :---- | :---- | :---- | :---- |
| **0** | **Kick** | 35, 36 | Low frequency transients |
| **1** | **Snare/Clap** | 38, 40, 37, 39 | Center vs. Wide (Spatial differentiation) |
| **2** | **HH Closed** | 42, 44 | Fast high-freq decay |
| **3** | **HH Open** | 46 | Sustained high-freq wash |
| **4** | **Tom High** | 48, 50 | Pitch-based grouping |
| **5** | **Tom Mid** | 45, 47 | Pitch-based grouping |
| **6** | **Tom Low** | 41, 43 | Pitch-based grouping |
| **7** | **Crash** | 49, 57 | Violent onset, exponential decay |
| **8** | **Ride** | 51, 53 | High-freq "ping" texture |
| **9** | **China** | 52 | Complex noisy/trashy spectrum |
| **10** | **Splash** | 55 | High-pitch fast-transient cymbal |

## **3\. Label Encoding: Causal Label Smearing**

This step creates the "Target" matrix. We use **Causal Smearing**—only smearing forward in time—to ensure the model doesn't learn to "predict the future" while still giving the optimizer a gradient to follow.

def midi\_to\_frame\_array(midi\_notes, total\_frames, hop\_length, sr):  
    """  
    Maps MIDI note objects to an \[11, Frames\] binary heatmap with causal smearing.  
    """  
    seconds\_per\_frame \= hop\_length / sr  
    labels \= torch.zeros((11, total\_frames))  
      
    \# Mapping based on the Hierarchical Table in Section 2  
    mapping \= {36:0, 35:0, 38:1, 40:1, 37:1, 39:1, 42:2, 44:2, 46:3,   
               48:4, 50:4, 45:5, 47:5, 41:6, 43:6, 49:7, 57:7,   
               51:8, 53:8, 52:9, 55:10}

    for note in midi\_notes:  
        if note.pitch in mapping:  
            \# Convert MIDI seconds to the nearest spectrogram frame  
            hit\_frame \= int(note.start\_time / seconds\_per\_frame)  
            idx \= mapping\[note.pitch\]  
              
            \# Causal Smear: Probability is 1.0 at impact, then decays  
            \# This allows the model to be 'close' and still receive partial credit  
            if hit\_frame \< total\_frames:  
                labels\[idx, hit\_frame\] \= 1.0       \# Precision Hit  
                if hit\_frame \+ 1 \< total\_frames: labels\[idx, hit\_frame+1\] \= 0.8  
                if hit\_frame \+ 2 \< total\_frames: labels\[idx, hit\_frame+2\] \= 0.5  
                if hit\_frame \+ 3 \< total\_frames: labels\[idx, hit\_frame+3\] \= 0.2  
                  
    return labels

## **4\. Verification Visualizer**

A utility to confirm that your get\_input\_tensor (Audio) and midi\_to\_frame\_array (Labels) are perfectly aligned in time.

import matplotlib.pyplot as plt

def plot\_alignment\_check(spec\_tensor, label\_tensor):  
    """  
    Overlays ground truth labels on the spectrogram.   
    Alignment check: Vertical lines in top plot must match hot spots in bottom plot.  
    """  
    fig, (ax1, ax2) \= plt.subplots(2, 1, figsize=(15, 8), sharex=True)  
      
    \# Plot Channel 0 (Left Spec)  
    ax1.imshow(spec\_tensor\[0\].numpy(), aspect='auto', origin='lower')  
    ax1.set\_title("Input Feature: Left Spectrogram")  
      
    \# Plot 11-Channel Label Matrix  
    ax2.imshow(label\_tensor.numpy(), aspect='auto', origin='lower', cmap='magma')  
    ax2.set\_yticks(range(11))  
    ax2.set\_yticklabels(\['Kick', 'Snare', 'HHC', 'HHO', 'TH', 'TM', 'TL', 'Cr', 'Ri', 'Ch', 'Sp'\])  
    ax2.set\_title("Target: 11-Channel Heatmap")  
      
    plt.tight\_layout()  
    plt.show()

## **5\. Model Architecture: CRNN Skeleton**

The "Neural Engine." It uses Convolutional layers to "see" frequency shapes and Recurrent (GRU) layers to "understand" rhythmic sequences.

import torch.nn as nn

class DrumTranscriber(nn.Module):  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_()  
        \# THE EYES: Conv block extracts frequency-domain features (transients)  
        self.conv \= nn.Sequential(  
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d((2, 1)), \# Reduce frequency height, preserve time resolution  
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool2d((2, 1))  
        )  
          
        \# THE BRAIN: GRU processes the timeline.   
        \# Bi-directional allows the model to look 'ahead' to see if a cymbal tail follows.  
        \# Input size (2048) comes from 64 filters \* (128 / 2 / 2\) freq bins.  
        self.rnn \= nn.GRU(2048, 128, batch\_first=True, bidirectional=True)  
          
        \# THE DECISION: Linear layer maps 256 GRU features to 11 probability bits  
        self.fc \= nn.Linear(256, 11\)

    def forward(self, x):  
        \# x shape: \[Batch, Channels, Freq, Time\]  
        x \= self.conv(x)   
          
        \# Prepare for RNN: Reorder to \[Batch, Time, Features\]  
        x \= x.permute(0, 3, 1, 2).flatten(2)   
          
        \# Temporal processing  
        x, \_ \= self.rnn(x)  
          
        \# Return probability (0.0 \- 1.0) for every frame  
        return torch.sigmoid(self.fc(x))

## **6\. Dynamic Calibration Loop**

Instead of a fixed 0.5 threshold, this step finds the "Optimal Trigger Point" for each drum type by looking at the variance between prediction confidence and ground truth.

1. **Prediction:** Model generates probability curves.  
2. **Peak-Finding:** Find local maxima in the probability stream.  
3. **Error Check:** Align predicted peaks with Ground Truth MIDI.  
4. **Threshold Tuning:** Iteratively find the threshold for each of the 11 channels that minimizes False Positives while maximizing True Hits.

## **7\. Inference Post-Processor**

The final stage to convert the Neural Heatmap back into a standard MIDI file.

1. **Heatmap Generation:** Pass audio through the trained .ckpt.  
2. **Probability Filtering:** Apply thresholds derived in Step 6\.  
3. **Onset Snapping:** Use the steepest part of the probability curve as the Note On event.  
4. **MIDI Writing:** Export as a .mid file for use in a DAW.

## **8\. Integration Test: Overfit Smoke Test**

Proves that the data pipe is leak-proof. If the model can't memorize one single 30-second file, there is a fundamental bug in Step 1 or Step 3\.

def run\_smoke\_test(audio\_path, midi\_notes):  
    """  
    A minimal training loop to prove the model can 'memorize' a single track.  
    If loss \< 0.01, the data flow is mathematically verified.  
    """  
    device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
      
    \# Prep data  
    input\_tensor \= get\_input\_tensor(audio\_path).unsqueeze(0).to(device)  
    target\_tensor \= midi\_to\_frame\_array(midi\_notes, input\_tensor.shape\[3\], 512, 44100\)  
    target\_tensor \= target\_tensor.unsqueeze(0).permute(0, 2, 1).to(device) \# \[Batch, Time, Classes\]

    model \= DrumTranscriber().to(device)  
    optimizer \= torch.optim.Adam(model.parameters(), lr=1e-3)  
    criterion \= nn.BCELoss() \# Binary Cross Entropy for multi-hot labels

    print("Running integration test (200 epochs)...")  
    for epoch in range(200):  
        optimizer.zero\_grad()  
        output \= model(input\_tensor)  
        loss \= criterion(output, target\_tensor)  
        loss.backward()  
        optimizer.step()  
          
        if epoch % 50 \== 0:  
            print(f"Epoch {epoch} | Training Loss: {loss.item():.6f}")  
              
    return "SUCCESS: Pipeline Verified" if loss \< 0.01 else "FAILURE: Check Alignment"  
