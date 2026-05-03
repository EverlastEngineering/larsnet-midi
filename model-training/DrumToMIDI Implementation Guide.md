# **DrumToMIDI: Implementation & Environment Guide**

This guide provides the technical setup and execution strategy for the DrumToMIDI Deep Learning pipeline, focusing on cross-platform compatibility (Mac Mini Metal vs. Windows Desktop).

Put all new files for this deep learning system in the `/model-training` folder.

## **1\. Environment Setup**

### **Mac Mini (MPS / Metal)**

The Mac Mini is your **Development & Debugging** machine. It uses the mps (Metal Performance Shaders) backend.

* **Requirements:** macOS 12.3+ and a version of PyTorch compiled with MPS support.  
* **Installation:**  
  pip install torch torchvision torchaudio

* **Verification Script:**  
  import torch  
  if torch.backends.mps.is\_available():  
      mps\_device \= torch.device("mps")  
      x \= torch.ones(1, device=mps\_device)  
      print("MPS (Metal) is active.")  
  else:  
      print("MPS not found. Check macOS version.")

### **Windows Desktop (AMD ROCm via WSL2)**

The 7900XT is your **Production Training** machine. To avoid driver headaches on Windows, using **WSL2 (Ubuntu)** is highly recommended.

1. **Install WSL2:** wsl \--install (Ubuntu).  
2. **AMD Drivers:** Ensure "AMD Software: Adrenalin Edition" is up to date on the Windows host.  
3. **ROCm in WSL2:** Inside the Ubuntu terminal, install the ROCm-compatible PyTorch:  
   pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/rocm6.0\](https://download.pytorch.org/whl/rocm6.0)

## **2\. Multi-Machine Workflow Strategy**

To maximize efficiency, follow this "Loop" between your two machines:

| Phase | Machine | Goal |
| :---- | :---- | :---- |
| **Logic Dev** | Mac Mini | Perfect the midi\_to\_frame\_array and DrumTranscriber code. |
| **Verification** | Mac Mini | Run the **Verification Visualizer** on 5-10 tracks. |
| **Smoke Test** | Mac Mini | Run Section 8 (Overfit) on a 10s loop to ensure zero code errors. |
| **Mass Training** | 7900XT | Sync the 90GB dataset and run the training loop for 24-48 hours. |
| **Deployment** | Mac Mini | Load the .ckpt from the PC and build the **Inference Post-Processor**. |

## **3\. Training Tips & Troubleshooting**

### **Memory Management (VRAM)**

* **Batch Size:** On the 7900XT (20GB VRAM), you can likely start with a batch\_size=32. On the Mac Mini, if it has 8GB RAM, stick to batch\_size=4 or 8\.  
* **Gradient Accumulation:** If you want the "intelligence" of a 64-batch size but only have memory for 8, use gradient accumulation to update weights every 8 steps.

### **Data Loading Bottlenecks**

With 90GB of audio, your GPU might sit idle while the CPU struggles to generate spectrograms.

* **The Pre-Process Strategy:** Do not calculate spectrograms inside the training loop. Run a script once that converts all .wav files into .pt (PyTorch Tensor) files on your SSD.  
* **Benefit:** Loading a .pt file is \~10x faster than loading a .wav and running an FFT.

### **Common Logic Pitfalls**

1. **Sample Rate Mismatch:** If your e-GMD files are 44.1kHz but your logic assumes 48kHz, your MIDI will drift out of sync by \~8% over the length of the song. **Strictly verify sr in every step.**  
2. **Normalization:** Always normalize your spectrograms. A simple (spec \- spec.mean()) / spec.std() helps the model converge much faster.  
3. **The "Silence" Problem:** High-accuracy models often fail because they are trained on too much silence. Use a simple RMS threshold to skip silent sections of the drum stems during training.

## **4\. Helpful LLM Prompting**

When asking an LLM to generate the actual files for the steps in drum\_to\_midi\_dl\_plan.md, use this specific phrasing:

"Using the logic in Step 1 (Feature Engineering) of the provided blueprint, generate a standalone Python script feature\_extractor.py that processes a folder of WAVs into PyTorch tensors. Ensure it handles stereo-width as described."

## **5\. Directory Structure Recommendation**

/DrumToMIDI\_Project  
  /data  
    /raw\_wavs      \<-- The 90GB dataset  
    /processed\_pt  \<-- Pre-calculated spectrograms  
  /models  
    /checkpoints   \<-- Saved .ckpt files  
  /scripts  
    dataset.py     \<-- PyTorch Dataset class  
    model.py       \<-- The CRNN Skeleton  
    train.py       \<-- The Training/Calibration loop  
    inference.py   \<-- The Post-Processor  
