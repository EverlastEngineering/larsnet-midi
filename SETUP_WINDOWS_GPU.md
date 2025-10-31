# Windows GPU Setup Guide (Docker + NVIDIA)

This guide walks through enabling NVIDIA GPU acceleration in Docker on Windows for significantly faster drum separation processing.

## Why GPU on Windows?

- **Performance**: GPU acceleration provides 10-20x speedup vs CPU
- **Docker Support**: Windows Docker can access NVIDIA GPUs (unlike Mac with Metal)
- **Expected Performance** (51.7s audio, overlap=8): ~40-60s vs 777s CPU-only

## Prerequisites

### System Requirements
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (GTX 1060 or better recommended)
- At least 6GB GPU VRAM (8GB+ recommended for batch processing)
- WSL 2 installed and configured

### Check Your GPU
```powershell
# In PowerShell, check if you have an NVIDIA GPU
nvidia-smi
```

If this fails, you need to install NVIDIA drivers first.

## Step 1: Install Prerequisites

### 1.1 Install WSL 2
```powershell
# In PowerShell (Administrator)
wsl --install
```

Restart your computer after installation.

### 1.2 Install Docker Desktop
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Run installer
3. During setup, ensure "Use WSL 2 instead of Hyper-V" is checked
4. Restart computer

### 1.3 Install NVIDIA Drivers
1. Go to https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install latest Game Ready or Studio driver
4. Restart computer

### 1.4 Install NVIDIA Container Toolkit (WSL 2)

Open Ubuntu/WSL terminal:
```bash
# Configure repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker Desktop
```

## Step 2: Verify GPU Access in Docker

Test that Docker can see your GPU:

```powershell
# In PowerShell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Expected output:**
- GPU information table
- Driver version
- CUDA version
- GPU name and memory

If this fails, check:
- Docker Desktop is running
- WSL 2 integration is enabled in Docker Desktop settings
- NVIDIA drivers are installed

## Step 3: Update Docker Compose Configuration

The project's `docker-compose.yaml` needs GPU runtime configuration:

```yaml
services:
  DrumToMIDI-midi:
    # ... existing config ...
    
    # Add GPU support (uncomment for NVIDIA GPUs)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Note:** The docker-compose.yaml in this repository may already have GPU support configured. Check the file for existing GPU settings.

## Step 4: Build and Test

### Build the Container
```powershell
# In PowerShell, from project root
docker-compose build
```

### Start the Container
```powershell
docker-compose up -d
```

### Verify GPU Inside Container
```powershell
docker exec -it DrumToMIDI-midi bash -c "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}\")'"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080 (or your GPU name)
```

## Step 5: Run Separation with GPU

```powershell
# Run separation with CUDA device
docker exec -it DrumToMIDI-midi bash -c "cd /app && python separate.py 1 --model mdx23c --overlap 8"
```

If GPU is not available, it will fall back to CPU:
```bash
docker exec -it DrumToMIDI-midi bash -c "cd /app && python separate.py 1 --device cuda --overlap 8"
```

The device will automatically detect CUDA if available. You can also explicitly specify:
```powershell
docker exec -it larsnet-midi bash -c "cd /app && python separate.py 1 --device cuda --overlap 8"
```

**Expected Performance** (51.7s audio):
- overlap=2: ~15-20s (similar to Mac MPS)
- overlap=8: ~40-60s (similar to Mac MPS)
- vs CPU: ~200s (overlap=2), ~777s (overlap=8)

## Troubleshooting

### Issue: "could not select device driver"
**Solution:** Ensure Docker Desktop has WSL 2 integration enabled:
1. Open Docker Desktop
2. Settings → Resources → WSL Integration
3. Enable integration with your WSL distro
4. Apply & Restart

### Issue: "nvidia-smi not found" in container
**Solution:** Verify NVIDIA Container Toolkit installed:
```bash
# In WSL terminal
dpkg -l | grep nvidia-docker2
```

If not installed, return to Step 1.4.

### Issue: CUDA out of memory
**Solution:** Reduce batch size or overlap:
```powershell
# Try overlap=4 or overlap=2
docker exec -it DrumToMIDI-midi bash -c "cd /app && python separate.py 1 --overlap 4"
```

### Issue: Slow performance despite GPU
**Solution:** Verify device is actually CUDA:
```powershell
# Check device detection
docker exec -it DrumToMIDI-midi bash -c "cd /app && python -c 'from device_utils import detect_best_device, print_device_info; d = detect_best_device(); print(f\"Detected: {d}\"); print_device_info(d)'"
```

Should show CUDA, not CPU.

### Issue: Docker can't access GPU after Windows update
**Solution:** Reinstall NVIDIA drivers and restart Docker Desktop.

## Performance Comparison

| Configuration | Time (51.7s audio) | RTF | Speedup vs CPU |
|---------------|-------------------|-----|----------------|
| CUDA overlap=2 | ~15-20s | 0.3-0.4x | 10-13x faster |
| CUDA overlap=8 | ~40-60s | 0.8-1.2x | 13-19x faster |
| CPU overlap=2 | ~200s | 3.9x | baseline |
| CPU overlap=8 | ~777s | 15x | baseline |

**Note:** Actual times depend on your specific GPU model. RTX 30-series or better recommended.

## Alternative: Native Windows Installation

For native Windows installation (without Docker):
1. Install Python 3.11
2. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
3. Follow conda environment setup similar to Mac native guide
4. GPU will be detected automatically

**Pros:**
- Potentially slightly faster
- Direct GPU access

**Cons:**
- More complex setup
- Environment management
- Windows-specific issues

Docker is recommended for Windows users due to easier setup and better reproducibility.

## Next Steps

- [ ] Verify GPU setup (Step 2)
- [ ] Test separation with GPU (Step 5)
- [ ] Compare performance to CPU
- [ ] Optimize overlap/batch settings for your GPU

## Support

If you encounter issues:
1. Check Docker Desktop is running and WSL 2 integration enabled
2. Verify nvidia-smi works in PowerShell
3. Check docker-compose.yaml has GPU configuration
4. Review Docker Desktop logs
5. Ensure Windows and NVIDIA drivers are up to date

## Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Desktop WSL 2 Backend](https://docs.docker.com/desktop/wsl/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
