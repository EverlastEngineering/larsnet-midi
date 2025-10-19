FROM condaforge/miniforge3

# Copy environment definition
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "larsnet", "/bin/bash", "-c"]

# Verify installation
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"
RUN python -c "import librosa; print('librosa version:', librosa.__version__)"
RUN python -c "import mido; print('mido version:', mido.__version__)"
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Configure shell to use environment by default
SHELL ["bash", "--login", "-c"]
RUN conda init bash
RUN echo "conda activate larsnet" >> ~/.bashrc
