FROM condaforge/miniforge3

# Copy environment definition
COPY environment.yml .

# Create the environment
RUN mamba env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "larsnet-midi", "/bin/bash", "-c"]

# Verify installation
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"
RUN python -c "import librosa; print('librosa version:', librosa.__version__)"
RUN python -c "import mido; print('mido version:', mido.version_info)"
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Configure shell to use environment by default
SHELL ["bash", "--login", "-c"]
RUN conda init bash
RUN echo "conda activate larsnet-midi" >> ~/.bashrc

# Create .bash_env for non-interactive shells (docker exec -c)
RUN echo 'eval "$(conda shell.bash hook)"' >> ~/.bash_env && \
    echo 'conda activate larsnet-midi' >> ~/.bash_env

# Set BASH_ENV to source our conda setup for all bash invocations
ENV BASH_ENV=~/.bash_env
