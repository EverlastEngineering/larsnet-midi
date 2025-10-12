FROM condaforge/miniforge3

# Copy environment definition
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "myenv-arm", "/bin/bash", "-c"]

# Verify
RUN python -c "import torch; print('Torch device:', torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))"

SHELL ["bash", "--login", "-c"]
RUN conda init bash
RUN echo "conda activate myenv-arm" >> ~/.bashrc
