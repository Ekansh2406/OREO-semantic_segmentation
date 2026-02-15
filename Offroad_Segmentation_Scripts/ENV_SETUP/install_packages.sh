#!/bin/bash

# Try activating the Conda environment
echo "Activating the Conda environment 'EDU'..."

# Get the conda executable path
CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
    echo "Could not determine conda base directory. Ensure that Conda is installed."
    exit 1
fi

# Source conda.sh to enable conda activate
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the EDU environment
conda activate EDU

if [ $? -ne 0 ]; then
    echo "Failed to activate the 'EDU' environment."
    exit 1
fi

# Install the required packages
echo "Installing PyTorch, Torchvision, CUDA 11.8, and Ultralytics..."
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics -y && \
pip install opencv-contrib-python && \
pip install tqdm

if [ $? -eq 0 ]; then
    echo "Environment setup complete. You can now run your code in the 'EDU' environment."
else
    echo "Package installation failed."
    exit 1
fi
