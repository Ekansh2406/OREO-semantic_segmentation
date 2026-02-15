#!/bin/bash

# Check if conda is available in the system
if ! command -v conda &> /dev/null; then
    echo "Conda is not found in your system. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create the environment
echo "Creating the Conda environment 'EDU' with Python 3.10..."
conda create --name EDU python=3.10 -y
