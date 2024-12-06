#!/bin/bash

# Twitter Analysis Setup and Run Script

# Make sure script stops on any error
set -e

echo "Setting up Twitter Network Analysis Environment..."

# Check if python and pip are installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create and activate virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv || {
    echo "Failed to create virtual environment. Continuing without it..."
}

# Try to activate virtual environment, continue if it fails
source venv/bin/activate || {
    echo "Continuing without virtual environment..."
}

# Install required packages
echo "Installing required packages..."
pip install numpy networkx matplotlib seaborn tqdm cython

# Check if twitter_combined.txt exists
if [ ! -f "data/twitter_combined.txt" ]; then
    echo "Error: twitter_combined.txt not found in data directory!"
    exit 1
fi

# Compile Cython code
echo "Compiling Cython code..."
python3 setup.py build_ext --inplace

# Run the analysis
echo "Running Twitter network analysis..."
python3 run_analysis.py data/twitter_combined.txt

echo "Analysis complete! Check the network_analysis_cy_* directory for results."

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate || true
fi
