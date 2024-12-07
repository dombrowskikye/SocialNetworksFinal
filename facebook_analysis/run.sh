#!/bin/bash

# Facebook Analysis Setup and Run Script

# Make sure script stops on any error
set -e

echo "Setting up Facebook Network Analysis Environment..."

# Function to check and install Python 3.11
install_python311() {
    if ! command -v python3.11 &> /dev/null; then
        echo "Python 3.11 not found. Installing..."
        if command -v apt &> /dev/null; then
            # Debian/Ubuntu
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3.11-dev
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum install -y epel-release
            sudo yum install -y python3.11 python3.11-devel
        else
            echo "Unsupported package manager. Please install Python 3.11 manually."
            exit 1
        fi
    fi
}

# Check if Python 3.11 is installed, if not install it
install_python311

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv venv || {
    echo "Failed to create virtual environment with Python 3.11. Please check the installation."
    exit 1
}

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || {
    echo "Failed to activate virtual environment."
    exit 1
}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install numpy==1.26.3  # Specific version known to work with Python 3.11
pip install networkx==3.2.1
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
pip install tqdm==4.66.1
pip install cython==3.0.8
pip install setuptools==69.0.3

# Check if facebook.txt exists
if [ ! -f "data/facebook_combined.txt" ]; then
    echo "Error: facebook_combined.txt not found in data directory!"
    exit 1
fi

# Compile Cython code
echo "Compiling Cython code..."
python setup.py build_ext --inplace

# Run the analysis
echo "Running Facebook network analysis..."
python run_facebook_analysis.py data/facebook_combined.txt

echo "Analysis complete! Check the network_analysis_cy_* directory for results."

# Deactivate virtual environment
deactivate

# Cleanup build artifacts (optional)
echo "Cleaning up build artifacts..."
rm -rf build/
find . -type f -name "*.so" -o -name "*.c" | while read f; do
    echo "Removing: $f"
    rm "$f"
done

echo "Setup and analysis completed successfully!"
