#!/bin/bash

# Network Analysis Setup and Run Script

# Make sure script stops on any error
set -e

echo "Setting up Network Analysis Environment..."

# Function to check and install Python 3.11 if not present
install_python() {
    if ! command -v python3.11 &> /dev/null; then
        echo "Python 3.11 not found. Installing..."
        if command -v apt &> /dev/null; then
            # Debian/Ubuntu
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum install -y epel-release
            sudo yum install -y python3.11
        else
            echo "Unsupported package manager. Please install Python 3.11 manually."
            exit 1
        fi
    fi
}

# Remove existing virtual environment if it exists
cleanup_existing_env() {
    if [ -d "venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    fi
}

# Create and activate virtual environment
setup_virtual_env() {
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip
}

# Install required packages
install_requirements() {
    echo "Installing required packages..."
    pip install streamlit==1.31.1
    pip install networkx==3.2.1
    pip install matplotlib==3.8.2
    pip install seaborn==0.13.1
    pip install numpy==1.26.3
    pip install python-louvain==0.16
    pip install tqdm==4.66.1
}

# Run the Streamlit app
run_app() {
    echo "Starting Streamlit app..."
    streamlit run app.py
}

# Main execution
main() {
    echo "Starting setup..."
    
    # Install Python if needed
    install_python
    
    # Clean up existing environment
    cleanup_existing_env
    
    # Setup virtual environment
    setup_virtual_env
    
    # Install requirements
    install_requirements
    
    # Run the app
    run_app
}

# Run main function
main
