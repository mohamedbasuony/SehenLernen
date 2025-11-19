#!/bin/bash

# Sehen Lernen - Automated Setup for macOS/Linux
# This script sets up the application for first-time use

clear

echo ""
echo "================================================"
echo "  SEHEN LERNEN - SETUP WIZARD"
echo "================================================"
echo ""

# Check if Python is installed
echo "Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "[ERROR] Python 3.9 or higher is required but not found!"
    echo ""
    echo "Install Python:"
    echo "  macOS: brew install python3"
    echo "  Linux: sudo apt-get install python3 python3-venv"
    echo ""
    echo "Visit: https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter after installing Python..."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating isolated Python environment..."
if [ -d "venv" ]; then
    echo "[OK] Virtual environment already exists"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Virtual environment created"
fi
echo ""

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python packages (this may take 5-10 minutes)..."
echo "Please be patient..."
echo ""

pip install --upgrade pip > /dev/null 2>&1

if [ -f "Backend/requirements.txt" ]; then
    echo "Installing backend dependencies..."
    pip install -q -r Backend/requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install backend dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Backend dependencies installed"
fi

if [ -f "Fronted/requirements.txt" ]; then
    echo "Installing frontend dependencies..."
    pip install -q -r Fronted/requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install frontend dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Frontend dependencies installed"
fi

echo ""
echo "================================================"
echo "  SETUP COMPLETE!"
echo "================================================"
echo ""
echo "To start the application:"
echo "  1. Run: bash start_app.sh"
echo "     OR"
echo "  2. Run these commands:"
echo "     source venv/bin/activate"
echo "     # Then follow the prompts"
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:8501"
echo ""
read -p "Press Enter to exit..."
