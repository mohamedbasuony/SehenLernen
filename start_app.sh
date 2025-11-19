#!/bin/bash

# Sehen Lernen - Application Launcher for macOS/Linux
# This script starts both the backend and frontend services

clear

echo ""
echo "================================================"
echo "  SEHEN LERNEN - APPLICATION LAUNCHER"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run 'bash setup_mac.sh' first to set up the application."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
python3 -c "import uvicorn" 2> /dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Dependencies not found!"
    echo ""
    echo "Please run 'bash setup_mac.sh' first to install dependencies."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[OK] Environment ready"
echo ""

# Function to start services
start_services() {
    echo "Starting Backend Server..."
    osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && source venv/bin/activate && cd Backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\""
    
    sleep 3
    
    echo "Starting Frontend Application..."
    osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && source venv/bin/activate && cd Fronted && streamlit run app.py --server.port 8501\""
    
    sleep 4
}

# Start services
start_services

echo ""
echo "================================================"
echo "  SERVICES STARTED SUCCESSFULLY!"
echo "================================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:8501"
echo ""
echo "Opening application in browser..."
sleep 1

# Open browser
open http://localhost:8501

echo ""
echo "Keep the Terminal windows open while using the application."
echo "To stop: Close both Terminal windows or press Ctrl+C in each."
echo ""
read -p "Press Enter to exit this window..."
