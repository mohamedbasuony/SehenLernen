# Sehen Lernen - Deployment Guide for 20 Student Machines

## Quick Decision Tree

**Time Available?**
- ⏱️ **< 2 hours prep**: Use **Option 1 (Standalone Executables)**
- ⏱️ **2-4 hours prep**: Use **Option 2 (Python + Scripts)**
- ⏱️ **> 4 hours prep & technical students**: Use **Option 3 (Docker)**

---

## 🏆 RECOMMENDED: Option 1 - Standalone Executables (EASIEST)

**Time to setup: 30 minutes prep + 5 minutes per machine**

### Why This Works Best for 20 Machines:
- ✅ No Python installation needed
- ✅ No dependency issues
- ✅ Single .exe file per app
- ✅ Double-click to run
- ✅ Automatic port management
- ✅ Can run on machines without internet (after initial setup)

### Setup Steps:

#### Step 1: Create Standalone Executables (Do This Once)

```bash
# Install PyInstaller
pip install pyinstaller

# Navigate to project
cd /Users/mobasuony/Desktop/SehenLernen-main

# Create backend executable
cd Backend
pyinstaller --onefile \
  --name SehrenLernen-Backend \
  --icon=../assets/icon.ico \
  --add-data "app:app" \
  -p . \
  app/main.py

# Create frontend executable  
cd ../Fronted
pyinstaller --onefile \
  --name SehenLernen-Frontend \
  --icon=../assets/icon.ico \
  app.py
```

Result: Two `.exe` files in `dist/` folders

#### Step 2: Create Deployment Package

```
SehenLernen-Deployment/
├── Backend/
│   ├── SehrenLernen-Backend.exe
│   └── README.txt (Backend on Port 8000)
├── Frontend/
│   ├── SehenLernen-Frontend.exe
│   └── README.txt (Frontend on Port 8501)
├── QUICK_START.txt
└── INSTALL.bat (batch script for Windows)
```

#### Step 3: Distribution Script (Windows Batch)

Create `INSTALL.bat`:
```batch
@echo off
echo Starting Sehen Lernen Backend...
start "Sehen Lernen Backend" Backend\SehrenLernen-Backend.exe
timeout /t 3

echo Starting Sehen Lernen Frontend...
start "Sehen Lernen Frontend" Frontend\SehenLernen-Frontend.exe
timeout /t 2

echo.
echo ✓ Both applications started!
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo.
echo Open your browser and go to: http://localhost:8501
pause
```

### Distribution to 20 Machines:

**Option A: USB Drives** (MOST RELIABLE)
1. Copy deployment folder to 20 USB drives
2. Give each student a USB
3. Have them run `INSTALL.bat`
4. Done! ✓

**Option B: Network Share**
1. Put deployment folder on shared network drive
2. Students run `\\network\SehenLernen\INSTALL.bat`
3. Everything runs locally from their machine

**Option C: Compressed Archive**
1. Create `SehenLernen-Deployment.zip`
2. Email or provide download link
3. Students extract and run `INSTALL.bat`

### Pros & Cons:
✅ **Pros:**
- Easiest setup (just run .exe)
- No dependencies
- Works offline
- Students can't accidentally break it
- Fast startup

❌ **Cons:**
- Larger file size (~200-300 MB)
- Need to rebuild if code changes
- Windows-specific (would need separate MacOS/Linux builds)

---

## Option 2 - Python + Automation Scripts (RECOMMENDED for Mixed Setup)

**Time to setup: 2 hours prep + 10 minutes per machine**

### Why Use This:
- ✅ One setup for all platforms (Windows, Mac, Linux)
- ✅ Easy to update code
- ✅ Students learn to use command line
- ✅ Medium complexity, manageable

### Setup Steps:

#### Step 1: Create Installation Script

Create `setup_windows.bat`:
```batch
@echo off
echo Installing Sehen Lernen...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.9+ is required but not installed!
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found!

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies (this may take 5 minutes)...
pip install -q -r Backend/requirements.txt
pip install -q -r Fronted/requirements.txt

echo.
echo ✓ Installation complete!
echo.
echo To start the app in the future, run: START.bat
pause
```

Create `START.bat`:
```batch
@echo off
REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start both servers
echo Starting Backend...
start "Sehen Lernen Backend" cmd /k "cd Backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

timeout /t 3

echo Starting Frontend...
start "Sehen Lernen Frontend" cmd /k "cd Fronted && streamlit run app.py --server.port 8501"

echo.
echo ✓ Both services started!
echo Open: http://localhost:8501
```

Create `setup_mac.sh`:
```bash
#!/bin/bash
echo "Installing Sehen Lernen..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.9+ required. Install from https://www.python.org/"
    exit 1
fi

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -q -r Backend/requirements.txt
pip install -q -r Fronted/requirements.txt

echo "✓ Installation complete!"
echo "To start: source venv/bin/activate && ./start.sh"
```

Create `start.sh`:
```bash
#!/bin/bash
source venv/bin/activate

echo "Starting Backend..."
open -a Terminal "cd $(pwd)/Backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

sleep 3

echo "Starting Frontend..."
open -a Terminal "cd $(pwd)/Fronted && streamlit run app.py --server.port 8501"

echo "Open: http://localhost:8501"
```

#### Step 2: Create Quick Start Guide

Create `QUICK_START.txt`:
```
FIRST TIME SETUP:
=================

Windows:
1. Double-click setup_windows.bat
2. Wait for installation (5 minutes)
3. It will show "✓ Installation complete!"
4. Click to close

STARTING THE APP:
=================

Windows:
1. Double-click START.bat
2. Wait 5 seconds for servers to start
3. Browser opens automatically
4. If not, go to: http://localhost:8501

Mac/Linux:
1. Open Terminal
2. cd to SehenLernen folder
3. chmod +x setup_mac.sh start.sh
4. ./setup_mac.sh (first time only)
5. ./start.sh (every time after)
6. Go to: http://localhost:8501

STOPPING:
=========
Close the Terminal/Command Prompt windows

PROBLEMS?
=========
See TROUBLESHOOTING.txt
```

### Distribution:

```
Distribution Package:
├── setup_windows.bat
├── setup_mac.sh
├── start.sh
├── START.bat
├── QUICK_START.txt
├── TROUBLESHOOTING.txt
├── Backend/
├── Fronted/
└── requirements.txt files
```

**Method:**
1. Create ZIP file
2. Email or USB to students
3. They run setup script once
4. They run start script each time

### Pros & Cons:

✅ **Pros:**
- Works on Windows, Mac, Linux
- Easy to update (just change files, re-run)
- Students learn Python workflow
- Smaller file size

❌ **Cons:**
- Requires Python installation
- Longer startup (needs to start Python)
- More troubleshooting needed

---

## Option 3 - Docker (Most Professional)

**Time to setup: 3-4 hours prep + 2 minutes per machine**

### Why Use This:
- ✅ Perfect reproducibility
- ✅ Guaranteed to work on all machines
- ✅ Easy scaling
- ✅ Professional deployment approach

### Step 1: Create Dockerfile

Create `Backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `Fronted/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
```

Create `docker-compose.yml` (root directory):
```yaml
version: '3.8'

services:
  backend:
    build: ./Backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./Backend/storage:/app/storage

  frontend:
    build: ./Fronted
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./Fronted:/app

volumes:
  backend_storage:
```

### Step 2: Quick Start with Docker

Create `docker-start.bat` (Windows):
```batch
@echo off
echo Starting Sehen Lernen with Docker...
docker-compose up

REM Open browser
timeout /t 5
start http://localhost:8501
```

Create `docker-start.sh` (Mac/Linux):
```bash
#!/bin/bash
docker-compose up
# Browser opens automatically for Mac
open http://localhost:8501
```

### Distribution:

Students need:
1. Docker Desktop installed (one-time)
2. Project folder
3. Run: `docker-compose up`

### Pros & Cons:

✅ **Pros:**
- Perfectly reproducible
- Works on all OS
- Professional setup
- Easy to scale

❌ **Cons:**
- Requires Docker installation (~500MB)
- Slightly slower startup
- Learning curve if students unfamiliar with Docker

---

## 🎯 FINAL RECOMMENDATION FOR YOUR SCENARIO:

### **USE OPTION 1 (Executables) IF:**
- Most students use Windows
- You want zero setup issues
- You have USB drives
- Time is critical

### **USE OPTION 2 (Python Scripts) IF:**
- Mixed OS (Windows + Mac + Linux)
- Students are tech-savvy
- You might need to update code during the day
- Network access available

### **USE OPTION 3 (Docker) IF:**
- This is a permanent installation
- You want professional setup
- Students will use this for weeks/months
- You have strong IT support

---

## 🚀 QUICK DEPLOYMENT CHECKLIST

### Day Before Deployment:

- [ ] Test deployment method on 2-3 different machines
- [ ] Create all necessary scripts/files
- [ ] Create ZIP/USB with deployment package
- [ ] Write simple QUICK_START guide
- [ ] Test both backend and frontend work together
- [ ] Create screenshot guide showing "Success" state
- [ ] Have backup plan (email, USB, network share)

### Deployment Day:

- [ ] Arrive 30 minutes early
- [ ] Test on first machine completely
- [ ] Have IT support on standby
- [ ] Create standardized setup checklist for students
- [ ] Have phone/email for troubleshooting

---

## 📋 MINIMAL TROUBLESHOOTING GUIDE

Create `TROUBLESHOOTING.txt`:

```
PROBLEM: Backend won't start
SOLUTION: Check port 8000 not in use
  Windows: netstat -ano | findstr :8000
  Mac/Linux: lsof -i :8000

PROBLEM: Frontend won't connect
SOLUTION: Make sure backend is running first
  Should see "Backend running on http://0.0.0.0:8000"

PROBLEM: Port already in use
SOLUTION: 
  Option 1: Close other app using that port
  Option 2: Change port in START script
    Windows: Change 8501 to 8502
    In start script

PROBLEM: Python not found
SOLUTION: Install Python 3.9+ from python.org

PROBLEM: Still doesn't work?
SOLUTION: Email screenshot of error to: [your email]
```

---

## FINAL SETUP TIME ESTIMATES:

| Method | Prep Time | Per-Machine Setup | Total for 20 |
|--------|-----------|------------------|--------------|
| Option 1 (Exe) | 45 min | 5 min | ~2 hours |
| Option 2 (Scripts) | 2 hours | 10 min | ~3.5 hours |
| Option 3 (Docker) | 3-4 hours | 2 min | ~1.5 hours |

**My Recommendation: Option 1 (Executables) for tomorrow!**
- Prep tonight: 1-2 hours
- Deploy tomorrow: 30 minutes setup, 5 min per machine = ~100 min total
- Students can't break it
- Works offline

**Backup Plan: Option 2 (Scripts) if executables don't work**

Let me know which option you want to implement and I'll help you create all the scripts right now!
