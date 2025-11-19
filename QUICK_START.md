# SEHEN LERNEN - QUICK START GUIDE

## ⚡ 30-Second Setup

### Windows:
```
1. Double-click: SETUP_WINDOWS.bat
2. Wait 5-10 minutes for installation
3. When done, double-click: START_APP.bat
4. Browser opens to http://localhost:8501
5. Done! ✓
```

### Mac/Linux:
```
1. Open Terminal
2. cd to this folder
3. bash setup_mac.sh
4. bash start_app.sh
5. Browser opens to http://localhost:8501
6. Done! ✓
```

---

## 📋 File Descriptions

| File | Purpose | Usage |
|------|---------|-------|
| `SETUP_WINDOWS.bat` | Initial setup (Windows) | Run ONCE before first use |
| `START_APP.bat` | Launch application (Windows) | Run every time to start |
| `setup_mac.sh` | Initial setup (Mac/Linux) | Run ONCE before first use |
| `start_app.sh` | Launch application (Mac/Linux) | Run every time to start |
| `DEPLOYMENT_GUIDE.md` | Full deployment options | Read for advanced setup |
| `TROUBLESHOOTING.md` | Common problems & fixes | Read if something goes wrong |

---

## 🖥️ System Requirements

- **Windows**: Windows 7 or later with Python 3.9+
- **Mac**: macOS 10.14 or later with Python 3.9+
- **Linux**: Ubuntu 18.04+ or similar, with Python 3.9+

### Check Python:
```
Windows: python --version
Mac/Linux: python3 --version
```

If Python not found, install from: https://www.python.org/downloads/

---

## 🚀 Starting for First Time

### Windows:

**Step 1: Setup** (only once)
```
1. Right-click on desktop
2. Select: New > Folder
3. Name it: SehenLernen
4. Extract all deployment files into this folder
5. Double-click: SETUP_WINDOWS.bat
6. Wait 5-10 minutes
7. Window closes automatically when done
```

**Step 2: Launch** (every time)
```
1. Open the SehenLernen folder
2. Double-click: START_APP.bat
3. Wait 10 seconds
4. Browser opens automatically to http://localhost:8501
5. Application is ready to use!
```

### Mac/Linux:

**Step 1: Setup** (only once)
```
1. Open Terminal
2. cd ~/Downloads/SehenLernen  (or wherever you extracted files)
3. bash setup_mac.sh
4. Wait 5-10 minutes
5. Press Enter when prompted
```

**Step 2: Launch** (every time)
```
1. Open Terminal
2. cd ~/Downloads/SehenLernen
3. bash start_app.sh
4. Wait 10 seconds
5. Browser opens automatically to http://localhost:8501
6. Application is ready to use!
```

---

## 🌐 Accessing the Application

### From Same Computer:
- Open browser → Go to: `http://localhost:8501`

### From Another Computer (on same network):
- Find your computer's IP: 
  - Windows: `ipconfig` in Command Prompt
  - Mac/Linux: `ifconfig` in Terminal
  - Look for: `192.168.x.x` or `10.0.x.x`
- Other computer opens: `http://YOUR_IP:8501`
- Example: `http://192.168.1.100:8501`

---

## ⏹️ Stopping the Application

### Windows:
- Close both Command Prompt windows
- Or press: Ctrl+C in each window

### Mac/Linux:
- Close both Terminal windows
- Or press: Ctrl+C in each terminal

---

## 🔍 Checking Services Are Running

### Backend (should print lots of logs):
```
Expected output shows:
✓ Uvicorn running on http://0.0.0.0:8000
Application startup complete
```

### Frontend (should show):
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

## ⚠️ Common Issues

### 1. "Python not found"
- **Problem**: Python is not installed or not in PATH
- **Solution**: Install Python from https://www.python.org/downloads/
- **On Windows**: Check "Add Python to PATH" during installation
- **Then**: Run SETUP_WINDOWS.bat again

### 2. "Port 8000 or 8501 already in use"
- **Problem**: Another application is using these ports
- **Windows Solution**:
  ```
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```
- **Mac/Linux Solution**:
  ```
  lsof -i :8000
  kill -9 <PID>
  ```

### 3. "Module not found" errors
- **Problem**: Dependencies not installed
- **Solution**: 
  - Windows: Run SETUP_WINDOWS.bat again
  - Mac/Linux: Run bash setup_mac.sh again

### 4. Browser shows "Connection refused"
- **Problem**: Backend not started yet
- **Solution**:
  - Wait 10 seconds after opening START_APP.bat/start_app.sh
  - Check that two Command Prompt/Terminal windows opened
  - If not, try starting manually (see TROUBLESHOOTING.md)

### 5. Application is very slow
- **Problem**: First run needs to load all models (~1-2 minutes)
- **Solution**: Be patient on first run, subsequent runs are faster
- **Note**: Feature extraction may take 30-60 seconds depending on image size

---

## 📞 Getting Help

If problems persist:

1. **Check**: TROUBLESHOOTING.md file
2. **Check**: Screenshots match expected output
3. **Screenshot**: Error messages and which window shows the error
4. **Contact**: [Your email/support channel]

Provide:
- Your operating system (Windows 7/10/11, Mac version, Linux distro)
- Python version (from `python --version`)
- Exact error message (copy-paste from Terminal/Command Prompt)
- Screenshot of what you see

---

## ✅ Success Indicators

### Setup Complete:
- No error messages
- Window closes automatically
- SETUP_WINDOWS.bat shows: "[OK] Backend dependencies installed"
- Setup complete message shows

### Application Started:
- Both Command Prompt/Terminal windows open
- Backend window shows: "Application startup complete"
- Frontend window shows: "Local URL: http://localhost:8501"
- Browser automatically opens the application
- You can interact with the interface

### Application Working:
- Can upload images
- Can select features
- Results display without errors
- Can download results

---

## 🔐 Security Notes

- Application runs locally on your computer
- Data does NOT leave your machine
- No internet connection required after initial setup
- Safe to run on public networks (only accessible from your computer)

---

## 📊 Storage

- Images stored in: `Backend/storage/images/`
- Results stored temporarily during session
- Data cleared when application closes (configurable)

---

## 🔄 Updating the Application

If code is updated:

**Windows:**
- Close START_APP.bat windows
- Run SETUP_WINDOWS.bat again to update dependencies
- Run START_APP.bat

**Mac/Linux:**
- Close Terminal windows
- `bash setup_mac.sh` (it will update if needed)
- `bash start_app.sh`

---

## 📖 More Information

For advanced deployment options, network setup, or customization:
- See: `DEPLOYMENT_GUIDE.md`
- See: `TROUBLESHOOTING.md`

For development/modification:
- See: `Backend/README.md`
- See: `Fronted/README.md`

---

**Last Updated**: $(date)
**Version**: 1.0
**Status**: Ready for Production

Good luck with your lab sessions! 🎓
