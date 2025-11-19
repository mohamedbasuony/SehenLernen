# SEHEN LERNEN - TROUBLESHOOTING GUIDE

## 🔴 Critical Issues

### Issue: Application won't start at all

#### Symptom:
- Command Prompt/Terminal closes immediately after opening
- Error messages appear and disappear too fast to read

#### Solutions:

**Windows - Slow down the process:**
1. Instead of double-clicking START_APP.bat
2. Right-click → Edit
3. Add this line at the end:
   ```
   pause
   ```
4. Save and try again
5. Now errors will pause for you to read

**Mac/Linux:**
```
bash start_app.sh 2>&1 | tee output.log
# This saves all errors to output.log file
```

---

## 🟠 Setup Issues

### Issue: "Python not found" during SETUP

#### Windows:

**Check if Python is installed:**
```
python --version
```

**If not found:**
1. Download Python from: https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" checkbox during install
3. Restart computer
4. Run SETUP_WINDOWS.bat again

**If "python" command doesn't work but "python3" does:**
- Edit SETUP_WINDOWS.bat
- Change `python` to `python3`
- Save and try again

#### Mac/Linux:

**Check if Python is installed:**
```
python3 --version
```

**If not found:**
```
# macOS with Homebrew:
brew install python3

# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

# Then run:
bash setup_mac.sh
```

---

### Issue: "Permission denied" on Mac/Linux

#### Symptom:
```
bash: ./setup_mac.sh: Permission denied
```

#### Solution:
```
chmod +x setup_mac.sh
chmod +x start_app.sh
bash setup_mac.sh
```

---

### Issue: Setup takes 30+ minutes

#### Symptom:
- stuck on "Installing dependencies"
- no progress for long time

#### Causes:
- Slow internet connection
- Large package downloads
- Disk space full

#### Solutions:

**Check disk space:**
- Windows: Right-click Drive C: → Properties
- Mac: Apple Menu → About This Mac → Storage
- Linux: `df -h`
- Need at least 5GB free

**Check internet:**
- Windows: Run `ping google.com`
- Mac/Linux: `ping google.com`
- Should show responses (not "Host unreachable")

**Retry with verbose output:**
- Windows: Edit SETUP_WINDOWS.bat, remove ">nul 2>&1" to see pip output
- Mac/Linux: Edit setup_mac.sh, remove "2>&1 | grep -v"

---

## 🟡 Runtime Issues

### Issue: "Port 8000 or 8501 already in use"

#### Symptom:
```
ERROR: Address already in use
bind: Address already in use
```

#### Windows Solution:

**Find what's using the port:**
```
netstat -ano | findstr :8000
netstat -ano | findstr :8501
```

**Output will show:** 
```
TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    12345
```

**Kill the process:**
```
taskkill /PID 12345 /F
```

**Alternative: Use different ports**

Edit START_APP.bat:
```
Change: --port 8000
To: --port 8010

Change: --server.port 8501
To: --server.port 8511
```

#### Mac/Linux Solution:

**Find what's using the port:**
```
lsof -i :8000
lsof -i :8501
```

**Kill the process:**
```
kill -9 12345
```

**Or use different ports (edit start_app.sh):**
```
--port 8010
--server.port 8511
```

---

### Issue: Backend starts but Frontend won't connect

#### Symptom:
- Backend Command Prompt/Terminal shows: "Application startup complete"
- Frontend Command Prompt/Terminal shows: "Local URL: http://localhost:8501"
- But browser shows: "Connection refused" or "This site can't be reached"
- Or Frontend keeps reloading indefinitely

#### Causes:
- Frontend trying to connect to backend before it's ready
- Firewall blocking ports

#### Solutions:

**Wait longer:**
- Wait 15-20 seconds after START_APP.bat opens
- Frontend may take time to initialize

**Check backend is really running:**
- Go to: http://localhost:8000/docs
- Should show API documentation (swagger UI)
- If shows "connection refused" → backend not running

**Fix firewall (Windows):**
1. Windows Defender Firewall → Allow app through firewall
2. Find "Python" in the list
3. Make sure it's checked for both Private and Public

**Manual frontend start:**
```
# Open new Terminal/Command Prompt
# Windows:
cd Fronted
streamlit run app.py --server.port 8501 --logger.level=debug

# Mac/Linux:
cd Fronted
streamlit run app.py --server.port 8501
```

---

### Issue: Frontend keeps reloading / showing error box

#### Symptom:
- Page loads for 1 second then refreshes
- Red error box at top of page
- Message like: "Connection Error: Failed to connect to backend"

#### Causes:
- Backend stopped or crashed
- Network/firewall issue
- Frontend URL doesn't match backend

#### Solutions:

**Check backend is running:**
1. Look at Backend Command Prompt/Terminal
2. Should show API logs (info about requests)
3. If empty or shows errors → restart backend

**Restart services:**
1. Close all Command Prompt/Terminal windows
2. Close browser
3. Run START_APP.bat / start_app.sh again
4. Wait 20 seconds
5. Browser opens automatically

**Check network communication:**
- Windows Command Prompt:
  ```
  curl http://localhost:8000/docs
  ```
- Mac/Linux Terminal:
  ```
  curl http://localhost:8000/docs
  ```
- Should show HTML (swagger docs page)

---

### Issue: "Module not found" or "ImportError"

#### Symptom:
```
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'streamlit'
```

#### Causes:
- Dependencies not installed
- Virtual environment not activated
- Using wrong Python version

#### Solutions:

**Reinstall dependencies:**
```
# Windows:
SETUP_WINDOWS.bat

# Mac/Linux:
bash setup_mac.sh
```

**Manually install missing module:**
```
# Windows:
venv\Scripts\activate.bat
pip install opencv-python

# Mac/Linux:
source venv/bin/activate
pip install opencv-python
```

**Verify virtual environment:**
```
# Windows:
where python

# Mac/Linux:
which python
```

Should show path like: `/path/to/project/venv/bin/python`

If shows `/usr/bin/python` → venv not activated!

**Activate it:**
```
# Windows:
venv\Scripts\activate.bat

# Mac/Linux:
source venv/bin/activate
```

---

## 🔵 Feature-Specific Issues

### Issue: Image upload fails / "Invalid file"

#### Symptom:
- Upload button does nothing
- Shows: "Invalid file format" or "File too large"

#### Solutions:

**Supported formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

**Size limits:**
- Max 50 MB per image
- Recommended: 1-20 MB for fast processing

**Try:**
1. Re-compress image (reduce resolution)
2. Use JPEG format instead of PNG
3. Try a different image file
4. Check image is not corrupted

---

### Issue: Feature extraction is very slow

#### Symptom:
- Upload image
- Select feature
- Stuck for 2-5 minutes
- Still processing

#### Expected behavior:
- Simple features: 5-10 seconds
- Complex features: 30-60 seconds
- Very complex (DeepLearn, Segmentation): 1-5 minutes
- First run of any feature loads models: +1-2 minutes

#### Solutions:

**Check system:**
- Is CPU usage at 100%? (normal for processing)
- Do you have enough RAM? (need 4GB+ for deep learning)
- Check disk space (need 1GB+ free)

**Try smaller image:**
- Resize image to 500x500 or smaller
- Processing time is proportional to image size

**Check task manager (Windows) or Activity Monitor (Mac):**
- Open while processing
- Should see high CPU usage
- This is normal

**Wait it out on first run:**
- First run of deep learning features downloads models (~500MB)
- May take 5-10 minutes to download
- Subsequent runs use cached models (much faster)

---

### Issue: Results don't save / download button doesn't work

#### Symptom:
- Click "Download Results"
- Nothing happens
- Or file downloads but is empty/corrupted

#### Solutions:

**Check browser downloads folder:**
- Some browsers default to ask where to save
- Files might be in ~/Downloads/

**Try different file format:**
- If "Download as CSV" fails → try "Download as Image"
- Try different browser (Chrome, Firefox, Edge, Safari)

**Check file permissions:**
- Make sure you have write permission to Downloads folder

**Manual file location:**
- Results also stored in: `Backend/storage/images/`
- Can manually copy files from there

---

## 🟢 Performance Issues

### Issue: Everything is very slow / freezing

#### Symptom:
- UI takes 10+ seconds to respond to clicks
- Results take 5+ minutes
- Computer very hot/fan loud

#### Causes:
- Low RAM available
- High CPU usage from other programs
- Disk is full or fragmented
- Using old/slow computer

#### Solutions:

**Close other programs:**
- Close browsers, media players, etc.
- Free up 2GB+ of RAM

**Check hardware:**
- Computer specs should be:
  - Processor: Intel i5/AMD Ryzen 5 or better
  - RAM: 8GB minimum, 16GB+ recommended
  - Disk: SSD preferred, at least 10GB free

**Restart computer:**
- Often fixes slow performance
- Clears memory cache

**Use smaller images:**
- 500x500 pixels process much faster than 4000x4000

---

## 🔴 Crash/Hang Issues

### Issue: Backend crashes with stack trace

#### Symptom:
```
Traceback (most recent call last):
  File "..."
RuntimeError: ...
```

Backend Command Prompt/Terminal shows error and stops

#### Short-term solution:
1. Restart backend: Close and run START_APP.bat again
2. Try different settings/image
3. If same error: Note the exact error message

#### Get detailed info:
- Screenshot the full error
- Open new Terminal/Command Prompt
- Try to reproduce the error
- Email screenshot to support

---

### Issue: Frontend crashes / "StreamlitAPIException"

#### Symptom:
```
StreamlitAPIException: Cannot add same element twice
```

Or page shows red error box with stack trace

#### Solution:
1. Close browser completely
2. Wait 2 seconds
3. Re-open browser
4. Go to: http://localhost:8501
5. Refresh page (Ctrl+R or Cmd+R)

---

### Issue: Application hangs (doesn't respond)

#### Symptom:
- Click button → nothing happens
- UI freezes for 5+ minutes
- No spinner, no error

#### Solution:

**Kill and restart:**

Windows:
```
# Open Command Prompt (Ctrl+Shift+Esc)
taskkill /F /IM python.exe
# Then re-run START_APP.bat
```

Mac/Linux:
```
# Press Ctrl+C in Terminal multiple times
# Or: killall python
# Then re-run: bash start_app.sh
```

---

## 🚫 Network/Network-Based Issues

### Issue: Accessing from another computer fails

#### Symptom:
- Same computer works: http://localhost:8501 ✓
- Other computer fails: http://192.168.x.x:8501 ✗

#### Causes:
- Firewall blocking port 8501
- Backend not accessible from network
- Wrong IP address

#### Solutions:

**Find your IP address:**
```
# Windows:
ipconfig
# Look for: IPv4 Address: 192.168.x.x or 10.0.x.x

# Mac/Linux:
ifconfig
# Look for: inet 192.168.x.x or inet 10.0.x.x
```

**Test from other computer:**
```
# Other computer Command Prompt/Terminal:
ping 192.168.x.x
# Should show responses

# If no response: Computer unreachable (network problem)
# If response: Network works, firewall might be blocking
```

**Open firewall (Windows):**
1. Settings → Privacy & Security → Windows Firewall
2. Click: "Allow app through firewall"
3. Find "Python"
4. Make sure it's checked for both Private and Public networks
5. Restart app

**Firewall (Mac):**
1. System Preferences → Security & Privacy → Firewall Options
2. Click: "Add exceptions"
3. Find Python application
4. Click: Allow

---

## 📊 Checking What's Wrong

### Quick Diagnostic Checklist:

- [ ] Computer has at least 4GB RAM free
- [ ] Computer is connected to internet (for first setup)
- [ ] Python version is 3.9 or higher: `python --version`
- [ ] Virtual environment exists: `venv` folder in project
- [ ] Both Command Prompt/Terminal windows are open
- [ ] Backend shows: "Application startup complete"
- [ ] Frontend shows: "Local URL: http://localhost:8501"
- [ ] Can reach backend: http://localhost:8000/docs works
- [ ] Can reach frontend: http://localhost:8501 opens (no error)

If all checked ✓ → Application should work

If any ✗ → See relevant section above

---

## 🆘 Still Not Working?

**Gather Information:**

1. **What command did you run?**
   - SETUP_WINDOWS.bat / setup_mac.sh
   - START_APP.bat / start_app.sh
   - Manual command

2. **What do you see?**
   - Error message? (copy-paste exact text)
   - Command Prompt/Terminal output? (screenshot)
   - Browser message? (screenshot)

3. **Your system:**
   - Operating system and version
   - Python version: `python --version`
   - How much RAM/disk space free

4. **When does it fail?**
   - During setup? → During installation of which dependency?
   - During startup? → Which service (backend/frontend)?
   - During first use? → What are you trying to do?
   - Randomly after working? → When/how often?

**Send this information to support with:**
- This checklist completed
- Screenshots of error messages
- Output from Terminal/Command Prompt (copy-paste or screenshot)
- Exact steps you took

---

## 📚 Learning Resources

**If you want to understand what's happening:**

- What is a virtual environment? → https://docs.python.org/3/tutorial/venv.html
- What is a port? → https://www.cloudflare.com/learning/network-layer/port/
- What is localhost? → https://en.wikipedia.org/wiki/Localhost
- How does the frontend-backend communication work? → See Backend/app/routers/ files

---

**Version**: 1.0
**Last Updated**: $(date)
**For**: Sehen Lernen Lab Application
