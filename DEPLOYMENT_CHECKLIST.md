# DEPLOYMENT DAY CHECKLIST

## 🎯 Pre-Deployment (Tonight/Before Lab)

### Code & Environment Setup
- [ ] All recent code changes committed/tested
- [ ] Both services tested locally and working
- [ ] No API errors or warnings
- [ ] Database/storage folders accessible
- [ ] Requirements.txt files updated and frozen

### Documentation
- [ ] QUICK_START.md created and clear
- [ ] TROUBLESHOOTING.md complete
- [ ] DEPLOYMENT_GUIDE.md reviewed
- [ ] Setup scripts created and tested:
  - [ ] SETUP_WINDOWS.bat
  - [ ] START_APP.bat
  - [ ] setup_mac.sh
  - [ ] start_app.sh

### Pre-Testing
- [ ] Tested SETUP_WINDOWS.bat on Windows machine
- [ ] Tested START_APP.bat on Windows machine  
- [ ] Tested setup_mac.sh on Mac machine
- [ ] Tested start_app.sh on Mac machine
- [ ] Verified both backend and frontend launch
- [ ] Tested at least 3 main features
- [ ] Verified downloads work
- [ ] Checked error handling on bad input

### Distribution Preparation
- [ ] Create deployment package:
  - [ ] All .bat and .sh files
  - [ ] Complete Backend/ folder
  - [ ] Complete Fronted/ folder
  - [ ] QUICK_START.md
  - [ ] TROUBLESHOOTING.md
  - [ ] All requirements.txt files
- [ ] Test ZIP file integrity (extract and verify)
- [ ] Copy to USB drives or upload to file share
- [ ] Label USB drives clearly: "Sehen Lernen - SETUP"
- [ ] Create 20 physical copies OR ensure network access

### Contingency Planning
- [ ] Have backup USB drive
- [ ] Have backup download link
- [ ] Know how to recover if USB corrupted
- [ ] Have Python 3.11 installers on USB (if needed)
- [ ] Have contact info for IT support
- [ ] Have screenshot of "successful setup" for reference

---

## 🏫 Lab Day (Morning Setup - 30-60 minutes before students arrive)

### Venue Setup
- [ ] Test internet/network connectivity
- [ ] Test power outlets at each machine
- [ ] Check projector/screen works for demo
- [ ] Verify 20 machines can be accessed
- [ ] Check student machine specs:
  - [ ] Windows/Mac/Linux OS
  - [ ] Python 3.9+ available (or installers)
  - [ ] At least 2GB RAM free
  - [ ] At least 5GB disk space free

### First Machine - Full Test
- [ ] Pick one machine as "template"
- [ ] Run SETUP_WINDOWS.bat / setup_mac.sh completely
- [ ] Wait for installation (should take 5-10 min)
- [ ] Run START_APP.bat / start_app.sh
- [ ] Verify both services start
- [ ] Test in browser:
  - [ ] Frontend loads: http://localhost:8501
  - [ ] Can upload test image
  - [ ] Can select features
  - [ ] Can download results
  - [ ] No errors in console

### Troubleshoot If Issues
- [ ] Check TROUBLESHOOTING.md
- [ ] Try alternative deployment method
- [ ] Check Python is in PATH
- [ ] Verify disk space / RAM
- [ ] Have IT support test on their system

### Go-Live Plan (if first machine works)
- [ ] Option 1 - USB Distribution: Give USB to each student
- [ ] Option 2 - Network Copy: Have students download from shared folder
- [ ] Option 3 - Manual: Help each student setup individually

---

## 👥 Student Lab Time (Deployment & Training)

### Greeting & Setup (10 minutes)
- [ ] Explain what the app does (brief overview)
- [ ] Show example of successful run on projector
- [ ] Hand out USB or provide download link
- [ ] Show where QUICK_START.md is

### Student Deployment (15-20 minutes)
- [ ] Each student extracts files to their Desktop/Documents
- [ ] Each student runs SETUP_WINDOWS.bat or setup_mac.sh
- [ ] Circulate and help with setup
- [ ] Once complete, each student runs START_APP.bat / start_app.sh
- [ ] Verify each machine opens application
- [ ] Confirm all 20 machines have working app

### Training & First Run (10-15 minutes)
- [ ] Show how to upload image
- [ ] Show how to select features
- [ ] Show how to download results
- [ ] Let students do one complete run
- [ ] Answer questions about each feature

### Supervised Practice (remainder of time)
- [ ] Students work through tasks
- [ ] Monitor for errors
- [ ] Use TROUBLESHOOTING.md for issues
- [ ] Document any problems for future

---

## 🚨 Emergency Procedures

### If Setup Fails on Multiple Machines

**Immediate actions:**
- [ ] Stop setup, don't waste time on individual machines
- [ ] Ask: Do you have Python installed?
- [ ] If no: Provide Python installers from USB
- [ ] If yes: Try deployment method 2 (manual commands)

**Alternative 1 - Central Server:**
- [ ] Pick one powerful machine as "server"
- [ ] Run backend and frontend on that machine
- [ ] All students access via browser: `http://server-ip:8501`
- [ ] Setup: 5 minutes
- [ ] Deployment: 1 minute per student

**Alternative 2 - Docker (if available):**
- [ ] Use docker-compose.yml from DEPLOYMENT_GUIDE.md
- [ ] Students who have Docker: `docker-compose up`
- [ ] All others: Use alternative 1 (central server)

**Alternative 3 - Web Version (if prepared):**
- [ ] Deploy to cloud (AWS, Heroku, DigitalOcean)
- [ ] Students access online version
- [ ] Requires 1-2 hours setup beforehand

### If Backend/Frontend Won't Connect

**Symptoms:** Frontend page keeps refreshing / shows error

**Fix:**
1. [ ] Close all windows
2. [ ] Wait 5 seconds
3. [ ] Re-run START_APP.bat / start_app.sh
4. [ ] Wait 20 seconds for startup
5. [ ] Browser should open automatically

### If Stuck on One Machine

**Time check:**
- After 15 minutes on one machine → Move to alternative method
- Can always come back to fix that machine after lab

**Suggested triage:**
- Students on working machines: Continue with lab
- One staff member: Help with problem machine(s)
- Use alternative deployment method for that machine

---

## 📋 Materials to Have Ready

### Physical Items
- [ ] 20 USB drives (labeled "Sehen Lernen - SETUP")
- [ ] Printed copies of QUICK_START.md (20 copies)
- [ ] Printed copies of TROUBLESHOOTING.md (5 copies)
- [ ] Power cords/adapters as needed
- [ ] Ethernet cables (if needed)

### Digital Backup
- [ ] ZIP file of entire deployment package
- [ ] Uploaded to cloud storage with download link
- [ ] Email link to all TAs/instructors
- [ ] Have on USB as backup
- [ ] Have on instructor laptop

### Documentation
- [ ] Printed DEPLOYMENT_GUIDE.md (for reference)
- [ ] Screenshots of successful setup
- [ ] Known issues and workarounds list
- [ ] Contact info for IT support

---

## ✅ Success Criteria

### Minimum Success: End of Day
- [ ] At least 18/20 machines have working app
- [ ] Students can upload images
- [ ] Students can run features
- [ ] Students can download results
- [ ] Students can work independently

### Ideal Success: End of First Day
- [ ] All 20/20 machines working
- [ ] All students completed at least one feature
- [ ] No critical bugs discovered
- [ ] Students understand basic workflow
- [ ] Feedback collected for improvements

---

## 📝 Post-Deployment

### Immediately After Lab
- [ ] Document any setup issues encountered
- [ ] Note which machines had problems
- [ ] Collect error messages/screenshots
- [ ] Ask students for feedback
- [ ] Update TROUBLESHOOTING.md with new solutions

### Before Next Lab Session
- [ ] Fix any bugs discovered
- [ ] Update setup scripts if needed
- [ ] Create summary of common issues
- [ ] Prepare improved QUICK_START for next group
- [ ] Test updated setup on fresh machine

### For Future Deployments
- [ ] Keep deployment package version-controlled
- [ ] Document all issues and solutions
- [ ] Create script to automate future setups
- [ ] Build in 15% extra time buffer for unknowns
- [ ] Have named "tech support" person on standby

---

## 🎓 Lab Session Tips

### Before You Start
- Have the TROUBLESHOOTING.md open on your laptop
- Have Python installer on USB just in case
- Have backup internet/network tested
- Brief IT staff on deployment approach
- Have contingency plan explained to students upfront

### During Setup
- Keep energy positive even if some fail
- Normalize setup problems ("this is normal")
- Help struggling students first
- Have them observe while you fix their machine
- Teaching opportunity about installations

### During Lab Work
- Walk around and observe
- Don't wait for students to ask for help
- Note which features cause issues
- Encourage experimentation
- Collect screenshots of good results

### Communication
- Clear timeline: "Setup 20 min, Demo 10 min, Work 30 min"
- Realistic expectations: "First run takes longer due to model loading"
- Troubleshooting: "If stuck, try X, Y, Z in that order"

---

## 📞 Key Contact Info

### Prepare These
- [ ] Your phone number
- [ ] IT support contact
- [ ] Lab administrator contact
- [ ] Python error help: [your support email]
- [ ] Backup person if you unavailable

### Leave Behind
- [ ] Email address for future questions
- [ ] How to report bugs/issues
- [ ] Where to get latest version if updated
- [ ] How to get help remotely

---

## Summary Timeline for Tomorrow

```
T-0:30    - Arrive early, test first machine
T-0:15    - Brief IT staff, have materials ready
T+0:00    - Students arrive, brief overview (5 min)
T+0:05    - Distribution & startup (15-20 min)
T+0:25    - Training & first run (15 min)
T+0:40    - Supervised practice (40+ min)
T+1:20    - Lab ends, collect feedback
T+1:30    - Debrief and document issues
```

**Total preparation time: ~2-3 hours**
**Total lab time: ~90 minutes**
**Actual deployment: 5-10 seconds per machine (after setup)**

---

**Version**: 1.0  
**Status**: Ready for Use  
**Created**: $(date)  
**For**: Sehen Lernen 20-Machine Lab Deployment

Good luck! 🚀
