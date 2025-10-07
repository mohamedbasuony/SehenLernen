# Alignment Error Fix - Round 2

## Changes Made

### 1. Enhanced Debug Logging in Sync Function
**File:** `Fronted/components/feature_selection.py`
**Function:** `_ensure_backend_sync()`

Added detailed logging at every step:
- Before clearing backend
- After clearing backend
- Number of files prepared for upload
- Backend response with IDs
- Session state update confirmation
- Final verification step

This will help identify exactly where the sync process is failing.

### 2. Triple-Verification in Classifier Training
**File:** `Fronted/components/feature_selection.py`
**Section:** Classifier Training Tab

Now checks THREE separate counts:
1. **Frontend images** - PIL Image objects in `st.session_state["images"]`
2. **Session IDs** - IDs stored in `st.session_state["uploaded_image_ids"]`  
3. **Backend IDs** - Actual IDs from backend API via `get_current_image_ids()`

This ensures we catch mismatches at any level of the sync process.

### 3. Force Re-upload Button
**File:** `Fronted/components/feature_selection.py`

Added a second button alongside "Refresh training dataset":
- **"Force re-upload images"** - Triggers `_ensure_backend_sync(force_reupload=True)` to forcefully clear backend and re-upload all images

## Debugging Steps

### Step 1: Check Terminal Output
Look at the **backend terminal** for any errors during upload. You should see:
```
INFO: ... POST /upload/images ... 200 OK
```

If you see 500 or other errors, the backend upload is failing.

### Step 2: Check Browser Console
Open browser DevTools (F12) and check the Console tab for any JavaScript errors or failed API calls.

### Step 3: Check Streamlit Logs
In the terminal where you're running the frontend, look for logging output:
```
Sync check: 3 frontend images, 1 backend IDs, force=False
Re-uploading: force=False, mismatch=True
Clearing backend storage...
Backend cleared successfully
Prepared 3 files for upload
Backend returned 3 IDs: ['image_1.png', 'image_2.png', 'image_3.png']
Updated session state with 3 IDs
Sync successful - all images uploaded and verified
```

If you don't see this output, logging might not be configured. Add this to your `Fronted/app.py`:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Step 4: Use the Force Re-upload Button
If automatic sync fails:
1. Click **"Force re-upload images"** button
2. Wait for the spinner
3. Check the debug caption that shows the counts
4. If still mismatched, check the backend terminal for errors

## Expected Behavior

### Before Sync
```
🔍 Debug: 3 images in session state, 1 image IDs from backend
```

### After Successful Sync
```
🔍 After sync: 3 images, 3 IDs in session, 3 actual backend IDs, sync_ok=True
```

## Common Issues & Solutions

### Issue 1: Backend Returns Wrong Number of IDs
**Symptom:** "Backend returned 1 IDs" in logs when 3 were uploaded
**Cause:** Backend upload endpoint might be failing silently or only processing first image
**Solution:** Check backend logs for exceptions during POST /upload/images

### Issue 2: Session State Not Updating
**Symptom:** sync_ok=True but session still has old IDs
**Cause:** Streamlit session state timing issue
**Solution:** Use "Force re-upload" button and then refresh page

### Issue 3: Backend Storage Not Cleared
**Symptom:** Old images remain in backend storage directory
**Cause:** `clear_all_backend_images()` might be failing
**Solution:** Check backend logs for DELETE /upload/clear-all-images errors

## Testing the Fix

1. **Clear everything:**
   - Click "Refresh training dataset" button
   - Go to Data Input tab
   - Upload 3 fresh images (or use Extractor)

2. **Go to Classifier Training tab:**
   - Should see: "3 images, 3 IDs, 3 backend IDs"
   - Should see 3 rows in the training table
   - No error message

3. **If still failing:**
   - Click "Force re-upload images"
   - Check all three debug outputs
   - Share the backend terminal output and Streamlit logs

## Next Steps if This Doesn't Work

If the issue persists even with detailed logging:

1. **Check image format:** Make sure all images are valid PNG/JPG
2. **Check file permissions:** Backend might not have write access to storage folder
3. **Check disk space:** Storage folder might be full
4. **Restart backend:** Kill and restart uvicorn
5. **Clear storage manually:** Delete all files in `Backend/storage/images/` and try again

## Files Modified
- `Fronted/components/feature_selection.py` - Enhanced sync function and classifier training verification
