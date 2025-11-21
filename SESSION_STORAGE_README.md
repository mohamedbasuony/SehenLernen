# Session-Based Image Storage

## Overview

The application now uses **per-session in-memory image storage** instead of shared persistent disk storage. This means:

- **Isolated uploads**: Each user's uploaded images are stored in memory with a unique session ID and are not visible to other users.
- **No shared state**: Images are cached in the backend's memory (dictionary keyed by session ID) and cleared when the backend restarts or the session is explicitly cleared.
- **Privacy**: No images are written to shared disk storage by default, preventing cross-user data leakage.

## How It Works

1. **Frontend (Streamlit)**:
   - On app load, `app.py` initializes a unique `session_id` (UUID) in `st.session_state`.
   - All API calls in `utils/api_client.py` automatically append `session_id` as a query parameter.

2. **Backend (FastAPI)**:
   - Endpoints in `routers/data_input.py` and `routers/features.py` accept an optional `session_id` query param.
   - `services/data_service.py` uses an in-memory dict `SESSION_IMAGES` to store images per session.
   - A context variable `CURRENT_SESSION_ID` allows feature services to retrieve session-scoped images without modifying function signatures.

3. **Persistence**:
   - Images are stored **only in RAM** (per session).
   - When the backend process restarts, all uploaded images are lost.
   - For long-term storage, you can modify `data_service.py` to write images to disk (opt-in behavior).

## Enabling Persistent Storage (Optional)

If you need to persist images across backend restarts:

1. **Remove or conditionally skip session-based logic**:
   - In `data_service.py`, modify `save_uploaded_images` and `save_and_extract_zip` to always write to disk (ignore `session_id` parameter or set a global flag).

2. **Separate storage per user (on disk)**:
   - Create subdirectories under `storage/images/<session_id>/` for each session.
   - Update `load_image` and `get_all_image_ids` to check disk paths keyed by session ID.

3. **Database-backed storage**:
   - Store image metadata (filename, upload timestamp, session_id) in a database.
   - Store binary data in a blob store (S3, Azure Blob, etc.) or filesystem.

## Testing Session Isolation

1. **Start the backend**:
   ```bash
   cd Backend
   uvicorn app.main:app --reload --port 8000
   ```

2. **Start the frontend**:
   ```bash
   cd Fronted
   streamlit run app.py --server.port 8501
   ```

3. **Simulate multiple users**:
   - Open the app in two different browser windows (or one normal + one incognito).
   - Upload images in each window.
   - Verify that images uploaded in one window **do not appear** in the other.

## Troubleshooting

- **Images disappear after backend restart**: This is expected behavior. Images are stored in memory and cleared on restart.
- **Session ID not being sent**: Check browser console for errors. Ensure `st.session_state["session_id"]` is initialized in `app.py`.
- **Old persistent images still visible**: Clear the `Backend/storage/images/` directory if you had previous uploads saved to disk.

## Migration Notes

- **Before this change**: All images were written to `Backend/storage/images/` and visible to all users.
- **After this change**: Images are stored per-session in memory (`SESSION_IMAGES` dict in `data_service.py`).
- **Fallback**: If `session_id` is not provided (e.g., direct API calls without frontend), the backend falls back to persistent disk storage (legacy behavior).
