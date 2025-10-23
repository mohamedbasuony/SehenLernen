import io
from pathlib import Path
from PIL import Image
import streamlit as st
from utils import api_client

APP_ROOT = Path(__file__).resolve().parent.parent


def _get_logo_path():
    """Find the logo file in the assets directory."""
    assets_dir = APP_ROOT / "assets"
    for filename in ("Logo.png", "logo.png"):
        candidate = assets_dir / filename
        if candidate.exists():
            return candidate
    return None


def render_sidebar():
    with st.sidebar:
        # Logo at top
        logo_path = _get_logo_path()
        if logo_path:
            st.image(str(logo_path), width=80)
        else:
            st.markdown("### Sehen Lernen")

        st.markdown("---")

        # Navigation removed per request – uploads only in sidebar

        # Image Upload Section
        st.subheader("Upload Images")

        # Initialize session state
        st.session_state.setdefault("images", [])
        st.session_state.setdefault("uploaded_image_ids", [])

        # Upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["Upload Images", "Upload ZIP", "Extract from CSV"],
            key="sidebar_upload_method",
        )

        if upload_method == "Upload Images":
            # Versioned key so we can clear the uploader after removing all images
            st.session_state.setdefault("sidebar_uploader_version", 0)
            uploader_key = f"sidebar_upload_images_{st.session_state['sidebar_uploader_version']}"

            image_files = st.file_uploader(
                "Choose image files",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=uploader_key,
                help="Files upload automatically when selected",
            )

            # Auto-upload on selection change
            sig = tuple(sorted([f.name for f in image_files])) if image_files else None
            last_sig = st.session_state.get("sidebar_last_image_sig")
            if image_files and sig != last_sig:
                with st.spinner("Uploading images..."):
                    try:
                        image_ids = api_client.upload_images(image_files, None)
                        st.session_state["uploaded_image_ids"] = image_ids
                        # Fetch images for preview from backend (preferred)
                        try:
                            images = api_client.get_all_images()
                            st.session_state["images"] = images
                        except Exception:
                            # Fallback to local previews
                            previews = []
                            for f in image_files:
                                try:
                                    previews.append(Image.open(io.BytesIO(f.getbuffer())))
                                except Exception:
                                    pass
                            if previews:
                                st.session_state["images"] = previews
                        st.session_state["sidebar_last_image_sig"] = sig
                        st.success(f"✅ Uploaded {len(image_ids)} images")
                        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        elif upload_method == "Upload ZIP":
            zip_file = st.file_uploader(
                "Choose a ZIP file",
                type=["zip"],
                key="sidebar_upload_zip",
            )

            if zip_file:
                if st.button("Upload ZIP", key="sidebar_btn_upload_zip", width='stretch'):
                    with st.spinner("Uploading ZIP..."):
                        try:
                            image_ids = api_client.upload_images(None, zip_file)
                            st.session_state["uploaded_image_ids"] = image_ids

                            # Fetch images for preview
                            images = api_client.get_all_images()
                            st.session_state["images"] = images

                            st.success(f"✅ Uploaded {len(image_ids)} images from ZIP")
                            (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")

        else:  # Extract from CSV
            csv_file = st.file_uploader(
                "Upload CSV with image URLs",
                type=["csv"],
                key="sidebar_csv_file",
                help=(
                    "CSV should contain image URLs in a column named 'url', 'image_url', 'link', "
                    "or the first column"
                ),
            )

            if csv_file:
                if st.button("Extract Images", key="sidebar_btn_extract_csv", width='stretch'):
                    with st.spinner("Extracting images from URLs..."):
                        try:
                            zip_bytes, image_ids, errors = api_client.extract_images_from_csv(csv_file)

                            if zip_bytes:
                                st.session_state["uploaded_image_ids"] = image_ids
                                st.session_state["extractor_zip"] = zip_bytes

                                # Fetch images for preview
                                images = api_client.get_all_images()
                                st.session_state["images"] = images

                                st.success(f"✅ Extracted {len(image_ids)} images")
                                if errors:
                                    st.warning(f"⚠️ {len(errors)} errors occurred")
                                    with st.expander("View errors"):
                                        for err in errors:
                                            st.text(f"• {err}")
                                (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                            else:
                                st.error("No images extracted")
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")

        st.markdown("---")

        # Helpers: re-upload current images after removals
        def _reupload_current_images():
            images = st.session_state.get("images", [])
            try:
                if not images:
                    api_client.clear_all_backend_images()
                    st.session_state["uploaded_image_ids"] = []
                    return
                class FileWrapper:
                    def __init__(self, name: str, data: bytes):
                        self.name = name
                        self.data = data
                        self.type = "image/png"
                    def getvalue(self) -> bytes:
                        return self.data
                files = []
                for idx, im in enumerate(images):
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    files.append(FileWrapper(f"image_{idx+1}.png", buf.getvalue()))
                try:
                    api_client.clear_all_backend_images()
                except Exception:
                    pass
                new_ids = api_client.upload_images(files, None)
                st.session_state["uploaded_image_ids"] = new_ids
            except Exception as exc:
                st.error(f"Failed to resync images after removal: {exc}")

        # Display uploaded images with per-item remove buttons
        if st.session_state.get("images"):
            st.subheader(f"Images ({len(st.session_state['images'])})")
            for i, img in enumerate(list(st.session_state["images"])):
                with st.container():
                    top = st.columns([6, 1])
                    with top[0]:
                        st.image(img, width='stretch')
                    with top[1]:
                        if st.button("✖", key=f"sidebar_remove_{i}", help="Remove this image"):
                            try:
                                # Remove locally
                                del st.session_state["images"][i]
                                # Reset the file_uploader selection so the "Browse files" list reflects removal
                                st.session_state["sidebar_uploader_version"] = st.session_state.get("sidebar_uploader_version", 0) + 1
                                st.session_state["sidebar_last_image_sig"] = None
                                # Resync remaining images to backend (or clear if none left)
                                _reupload_current_images()
                                # Refresh UI
                                (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()
                            except Exception as exc:
                                st.error(f"Failed to remove image: {exc}")
        else:
            st.info("No images uploaded yet. Use the uploaders above to begin.")
