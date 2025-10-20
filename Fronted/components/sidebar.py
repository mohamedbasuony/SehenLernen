import io
import streamlit as st
from pathlib import Path
from PIL import Image
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


# Sidebar navigation component with upload functionality
def render_sidebar():
    with st.sidebar:
        # Logo at top
        logo_path = _get_logo_path()
        if logo_path:
            st.image(str(logo_path), width=80)
        else:
            st.markdown("### Sehen Lernen")
        
        st.markdown("---")
        
        # Navigation buttons
        st.subheader("Navigation")
        if st.button("📤 Data Input", key="nav_data_input", use_container_width=True):
            st.session_state["active_section"] = "Data Input"
            st.rerun()
        if st.button("🔍 Feature Selection", key="nav_feature_selection", use_container_width=True):
            st.session_state["active_section"] = "Feature Selection"
            st.rerun()
        
        st.markdown("---")
        
        # Image Upload Section
        st.subheader("Upload Images")
        
        # Initialize session state
        st.session_state.setdefault("images", [])
        st.session_state.setdefault("uploaded_image_ids", [])
        
        # Upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["Upload Images", "Upload ZIP", "Extract from CSV"],
            key="sidebar_upload_method"
        )
        
        if upload_method == "Upload Images":
            image_files = st.file_uploader(
                "Choose image files",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="sidebar_upload_images"
            )
            
            if image_files:
                # Show live preview
                try:
                    st.session_state["images"] = [
                        Image.open(io.BytesIO(f.getbuffer())) for f in image_files
                    ]
                except Exception as e:
                    st.error(f"Failed to read images: {e}")
                    st.session_state["images"] = []
                
                if st.button("Upload", key="sidebar_btn_upload_images", use_container_width=True):
                    with st.spinner("Uploading images..."):
                        try:
                            image_ids = api_client.upload_images(image_files, None)
                            st.session_state["uploaded_image_ids"] = image_ids
                            st.success(f"✅ Uploaded {len(image_ids)} images")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
        
        elif upload_method == "Upload ZIP":
            zip_file = st.file_uploader(
                "Choose a ZIP file",
                type=["zip"],
                key="sidebar_upload_zip"
            )
            
            if zip_file:
                if st.button("Upload ZIP", key="sidebar_btn_upload_zip", use_container_width=True):
                    with st.spinner("Uploading ZIP..."):
                        try:
                            image_ids = api_client.upload_images(None, zip_file)
                            st.session_state["uploaded_image_ids"] = image_ids
                            
                            # Fetch images for preview
                            images = api_client.get_all_images()
                            st.session_state["images"] = images
                            
                            st.success(f"✅ Uploaded {len(image_ids)} images from ZIP")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
        
        else:  # Extract from CSV
            csv_file = st.file_uploader(
                "Upload CSV with image URLs",
                type=["csv"],
                key="sidebar_csv_file",
                help="CSV should contain image URLs in a column named 'url', 'image_url', 'link', or the first column"
            )
            
            if csv_file:
                if st.button("Extract Images", key="sidebar_btn_extract_csv", use_container_width=True):
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
                                st.rerun()
                            else:
                                st.error("No images extracted")
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
        
        st.markdown("---")
        
        # Display uploaded images in a scrollable view
        if st.session_state.get("images"):
            st.subheader(f"Images ({len(st.session_state['images'])})")
            
            # Scrollable container with thumbnails
            with st.container():
                for i, img in enumerate(st.session_state["images"]):
                    with st.expander(f"Image {i+1}", expanded=False):
                        st.image(img, use_container_width=True)
