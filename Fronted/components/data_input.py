# Fronted/components/data_input.py

import streamlit as st


def render_data_input():
    # CRITICAL: Force sidebar to be visible and override any home page CSS
    st.markdown(
        """
        <style>
        /* FORCE sidebar to be visible - override any home page CSS */
        [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            width: auto !important;
            height: auto !important;
        }
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Reset background from home page */
        .stApp {
            background: var(--background-color) !important;
            animation: none !important;
            height: auto !important;
            overflow: auto !important;
        }
        
        /* Reset containers from home page */
        html, body, [data-testid="stAppViewContainer"], .main {
            height: auto !important;
            overflow: auto !important;
            position: static !important;
        }
        
        .block-container {
            padding: 3rem 1rem !important;
            max-width: 1200px !important;
            position: static !important;
        }
        
        /* Remove any fixed positioning */
        .landing-container {
            position: static !important;
            width: auto !important;
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("📤 Data Input")
    
    st.info("👈 Use the sidebar to upload images via Images, ZIP, or CSV extraction.")
    
    # Show uploaded images if any
    images = st.session_state.get("images", [])
    
    if not images:
        st.warning("No images uploaded yet. Please use the sidebar to upload images.")
        st.markdown("""
        ### Upload Methods Available:
        1. **Upload Images** - Select multiple image files (PNG, JPG, JPEG)
        2. **Upload ZIP** - Upload a ZIP file containing images
        3. **Extract from CSV** - Provide a CSV file with image URLs to download and extract images
        
        All uploaded images will appear in the sidebar and be available for feature extraction.
        """)
    else:
        st.success(f"✅ {len(images)} images uploaded and ready for analysis")
        
        st.subheader("Preview of Uploaded Images")
        
        # Display images in a grid
        cols_per_row = 4
        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images):
                    with col:
                        st.image(images[idx], caption=f"Image {idx+1}", use_container_width=True)
        
        st.markdown("---")
        st.info("💡 Navigate to **Feature Selection** using the sidebar to analyze these images.")
    
    # Show download button for extracted images if available
    if "extractor_zip" in st.session_state:
        st.download_button(
            label="📥 Download Extracted Images ZIP",
            data=st.session_state["extractor_zip"],
            file_name="extracted_images.zip",
            mime="application/zip",
            key="btn_download_extractor_zip"
        )
