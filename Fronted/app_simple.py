import streamlit as st
import io
from pathlib import Path
from PIL import Image

# Simple, clean app that definitely shows sidebar
st.set_page_config(page_title="Sehen Lernen", layout="wide")

# Initialize session state
if "show_sidebar" not in st.session_state:
    st.session_state["show_sidebar"] = False
if "images" not in st.session_state:
    st.session_state["images"] = []

# ALWAYS show sidebar if show_sidebar is True - NO CSS HIDING
if st.session_state["show_sidebar"]:
    with st.sidebar:
        st.title("🎉 SIDEBAR WORKING!")
        st.write("Logo would go here")
        
        st.subheader("Upload Images")
        upload_method = st.radio("Method:", ["Images", "ZIP", "CSV"])
        
        if upload_method == "Images":
            files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if files:
                try:
                    st.session_state["images"] = [Image.open(io.BytesIO(f.getbuffer())) for f in files]
                    st.success(f"Loaded {len(files)} images")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state["images"]:
            st.subheader(f"Images ({len(st.session_state['images'])})")
            for i, img in enumerate(st.session_state["images"]):
                with st.expander(f"Image {i+1}"):
                    st.image(img, use_container_width=True)
        
        if st.button("Back to Home"):
            st.session_state["show_sidebar"] = False
            st.rerun()

# Main content
if not st.session_state["show_sidebar"]:
    # Landing page - NO CSS COMPLICATIONS
    st.markdown("""
    <div style="text-align: center; padding: 5rem 2rem;">
        <h1 style="font-size: 3rem; color: #2563eb;">Welcome to Sehen Lernen</h1>
        <p style="font-size: 1.2rem; margin: 2rem 0;">AI-powered image analysis workspace</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Learning", key="start_btn"):
        st.session_state["show_sidebar"] = True
        st.rerun()
else:
    # Main app content
    st.title("📊 Data Analysis")
    
    if st.session_state["images"]:
        st.success(f"✅ {len(st.session_state['images'])} images ready for analysis")
        
        cols = st.columns(4)
        for i, img in enumerate(st.session_state["images"]):
            with cols[i % 4]:
                st.image(img, caption=f"Image {i+1}", use_container_width=True)
    else:
        st.info("👈 Use the sidebar to upload images")
