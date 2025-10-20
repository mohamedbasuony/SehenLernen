import streamlit as st
import io
from pathlib import Path
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Sehen Lernen", layout="wide")

APP_ROOT = Path(__file__).resolve().parent

def _get_logo_path():
    """Find the logo file in the assets directory."""
    assets_dir = APP_ROOT / "assets"
    for filename in ("Logo.png", "logo.png"):
        candidate = assets_dir / filename
        if candidate.exists():
            return candidate
    return None

# --- Initialize Session State ---
if "show_main_app" not in st.session_state:
    st.session_state["show_main_app"] = False
if "images" not in st.session_state:
    st.session_state["images"] = []
if "uploaded_image_ids" not in st.session_state:
    st.session_state["uploaded_image_ids"] = []

# --- SIDEBAR (only show when main app is active) ---
if st.session_state["show_main_app"]:
    with st.sidebar:
        # Logo
        logo_path = _get_logo_path()
        if logo_path:
            st.image(str(logo_path), width=80)
        else:
            st.markdown("### Sehen Lernen")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        section = st.radio("Go to:", ["Data Input", "Feature Selection"], key="nav_section")
        
        st.markdown("---")
        
        # Upload section
        st.subheader("Upload Images")
        upload_method = st.radio("Method:", ["Upload Images", "Upload ZIP", "Extract from CSV"], key="upload_method")
        
        if upload_method == "Upload Images":
            uploaded_files = st.file_uploader(
                "Choose images", 
                type=["png", "jpg", "jpeg"], 
                accept_multiple_files=True,
                key="file_uploader"
            )
            
            if uploaded_files:
                try:
                    new_images = []
                    for uploaded_file in uploaded_files:
                        img = Image.open(io.BytesIO(uploaded_file.getbuffer()))
                        new_images.append(img)
                    
                    st.session_state["images"] = new_images
                    st.success(f"Loaded {len(new_images)} images")
                except Exception as e:
                    st.error(f"Error loading images: {e}")
        
        elif upload_method == "Upload ZIP":
            st.info("ZIP upload functionality coming soon")
        
        elif upload_method == "Extract from CSV":
            st.info("CSV extraction functionality coming soon")
        
        # Display uploaded images
        if st.session_state["images"]:
            st.markdown("---")
            st.subheader(f"Images ({len(st.session_state['images'])})")
            
            for i, img in enumerate(st.session_state["images"]):
                with st.expander(f"Image {i+1}"):
                    st.image(img, use_container_width=True)
        
        # Back to Home button
        st.markdown("---")
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state["show_main_app"] = False
            st.rerun()

# --- MAIN CONTENT ---
if not st.session_state["show_main_app"]:
    # Simple centered landing page - NO SCROLLING
    
    # Center everything with columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Minimal top spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Logo - positioned slightly to the right
        logo_col1, logo_col2, logo_col3 = st.columns([0.8, 1, 1.2])
        with logo_col2:
            try:
                st.image("assets/Logo.png", width=120)
            except:
                # Fallback to emoji if logo doesn't exist
                st.markdown("<div style='text-align: center; font-size: 3rem; margin-bottom: 0.5rem;'>👁️</div>", unsafe_allow_html=True)
        
        # Thin separator line after logo
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, #ccc, transparent); margin: 0.8rem 0;'>", unsafe_allow_html=True)
        
        # Title - compact sizing
        st.markdown("<h1 style='text-align: center; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600; color: #2c3e50;'>Welcome to Sehen Lernen</h1>", unsafe_allow_html=True)
        
        # Subtitle - compact sizing
        st.markdown("<p style='text-align: center; font-size: 1rem; margin-bottom: 1rem; line-height: 1.4; color: #555; max-width: 450px; margin-left: auto; margin-right: auto;'>Explore the intersection of human and artificial intelligence in visual perception.</p>", unsafe_allow_html=True)
        
        # Thin separator line before button
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, #ddd, transparent); margin: 0.8rem 0;'>", unsafe_allow_html=True)
        
        # Start button - centered and compact
        start_col1, start_col2, start_col3 = st.columns([1, 1, 1])
        with start_col2:
            if st.button("🚀 Start Learning", key="start_button", use_container_width=True, 
                        help="Begin your visual learning journey"):
                st.session_state["show_main_app"] = True
                st.rerun()
        
        # Thin separator line after button
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, #ddd, transparent); margin: 0.8rem 0;'>", unsafe_allow_html=True)
        
        # Features in compact layout
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.4rem;'>🔍</div>
                <h3 style='font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; color: #2c3e50;'>Visual Analysis</h3>
                <p style='font-size: 0.8rem; line-height: 1.3; color: #666; margin: 0;'>Advanced computer vision</p>
            </div>
            """, unsafe_allow_html=True)
            
        with feature_col2:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.4rem;'>🧠</div>
                <h3 style='font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; color: #2c3e50;'>AI Comparison</h3>
                <p style='font-size: 0.8rem; line-height: 1.3; color: #666; margin: 0;'>Human vs AI perception</p>
            </div>
            """, unsafe_allow_html=True)
            
        with feature_col3:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.4rem;'>📊</div>
                <h3 style='font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; color: #2c3e50;'>Interactive Learning</h3>
                <p style='font-size: 0.8rem; line-height: 1.3; color: #666; margin: 0;'>Hands-on exploration</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Large fixed footer at bottom - 25% of viewport height
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f0f2f6;
        border-top: 2px solid #e6e9ef;
        padding: 2rem 2rem;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        height: 25vh;
        display: flex;
        align-items: center;
    ">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3rem; max-width: 1200px; margin: 0 auto; width: 100%;">
            <div style="text-align: center;">
                <strong style='color: #2c3e50; font-size: 1.4rem; margin-bottom: 1rem; display: block;'>About the Project</strong>
                <p style='color: #555; font-size: 1rem; line-height: 1.5; margin: 0;'>
                    Sehen Lernen explores the fascinating intersection of human and artificial intelligence in visual perception. 
                    Our platform provides tools for comparative analysis and interactive learning.
                </p>
            </div>
            <div style="text-align: center;">
                <strong style='color: #2c3e50; font-size: 1.4rem; margin-bottom: 1rem; display: block;'>Team Members</strong>
                <div style='color: #555; font-size: 1rem; line-height: 1.8;'>
                    <div>Prof. Dr. Martin Langner</div>
                    <div>Mohamed Basuony</div>
                    <div>Marta Kipke</div>
                    <div>Luana Costa</div>
                    <div>Alexander Zeckey</div>
                </div>
            </div>
            <div style="text-align: center;">
                <strong style='color: #2c3e50; font-size: 1.4rem; margin-bottom: 1rem; display: block;'>Contact & Info</strong>
                <div style='color: #555; font-size: 1rem; line-height: 1.8;'>
                    <div>📧 martin.langner@uni-goettingen.de</div>
                    <div>🌐 <a href="https://www.uni-goettingen.de/de/597374.html" target="_blank" style="color: #555; text-decoration: none;">uni-goettingen.de</a></div>
                    <div>📍 Institut für Digital Humanities</div>
                    <div>🏛️ University of Göttingen</div>
                    <div style='margin-top: 1rem; font-weight: 600;'>© 2025 Sehen Lernen Project</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # Main application content when sidebar is shown
    st.title("🔍 Data Analysis Platform")
    
    if st.session_state["images"]:
        st.success(f"✅ {len(st.session_state['images'])} images ready for analysis")
        
        # Display images in a grid
        cols = st.columns(4)
        for i, img in enumerate(st.session_state["images"]):
            with cols[i % 4]:
                st.image(img, caption=f"Image {i+1}", use_container_width=True)
        
        # Add analysis options
        st.markdown("---")
        st.subheader("Analysis Options")
        analysis_type = st.selectbox("Choose analysis:", ["Feature Extraction", "Object Detection", "Similarity Analysis"])
        
        if st.button("Run Analysis"):
            st.info(f"Running {analysis_type}... (Feature coming soon)")
    
    else:
        st.info("👈 Use the sidebar to upload images to get started")
        
        # Show sample images or demo
        st.markdown("### Welcome to Sehen Lernen")
        st.write("Upload images using the sidebar to begin your visual perception analysis journey.")