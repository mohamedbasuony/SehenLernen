import streamlit as st

# Home page component with responsive design
def render_home():
    # Add responsive CSS
    st.markdown("""
    <style>
    /* Global reset and base styles */
    .home-container {
        padding: 1rem;
        padding-bottom: 900px; /* Reduced by another 20px */
        max-width: 1200px;
        margin: 0 auto;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    .project-info {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 3rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .welcome-section {
        text-align: center;
        margin: 3rem 0 4rem 0;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .welcome-title {
        font-size: clamp(2rem, 5vw, 3rem);
        margin-bottom: 1.5rem; 
        color: white;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .welcome-subtitle {
        font-size: clamp(1.1rem, 3vw, 1.4rem);
        margin-bottom: 2.5rem;
        color: rgba(255,255,255,0.95);
        line-height: 1.6;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .start-button-container {
        margin: 3rem 0 4rem 0;
        text-align: center;
        padding: 0 1rem;
    }
    
    .info-section {
        margin-top: 4rem;
        margin-bottom: 3rem;
        padding: 3rem 0;
        border-top: 3px solid #e3f2fd;
        background: #fafafa;
        border-radius: 15px;
        clear: both;
        position: relative;
        z-index: 1;
    }
    
    .info-section-title {
        text-align: center;
        font-size: clamp(1.5rem, 3vw, 2rem);
        color: #2c3e50;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
        height: auto;
        min-height: 180px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .info-card h3 {
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.5rem;
    }
    
    .info-card p {
        margin-bottom: 0.8rem;
        line-height: 1.6;
        color: #555;
        font-size: 1rem;
    }
    
    .info-card a {
        color: #3498db;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .info-card a:hover {
        color: #2980b9;
        text-decoration: underline;
    }
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .home-container {
            padding: 0.5rem;
            padding-bottom: 980px; /* Reduced by another 20px on tablet */
        }
        
        .welcome-section {
            padding: 2rem 1rem;
            margin: 2rem 0 3rem 0;
        }
        
        .project-info {
            padding: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .info-card {
            padding: 1.5rem;
            min-height: 150px;
            margin-bottom: 1rem;
        }
        
        .info-section {
            margin-top: 3rem;
            padding: 2rem 1rem;
        }
        
        .start-button-container {
            margin: 2rem 0 3rem 0;
        }
    }
    
    @media (max-width: 480px) {
        .home-container {
            padding: 0.25rem;
            padding-bottom: 1060px; /* Reduced by another 20px on mobile */
        }
        
        .welcome-section {
            padding: 1.5rem 0.8rem;
            margin: 1.5rem 0 2rem 0;
        }
        
        .info-card {
            padding: 1.2rem;
            min-height: 130px;
        }
        
        .info-section {
            margin-top: 2rem;
            padding: 1.5rem 0.5rem;
        }
    }
    
    /* Ensure proper stacking context */
    .main > div {
        position: relative;
        z-index: auto;
    }
    
    /* Add bottom padding to prevent masking by sticky footer */
    .home-container::after {
        content: "";
        height: 220px;
        display: block;
    }
    
    /* Responsive bottom padding adjustments */
    @media (max-width: 768px) {
        .home-container::after {
            height: 270px;
        }
    }
    
    @media (max-width: 480px) {
        .home-container::after {
            height: 300px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    
    # Project information section
    st.markdown("""
    <div class="project-info">
        <h2 style="margin-top: 0; color: #2c3e50;">Sehen Lernen: Human and AI Comparison</h2>
        <p><strong>Funded by:</strong> Innovation in Teaching Foundation as part of the Freedom 2023 program</p>
        <p><strong>Contact Person:</strong> Prof. Dr. Martin Langner</p>
        <p><strong>Team Members:</strong> Luana Moraes Costa, Firmin Forster, M. Kipke, Alexander Eric Wilhelm, Alexander Zeckey</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    <div class="welcome-section">
        <h1 class="welcome-title">Welcome to Sehen Lernen Platform</h1>
        <p class="welcome-subtitle">A comprehensive platform for image analysis, computer vision, and machine learning. Upload, process, and analyze images with advanced algorithms and AI techniques.</p>
    </div>
    """, unsafe_allow_html=True)

    # Start button
    st.markdown('<div class="start-button-container">', unsafe_allow_html=True)
    if st.button("🚀 Start Analyzing Images", key="home_start_button", type="primary"):
        st.session_state["active_section"] = "Data Input"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Information section with responsive layout
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    
    # Add section title
    st.markdown('<h2 class="info-section-title">📋 Project Information & Resources</h2>', unsafe_allow_html=True)
    
    # Use responsive columns - 2x2 layout for better readability
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📚 About</h3>
            <p>Sehen Lernen Project</p>
            <p>Institut für Digital Humanities</p>
            <p>Georg-August-Universität Göttingen</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📋 Policies</h3>
            <p><a href="#">Privacy Policy</a></p>
            <p><a href="#">Terms of Service</a></p>
            <p><a href="#">Cookie Policy</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🛠️ Tool Information</h3>
            <p>Version 1.0</p>
            <p>Streamlit Platform</p>
            <p>Machine Learning Application</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📞 Contact</h3>
            <p><a href="#">Contact Us</a></p>
            <p><a href="#">Support Email</a></p>
            <p><a href="#">FAQ</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
