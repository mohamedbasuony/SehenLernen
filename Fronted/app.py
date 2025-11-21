import streamlit as st
from pathlib import Path

from components.sidebar import render_sidebar
from components.data_input import render_data_input
from components.feature_selection import render_feature_selection
from components.stats_analysis import render_stats_analysis
from components.visualization import render_visualization
from components.seher_smart_chat import render_smart_chat, update_chat_context, add_chat_message

import uuid

# Per-session identifier used to isolate uploads from other users
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

# --- Page Configuration ---
st.set_page_config(
    page_title="Sehen Lernen", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Responsive CSS and Button Styling ---
st.markdown("""
<style>
    /* Global responsive settings */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Responsive container adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            padding-top: 1rem;
        }
        
        /* Adjust sidebar for mobile */
        .css-1d391kg {
            padding-top: 1rem;
        }
    }
    
    /* Ensure columns stack properly on mobile */
    @media (max-width: 640px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        
        .row-widget.stHorizontal > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    /* Style all buttons with the specified background color */
    .stButton > button {
        background-color: #F4F4F4 !important;
        color: #333333 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #E8E8E8 !important;
        border-color: #AAAAAA !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button:active {
        background-color: #DDDDDD !important;
        transform: translateY(0px) !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #F4F4F4 !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        border: 2px solid #2c3e50 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #E8E8E8 !important;
        border-color: #1a252f !important;
    }
    
    /* Radio buttons and checkboxes styling */
    .stRadio > div, .stCheckbox > div {
        text-align: center !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        text-align: center !important;
    }
    
    /* Keep form elements naturally aligned - don't force center */
</style>
""", unsafe_allow_html=True)

APP_ROOT = Path(__file__).resolve().parent


def _get_logo_path() -> Path | None:
    assets_dir = APP_ROOT / "assets"
    for name in ("Logo.png", "logo.png"):
        candidate = assets_dir / name
        if candidate.exists():
            return candidate
    return None


# --- Initialize Session State ---
st.session_state.setdefault("active_section", "Home")
st.session_state.setdefault("show_main_app", False)


def _render_landing() -> None:
    """Landing page with centered logo, start button, and fixed footer."""
    # Center block
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        l1, l2, l3 = st.columns([1.8, 1, 1.5])
        with l2:
            logo = _get_logo_path()
            if logo:
                st.image(str(logo), width=120)
            else:
                st.markdown(
                    "<div style='text-align:center;font-size:3rem;margin-bottom:0.5rem;'>👁️</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,#ccc,transparent);margin:0.8rem 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h1 style='text-align:center;font-size:2.5rem;margin-bottom:0.5rem;font-weight:600;color:#2c3e50;'>Welcome to Sehen Lernen</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;font-size:1rem;margin-bottom:1rem;line-height:1.4;color:#555;max-width:450px;margin-left:auto;margin-right:auto;'>Explore the intersection of human and artificial intelligence in visual perception.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,#ddd,transparent);margin:0.8rem 0;'>",
            unsafe_allow_html=True,
        )

        b1, b2, b3 = st.columns([1, 1, 1])
        with b2:
            if st.button("🚀 Start Learning", key="start_button", width='stretch',
                         help="Begin your visual learning journey"):
                st.session_state["show_main_app"] = True
                st.session_state["active_section"] = "Data Input"  # take users to uploads first
                # Streamlit 1.25+ uses st.rerun; older versions used experimental_rerun
                rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                if callable(rerun_fn):
                    rerun_fn()

        st.markdown(
            "<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,#ddd,transparent);margin:0.8rem 0;'>",
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                """
                <div style='text-align:center;padding:0.5rem;'>
                    <div style='font-size:2rem;margin-bottom:0.4rem;'>🔍</div>
                    <h3 style='font-size:1.3rem;font-weight:600;margin-bottom:0.3rem;color:#2c3e50;'>Visual Analysis</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                """
                <div style='text-align:center;padding:0.5rem;'>
                    <div style='font-size:2rem;margin-bottom:0.4rem;'>🧠</div>
                    <h3 style='font-size:1.3rem;font-weight:600;margin-bottom:0.3rem;color:#2c3e50;'>AI Comparison</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                """
                <div style='text-align:center;padding:0.5rem;'>
                    <div style='font-size:2rem;margin-bottom:0.4rem;'>📊</div>
                    <h3 style='font-size:1.3rem;font-weight:600;margin-bottom:0.3rem;color:#2c3e50;'>Interactive Learning</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Sticky footer that sticks to bottom and fits all screen widths
    st.markdown(
        """
        <style>
        .sticky-footer {
            position: relative;
            width: 100vw; /* Full viewport width */
            background-color: #f0f2f6;
            border-top: 2px solid #e6e9ef;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
            margin-top: 4.75rem; /* Reduced by another ~20px */
            margin-left: calc(-50vw + 50%); /* Center full-width element */
            margin-bottom: 0 !important; /* Force no bottom margin */
            padding: 1rem 0.8rem;
            padding-bottom: 1rem !important; /* Exact padding, no more */
            /* Make this the absolute last element */
            display: block;
            clear: both;
        }
        
        /* Ensure nothing comes after the footer */
        .sticky-footer::after {
            content: "";
            display: block;
            height: 0;
            clear: both;
        }
        
        .footer-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            max-width: 1400px;
            margin: 0 auto;
            width: calc(100% - 1.6rem); /* Account for padding */
            padding: 0 0.8rem;
        }
        
        .footer-section {
            text-align: center;
            padding: 0.3rem; /* More compact padding */
        }
        
        .footer-title {
            color: #2c3e50;
            font-size: clamp(1rem, 2.2vw, 1.2rem); /* Smaller title */
            font-weight: 600;
            margin-bottom: 0.5rem; /* Less margin */
            display: block;
            border-bottom: 1px solid #3498db; /* Thinner border */
            padding-bottom: 0.2rem;
        }
        
        .footer-text {
            color: #555;
            font-size: clamp(0.8rem, 1.8vw, 0.9rem); /* Smaller text */
            line-height: 1.4; /* Tighter line height */
            margin: 0;
        }
        
        .footer-links {
            color: #555;
            font-size: clamp(0.8rem, 1.8vw, 0.9rem); /* Smaller text */
            line-height: 1.4; /* Tighter line height */
        }
        
        .footer-links a {
            color: #3498db;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer-links a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .sticky-footer {
                padding: 0.8rem 0.4rem;
                padding-bottom: 0.8rem; /* No extra bottom padding */
                margin-top: 3.25rem; /* Brought up by another ~20px on tablet */
                margin-bottom: 0;
            }
            
            .footer-content {
                grid-template-columns: 1fr;
                gap: 0.8rem;
                width: calc(100% - 0.8rem);
                padding: 0 0.4rem;
            }
            
            .footer-section {
                padding: 0.2rem;
            }
        }
        
        @media (max-width: 480px) {
            .sticky-footer {
                padding: 0.6rem 0.3rem;
                padding-bottom: 0.6rem; /* No extra bottom padding */
                margin-top: 1.25rem; /* Brought up by another ~20px on mobile */
                margin-bottom: 0;
            }
            
            .footer-content {
                gap: 0.6rem;
                width: calc(100% - 0.6rem);
                padding: 0 0.3rem;
            }
        }
        
        /* Ensure proper page flow - page ends at footer */
        .main .block-container {
            min-height: auto; /* Remove fixed height */
        }
        
        /* Force page height to end exactly at footer */
        html {
            height: auto !important;
        }
        
        body {
            height: auto !important;
            min-height: auto !important;
        }
        
        #root {
            height: auto !important;
            min-height: auto !important;
        }
        
        /* Remove padding since footer is no longer floating */
        .stApp > div {
            padding-bottom: 0 !important;
        }
        
        .main .block-container {
            padding-bottom: 0 !important;
        }
        
        /* Ensure no whitespace below footer - eliminate all trailing space */
        body, html {
            margin: 0 !important;
            padding: 0 !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            height: auto !important;
            overflow-x: hidden !important;
        }
        
        .stApp {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Force page to end exactly at footer */
        .stApp > div:last-child {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Remove any default Streamlit bottom spacing */
        .main, .main > div, .block-container {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Remove body margins since footer is no longer floating */
        body {
            margin-bottom: 0;
        }
        </style>
        
        <div class="sticky-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <strong class="footer-title">About the Project</strong>
                    <p class="footer-text">
                        Sehen Lernen explores the fascinating intersection of human and artificial intelligence in visual perception.
                        Our platform provides tools for comparative analysis and interactive learning.
                    </p>
                </div>
                <div class="footer-section">
                    <strong class="footer-title">Team Members</strong>
                    <div class="footer-links">
                        <div>Prof. Dr. Martin Langner</div>
                        <div>Mohamed Basuony</div>
                        <div>Marta Kipke</div>
                        <div>Luana Costa</div>
                        <div>Alexander Zeckey</div>
                    </div>
                </div>
                <div class="footer-section">
                    <strong class="footer-title">Contact & Info</strong>
                    <div class="footer-links">
                        <div>📧 <a href="mailto:martin.langner@uni-goettingen.de">martin.langner@uni-goettingen.de</a></div>
                        <div>🌐 <a href="https://www.uni-goettingen.de/de/597374.html" target="_blank">uni-goettingen.de</a></div>
                        <div>📍 Institut für Digital Humanities</div>
                        <div>🏛️ University of Göttingen</div>
                        <div style="margin-top: 0.8rem; font-weight: 600; color: #2c3e50;">© 2025 Sehen Lernen Project</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_main() -> None:
    """Main app: show sidebar and route sections without changing feature API calls."""
    # Sidebar (existing navigation/buttons live here)
    render_sidebar()

    # Route like before
    section = st.session_state.get("active_section", "Feature Selection")
    if section == "Home":
        # Show features page; it will prompt to upload if empty
        update_chat_context("general")
        render_feature_selection()
    elif section == "Data Input":
        # Route Data Input to Feature Selection view; uploads live in the sidebar now
        update_chat_context("general")
        render_feature_selection()
    elif section == "Feature Selection":
        update_chat_context("general")
        render_feature_selection()
    elif section == "Statistics Analysis":
        update_chat_context("stats")
        render_stats_analysis()
    elif section == "Visualization":
        update_chat_context("visualization")
        render_visualization()
    else:
        st.error(f"Unknown section: {section}")
    
    # Always render the chat island (it will float independently)
    # Position it at the bottom right of the screen
    with st.container():
        render_smart_chat()


# --- Render ---
if not st.session_state["show_main_app"]:
    _render_landing()
else:
    _render_main()
