import streamlit as st
from pathlib import Path

from components.sidebar import render_sidebar
from components.data_input import render_data_input
from components.feature_selection import render_feature_selection
from components.stats_analysis import render_stats_analysis
from components.visualization import render_visualization

# --- Page Configuration ---
st.set_page_config(page_title="Sehen Lernen", layout="wide")

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
            if st.button("🚀 Start Learning", key="start_button", use_container_width=True,
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

    # Fixed footer
    st.markdown(
        """
        <div style="
            position: fixed; bottom: 0; left: 0; right: 0;
            background-color: #f0f2f6; border-top: 2px solid #e6e9ef;
            padding: 2rem 2rem; z-index: 1000; box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            height: 25vh; display: flex; align-items: center;">
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
                        <div>🌐 <a href="https://www.uni-goettingen.de/de/597374.html" target="_blank" style="color:#555;text-decoration:none;">uni-goettingen.de</a></div>
                        <div>📍 Institut für Digital Humanities</div>
                        <div>🏛️ University of Göttingen</div>
                        <div style='margin-top: 1rem; font-weight: 600;'>© 2025 Sehen Lernen Project</div>
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
        render_feature_selection()
    elif section == "Data Input":
        # Route Data Input to Feature Selection view; uploads live in the sidebar now
        render_feature_selection()
    elif section == "Feature Selection":
        render_feature_selection()
    elif section == "Statistics Analysis":
        render_stats_analysis()
    elif section == "Visualization":
        render_visualization()
    else:
        st.error(f"Unknown section: {section}")


# --- Render ---
if not st.session_state["show_main_app"]:
    _render_landing()
else:
    _render_main()
