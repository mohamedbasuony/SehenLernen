import streamlit as st
import logging

# Suppress warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Page config
st.set_page_config(
    page_title="Sehen Lernen", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def main():
    """Main application function"""
    try:
        # Import and render sidebar
        from components.sidebar import render_sidebar
        page = render_sidebar()
        
        # Route to pages
        if page == 'home':
            try:
                from components.home import render_home
                render_home()
            except ImportError:
                st.info("🏠 Home page")
        elif page == 'data_input':
            from components.data_input import render_data_input
            render_data_input()
        elif page == 'feature_selection':
            from components.feature_selection import render_feature_selection
            render_feature_selection()
        elif page == 'stats_analysis':
            from components.stats_analysis import render_stats_analysis
            render_stats_analysis()
        elif page == 'visualization':
            from components.visualization import render_visualization
            render_visualization()
        
        # Render chat
        try:
            from components.seher_smart_chat import render_smart_chat
            render_smart_chat()
        except:
            pass
            
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")

if __name__ == "__main__":
    main()
