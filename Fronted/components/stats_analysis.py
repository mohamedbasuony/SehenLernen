import streamlit as st

# Statistics Analysis page component
def render_stats_analysis():
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem; margin-bottom: 1.5rem; color: #2c3e50;'>Statistics Analysis</h1>", unsafe_allow_html=True)
    
    # Ensure feature embeddings or metrics available
    if not st.session_state.get("feature_results"):
        st.warning("Please run feature extraction first.")
        return

    st.info("Statistics analysis placeholder – integrate statistical tests or metrics here.")

    # Download and navigation buttons
    col1, col2, col3 = st.columns([2,1,1])
    with col2:
        if st.button("Download Result", key="stats_download_result"):
            pass
    with col3:
        if st.button("Download Settings", key="stats_download_settings"):
            pass

    if st.button("Next: Visualization", key="stats_next"):
        st.session_state["active_section"] = "Visualization"
