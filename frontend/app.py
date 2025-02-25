import streamlit as st

# --- Custom CSS for sidebar styling with slide-down animation ---
st.markdown("""
    <style>
    /* Sidebar background color */
    section[data-testid="stSidebar"] {
        background-color: #2f2f2f;  /* Dark Grey */
        padding: 10px;
    }

    /* Main buttons in the sidebar */
    .sidebar .button {
        font-size: 18px;
        padding: 10px;
        margin: 10px 0;
        background-color: #91C7EF;
        color: white;
        text-align: left;
        border: none;
        width: 100%;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .sidebar .button:hover {
        background-color: #81b0d4;  /* Hover color */
    }

    /* Sub-buttons with animation and indentation */
    .sidebar .sub-button-container {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.5s ease, opacity 0.5s ease;
    }

    /* When container is expanded, show sub-buttons */
    .sidebar .sub-button-container.expanded {
        max-height: 500px;  /* Set a large enough height for sliding effect */
        opacity: 1;
    }

    .sidebar .sub-button {
        font-size: 16px;
        padding: 8px;
        margin-left: 30px;
        margin-bottom: 8px;
        background-color: #A99FBF;
        color: white;
        border: none;
        width: 85%;
        border-radius: 5px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    /* When sub-buttons become visible */
    .sidebar .sub-button.visible {
        opacity: 1;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Toolbox with Expandable Buttons ---
st.sidebar.title("Toolboxes")

# Helper function to control the visibility of sub-buttons
def display_subbuttons(main_button_text, sub_buttons, key):
    expanded = st.session_state.get(f"expanded_{key}", False)

    if st.sidebar.button(main_button_text, key=key):
        st.session_state[f"expanded_{key}"] = not expanded

    expanded = st.session_state.get(f"expanded_{key}", False)

    sub_button_container_class = "expanded" if expanded else ""
    sub_button_class = "visible" if expanded else ""

    # Create sub-button container with sliding effect
    st.sidebar.markdown(f'<div class="sub-button-container {sub_button_container_class}">', unsafe_allow_html=True)

    if expanded:
        for sub_button in sub_buttons:
            st.sidebar.markdown(f'<button class="sub-button {sub_button_class}">{sub_button}</button>', unsafe_allow_html=True)

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Data Preparation with sub-buttons
display_subbuttons("Data preparation", ["Input", "Download", "Sampling"], "data_prep")

# Feature Extraction with sub-buttons
display_subbuttons("Feature extraction", ["Image Processing", "CNNs"], "feature_extraction")

# Statistics and Classification with sub-buttons
display_subbuttons("Statistics and classification", ["Statistical Algorithms", "Statistical Evaluation"], "statistics_classification")

# Visualization with sub-buttons
display_subbuttons("Visualization", ["Visualization"], "visualization")

# --- Main Area: Tabs for Different Views ---
st.title("Current View")

tab1, tab2, tab3 = st.tabs(["Full View Original", "Full View Edited Image", "Diagram of Visualization"])

with tab1:
    st.write("Original Image View")

with tab2:
    st.write("Edited Image View")

with tab3:
    st.write("Visualization Diagram")

# --- Draggable Icons Section ---
st.subheader("Workflow Bar")
icon1, icon2, icon3 = st.columns(3)
with icon1:
    st.write("🔧")
with icon2:
    st.write("🔍")
with icon3:
    st.write("📊")

# --- Thumbnail Section ---
st.subheader("Thumbnails ")
st.image(["https://via.placeholder.com/100", "https://via.placeholder.com/100", "https://via.placeholder.com/100"], width=100)

# --- Sidebar: Options for the Selected Tool ---
st.sidebar.title("Options of the Selected Tool")
st.sidebar.write("Options will appear here based on the selected tool")
