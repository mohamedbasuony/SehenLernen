import streamlit as st

# Test the sidebar logic
st.set_page_config(page_title="Sidebar Test", layout="wide")

# Initialize state
if "has_started" not in st.session_state:
    st.session_state["has_started"] = False

# Show current state
st.markdown(f"**Current state:** has_started = {st.session_state['has_started']}")

# Sidebar logic
if st.session_state["has_started"]:
    with st.sidebar:
        st.success("🎉 SIDEBAR IS WORKING!")
        st.write("This proves the sidebar can show")
        if st.button("Reset"):
            st.session_state["has_started"] = False
            st.rerun()

# Main content
if not st.session_state["has_started"]:
    st.title("Test: Sidebar Visibility")
    st.write("The sidebar should NOT be visible now.")
    if st.button("Show Sidebar"):
        st.session_state["has_started"] = True
        st.rerun()
else:
    st.title("Sidebar Test Active")
    st.success("✅ Check the sidebar on the left!")
    st.write("If you can see the sidebar, the logic works correctly.")