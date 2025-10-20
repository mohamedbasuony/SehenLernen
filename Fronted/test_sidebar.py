#!/usr/bin/env python3
"""
Minimal test to verify sidebar functionality works
"""
import streamlit as st

# Page config
st.set_page_config(page_title="Sidebar Test", layout="wide")

# Initialize session state
if "has_started" not in st.session_state:
    st.session_state["has_started"] = False

# Show current state
st.write(f"**Current state:** has_started = {st.session_state['has_started']}")

# Sidebar logic
if st.session_state["has_started"]:
    with st.sidebar:
        st.title("🎉 SIDEBAR IS WORKING!")
        st.write("Logo would go here")
        st.button("📤 Upload Images")
        st.button("🔍 Feature Selection")
        st.write("Upload methods:")
        st.radio("Method", ["Images", "ZIP", "CSV"])

# Main content
if not st.session_state["has_started"]:
    st.title("Welcome to Sehen Lernen")
    if st.button("Start Learning"):
        st.session_state["has_started"] = True
        st.rerun()
else:
    st.title("Main Application")
    st.success("✅ You've started! Check the sidebar on the left!")
    
    if st.button("Reset to Home"):
        st.session_state["has_started"] = False
        st.rerun()