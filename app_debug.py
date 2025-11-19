import streamlit as st

st.set_page_config(page_title="Sehen Lernen", layout="wide")

st.title("🎯 Sehen Lernen - Loading...")

try:
    st.write("✅ Streamlit is running correctly!")
    st.info("Testing app initialization...")
    
    # Try importing components one by one
    try:
        from components.sidebar import render_sidebar
        st.write("✅ Sidebar imported")
    except Exception as e:
        st.error(f"❌ Sidebar import failed: {e}")
    
    try:
        from components.data_input import render_data_input
        st.write("✅ Data input imported")
    except Exception as e:
        st.error(f"❌ Data input import failed: {e}")
    
    try:
        from components.feature_selection import render_feature_selection
        st.write("✅ Feature selection imported")
    except Exception as e:
        st.error(f"❌ Feature selection import failed: {e}")
        st.write("Error details:")
        st.write(str(e))
    
    try:
        from components.seher_smart_chat import render_smart_chat
        st.write("✅ Chat imported")
    except Exception as e:
        st.error(f"❌ Chat import failed: {e}")
    
    st.success("All components imported successfully!")
    
except Exception as e:
    st.error(f"Critical error: {e}")
    import traceback
    st.code(traceback.format_exc())
