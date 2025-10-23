"""
Modern floating chatbot widget for Sehen Lernen
"""
import streamlit as st
from datetime import datetime

class SeherSmartChat:
    def __init__(self):
        self.messages_key = "seher_messages"
        self.context_key = "seher_context"
        self.visible_key = "seher_visible"
    
    def _ensure_initialized(self):
        if self.messages_key not in st.session_state:
            st.session_state[self.messages_key] = [
                {"role": "assistant", "content": "👋 Hi! I'm Seher, your computer vision assistant! I can help you understand and use all the features in Sehen Lernen. Ask me about similarity search, histograms, clustering, texture analysis, or any other tool!", "timestamp": datetime.now()}
            ]
        if self.context_key not in st.session_state:
            st.session_state[self.context_key] = {"tab": "general"}
        if self.visible_key not in st.session_state:
            st.session_state[self.visible_key] = False
    
    def update_context(self, tab: str, details: list = None):
        self._ensure_initialized()
        st.session_state[self.context_key] = {"tab": tab, "details": details or []}
    
    def render(self):
        self._ensure_initialized()
        is_expanded = st.session_state.get(self.visible_key, False)
        
        if not is_expanded:
            # Collapsed - show button in sidebar or bottom
            col1, col2, col3 = st.columns([8, 1, 1])
            with col3:
                if st.button("💬", key="seher_toggle", help="Open Seher Assistant"):
                    st.session_state[self.visible_key] = True
                    st.rerun()
        else:
            # Expanded - show chat window
            col1, col2 = st.columns([6, 4])
            with col2:
                # Header
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown("### 💫 Seher")
                with h2:
                    if st.button("✕", key="seher_close"):
                        st.session_state[self.visible_key] = False
                        st.rerun()
                
                # Messages
                messages = st.session_state[self.messages_key]
                for msg in messages[-6:]:
                    if msg['role'] == 'assistant':
                        st.info(f"🤖 {msg['content']}")
                    else:
                        st.success(f"You: {msg['content']}")
                
                # Input
                with st.form("chat", clear_on_submit=True):
                    user_input = st.text_input("Ask me...", label_visibility="collapsed")
                    if st.form_submit_button("Send") and user_input.strip():
                        st.session_state[self.messages_key].append({
                            "role": "user", "content": user_input.strip(), "timestamp": datetime.now()
                        })
                        response = self._generate_response(user_input.strip())
                        st.session_state[self.messages_key].append({
                            "role": "assistant", "content": response, "timestamp": datetime.now()
                        })
                        st.rerun()
    
    def _generate_response(self, user_input: str) -> str:
        """Generate comprehensive, detailed responses based on user input"""
        user_input_lower = user_input.lower()
        current_context = st.session_state.get(self.context_key, {})
        current_tab = current_context.get("tab", "general")
        
        # SIMILARITY SEARCH QUERIES
        if any(keyword in user_input_lower for keyword in ["similarity search", "similarity", "similar images", "find similar"]):
            return """🔍 **Similarity Search** finds visually similar images in your dataset - like a reverse image search for your own photos!

**How it works:**
1. **Upload your images** in the Data Input tab first
2. **Choose a feature method** that captures what "similar" means to you:
   • **CNN**: Best for semantic similarity (same objects/scenes even with different colors)
   • **SIFT**: Perfect for same objects from different angles/lighting  
   • **HOG**: Great for similar shapes and structural patterns
   • **Histogram**: Finds images with similar color distributions
   • **Manuscript**: Specialized for handwritten text analysis

3. **Select distance metric** (Cosine is usually best)
4. **Choose your query image** and set similarity threshold
5. **Click search** to find matches!

**Pro tip:** Use "Precompute Features" first for faster searches on large datasets."""

        # HISTOGRAM ANALYSIS
        elif any(keyword in user_input_lower for keyword in ["histogram", "color distribution", "brightness"]):
            return """� **Histogram Analysis** shows how pixel intensities are distributed in your images.

**Two types available:**
• **Colored Histograms**: Show Red, Green, Blue channels separately
  - Reveals color composition and dominant colors
  - Peaks in red channel = lots of red pixels, etc.

• **Black & White Histograms**: Show overall brightness distribution  
  - Left side (0) = dark pixels, Right side (255) = bright pixels
  - Helps identify exposure and contrast issues

**What histograms tell you:**
- Peak on left = dark/underexposed image
- Peak on right = bright/overexposed image  
- Peak in middle = well-exposed image
- Multiple peaks = varied lighting or distinct regions

**Practical uses:** Exposure correction, color grading, image quality assessment"""

        # K-MEANS CLUSTERING  
        elif any(keyword in user_input_lower for keyword in ["kmeans", "k-means", "clustering", "segmentation"]):
            return """🎯 **K-means Clustering** groups pixels with similar colors to segment your image into distinct regions.

**How to use:**
1. **Choose number of clusters (k)**:
   - k=2-3: Simple images (sky + ground, object + background)
   - k=4-6: Moderate complexity (multiple objects)
   - k=8-12: Complex scenes with many colors

2. **Select images** to cluster (single or batch mode)

**What you get:**
- **Color palette** of dominant colors in your image
- **Segmented image** showing distinct regions
- **Cluster centers** (the representative colors)

**Applications:**
- Image segmentation for object isolation
- Color quantization for artistic effects  
- Background removal preparation
- Dominant color extraction for design"""

        # HARALICK TEXTURE FEATURES
        elif any(keyword in user_input_lower for keyword in ["haralick", "texture", "glcm", "co-occurrence"]):
            return """🧵 **Haralick Texture Features** analyze surface patterns using Gray-Level Co-occurrence Matrix (GLCM).

**What it measures:**
- **Contrast**: How much variation between neighboring pixels
- **Homogeneity**: How uniform the texture appears  
- **Energy**: How ordered/regular the texture is
- **Correlation**: How linearly dependent pixel pairs are
- **Dissimilarity**: Average difference between pixel pairs

**Parameters you can adjust:**
- **Distances**: How far apart to compare pixels (1,2,3,5 pixels)
- **Angles**: Directions to analyze (0°, 45°, 90°, 135°)  
- **Quantization levels**: How many gray levels to use (16-256)

**Perfect for:** Medical imaging, material classification, surface quality control, fabric analysis"""

        # SHAPE FEATURES
        elif any(keyword in user_input_lower for keyword in ["shape", "hog", "embedding", "features"]):
            return """� **Shape Features** capture geometric and structural information from images.

**Available methods:**

**1. HOG (Histogram of Oriented Gradients)**
- Analyzes edge directions and magnitudes
- Excellent for object detection (especially people, vehicles)
- Creates feature vectors describing shape patterns

**2. Deep Learning Embeddings**  
- **ResNet-50**: 2048-dimensional rich features, best quality
- **ResNet-18**: 512-dimensional, faster processing
- **MobileNet-v2**: 1280-dimensional, optimized for mobile

**3. FAST Corner Detection**
- Finds distinctive corner points rapidly
- Great for keypoint matching and tracking
- Useful as features for further analysis

**When to use what:**
- HOG: Object detection, structural analysis
- Embeddings: Similarity search, classification  
- FAST: Image matching, motion tracking"""

        # GENERAL HELP OR UNCLEAR QUESTIONS
        else:
            return f"""👋 **Welcome to Sehen Lernen!** I'm your computer vision assistant.

**I can help you with:**

🔍 **Similarity Search** - Find visually similar images using CNN, SIFT, HOG, or histogram features

📊 **Histogram Analysis** - Understand color and brightness distributions in your images

🎯 **K-means Clustering** - Segment images by grouping similar-colored pixels

🧵 **Haralick Texture** - Analyze surface patterns and texture properties using GLCM

🔲 **Shape Features** - Extract HOG, deep learning embeddings, or corner features

🔢 **Local Binary Patterns (LBP)** - Encode local texture patterns for classification

⭕ **Contour Extraction** - Find object boundaries and shapes

🤖 **Classifier Training** - Build custom image classification models

**Current context:** You're in the {current_tab} section. Ask me anything specific about the tools or concepts!

**Example questions:** "How does similarity search work?", "What's the difference between CNN and SIFT features?", "How do I choose the right number of clusters?"
"""

# Global instance
_seher_chat = SeherSmartChat()

def render_smart_chat():
    _seher_chat.render()

def update_chat_context(tab: str, details: list = None):
    _seher_chat.update_context(tab, details)

def add_chat_message(message: str):
    _seher_chat._ensure_initialized()
    st.session_state[_seher_chat.messages_key].append({
        "role": "assistant", "content": message, "timestamp": datetime.now()
    })
