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
                {"role": "assistant", "content": "👋 Hi! I'm Seher, your intelligent computer vision assistant! I understand natural language and can help you with all features in Sehen Lernen. Ask me anything - like 'What is similarity search?', 'How do I use histograms?', or 'Which clustering method should I choose?' I'll give you contextual, detailed answers!", "timestamp": datetime.now()}
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
            # Collapsed - show button in bottom-right with responsive positioning
            col1, col2, col3 = st.columns([7, 2, 1])
            with col3:
                if st.button("💬", key="seher_toggle", help="Open Seher Assistant"):
                    st.session_state[self.visible_key] = True
                    st.rerun()
        else:
            # Expanded - show chat window with responsive layout
            # On mobile: full width, on desktop: right column
            if st.session_state.get('mobile_view', False):
                # Full width on mobile
                chat_container = st.container()
            else:
                # Right column on desktop
                col1, col2 = st.columns([5, 5])  # More balanced columns
                chat_container = col2
            
            with chat_container:
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
        """Generate intelligent, contextual responses based on user input"""
        user_input_lower = user_input.lower()
        current_context = st.session_state.get(self.context_key, {})
        current_tab = current_context.get("tab", "general")
        
        # Determine question type for better responses
        is_how_question = any(phrase in user_input_lower for phrase in ["how", "steps", "tutorial", "guide"])
        is_what_question = any(phrase in user_input_lower for phrase in ["what", "define", "explain", "tell me"])
        is_which_question = any(phrase in user_input_lower for phrase in ["which", "recommend", "best", "choose"])
        is_comparison = any(phrase in user_input_lower for phrase in ["difference", "compare", "vs", "versus", "between"])
        
        # SIMILARITY SEARCH QUERIES - Enhanced pattern matching
        if any(keyword in user_input_lower for keyword in ["similarity", "similar", "search", "match", "find", "lookup", "reverse", "cnn", "sift", "hog", "resnet"]):
            if is_what_question:
                return """🔍 **Similarity Search** finds visually similar images in your dataset - like a reverse image search for your own photos!

It analyzes visual features (colors, shapes, textures, objects) to find images that look similar to a query image, without needing filename matching or tags.

**Core concept**: Convert images into mathematical feature vectors, then measure distances to find closest matches."""
            elif is_how_question:
                return """📋 **Step-by-Step Similarity Search:**

1. **Upload images** → Data Input tab (need 10+ for good results)
2. **Choose method**: CNN (objects), SIFT (angles), HOG (shapes), Histogram (colors)
3. **Set parameters**: Cosine distance, 0.5 threshold (adjust as needed)
4. **Search**: Select query → "Find Similar Images"
5. **Review**: Check ranked results with similarity scores

💡 **Pro tip**: Use "Precompute Features" for faster repeated searches!"""
            elif is_which_question:
                return """🎯 **Choosing the Right Similarity Method:**

**For natural photos**: CNN (best overall - understands objects and scenes)
**For technical images**: SIFT (logos, diagrams) or HOG (shapes, patterns)  
**For color matching**: Histogram (fast color-based similarity)
**For documents**: Manuscript (specialized for handwritten text)

**Distance metrics**: Cosine (recommended), Euclidean (histograms), Manhattan (robust)"""
            elif is_comparison:
                return """⚖️ **Similarity Method Comparison:**

**CNN**: Semantic understanding, slower but most accurate
**SIFT**: Geometric invariance, perfect for same object/different angles  
**HOG**: Structural patterns, fast processing, good for shapes
**Histogram**: Color similarity, very fast but ignores spatial relationships
**Manuscript**: Text-specific features for handwritten documents

**Best for most cases**: CNN → SIFT → HOG → Histogram"""
            else:
                return """🔍 **Similarity Search** finds visually similar images in your dataset - like a reverse image search for your own photos!

**Quick guide**: Upload images → Choose method (CNN for photos, SIFT for objects, HOG for shapes, Histogram for colors) → Set threshold → Search!

**Methods**: CNN (semantic), SIFT (geometric), HOG (structural), Histogram (color), Manuscript (text)"""

        # HISTOGRAM ANALYSIS - Enhanced pattern matching  
        elif any(keyword in user_input_lower for keyword in ["histogram", "color", "brightness", "distribution", "channels", "rgb", "pixel", "intensity", "exposure"]):
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

        # K-MEANS CLUSTERING - Enhanced pattern matching
        elif any(keyword in user_input_lower for keyword in ["cluster", "kmeans", "k-means", "segment", "group", "palette", "colors", "regions", "centroids"]):
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

        # HARALICK TEXTURE FEATURES - Enhanced pattern matching
        elif any(keyword in user_input_lower for keyword in ["haralick", "texture", "glcm", "co-occurrence", "pattern", "surface", "roughness", "contrast", "homogeneity", "energy"]):
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

        # SHAPE FEATURES - Enhanced pattern matching
        elif any(keyword in user_input_lower for keyword in ["shape", "hog", "embedding", "features", "resnet", "mobilenet", "corner", "fast", "geometric", "structural"]):
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

# Module-level variable to hold instance (lazy loaded)
_seher_chat = None

def _get_seher_chat():
    global _seher_chat
    if _seher_chat is None:
        _seher_chat = SeherSmartChat()
    return _seher_chat

def render_smart_chat():
    _get_seher_chat().render()

def update_chat_context(tab: str, details: list = None):
    _get_seher_chat().update_context(tab, details)

def add_chat_message(message: str):
    chat = _get_seher_chat()
    chat._ensure_initialized()
    st.session_state[chat.messages_key].append({
        "role": "assistant", "content": message, "timestamp": datetime.now()
    })
