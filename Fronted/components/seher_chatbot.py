# Fronted/components/seher_chatbot.py
"""
Seher - Dynamic AI Chatbot for SehenLernen
Comprehensive, context-aware explanations for all image processing features
"""

import streamlit as st
from typing import Dict, List, Optional


class SeherChatbot:
    """
    Seher AI Chatbot - Dynamic, intelligent explanations for all SehenLernen features
    """
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.conversation_history = []
    
    def _build_knowledge_base(self) -> Dict[str, Dict]:
        """
        Comprehensive knowledge base for ALL SehenLernen features
        """
        return {
            "histogram": {
                "title": "📊 Histogram Analysis",
                "what_is": """A graph showing pixel brightness distribution from 0 (black) to 255 (white).

**Think of it like a census of brightness:**
• X-axis: Brightness values (0=pure black, 128=medium gray, 255=pure white)
• Y-axis: How many pixels have each brightness level
• Peaks: The most common brightness values in your image

**Visual mood indicators:**
• Peak on left (0-85): Dark, moody image (like Caravaggio or film noir)
• Peak in center (85-170): Balanced, evenly lit (typical portraits)
• Peak on right (170-255): Bright, high-key (outdoor scenes)
• Two peaks: High contrast (dramatic lighting with darks AND brights)""",
                
                "features_in_app": {
                    "histogram_type": "Choose 'B&W' for overall brightness or 'Colored' for separate R, G, B channels. B&W simplifies to just lighting analysis. Colored reveals color temperature (warm/cool tones).",
                    "all_images": "Check this to analyze all uploaded images at once for batch comparison across your entire collection.",
                    "selected_image": "Focus on analyzing one specific image when you want detailed study of a single piece."
                },
                
                "interpretation": "Peak left=dark image, peak center=balanced, peak right=bright image, two peaks=high contrast",
                
                "use_cases": [
                    "Compare lighting moods across different paintings",
                    "Analyze photographic exposure techniques over time",
                    "Study manuscript illumination brightness patterns",
                    "Identify similar lighting conditions in historical documents"
                ]
            },
            
            "hog": {
                "title": "📐 HOG - Histogram of Oriented Gradients",
                "what_is": """HOG captures the "shape signature" of objects by analyzing edge directions.

**How it works - Simple analogy:**
Imagine tracing building outlines with arrows showing which way each wall faces. HOG does this digitally:
1. Finds edges in the image (where brightness changes sharply)
2. Determines the direction each edge points (0°, 45°, 90°, 135°, etc.)
3. Counts how many edges point in each direction
4. Creates a "directional fingerprint" of the shape

**Why it's powerful:**
• Captures object structure without needing exact pixel values
• Robust to lighting changes
• Excellent for finding similar shapes even if rotated slightly""",
                
                "features_in_app": {
                    "orientations": """**ORIENTATION BINS - Number of direction buckets:**

• **6 bins** = 30° per bin → Coarse shape capture, fastest processing
  Good for: Simple objects, quick analysis

• **9 bins** = 20° per bin → **DEFAULT - RECOMMENDED**
  Best balance for most use cases

• **12 bins** = 15° per bin → More detailed shape representation
  Good for: Complex architectural features

• **18 bins** = 10° per bin → Very detailed, captures subtle direction changes
  Good for: Fine geometric analysis

**What this means:** More bins give you a more precise "directional fingerprint" but take longer to process. Think of it like dividing a compass into more segments - 9 segments is usually perfect!""",
                    
                    "pixels_per_cell": """Size of small patches (cells) to analyze:
• (8, 8) - DEFAULT, good for most images
• (4, 4) - Captures very fine details, slower
• (16, 16) - Faster, coarser features

Smaller cells = finer details captured. Larger cells = broader patterns.""",
                    
                    "cells_per_block": """How many cells to group for normalization:
• (2, 2) - DEFAULT, standard normalization
• (3, 3) - More robust to lighting variations
• (1, 1) - No grouping, faster

Grouping helps handle lighting differences between image regions. Keep at (2, 2) unless you have extreme lighting variations."""
                },
                
                "orientation_bins_explained": """**🧭 ORIENTATION BINS - Deep Dive:**

Orientation bins divide 180° (or 360°) into equal angular segments.

**The bins divide up directions like slicing a pie:**
• 6 bins: Each slice is 30° wide (like cutting a pie into 6 pieces)
• 9 bins: Each slice is 20° wide (9 pieces) ← **MOST COMMON**
• 12 bins: Each slice is 15° wide (12 pieces)
• 18 bins: Each slice is 10° wide (18 pieces)

**Example with 9 bins (20° each):**
Bin 0: 0°-20° (horizontal edges pointing right)
Bin 1: 20°-40° (slightly diagonal)
Bin 2: 40°-60° (diagonal)
Bin 3: 60°-80° (mostly vertical)
Bin 4: 80°-100° (vertical edges pointing up)
...and so on

**Why it matters:**
• More bins = more precise direction measurement
• More bins = larger feature vector (more data to process)
• 9 bins is the sweet spot for most computer vision tasks

**Real-world analogy:**
Imagine describing a building's architecture. With 6 bins, you might say "it has vertical and horizontal lines." With 18 bins, you could say "it has lines at 15°, 35°, 75°, and 165°" - much more specific!""",
                
                "interpretation": "HOG visualization shows gradient directions as patterns. Bright areas = strong edges. Vertical lines = vertical structures, horizontal = horizontal structures.",
                
                "use_cases": [
                    "Detect and compare architectural features across buildings",
                    "Analyze compositional structures in paintings",
                    "Match similar object shapes across different images",
                    "Extract structural features for classification"
                ]
            },
            
            "kmeans": {
                "title": "🎨 K-means Clustering",
                "what_is": """K-means automatically groups your images into visual categories - like an AI curator!

**Simple explanation:**
Imagine sorting a box of colored candies by color. K-means does this with images:
• You say "I want 3 groups" (k=3)
• The computer examines all images
• It finds the 3 most natural groupings based on visual similarity
• Each image gets assigned to its closest group

**What makes images similar:**
Color palettes, brightness levels, visual patterns, textures, overall appearance""",
                
                "features_in_app": {
                    "n_clusters": """Number of groups (clusters) to create:
• 2-3 clusters: Broad categories (e.g., dark vs light images)
• 4-7 clusters: Moderate grouping - MOST COMMON USE
• 8-12 clusters: Fine-grained categories
• 13+ clusters: Very specific groupings (risk: groups become too small)

**Tip:** Start with 3-5 clusters. Too many = overly specific groups with few images each.""",
                    
                    "random_state": "A seed number for reproducibility. Ensures you get the same grouping every time. Leave as default unless experimenting.",
                    
                    "select_all_images": "Check this to include all uploaded images in clustering and discover patterns across your full dataset.",
                    
                    "image_selection": "Choose specific subset of images to cluster when testing or analyzing a specific group."
                },
                
                "interpretation": "Scatter plot where each dot is an image. Same color = same cluster (similar images). Closer dots = more similar images.",
                
                "use_cases": [
                    "Organize large collections of artworks automatically",
                    "Discover visual patterns you might have missed",
                    "Group historical documents by visual style",
                    "Find images with similar color palettes"
                ]
            },
            
            "similarity": {
                "title": "🔍 Image Similarity Search",
                "what_is": """Like Google image search for your collection - finds visually similar images!

**How it works:**
1. Select a "query image" (your search)
2. Choose what aspect to compare (color, shape, texture)
3. System ranks all other images by similarity
4. Most similar images appear first

You can search by color, shape (HOG), texture, brightness, or overall content.""",
                
                "features_in_app": {
                    "query_image": "The image you want to find matches for - this is your 'search query'. Choose one with distinctive features you want to match.",
                    
                    "feature_method": """Which visual aspect to compare:
• **Histogram**: Brightness distributions (lighting similarity)
• **HOG**: Shapes and structures (compositional similarity)
• **Color Histogram**: Color palettes (color mood matching)
• **SIFT**: Distinctive feature points (detail matching)
• **Deep Learning**: Overall visual content (AI-powered)

**Tip:** Use Histogram for lighting, HOG for composition, Color for palette matching.""",
                    
                    "distance_metric": """How to measure similarity:
• **euclidean**: Straight-line distance - DEFAULT, most intuitive
• **cosine**: Angle-based - good for normalized features
• **manhattan**: Grid-based distance - robust to outliers

**Tip:** Start with euclidean (most intuitive).""",
                    
                    "max_results": "How many similar images to return (1-20). **Tip:** 5-10 results usually sufficient for analysis.",
                    
                    "threshold": "Optional minimum similarity score. Only show results above this level. **Tip:** Leave unchecked initially to see the full range.",
                    
                    "hog_orientations": "**[Only appears when Feature Method = HOG]** Number of direction bins for HOG search. Default 9. See HOG section for detailed explanation of orientation bins.",
                    
                    "hist_bins": "**[Only appears for Histogram-based searches]** Number of brightness levels to compare (8-256). **Tip:** 64 bins is a good balance - detailed but not too slow."
                },
                
                "interpretation": "Results ranked by similarity (most similar first). Lower distance scores = more similar (0 = identical).",
                
                "use_cases": [
                    "Find paintings with similar compositions",
                    "Locate photos taken in similar lighting",
                    "Discover visual connections across time periods",
                    "Identify similar decorative motifs"
                ]
            },
            
            "texture": {
                "title": "🧵 Texture Analysis (Haralick & GLCM)",
                "what_is": """Analyzes surface patterns and "texture feel" in images - like digitally touching the surface!

**What is texture?**
• Smoothness: Like silk (uniform pixel values)
• Roughness: Like sandpaper (varying pixel values)
• Regularity: Like woven fabric (repeating patterns)
• Randomness: Like gravel (chaotic patterns)

**GLCM = Gray-Level Co-occurrence Matrix:**
Examines how often pixel brightness values appear next to each other.
• Smooth texture: Similar values occur together
• Rough texture: Different values occur together""",
                
                "features_in_app": {
                    "distances": """How far apart to compare pixels:
• 1 pixel: Fine, immediate texture (thread patterns)
• 2-5 pixels: Medium texture (brush strokes, weave)
• 10+ pixels: Coarse texture (large patterns)

**Tip:** Use [1, 2, 4] to capture multiple scales at once.""",
                    
                    "angles": """Directions to analyze texture patterns:
• 0°: Horizontal patterns
• 45°: Diagonal (SW to NE)
• 90°: Vertical patterns
• 135°: Diagonal (NW to SE)

**Tip:** Use all 4 angles [0, 45, 90, 135] for complete texture characterization. Textures can have directional bias (like wood grain).""",
                    
                    "quantization_levels": """Number of brightness levels to use:
• 16 levels: Fast, coarse texture analysis
• 32-64 levels: Good balance for most uses
• 256 levels: Maximum detail, slower processing

**Tip:** 256 for precise analysis, 64 for general use, 16 for quick tests."""
                },
                
                "measurements": {
                    "Contrast": "High = rough/varied texture (sandpaper), Low = smooth (glass)",
                    "Homogeneity": "High = regular, repeating (fabric weave), Low = chaotic (cracked paint)",
                    "Energy": "High = simple, repetitive (stripes), Low = complex (natural textures)",
                    "Correlation": "High = organized patterns (brickwork), Low = random (clouds)",
                    "Dissimilarity": "Similar to contrast - measures texture variation",
                    "ASM": "Angular Second Moment - texture uniformity"
                },
                
                "use_cases": [
                    "Compare fabric textures in historical clothing",
                    "Analyze brush stroke patterns in paintings",
                    "Study surface treatments in ceramics",
                    "Compare paper textures in manuscripts",
                    "Classify materials by surface patterns"
                ]
            },
            
            "lbp": {
                "title": "🔬 LBP - Local Binary Patterns",
                "what_is": """Identifies tiny, local texture patterns - like having a magnifying glass for micro-textures!

**How it works - Simple analogy:**
For each pixel, look at its surrounding neighbors:
• Is each neighbor brighter or darker than the center?
• Brighter = 1, Darker = 0
• Creates a binary code (like 10110101) for each pixel
• Same codes = same local pattern types

Detects edges, corners, repeating micro-patterns, and surface irregularities at small scales.""",
                
                "features_in_app": {
                    "radius": """Size of the neighborhood to examine:
• 1: Immediate neighbors (3×3 grid) - finest details
• 2: Slightly larger (5×5 area) - DEFAULT
• 3-4: Medium-scale patterns
• 8+: Large-scale local patterns

**Tip:** Radius=2 works well for most texture analysis. Increase for coarser patterns.""",
                    
                    "n_points": """Number of surrounding points to compare:
• 8 points: Standard, fast - DEFAULT
• 16 points: More detailed, common
• 24 points: Very detailed, comprehensive

**Tip:** 8 points sufficient for most uses. Use 16+ for complex textures.""",
                    
                    "method": """Which pattern types to keep:
• **default**: All patterns included
• **uniform**: Only common, stable patterns - RECOMMENDED
• **ror**: Rotation invariant (same pattern regardless of angle)
• **var**: Variance-based patterns

**Tip:** 'uniform' is best for most texture analysis - focuses on meaningful patterns."""
                },
                
                "interpretation": "Histogram shows distribution of pattern types. Peaks = most common local patterns. Similar histograms = similar micro-textures.",
                
                "use_cases": [
                    "Analyze fine details in manuscript illuminations",
                    "Compare printing techniques or paper textures",
                    "Study surface patterns in archaeological artifacts",
                    "Identify similar decorative motifs",
                    "Classify materials by micro-texture"
                ]
            },
            
            "contours": {
                "title": "✏️ Contour Extraction",
                "what_is": """Traces outlines and boundaries of objects - like automatically drawing around shapes!

**What it does:**
1. Converts image to black & white (thresholding)
2. Finds edges where black meets white
3. Traces these edges to create outlines
4. Each closed outline = one contour

**Output:** Visual overlay with detected boundaries, bounding boxes around shapes, area measurements, hierarchical relationships.""",
                
                "features_in_app": {
                    "mode": """Which contours to retrieve:
• **RETR_EXTERNAL**: Only outermost boundaries - DEFAULT
• **RETR_LIST**: All contours as flat list
• **RETR_TREE**: Full hierarchy (nested shapes)
• **RETR_CCOMP**: Two-level hierarchy

**Tip:** Use RETR_EXTERNAL to avoid duplicate nested shapes.""",
                    
                    "method": """How to approximate the contour shape:
• **CHAIN_APPROX_SIMPLE**: Compress to key points - DEFAULT, efficient
• **CHAIN_APPROX_NONE**: Store all boundary points - precise but slow
• **CHAIN_APPROX_TC89_L1**: Teh-Chin chain approximation

**Tip:** CHAIN_APPROX_SIMPLE is best for most uses - good detail, fast.""",
                    
                    "min_area": "Ignore contours smaller than this pixel area. Filters out noise and tiny artifacts. **Tip:** Start with 10, increase if too many small contours detected.",
                    
                    "return_bounding_boxes": "Include rectangular boxes around each contour. Useful for object detection. Gives (x, y, width, height) for each shape.",
                    
                    "return_hierarchy": "Include shape nesting information (which shapes contain other shapes). Enable for complex images with nested objects."
                },
                
                "interpretation": "Colored outlines overlaid on image. Count = number of distinct shapes detected. Areas = size of each shape in pixels.",
                
                "use_cases": [
                    "Count objects in archaeological photos",
                    "Analyze spatial composition in paintings",
                    "Extract decorative motifs or patterns",
                    "Study architectural element arrangements",
                    "Measure and compare shape proportions"
                ]
            }
        }
    
    def analyze_question(self, question: str) -> Dict[str, any]:
        """Intelligently analyze user question"""
        q_lower = question.lower()
        
        # Detect feature
        feature = None
        if any(t in q_lower for t in ["histogram", "brightness", "distribution"]):
            feature = "histogram"
        elif any(t in q_lower for t in ["hog", "oriented gradient", "orientation", "gradient"]):
            feature = "hog"
        elif "orientation bin" in q_lower or ("bin" in q_lower and "hog" in q_lower):
            feature = "hog"  # orientation bins are part of HOG
        elif any(t in q_lower for t in ["kmeans", "k-means", "clustering", "cluster"]):
            feature = "kmeans"
        elif any(t in q_lower for t in ["similarity", "similar", "search", "compare"]):
            feature = "similarity"
        elif any(t in q_lower for t in ["texture", "haralick", "glcm"]):
            feature = "texture"
        elif any(t in q_lower for t in ["lbp", "local binary"]):
            feature = "lbp"
        elif any(t in q_lower for t in ["contour", "outline", "boundary"]):
            feature = "contours"
        
        # Detect question type
        q_type = "general"
        if any(q in q_lower for q in ["what is", "what are", "explain"]):
            q_type = "what_is"
        elif any(q in q_lower for q in ["how", "use"]):
            q_type = "how_to"
        elif any(q in q_lower for q in ["field", "parameter", "option", "setting", "available"]):
            q_type = "field"
        
        # Check for orientation bins specifically
        is_orientation_bins = "orientation bin" in q_lower or ("bin" in q_lower and feature == "hog")
        
        return {
            "feature": feature,
            "type": q_type,
            "question": question,
            "is_orientation_bins": is_orientation_bins
        }
    
    def generate_response(self, question: str) -> str:
        """Generate intelligent, context-aware response"""
        analysis = self.analyze_question(question)
        feature = analysis["feature"]
        q_type = analysis["type"]
        is_orientation_bins = analysis.get("is_orientation_bins", False)
        
        if not feature:
            return self._general_help()
        
        kb = self.knowledge_base.get(feature, {})
        if not kb:
            return f"❓ I don't have information about '{feature}' yet. Try asking about: histogram, HOG, k-means, similarity, texture, LBP, or contours."
        
        # Build comprehensive response
        response = f"{kb.get('title', feature.upper())}\n\n"
        
        # Special handling for orientation bins
        if is_orientation_bins and feature == "hog":
            response += f"**What it is:**\n{kb['what_is']}\n\n"
            
            if "orientation_bins_explained" in kb:
                response += f"{kb['orientation_bins_explained']}\n\n"
            
            if "features_in_app" in kb and "orientations" in kb["features_in_app"]:
                response += f"**Available in SehenLernen:**\n{kb['features_in_app']['orientations']}\n\n"
            
            return response
        
        # What is it?
        if "what_is" in kb:
            response += f"**What it is:**\n{kb['what_is']}\n\n"
        
        # Fields/Parameters available in SehenLernen
        if "features_in_app" in kb and q_type == "field":
            response += "**🎛️ Available Fields in SehenLernen:**\n\n"
            for field_name, field_desc in kb["features_in_app"].items():
                response += f"**{field_name.replace('_', ' ').title()}:**\n{field_desc}\n\n"
        elif "features_in_app" in kb:
            response += "**Fields available in SehenLernen:**\n"
            for field_name in kb["features_in_app"].keys():
                response += f"• {field_name.replace('_', ' ').title()}\n"
            response += "\n*Ask me about specific fields for more details!*\n\n"
        
        # Interpretation guide
        if "interpretation" in kb:
            response += f"**How to interpret:**\n{kb['interpretation']}\n\n"
        
        # Measurements (for texture)
        if "measurements" in kb:
            response += "**Key measurements:**\n\n"
            for measure, desc in kb["measurements"].items():
                response += f"• **{measure}**: {desc}\n"
            response += "\n"
        
        # Use cases
        if "use_cases" in kb:
            response += "**💡 Use cases:**\n"
            for use_case in kb["use_cases"]:
                response += f"• {use_case}\n"
        
        return response
    
    def _general_help(self) -> str:
        """General help when feature not detected"""
        return """## 🤖 Seher - Your SehenLernen AI Assistant

I can explain all features in SehenLernen! Ask me about:

**📊 Analysis Features:**
• **Histogram** - Brightness distribution analysis
• **HOG** - Shape and gradient analysis (includes orientation bins!)
• **Texture (Haralick/GLCM)** - Surface pattern analysis
• **LBP** - Local binary patterns for micro-textures

**🔍 Comparison Features:**
• **K-means** - Automatic image clustering
• **Similarity Search** - Find visually similar images

**🎯 Other Features:**
• **Contours** - Object outline extraction

**Example questions:**
- "What is histogram and what features does SehenLernen have?"
- "Explain HOG and all its fields"
- "What are orientation bins in HOG?"
- "How do I use k-means clustering?"
- "What fields are available for texture analysis?"

Ask away! 🚀"""
    
    def chat_interface(self):
        """Render chat interface"""
        st.markdown("### 🤖 Seher AI Assistant")
        st.caption("Ask me about any SehenLernen feature!")
        
        # Initialize chat history
        if "seher_messages" not in st.session_state:
            st.session_state.seher_messages = []
        
        # Display chat history
        for message in st.session_state.seher_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about features, fields, or how to use something..."):
            # Add user message
            st.session_state.seher_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            response = self.generate_response(prompt)
            st.session_state.seher_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


# Global instance
seher = SeherChatbot()
