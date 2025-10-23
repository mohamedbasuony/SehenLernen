# Fronted/components/seher_chatbot.py
"""
Seher - Dynamic AI Chatbot for SehenLernen
Comprehensive, context-aware explanations for all image processing features
"""

import streamlit as st
from typing import Dict, List, Optional
import re


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
                "what_is": "A graph showing pixel brightness distribution from 0 (black) to 255 (white)",
                "features_in_app": {
                    "histogram_type": "Choose 'B&W' for overall brightness or 'Colored' for RGB channels",
                    "all_images": "Analyze all images at once for batch comparison",
                    "selected_image": "Focus on analyzing one specific image"
                },
                "interpretation": "Peak left=dark image, peak center=balanced, peak right=bright, two peaks=high contrast",
                "use_cases": ["Compare lighting across paintings", "Analyze photo exposure", "Study manuscript brightness"]
            },
            "hog": {
                "what_is": "Histogram of Oriented Gradients - captures shape by analyzing edge directions",
                "how_it_works": "Detects edges, calculates their directions (0°-180°), counts edges in each direction bin",
                "features_in_app": {
                    "orientations": "Direction bins: 6 bins (30° each, fast), 9 bins (20° each, DEFAULT), 12 bins (15° each, detailed), 18 bins (10° each, very detailed)",
                    "pixels_per_cell": "Cell size: 8×8 (DEFAULT, good balance), 4×4 (fine details), 16×16 (faster, coarser)",
                    "cells_per_block": "Normalization groups: 2×2 (DEFAULT), 3×3 (more robust to lighting)"
                },
                "orientation_bins_explained": """
Orientation bins divide 180° (or 360° for signed gradients) into equal angular segments.
- 6 bins = 30° per bin → Coarse shape capture, faster processing
- 9 bins = 20° per bin → **DEFAULT - Best balance for most images**
- 12 bins = 15° per bin → More detailed shape representation
- 18 bins = 10° per bin → Very detailed, captures subtle direction changes

Each bin counts how many edges point in that direction range. More bins = more precise shape "fingerprint" but slower processing.""",
                "use_cases": ["Detect architectural features", "Compare compositions", "Match similar shapes"]
            },
            "kmeans": {
                "what_is": "Automatic image grouping by visual similarity - like an AI curator",
                "how_it_works": "Examines images, finds natural visual categories, assigns each to most similar group",
                "features_in_app": {
                    "n_clusters": "Number of groups: 2-3 (broad), 4-7 (moderate, RECOMMENDED), 8-12 (fine-grained), 13+ (very specific)",
                    "random_state": "Seed for reproducibility - leave as default unless experimenting",
                    "select_all_images": "Include all images in clustering analysis",
                    "image_selection": "Choose specific subset to cluster"
                },
                "interpretation": "Scatter plot where same-colored dots = similar images, distance between dots = similarity",
                "use_cases": ["Auto-organize large collections", "Discover visual patterns", "Group by color palette"]
            },
            "similarity": {
                "what_is": "Find images visually similar to a query image - like Google Images for your collection",
                "features_in_app": {
                    "query_image": "The image to find matches for (your 'search query')",
                    "feature_method": "What to compare - Histogram (brightness), HOG (shape), Color Histogram (palette), SIFT (details), Deep Learning (overall content)",
                    "distance_metric": "euclidean (straight-line, DEFAULT), cosine (angle-based), manhattan (grid-based)",
                    "max_results": "How many similar images to return (typically 5-10)",
                    "threshold": "Optional minimum similarity cutoff",
                    "hog_orientations": "(Appears when HOG selected) Direction bins for shape comparison, default 9",
                    "hist_bins": "(Appears for Histogram methods) Number of brightness levels, default 64"
                },
                "interpretation": "Results ranked by similarity, lower scores = more similar (0=identical)",
                "use_cases": ["Find paintings with similar compositions", "Discover visual influences", "Group similar photos"]
            },
            "texture": {
                "what_is": "Analyzes surface patterns using GLCM (Gray-Level Co-occurrence Matrix) - Haralick features",
                "how_it_works": "Examines how often pixel values appear next to each other at various distances and angles",
                "features_in_app": {
                    "distances": "Pixel separation: 1 (fine texture), 2-5 (medium), 10+ (coarse). Use [1,2,4] for multi-scale",
                    "angles": "Directions: 0° (horizontal), 45° (diagonal), 90° (vertical), 135° (diagonal). Use all for complete analysis",
                    "quantization_levels": "Brightness levels: 16 (fast), 64 (balanced), 256 (detailed)"
                },
                "measurements": {
                    "contrast": "High=rough texture (sandpaper), Low=smooth (glass)",
                    "homogeneity": "High=regular patterns (fabric weave), Low=chaotic (cracked paint)",
                    "energy": "High=simple/repetitive (stripes), Low=complex (natural textures)",
                    "correlation": "High=organized (brickwork), Low=random (clouds)"
                },
                "use_cases": ["Compare fabric textures", "Analyze brush strokes", "Study ceramic surfaces", "Compare paper textures"]
            },
            "lbp": {
                "what_is": "Local Binary Patterns - identifies micro-textures by comparing each pixel to neighbors",
                "how_it_works": "For each pixel, checks if neighbors are brighter/darker → creates binary code → histogram of pattern types",
                "features_in_app": {
                    "radius": "Neighborhood size: 1 (finest, 3×3), 2 (DEFAULT, 5×5), 3-4 (medium), 8+ (large patterns)",
                    "n_points": "Number of comparison points: 8 (standard, DEFAULT), 16 (detailed), 24 (very detailed)",
                    "method": "Pattern types: 'default' (all), 'uniform' (common patterns, RECOMMENDED), 'ror' (rotation invariant), 'var' (variance-based)"
                },
                "interpretation": "Histogram shows pattern distribution; similar histograms = similar micro-textures",
                "use_cases": ["Analyze manuscript details", "Compare printing techniques", "Study artifact surfaces", "Classify materials"]
            },
            "contours": {
                "what_is": "Traces object outlines and boundaries - automatically draws around shapes",
                "how_it_works": "Converts to B&W, finds edges where black meets white, traces closed boundaries",
                "features_in_app": {
                    "mode": "RETR_EXTERNAL (outer boundaries only, DEFAULT), RETR_LIST (all as flat list), RETR_TREE (nested hierarchy)",
                    "method": "CHAIN_APPROX_SIMPLE (compress to key points, DEFAULT), CHAIN_APPROX_NONE (all points, precise)",
                    "min_area": "Ignore shapes smaller than this (pixels), typically 10-100 to filter noise",
                    "return_bounding_boxes": "Include rectangular boxes around each shape (x, y, width, height)",
                    "return_hierarchy": "Include shape nesting information (which shapes contain others)"
                },
                "interpretation": "Colored outlines + bounding boxes; count = number of detected shapes",
                "use_cases": ["Count objects", "Analyze composition", "Extract decorative motifs", "Study architectural arrangements"]
            },
            "sift": {
                "what_is": "Scale-Invariant Feature Transform - finds distinctive 'landmark' points recognizable across transformations",
                "how_it_works": "Detects interest points (corners, blobs), describes local area, creates unique fingerprints, matches across images",
                "features_in_app": {
                    "sift_features": "Extract keypoints and descriptors for robust matching",
                    "edge_detection": "Sobel (gradient-based), Canny (multi-stage, cleanest), Laplacian (fine details)",
                    "feature_matching": "Find corresponding points between two images - shows connecting lines"
                },
                "use_cases": ["Match architectural details", "Find repeated motifs", "Compare compositional elements", "Analyze edge patterns"]
            },
            "embedding": {
                "what_is": "Deep learning 'semantic fingerprint' - neural network captures image meaning, not just pixels",
                "how_it_works": "Pre-trained CNN extracts high-level features representing content and concepts",
                "features_in_app": {
                    "model": "ResNet50 (50-layer, balanced, DEFAULT), VGG16 (16-layer, detailed), MobileNetV2 (fast), EfficientNetB0 (accurate)"
                },
                "use_cases": ["Semantic similarity search", "Conceptually similar images", "ML pre-processing", "Content-based clustering"]
            },
            "classification": {
                "what_is": "Train custom AI to auto-categorize images based on labeled examples",
                "workflow": "Label images → Choose features → Train classifier → Predict new labels",
                "features_in_app": {
                    "labels": "Category names for each image (must label all training data)",
                    "train_split": "Mark 'train' (teaches model, ~70-80%) vs 'test' (evaluates accuracy, ~20-30%)",
                    "feature_type": "Histogram (lighting-based), HOG (shape-based), LBP (texture-based), Haralick (advanced texture), Embedding (semantic content)"
                },
                "use_cases": ["Auto-categorize photo archives", "Classify manuscript types", "Sort artworks by style", "Identify artifact types"]
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
        elif "orientation bin" in q_lower or "bins" in q_lower:
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
        elif any(t in q_lower for t in ["sift", "keypoint", "edge"]):
            feature = "sift"
        elif any(t in q_lower for t in ["embedding", "deep learning", "neural"]):
            feature = "embedding"
        elif any(t in q_lower for t in ["classification", "classify", "train", "label"]):
            feature = "classification"
        
        # Detect question type
        q_type = "general"
        if any(q in q_lower for q in ["what is", "what are", "explain"]):
            q_type = "what_is"
        elif any(q in q_lower for q in ["how", "use"]):
            q_type = "how_to"
        elif any(q in q_lower for q in ["field", "parameter", "option", "setting"]):
            q_type = "field"
        
        return {"feature": feature, "type": q_type, "question": question}
    
    def generate_response(self, question: str) -> str:
        """Generate intelligent, context-aware response"""
        analysis = self.analyze_question(question)
        feature = analysis["feature"]
        q_type = analysis["type"]
        
        if not feature:
            return self._general_help()
        
        kb = self.knowledge_base.get(feature, {})
        if not kb:
            return f"❓ I don't have information about '{feature}' yet. Try asking about: histogram, HOG, k-means, similarity, texture, LBP, contours, SIFT, embeddings, or classification."
        
        # Build comprehensive response
        response = f"## {kb.get('title', feature.upper())}\n\n"
        
        # What is it?
        if "what_is" in kb:
            response += f"**📚 What it is:**\n{kb['what_is']}\n\n"
        
        # Special handling for orientation bins (HOG)
        if "orientation bin" in question.lower() or ("hog" in question.lower() and "bin" in question.lower()):
            if feature == "hog" and "orientation_bins_explained" in kb:
                response += f"**🧭 Orientation Bins Explained:**\n\n{kb['orientation_bins_explained']}\n\n"
        
        # How it works
        if "how_it_works" in kb:
            response += f"**⚙️ How it works:**\n{kb['how_it_works']}\n\n"
        
        # Fields/Parameters available in SehenLernen
        if "features_in_app" in kb:
            response += "**🎛️ Available Fields in SehenLernen:**\n\n"
            for field_name, field_desc in kb["features_in_app"].items():
                response += f"• **{field_name.replace('_', ' ').title()}**: {field_desc}\n"
            response += "\n"
        
        # Interpretation guide
        if "interpretation" in kb:
            response += f"**📊 How to interpret:**\n{kb['interpretation']}\n\n"
        
        # Measurements (for texture)
        if "measurements" in kb:
            response += "**📏 Key measurements:**\n\n"
            for measure, desc in kb["measurements"].items():
                response += f"• **{measure.title()}**: {desc}\n"
            response += "\n"
        
        # Workflow (for classification)
        if "workflow" in kb:
            response += f"**🔄 Workflow:**\n{kb['workflow']}\n\n"
        
        # Use cases
        if "use_cases" in kb:
            response += "**💡 Use cases:**\n"
            for use_case in kb["use_cases"]:
                response += f"• {use_case}\n"
        
        return response
    
    def _general_help(self) -> str:
        """General help when feature not detected"""
        return """
## 🤖 Seher - Your SehenLernen AI Assistant

I can explain all features in SehenLernen! Ask me about:

**📊 Analysis Features:**
• **Histogram** - Brightness distribution analysis
• **HOG** - Shape and gradient analysis (includes orientation bins!)
• **Texture (Haralick/GLCM)** - Surface pattern analysis
• **LBP** - Local binary patterns for micro-textures

**🔍 Comparison Features:**
• **K-means** - Automatic image clustering
• **Similarity Search** - Find visually similar images
• **SIFT** - Keypoint detection and matching

**🧠 Advanced Features:**
• **Contours** - Object outline extraction
• **Embeddings** - Deep learning image representations
• **Classification** - Train custom image classifiers

**Example questions:**
- "What is histogram and what features does SehenLernen have?"
- "Explain HOG and all its fields"
- "What are orientation bins in HOG?"
- "How do I use k-means clustering?"
- "What fields are available for texture analysis?"

Ask away! 🚀
"""
    
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
