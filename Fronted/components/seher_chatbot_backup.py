# Fronted/components/seher_chatbot.py
"""
Seher - AI Chatbot for explaining image processing features
Designed specifically for humanities users with no technical background
"""

import streamlit as st
from typing import Dict, List, Optional
import re


class SeherChatbot:
    """
    Seher AI Chatbot - Dynamic, context-aware explanations for all SehenLernen features
    """
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.field_mappings = self._build_field_mappings()
        self.conversation_history = []
    
    def _build_knowledge_base(self) -> Dict[str, Dict]:
    def _build_knowledge_base(self) -> Dict[str, Dict]:
        """
        Comprehensive knowledge base for ALL SehenLernen features and fields
        """
        return {
            # ============ HISTOGRAM ANALYSIS ============
            "histogram": {
                "title": "📊 Histogram Analysis",
                "what_is": """
                A histogram is a graph that shows the distribution of pixel brightness in your image.
                
                **Think of it like a census of brightness:**
                - X-axis: Brightness values from 0 (pure black) to 255 (pure white)
                - Y-axis: How many pixels have each brightness level
                - Peaks: The most common brightness values in your image
                
                **Visual mood indicators:**
                - Peak on left (0-85): Dark, moody image (like Caravaggio or film noir)
                - Peak in center (85-170): Balanced, evenly lit (typical portraits)
                - Peak on right (170-255): Bright, high-key (outdoor scenes)
                - Two peaks: High contrast (dramatic lighting with darks AND brights)
                """,
                "types": {
                    "black_and_white": "Shows overall brightness distribution regardless of color. Best for analyzing lighting and contrast.",
                    "colored": "Shows separate histograms for Red, Green, and Blue channels. Reveals color dominance and color temperature."
                },
                "fields": {
                    "histogram_type": {
                        "what": "Choose between Black & White or Colored histogram",
                        "why": "B&W simplifies analysis to just brightness. Colored reveals color biases (warm/cool tones).",
                        "values": ["B&W", "Colored"],
                        "tip": "Start with B&W to understand lighting, then use Colored to analyze color mood"
                    },
                    "all_images": {
                        "what": "Analyze all uploaded images at once",
                        "why": "Enables batch comparison across your entire collection",
                        "tip": "Check this when comparing lighting across multiple artworks or photos"
                    },
                    "image_selection": {
                        "what": "Pick a specific image to analyze",
                        "why": "Focus on one image for detailed study",
                        "tip": "Use this when you want in-depth analysis of a single piece"
                    }
                },
                "use_cases": [
                    "Compare lighting moods across different paintings",
                    "Analyze photographic exposure techniques over time",
                    "Study manuscript illumination brightness patterns",
                    "Identify similar lighting conditions in historical documents"
                ]
            },
            
            # ============ HOG (HISTOGRAM OF ORIENTED GRADIENTS) ============
            "hog": {
                "title": "📐 HOG - Histogram of Oriented Gradients",
                "what_is": """
                HOG captures the "shape signature" of objects by analyzing edge directions.
                
                **How it works - Simple analogy:**
                Imagine tracing the outlines of a building with arrows showing which way each wall faces.
                HOG does this digitally:
                1. Finds edges in the image (where brightness changes)
                2. Determines the direction each edge points (0°, 45°, 90°, 135°, etc.)
                3. Counts how many edges point in each direction
                4. Creates a "directional fingerprint" of the shape
                
                **Why it's powerful:**
                - Captures object structure without needing exact pixel values
                - Robust to lighting changes
                - Excellent for finding similar shapes even if rotated slightly
                """,
                "fields": {
                    "orientations": {
                        "what": "Number of direction bins (buckets) to group edge angles",
                        "why": "More bins = more detailed shape capture, but also more complex",
                        "values": {
                            "6 bins": "30° per bin - Fast, coarse shape capture. Good for simple objects.",
                            "9 bins": "20° per bin - DEFAULT. Best balance for most use cases.",
                            "12 bins": "15° per bin - More detailed. Good for complex architectural features.",
                            "18 bins": "10° per bin - Very detailed. For fine geometric analysis."
                        },
                        "range": "1-32 (typical: 6, 9, 12, or 18)",
                        "tip": "Start with 9 bins. Increase if you need more shape detail, decrease for speed."
                    },
                    "pixels_per_cell": {
                        "what": "Size of small patches (cells) to analyze, e.g., 8×8 pixels",
                        "why": "Smaller cells = finer details captured, larger cells = broader patterns",
                        "common_values": ["(8, 8) - DEFAULT, good for most images", "(16, 16) - Faster, coarser features", "(4, 4) - Very detailed, slower"],
                        "tip": "Use 8×8 for standard analysis, 4×4 for tiny details, 16×16 for large images"
                    },
                    "cells_per_block": {
                        "what": "How many cells to group together for normalization (e.g., 2×2)",
                        "why": "Grouping helps handle lighting variations between image regions",
                        "common_values": ["(2, 2) - DEFAULT, standard normalization", "(3, 3) - More robust to lighting", "(1, 1) - No grouping, faster"],
                        "tip": "Keep at (2, 2) unless you have extreme lighting variations"
                    }
                },
                "interpretation": {
                    "results": "HOG visualization shows gradient directions as patterns of lines and intensities",
                    "bright_areas": "Strong edges or shape boundaries detected",
                    "patterns": "Vertical lines = vertical structures, horizontal = horizontal structures, etc.",
                    "use_for": "Comparing architectural elements, analyzing compositions, matching similar shapes"
                },
                "use_cases": [
                    "Detect and compare architectural features across buildings",
                    "Analyze compositional structures in paintings",
                    "Match similar object shapes across different images",
                    "Extract structural features for classification"
                ]
            },
            
            # ============ K-MEANS CLUSTERING ============
            "kmeans": {
                "title": "🎨 K-means Clustering",
                "what_is": """
                K-means automatically groups your images into visual categories - like an AI curator!
                
                **Simple explanation:**
                Imagine sorting a box of colored candies by color. K-means does this with images:
                - You say "I want 3 groups" (k=3)
                - The computer examines all images
                - It finds the 3 most natural groupings based on visual similarity
                - Each image gets assigned to its closest group
                
                **What makes images similar:**
                - Color palettes (warm vs cool tones)
                - Brightness levels
                - Visual patterns and textures
                - Overall appearance
                """,
                "fields": {
                    "n_clusters": {
                        "what": "Number of groups (clusters) to create",
                        "why": "Controls how finely images are categorized",
                        "guidance": {
                            "2-3": "Broad categories (e.g., dark vs light images)",
                            "4-7": "Moderate grouping (most common use)",
                            "8-12": "Fine-grained categories",
                            "13+": "Very specific groupings (risk: groups become too small)"
                        },
                        "tip": "Start with 3-5 clusters. Too many = overly specific groups with few images each."
                    },
                    "random_state": {
                        "what": "A seed number for reproducibility",
                        "why": "Ensures you get the same grouping every time you run it",
                        "tip": "Leave as default. Only change if experimenting with different random starts."
                    },
                    "select_all_images": {
                        "what": "Include all uploaded images in clustering",
                        "why": "Analyze your entire collection at once",
                        "tip": "Check this to discover patterns across your full dataset"
                    },
                    "image_selection": {
                        "what": "Choose specific images to cluster",
                        "why": "Focus analysis on a subset of images",
                        "tip": "Use when testing or analyzing a specific group"
                    }
                },
                "interpretation": {
                    "visualization": "Scatter plot where each dot is an image",
                    "colors": "Same color = same cluster (similar images)",
                    "distance": "Closer dots = more similar images",
                    "patterns": "Clusters reveal natural visual groupings in your collection"
                },
                "use_cases": [
                    "Organize large collections of artworks or photographs automatically",
                    "Discover visual patterns you might have missed",
                    "Group historical documents by visual style",
                    "Find images with similar color palettes"
                ]
            },
            
            # ============ SIMILARITY SEARCH ============
            "similarity": {
                "title": "🔍 Image Similarity Search",
                "what_is": """
                Like Google image search for your collection - finds visually similar images!
                
                **How it works:**
                1. You select a "query image" (your search)
                2. Choose what aspect to compare (color, shape, texture, etc.)
                3. The system ranks all other images by similarity
                4. Most similar images appear first
                
                **Search by different features:**
                - Color: Find images with similar color schemes
                - Shape (HOG): Find similar compositions or objects
                - Texture: Find similar surface patterns
                - Overall (Histogram): Find similar brightness distributions
                """,
                "fields": {
                    "query_image": {
                        "what": "The image you want to find matches for",
                        "why": "This is your 'search query' - what to compare against",
                        "tip": "Choose an image with distinctive features you want to match"
                    },
                    "feature_method": {
                        "what": "Which visual aspect to compare",
                        "options": {
                            "Histogram": "Compare brightness distributions (lighting similarity)",
                            "HOG": "Compare shapes and structures (compositional similarity)",
                            "Color Histogram": "Compare color palettes (color mood)",
                            "SIFT": "Compare distinctive feature points (detail matching)",
                            "Deep Learning": "Compare overall visual content (AI-powered)"
                        },
                        "tip": "Use Histogram for lighting, HOG for composition, Color for palette matching"
                    },
                    "distance_metric": {
                        "what": "How to measure 'similarity'",
                        "options": {
                            "euclidean": "Straight-line distance - DEFAULT, intuitive",
                            "cosine": "Angle-based - good for normalized features",
                            "manhattan": "Grid-based distance - robust to outliers"
                        },
                        "tip": "Start with euclidean (most intuitive)"
                    },
                    "max_results": {
                        "what": "How many similar images to return",
                        "range": "1-20",
                        "tip": "5-10 results usually sufficient for analysis"
                    },
                    "threshold": {
                        "what": "Minimum similarity score (optional filter)",
                        "why": "Only show results above a certain similarity level",
                        "tip": "Leave unchecked initially to see the full range"
                    },
                    "hog_orientations": {
                        "what": "Number of direction bins for HOG search",
                        "when": "Only appears when Feature Method = HOG",
                        "default": 9,
                        "tip": "See HOG section for details on orientation bins"
                    },
                    "hist_bins": {
                        "what": "Number of brightness levels to compare",
                        "when": "Appears for Histogram-based searches",
                        "range": "8-256",
                        "tip": "64 bins is a good balance - detailed but not too slow"
                    }
                },
                "interpretation": {
                    "results": "Images ranked by similarity (most similar first)",
                    "scores": "Lower distance = more similar (0 = identical)",
                    "use_for": "Finding related artworks, discovering visual influences, grouping similar photos"
                },
                "use_cases": [
                    "Find paintings with similar compositions",
                    "Locate photos taken in similar lighting conditions",
                    "Discover visual connections across time periods",
                    "Identify similar decorative motifs"
                ]
            },
            
            # ============ TEXTURE ANALYSIS (HARALICK/GLCM) ============
            "texture": {
                "title": "🧵 Texture Analysis (Haralick & GLCM)",
                "what_is": """
                Analyzes surface patterns and "texture feel" in images - like digitally touching the surface!
                
                **What is texture?**
                - Smoothness: Like silk or glass (uniform pixel values)
                - Roughness: Like sandpaper or bark (varying pixel values)
                - Regularity: Like woven fabric (repeating patterns)
                - Randomness: Like gravel or clouds (chaotic patterns)
                
                **GLCM = Gray-Level Co-occurrence Matrix:**
                Examines how often pixel brightness values appear next to each other.
                - Smooth texture: Similar brightness values occur together
                - Rough texture: Different brightness values occur together
                """,
                "fields": {
                    "distances": {
                        "what": "How far apart to compare pixels (in pixels)",
                        "why": "Different distances capture different pattern scales",
                        "values": {
                            "1": "Fine, immediate texture (thread patterns)",
                            "2-5": "Medium texture (brush strokes, weave)",
                            "10+": "Coarse texture (large patterns)"
                        },
                        "tip": "Use [1, 2, 4] to capture multiple scales at once"
                    },
                    "angles": {
                        "what": "Directions to analyze texture patterns",
                        "why": "Textures can have directional bias (like wood grain)",
                        "values": {
                            "0°": "Horizontal patterns",
                            "45°": "Diagonal (SW to NE)",
                            "90°": "Vertical patterns",
                            "135°": "Diagonal (NW to SE)"
                        },
                        "tip": "Use all 4 angles [0, 45, 90, 135] for complete texture characterization"
                    },
                    "quantization_levels": {
                        "what": "Number of brightness levels to use",
                        "why": "Reduces complexity while preserving texture information",
                        "values": {
                            "16": "Fast, coarse texture analysis",
                            "32-64": "Good balance for most uses",
                            "256": "Maximum detail, slower processing"
                        },
                        "tip": "256 for precise analysis, 64 for general use, 16 for quick tests"
                    }
                },
                "measurements": {
                    "contrast": "High = rough/varied texture (sandpaper). Low = smooth (glass)",
                    "homogeneity": "High = regular, repeating (fabric weave). Low = chaotic (cracked paint)",
                    "energy": "High = simple, repetitive (stripes). Low = complex (natural textures)",
                    "correlation": "High = organized patterns (brickwork). Low = random (clouds)",
                    "dissimilarity": "Similar to contrast - texture variation",
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
            
            # ============ LBP (LOCAL BINARY PATTERNS) ============
            "lbp": {
                "title": "🔬 LBP - Local Binary Patterns",
                "what_is": """
                Identifies tiny, local texture patterns - like having a magnifying glass for micro-textures!
                
                **How it works - Simple analogy:**
                Imagine looking at a pixel and its surrounding neighbors:
                - Is each neighbor brighter or darker than the center?
                - Brighter = 1, Darker = 0
                - Creates a binary code (like 10110101) for each pixel
                - Same codes = same local pattern types
                
                **What it detects:**
                - Edges and corners at small scales
                - Repeating micro-patterns
                - Local texture primitives
                - Surface irregularities
                """,
                "fields": {
                    "radius": {
                        "what": "Size of the neighborhood to examine",
                        "why": "Controls the scale of patterns detected",
                        "values": {
                            "1": "Immediate neighbors (3×3 grid) - finest details",
                            "2": "Slightly larger neighborhood (5×5 area) - DEFAULT",
                            "3-4": "Medium-scale patterns",
                            "8+": "Large-scale local patterns"
                        },
                        "range": "1-16",
                        "tip": "Radius=2 works well for most texture analysis. Increase for coarser patterns."
                    },
                    "n_points": {
                        "what": "Number of surrounding points to compare",
                        "why": "More points = more detailed pattern encoding",
                        "values": {
                            "8": "Standard, fast - DEFAULT",
                            "16": "More detailed, common",
                            "24": "Very detailed, comprehensive"
                        },
                        "tip": "8 points sufficient for most uses. Use 16+ for complex textures."
                    },
                    "method": {
                        "what": "Which pattern types to keep",
                        "options": {
                            "default": "All patterns included",
                            "uniform": "Only common, stable patterns - RECOMMENDED",
                            "ror": "Rotation invariant - same pattern regardless of angle",
                            "var": "Variance-based patterns"
                        },
                        "tip": "'uniform' is best for most texture analysis - focuses on meaningful patterns"
                    }
                },
                "interpretation": {
                    "histogram": "Shows distribution of pattern types in the image",
                    "peaks": "Most common local patterns",
                    "similarity": "Similar histograms = similar micro-textures",
                    "use_for": "Texture classification, material identification, printing technique analysis"
                },
                "use_cases": [
                    "Analyze fine details in manuscript illuminations",
                    "Compare printing techniques or paper textures",
                    "Study surface patterns in archaeological artifacts",
                    "Identify similar decorative motifs across artworks",
                    "Classify materials by micro-texture"
                ]
            },
            
            # ============ CONTOUR EXTRACTION ============
            "contours": {
                "title": "✏️ Contour Extraction",
                "what_is": """
                Traces outlines and boundaries of objects - like automatically drawing around shapes!
                
                **What it does:**
                1. Converts image to black & white (thresholding)
                2. Finds edges where black meets white
                3. Traces these edges to create outlines
                4. Each closed outline = one contour
                
                **Output:**
                - Visual overlay showing detected boundaries
                - Bounding boxes around each shape
                - Area measurements for each contour
                - Hierarchical relationships (shape within shape)
                """,
                "fields": {
                    "mode": {
                        "what": "Which contours to retrieve",
                        "options": {
                            "RETR_EXTERNAL": "Only outermost boundaries - DEFAULT",
                            "RETR_LIST": "All contours as flat list",
                            "RETR_TREE": "Full hierarchy (nested shapes)",
                            "RETR_CCOMP": "Two-level hierarchy"
                        },
                        "tip": "Use RETR_EXTERNAL to avoid duplicate nested shapes"
                    },
                    "method": {
                        "what": "How to approximate the contour shape",
                        "options": {
                            "CHAIN_APPROX_SIMPLE": "Compress contour to key points - DEFAULT, efficient",
                            "CHAIN_APPROX_NONE": "Store all boundary points - precise but slow",
                            "CHAIN_APPROX_TC89_L1": "Teh-Chin chain approximation"
                        },
                        "tip": "CHAIN_APPROX_SIMPLE is best for most uses - good detail, fast"
                    },
                    "min_area": {
                        "what": "Ignore contours smaller than this pixel area",
                        "why": "Filters out noise and tiny artifacts",
                        "typical": "10-100 pixels",
                        "tip": "Start with 10, increase if too many small contours detected"
                    },
                    "return_bounding_boxes": {
                        "what": "Include rectangular boxes around each contour",
                        "why": "Useful for object detection and measurement",
                        "tip": "Enable to get (x, y, width, height) for each shape"
                    },
                    "return_hierarchy": {
                        "what": "Include shape nesting information",
                        "why": "Shows which shapes contain other shapes",
                        "tip": "Enable for complex images with nested objects"
                    }
                },
                "interpretation": {
                    "visualization": "Colored outlines overlaid on original image",
                    "count": "Number of distinct shapes/objects detected",
                    "areas": "Size of each shape in pixels",
                    "bounding_boxes": "Rectangular coordinates for each shape",
                    "use_for": "Object counting, composition analysis, shape extraction"
                },
                "use_cases": [
                    "Count objects in archaeological photos",
                    "Analyze spatial composition in paintings",
                    "Extract decorative motifs or patterns",
                    "Study architectural element arrangements",
                    "Measure and compare shape proportions"
                ]
            },
            
            # ============ SIFT & EDGE DETECTION ============
            "sift": {
                "title": "🎯 SIFT & Edge Detection",
                "what_is": """
                SIFT = Scale-Invariant Feature Transform
                
                Finds distinctive "landmark" points that are recognizable even when image is:
                - Rotated
                - Scaled (zoomed in/out)
                - Partially occluded
                
                **How it works:**
                1. Finds interest points (corners, blobs, edges)
                2. Describes the local area around each point
                3. Creates unique "fingerprints" for each point
                4. Can match these fingerprints across different images
                
                **Edge Detection:**
                Highlights boundaries where brightness changes sharply:
                - Object outlines
                - Structural elements
                - Transitions between regions
                """,
                "fields": {
                    "sift_features": {
                        "what": "Extract SIFT keypoints and descriptors",
                        "why": "Enables robust image matching and feature recognition",
                        "output": "List of keypoint coordinates and their descriptors",
                        "tip": "Use for finding matching details across different images"
                    },
                    "edge_detection": {
                        "what": "Detect edges using Sobel, Canny, or other methods",
                        "options": {
                            "Sobel": "Gradient-based, shows edge strength",
                            "Canny": "Multi-stage, clean edges - most popular",
                            "Laplacian": "Second derivative, sensitive to fine details"
                        },
                        "tip": "Canny produces cleanest edge maps"
                    },
                    "feature_matching": {
                        "what": "Find corresponding points between two images",
                        "why": "Identifies similar details or repeated motifs",
                        "output": "Lines connecting matched keypoints",
                        "tip": "Great for comparing two artworks or finding similar architectural features"
                    }
                },
                "use_cases": [
                    "Match architectural details across different buildings",
                    "Find repeated decorative motifs in art collections",
                    "Compare compositional elements between paintings",
                    "Identify similar structural features in manuscripts",
                    "Analyze edge patterns and linear structures"
                ]
            },
            
            # ============ IMAGE EMBEDDING ============
            "embedding": {
                "title": "🧠 Image Embeddings (Deep Learning)",
                "what_is": """
                Uses pre-trained neural networks to create a "semantic fingerprint" of your image.
                
                **What's an embedding?**
                A vector (list of numbers) that captures the image's meaning and content:
                - Not just pixels, but concepts (e.g., "outdoor", "portrait", "architecture")
                - Similar images have similar embeddings
                - Enables AI-powered similarity search
                
                **Available models:**
                - ResNet: Classic architecture, good for general images
                - VGG: Deep network, captures fine details
                - MobileNet: Lightweight, fast processing
                - EfficientNet: State-of-the-art accuracy
                """,
                "fields": {
                    "model": {
                        "what": "Which neural network to use",
                        "options": {
                            "ResNet50": "50-layer network, good balance",
                            "VGG16": "16-layer network, detailed features",
                            "MobileNetV2": "Efficient, fast",
                            "EfficientNetB0": "Modern, accurate"
                        },
                        "tip": "ResNet50 is a solid default choice"
                    }
                },
                "use_cases": [
                    "Semantic image similarity (finds conceptually similar images)",
                    "Pre-processing for machine learning classification",
                    "Finding images with similar content regardless of exact appearance",
                    "Creating feature vectors for clustering or search"
                ]
            },
            
            # ============ CLASSIFICATION ============
            "classification": {
                "title": "🏷️ Image Classification & Training",
                "what_is": """
                Train a custom AI model to categorize your images automatically!
                
                **Workflow:**
                1. Label your images (e.g., "portrait", "landscape", "architecture")
                2. Choose which features to extract (HOG, LBP, Histogram, etc.)
                3. Train a classifier (the AI learns patterns in each category)
                4. Predict labels for new, unlabeled images
                
                **When to use:**
                - Large collections needing automated categorization
                - Consistent visual patterns across categories
                - Repetitive classification tasks
                """,
                "fields": {
                    "labels": {
                        "what": "Category names for your images",
                        "tip": "Use consistent, meaningful names. Each image needs a label to train."
                    },
                    "train_split": {
                        "what": "Mark images as 'train' or 'test'",
                        "why": "Train data teaches the model, test data evaluates accuracy",
                        "typical": "70-80% train, 20-30% test",
                        "tip": "Ensure all categories represented in both train and test"
                    },
                    "feature_type": {
                        "what": "Which features to extract for classification",
                        "options": {
                            "Histogram": "Brightness distribution - good for lighting-based categories",
                            "HOG": "Shape structure - best for shape-based categories",
                            "LBP": "Texture patterns - best for texture-based categories",
                            "Haralick": "GLCM texture - advanced texture features",
                            "Embedding": "Deep learning - captures semantic content"
                        },
                        "tip": "Match feature type to what distinguishes your categories"
                    }
                },
                "use_cases": [
                    "Auto-categorize large photo archives",
                    "Classify manuscript types by visual features",
                    "Sort artworks by style or period",
                    "Identify artifact types in archaeology"
                ]
            }
        }
    
    def _build_field_mappings(self) -> Dict[str, List[str]]:
        """Map feature categories to their fields for dynamic lookup"""
        return {
            "histogram": ["histogram_type", "all_images", "image_selection"],
            "hog": ["orientations", "pixels_per_cell", "cells_per_block", "orientation_bins"],
            "kmeans": ["n_clusters", "random_state", "select_all_images", "image_selection"],
            "similarity": ["query_image", "feature_method", "distance_metric", "max_results", 
                          "threshold", "hog_orientations", "hist_bins"],
            "texture": ["distances", "angles", "quantization_levels"],
            "lbp": ["radius", "n_points", "method", "neighbors"],
            "contours": ["mode", "method", "min_area", "return_bounding_boxes", "return_hierarchy"],
            "sift": ["sift_features", "edge_detection", "feature_matching"],
            "embedding": ["model"],
            "classification": ["labels", "train_split", "feature_type"]
        }
    
    def analyze_question(self, question: str) -> Dict[str, any]:
        """
        Intelligently analyze user question to determine intent and relevant features
        """
        question_lower = question.lower()
        
        # Detect question type
        question_type = "general"
        if any(q in question_lower for q in ["what is", "what are", "what does", "explain"]):
            question_type = "what_is"
        elif any(q in question_lower for q in ["how do", "how to", "how can", "how does"]):
            question_type = "how_to"
        elif any(q in question_lower for q in ["why", "reason", "purpose"]):
            question_type = "why"
        elif any(q in question_lower for q in ["which", "what should", "recommend"]):
            question_type = "recommendation"
        elif "field" in question_lower or "parameter" in question_lower or "option" in question_lower:
            question_type = "field_explanation"
        
        # Detect feature category
        feature = None
        if any(term in question_lower for term in ["histogram", "brightness", "distribution"]):
            feature = "histogram"
        elif any(term in question_lower for term in ["hog", "oriented gradient", "orientation", "gradient"]):
            feature = "hog"
        elif any(term in question_lower for term in ["kmeans", "k-means", "clustering", "cluster", "group"]):
            feature = "kmeans"
        elif any(term in question_lower for term in ["similarity", "similar", "search", "find similar", "compare"]):
            feature = "similarity"
        elif any(term in question_lower for term in ["texture", "haralick", "glcm", "co-occurrence"]):
            feature = "texture"
        elif any(term in question_lower for term in ["lbp", "local binary", "binary pattern"]):
            feature = "lbp"
        elif any(term in question_lower for term in ["contour", "outline", "boundary", "shape extraction"]):
            feature = "contours"
        elif any(term in question_lower for term in ["sift", "keypoint", "edge", "edge detection"]):
            feature = "sift"
        elif any(term in question_lower for term in ["embedding", "deep learning", "neural", "resnet", "vgg"]):
            feature = "embedding"
        elif any(term in question_lower for term in ["classification", "classify", "train", "predict", "label"]):
            feature = "classification"
        
        # Detect specific field being asked about
        field = None
        for potential_field in ["orientations", "orientation bins", "bins", "pixels per cell", "cells per block",
                                "n_clusters", "clusters", "radius", "neighbors", "points", "method", "mode",
                                "distance", "metric", "threshold", "quantization", "levels"]:
            if potential_field in question_lower:
                field = potential_field.replace(" ", "_")
                break
        
        return {
            "question_type": question_type,
            "feature": feature,
            "field": field,
            "original": question
        }
    
    def generate_response(self, question: str) -> str:
                "title": "Histogram Analysis",
                "simple_explanation": """
                Think of a histogram like a survey of all the colors (or brightness levels) in your image. 
                
                **What it does:** It counts how many pixels in your image are very dark, medium, or very bright.
                
                **Why it's useful for humanities:** 
                - Compare lighting in different paintings or photographs
                - Analyze the mood or atmosphere of artworks (dark vs bright images)
                - Study photographic techniques across time periods
                - Identify similar lighting conditions in historical documents
                """,
                "fields_explanation": {
                    "histogram_type": "Choose 'Black and White' to analyze brightness only, or 'Colored' to see red, green, and blue separately",
                    "all_images": "Check this to analyze all your uploaded images at once, or leave unchecked to analyze just one",
                    "image_selection": "Pick which specific image you want to analyze if you didn't select 'all images'"
                },
                "how_to_interpret": """
                **Reading the results:**
                - **Left side of graph:** Dark areas of your image
                - **Right side of graph:** Bright areas of your image  
                - **Tall peaks:** Many pixels of that brightness level
                - **Short areas:** Few pixels of that brightness level
                
                **What different shapes mean:**
                - **Peak on left:** Dark, moody image (like a Caravaggio painting)
                - **Peak on right:** Bright, well-lit image (like an outdoor photograph)
                - **Peak in middle:** Balanced lighting (like most portrait paintings)
                - **Two peaks:** High contrast (both very dark and very bright areas)
                """
            },
            
            "kmeans_clustering": {
                "title": "K-means Clustering",
                "simple_explanation": """
                Imagine you're organizing a collection of paintings by their dominant colors. K-means clustering does this automatically!
                
                **What it does:** Groups your images based on their visual similarities, like an automated curator.
                
**Why it's useful for humanities:**
                - Automatically organize large collections of artworks or photographs
                - Discover visual patterns you might have missed
                - Group historical documents by visual style
                - Find images with similar color palettes or compositions
                """,
                "fields_explanation": {
                    "n_clusters": "How many groups do you want? Start with 3-5. More groups = more specific categories",
                    "random_state": "A technical setting - just leave it as is. It ensures you get the same results each time",
                    "select_all_images": "Check this to group ALL your images, or select specific ones to compare",
                    "image_selection": "Choose which images to include in the grouping analysis"
                },
                "how_to_interpret": """
                **Reading the results:**
                - Each dot represents one of your images
                - **Same color dots = similar images** that the computer grouped together
                - **Distance between dots:** How similar images are (closer = more similar)
                - **Different colored groups:** Distinct visual categories the computer found
                
                **What you can discover:**
                - Images with similar color schemes
                - Artworks from the same period or style
                - Photographs taken under similar conditions
                - Documents with similar visual layouts
                """
            },
            
            "shape_features": {
                "title": "Shape Feature Extraction", 
                "simple_explanation": """
                This tool analyzes the shapes, lines, and patterns in your images - like having a computer describe what it "sees."
                
                **What it does:** Identifies geometric patterns, edges, and structural elements in images.
                
                **Why it's useful for humanities:**
                - Compare architectural styles across different periods
                - Analyze composition techniques in paintings
                - Study geometric patterns in decorative arts
                - Identify similar structural elements in manuscripts or maps
                """,
                "fields_explanation": {
                    "method_hog": "HOG (Histogram of Gradients): Great for detecting overall shapes and object outlines - like identifying architectural features",
                    "method_sift": "SIFT: Finds distinctive points and features - excellent for matching similar details across different images",
                    "method_fast": "FAST: Quickly identifies corner points and sharp features - useful for geometric analysis"
                },
                "how_to_interpret": """
                **Reading HOG results:**
                - Shows the main directional patterns (vertical lines, horizontal lines, diagonals)
                - Bright areas = strong edges or shapes detected
                - Useful for comparing architectural elements or compositional structures
                
                **Reading SIFT results:**
                - Identifies unique points that can be matched across images
                - Great for finding similar details in different artworks
                - Each point represents a distinctive visual feature
                
                **Reading FAST results:**
                - Highlights corners and sharp transitions
                - Useful for analyzing geometric patterns and architectural details
                """
            },
            
            "similarity_search": {
                "title": "Image Similarity Search",
                "simple_explanation": """
                Like having a research assistant that can instantly find images similar to one you're studying!
                
                **What it does:** Compares images and finds the most visually similar ones in your collection.
                
                **Why it's useful for humanities:**
                - Find artworks with similar compositions or styles
                - Locate related historical photographs or documents
                - Discover visual influences between different artists or periods
                - Group related materials for comparative analysis
                """,
                "fields_explanation": {
                    "query_image": "The image you want to find matches for - your 'search query'",
                    "feature_method": "How to compare images: by color, shape, texture, or overall appearance",
                    "distance_metric": "How strictly to match - 'euclidean' is usually best for beginners",
                    "similarity_threshold": "How similar results must be (higher = more strict matching)"
                },
                "how_to_interpret": """
                **Reading the results:**
                - **Images shown:** Most similar to your query image
                - **Similarity scores:** Lower numbers = more similar (0 = identical)
                - **Order:** Usually arranged from most to least similar
                
                **What you can discover:**
                - Artworks with similar color palettes
                - Photographs taken in similar settings
                - Documents with comparable layouts
                - Images sharing compositional elements
                """
            },
            
            "texture_analysis": {
                "title": "Texture Analysis (Haralick & Co-occurrence)",
                "simple_explanation": """
                Analyzes the "feel" or surface patterns in images - like digitally touching the texture of a painting or fabric.
                
                **What it does:** Measures how smooth, rough, regular, or chaotic the surface patterns appear.
                
                **Why it's useful for humanities:**
                - Compare fabric textures in historical clothing or tapestries
                - Analyze brush stroke patterns in paintings
                - Study surface treatments in ceramics or sculptures
                - Compare paper textures in manuscripts or prints
                """,
                "fields_explanation": {
                    "distances": "How far apart to compare pixels - smaller numbers detect fine textures, larger numbers detect broader patterns",
                    "angles": "Different directions to analyze texture (0°, 45°, 90°, 135°) - covers all orientations",
                    "quantization_levels": "How many gray levels to use - 256 is most detailed, 16 is faster but less precise"
                },
                "how_to_interpret": """
                **Key texture measurements:**
                - **Contrast:** High = very rough or varied texture, Low = smooth or uniform
                - **Homogeneity:** High = regular, repeating patterns, Low = chaotic or random
                - **Energy:** High = simple, repetitive textures, Low = complex patterns
                - **Correlation:** High = organized patterns, Low = scattered or random textures
                
                **Practical applications:**
                - Compare canvas textures across different paintings
                - Analyze weaving patterns in textiles
                - Study surface treatments in archaeological artifacts
                """
            },
            
            "lbp_analysis": {
                "title": "Local Binary Patterns (LBP)",
                "simple_explanation": """
                Identifies tiny local patterns and textures - like having a magnifying glass that spots repeating micro-patterns.
                
                **What it does:** Looks at small neighborhoods of pixels to find characteristic local patterns.
                
                **Why it's useful for humanities:**
                - Analyze fine details in manuscript illuminations
                - Compare printing techniques or paper textures
                - Study surface patterns in archaeological artifacts
                - Identify similar decorative motifs across artworks
                """,
                "fields_explanation": {
                    "radius": "Size of the neighborhood to analyze - smaller = finer details, larger = broader patterns",
                    "n_points": "Number of surrounding points to compare - more points = more detailed analysis",
                    "method": "'uniform' focuses on the most common patterns, 'default' includes all patterns"
                },
                "how_to_interpret": """
                **Reading LBP results:**
                - **Histogram peaks:** Show the most common local patterns in your image
                - **Different peak patterns:** Indicate different types of local textures
                - **Similar histograms:** Suggest images have similar micro-textures
                
                **What you can discover:**
                - Consistent patterns in artistic techniques
                - Similar material properties across objects
                - Printing or manufacturing methods
                - Characteristic surface treatments
                """
            },
            
            "contour_extraction": {
                "title": "Contour Extraction",
                "simple_explanation": """
                Traces the outlines and edges of objects in your images - like having someone draw around all the important shapes.
                
                **What it does:** Finds and highlights the boundaries of objects, figures, and architectural elements.
                
                **Why it's useful for humanities:**
                - Analyze composition and spatial arrangements
                - Study architectural outlines and proportions
                - Compare figure poses or object shapes across artworks
                - Extract decorative patterns and motifs
                """,
                "fields_explanation": {
                    "mode": "What type of outlines to find - 'external' gets outer boundaries, 'tree' gets nested shapes",
                    "method": "How precise to be - 'simple' gives clean lines, 'accurate' preserves all details",
                    "min_area": "Ignore tiny shapes smaller than this size (reduces noise)"
                },
                "how_to_interpret": """
                **Reading contour results:**
                - **Colored outlines:** Show detected object boundaries
                - **Bounding boxes:** Rectangular frames around detected objects
                - **Contour count:** Number of separate shapes or objects found
                
                **What you can analyze:**
                - Compositional elements and their relationships
                - Object proportions and sizes
                - Spatial arrangements in scenes
                - Architectural elements and their organization
                """
            },
            
            "sift_edge_detection": {
                "title": "SIFT & Edge Detection",
                "simple_explanation": """
                Finds distinctive points and edges that make each image unique - like identifying landmarks in a visual landscape.
                
                **What it does:** Locates important visual features and edge patterns that can be used for matching and analysis.
                
                **Why it's useful for humanities:**
                - Match similar architectural details across buildings
                - Find repeated motifs in decorative arts
                - Compare compositional elements between artworks
                - Identify similar structural features in manuscripts
                """,
                "fields_explanation": {
                    "sift_features": "SIFT finds unique points that remain recognizable even if image is rotated or scaled",
                    "edge_detection": "Highlights important boundaries and transitions in the image",
                    "feature_matching": "Compares distinctive points between different images"
                },
                "how_to_interpret": """
                **Reading SIFT results:**
                - **Key points:** Distinctive features marked with circles or crosses
                - **Feature descriptors:** Mathematical descriptions of each unique point
                - **Matches between images:** Lines connecting similar features across images
                
                **Reading edge detection:**
                - **White lines:** Strong edges and boundaries detected
                - **Thick lines:** More prominent edges
                - **Connected patterns:** Show structural relationships
                
                **Applications:**
                - Finding architectural similarities across buildings
                - Matching decorative patterns between artworks
                - Identifying repeated elements in manuscript collections
                """
            }
        }
    
    def show_help_icon(self, feature_key: str, size: str = "small") -> bool:
        """
        Display a help icon that opens Seher when clicked
        Returns True if the help was requested
        """
        icon = "🤖" if size == "large" else "❓"
        help_key = f"seher_help_{feature_key}"
        
        if st.button(icon, key=help_key, help=f"Ask Seher about {feature_key.replace('_', ' ').title()}"):
            st.session_state[f"seher_active_{feature_key}"] = True
            return True
        return False
    
    def show_explanation(self, feature_key: str):
        """Display Seher's explanation for a specific feature"""
        if feature_key not in self.explanations:
            st.error(f"Sorry, I don't have information about {feature_key} yet!")
            return
        
        explanation = self.explanations[feature_key]
        
        # Seher's conversational interface
        st.markdown("### 🤖 Seher - Your AI Research Assistant")
        st.markdown("*Hi! I'm Seher, and I'm here to help you understand image analysis tools. Let me explain this feature in simple terms.*")
        
        st.markdown("---")
        
        # Main explanation
        st.markdown(f"## {explanation['title']}")
        st.markdown(explanation["simple_explanation"])
        
        # Interactive expandable sections
        with st.expander("📝 Understanding the Controls & Fields"):
            st.markdown("**What do all these options mean?**")
            for field, desc in explanation["fields_explanation"].items():
                st.markdown(f"- **{field.replace('_', ' ').title()}:** {desc}")
        
        with st.expander("📊 How to Read & Interpret Results"):
            st.markdown(explanation["how_to_interpret"])
        
        with st.expander("💡 Tips for Humanities Research"):
            if feature_key == "histogram_analysis":
                st.markdown("""
                **Research Ideas:**
                - Compare the lighting moods across different artists' works
                - Analyze how photographic exposure changed over decades
                - Study the brightness patterns in manuscript illuminations
                - Compare natural vs artificial lighting in historical photos
                """)
            elif feature_key == "kmeans_clustering":
                st.markdown("""
                **Research Ideas:**
                - Organize large digital archives automatically
                - Discover visual connections between artworks you hadn't noticed
                - Group historical photos by visual similarity
                - Find patterns in decorative art collections
                """)
            elif feature_key == "similarity_search":
                st.markdown("""
                **Research Ideas:**
                - Track artistic influences across time periods
                - Find related images in large digital collections
                - Compare compositions between different artists
                - Identify visual citations or references between works
                """)
            else:
                st.markdown("""
                **Research Ideas:**
                - Use these technical analyses to support visual arguments
                - Quantify visual differences you observe qualitatively
                - Find patterns across large image collections
                - Create data-driven humanities research
                """)
        
        # Close button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Thanks Seher, I understand now!", 
                        key=f"close_seher_{feature_key}", 
                        type="primary", 
                        width='stretch'):
                st.session_state[f"seher_active_{feature_key}"] = False
                st.rerun()
    
    def is_active(self, feature_key: str) -> bool:
        """Check if Seher is currently explaining a specific feature"""
        return st.session_state.get(f"seher_active_{feature_key}", False)
    
    def initialize_session_state(self):
        """Initialize all Seher-related session state variables"""
        features = [
            "histogram_analysis", "kmeans_clustering", "shape_features", 
            "similarity_search", "texture_analysis", "lbp_analysis",
            "contour_extraction", "sift_edge_detection"
        ]
        
        for feature in features:
            st.session_state.setdefault(f"seher_active_{feature}", False)


# Global instance
seher = SeherChatbot()