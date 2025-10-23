# Fronted/components/seher_chat_window_clean.py

import streamlit as st
import time
from datetime import datetime
import random

class SeherChatWindow:
    """AI Chat Assistant for SehenLernen - Clean Implementation"""
    
    def __init__(self):
        """Initialize Seher AI Chat Window"""
        # Initialize session state for chat functionality
        st.session_state.setdefault('seher_chat_open', False)
        st.session_state.setdefault('seher_chat_history', [])
        st.session_state.setdefault('seher_current_context', {"tab": "general", "fields": []})
        
        # Context-aware knowledge base for each feature
        self.feature_knowledge = {
            "histogram": {
                "simple": "Histograms show the distribution of pixel intensities in your images. Use them to understand brightness and contrast patterns.",
                "detailed": "Histograms are like fingerprints for images - they tell us about the distribution of light and color. If you're studying historical paintings, a histogram can reveal if the artist preferred dark or bright tones, or used a limited color palette.",
            },
            "kmeans": {
                "simple": "K-means clustering groups similar colors or patterns together. It's great for segmenting images into distinct regions.",
                "detailed": "K-means is like organizing a messy drawer - it groups similar things together. For images, it can separate sky from land, or group similar textures. The 'k' number tells it how many groups to make.",
            },
            "shape": {
                "simple": "Shape analysis helps identify and measure geometric features in your images like circles, edges, and contours.",
                "detailed": "Shape analysis is like being a detective for geometric patterns. It can find repeated architectural elements, count objects, or measure biological structures with mathematical precision.",
            },
            "similarity": {
                "simple": "Similarity search finds images that look alike based on visual features like color, texture, or shape.",
                "detailed": "Think of similarity search as having a visual memory like an art historian - it can find paintings with similar compositions, architectural styles, or even detect forgeries by comparing brushwork patterns.",
            },
            "texture": {
                "simple": "Texture analysis examines surface patterns and roughness in images, useful for material identification.",
                "detailed": "Texture analysis reads the 'skin' of images - it can distinguish between silk and burlap, smooth marble and rough stone, helping categorize materials or surfaces in your visual research.",
            },
            "lbp": {
                "simple": "Local Binary Patterns (LBP) detect micro-textures and patterns that are invisible to the naked eye.",
                "detailed": "LBP is like having a microscopic texture reader - it captures tiny patterns that our eyes miss, perfect for analyzing fabric weaves, skin textures, or surface deterioration in artifacts.",
            },
            "contours": {
                "simple": "Contour detection finds object boundaries and outlines in your images.",
                "detailed": "Contours trace the edges of objects like an artist's outline sketch. They're essential for measuring shapes, counting objects, or preparing images for further geometric analysis.",
            },
        }
    
    def render_chat_sidebar(self):
        """Main method to render the chat interface"""
        self._render_social_chat_widget()
    
    def _render_social_chat_widget(self):
        """Render a completely self-contained chat interface as a collapsible sidebar"""
        # Ensure session state is initialized
        if 'seher_chat_open' not in st.session_state:
            st.session_state.seher_chat_open = False
        if 'seher_chat_history' not in st.session_state:
            st.session_state.seher_chat_history = []
        if 'seher_current_context' not in st.session_state:
            st.session_state.seher_current_context = {"tab": "general", "fields": []}
            
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        # Map internal tab names to display names
        tab_display_names = {
            "histogram": "Histogram Analysis",
            "kmeans": "K-Means Clustering", 
            "shape": "Shape Analysis",
            "similarity": "Similarity Search",
            "texture": "Texture Analysis",
            "lbp": "LBP Analysis",
            "contours": "Contour Detection",
            "classifier": "Classifier Training",
            "general": "General"
        }
        
        current_tab_display = tab_display_names.get(current_tab.lower(), current_tab.title())
        
        # Use CSS to position chat as a fixed sidebar on the right
        st.markdown("""
        <style>
        .seher-chat-sidebar {
            position: fixed;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 350px;
            max-height: 80vh;
            background: white;
            border-radius: 15px 0 0 15px;
            box-shadow: -5px 0 20px rgba(0,0,0,0.15);
            z-index: 1000;
            border: 1px solid #e1e5e9;
        }
        
        .seher-chat-collapsed {
            width: 60px;
            height: 120px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a container that will be positioned as sidebar
        with st.container():
            # When collapsed, show only a compact toggle button
            if not st.session_state.seher_chat_open:
                if st.button(
                    f"🤖 Seher AI ▼\nOnline • {current_tab_display}",
                    key="seher_toggle_chat",
                    width='stretch',
                    type="primary"
                ):
                    st.session_state.seher_chat_open = True
                    st.rerun()
            
            # When expanded, show the complete chat interface
            else:
                # Create a bordered container for the chat
                with st.container():
                    # Header with close button
                    header_col1, header_col2 = st.columns([4, 1])
                    with header_col1:
                        st.markdown("**🤖 Seher AI** - " + f"Online • {current_tab_display}")
                    with header_col2:
                        if st.button("▲", key="seher_close_chat", help="Collapse chat"):
                            st.session_state.seher_chat_open = False
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Chat messages display area
                    with st.container():
                        if not st.session_state.seher_chat_history:
                            st.info(f"🎯 **Welcome!** I'm Seher, your guide for Sehen Lernen! I'm here to help you master this powerful image analysis tool. I can see you're exploring **{current_tab_display}** - ready to dive deep into what makes this feature amazing?")
                        
                        # Display chat history using Streamlit's chat message components
                        for message in st.session_state.seher_chat_history[-6:]:  # Show last 6 messages
                            if message["role"] == "user":
                                with st.chat_message("user"):
                                    st.write(message["content"])
                            else:
                                with st.chat_message("assistant"):
                                    if message.get("thinking"):
                                        st.write("🤔 *Seher is thinking...*")
                                    else:
                                        # Display the message content
                                        st.write(message["content"])
                                        
                                        if message.get('typing_delay'):
                                            st.caption(f"⏱️ {message['typing_delay']:.1f}s")
                    
                    # Chat input form - completely contained within this widget
                    with st.form("seher_isolated_chat_form", clear_on_submit=True):
                        user_input = st.text_area(
                            "Message Seher...",
                            placeholder=f"Ask anything about {current_tab} in natural language!",
                            height=60,
                            key="seher_isolated_input"
                        )
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            send_btn = st.form_submit_button("💬 Send", width='stretch', type="primary")
                        with col2:
                            help_btn = st.form_submit_button("❓ Help", width='stretch')
                        with col3:
                            clear_btn = st.form_submit_button("🗑️ Clear", width='stretch')
                        
                        # Handle form submissions within this isolated context
                        if send_btn and user_input.strip():
                            # Immediately add user message to show it right away
                            st.session_state.seher_chat_history.append({
                                "role": "user", 
                                "content": user_input.strip(),
                                "timestamp": datetime.now()
                            })
                            
                            # Add thinking indicator
                            st.session_state.seher_chat_history.append({
                                "role": "assistant",
                                "content": "🤔 *Seher is thinking...*", 
                                "timestamp": datetime.now(),
                                "thinking": True
                            })
                            
                            # Rerun to show user message and thinking immediately
                            st.rerun()
                            
                        elif help_btn:
                            question = f"What is {current_tab_display} and how do I use it?"
                            st.session_state.seher_chat_history.append({
                                "role": "user", 
                                "content": question,
                                "timestamp": datetime.now()
                            })
                            st.session_state.seher_chat_history.append({
                                "role": "assistant",
                                "content": "🤔 *Seher is thinking...*", 
                                "timestamp": datetime.now(),
                                "thinking": True
                            })
                            st.rerun()
                            
                        elif clear_btn:
                            st.session_state.seher_chat_history = []
                            st.rerun()
        
        # Handle the actual processing after rerun (when thinking indicator is shown)
        if (st.session_state.seher_chat_history and 
            len(st.session_state.seher_chat_history) >= 2 and
            st.session_state.seher_chat_history[-1].get("thinking") and
            st.session_state.seher_chat_history[-2]["role"] == "user"):
            
            # Get the user's question
            user_question = st.session_state.seher_chat_history[-2]["content"]
            
            # Remove thinking indicator
            st.session_state.seher_chat_history.pop()
            
            # Process the response with delay
            start_time = time.time()
            thinking_delay = random.uniform(1.5, 3.5)
            time.sleep(thinking_delay)
            
            response = self._generate_contextual_response(user_question)
            total_time = time.time() - start_time
            
            # Add real response
            st.session_state.seher_chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(), 
                "typing_delay": total_time
            })
            
            st.rerun()
    

    
    def _generate_contextual_response(self, user_input: str) -> str:
        """Generate engaging, context-aware responses as Seher, the Sehen Lernen expert"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general").lower()
        user_input_lower = user_input.lower()
        
        # Map internal tab names to display names
        tab_display_names = {
            "histogram": "Histogram Analysis",
            "kmeans": "K-Means Clustering", 
            "shape": "Shape Analysis",
            "similarity": "Similarity Search",
            "texture": "Texture Analysis",
            "lbp": "LBP Analysis",
            "contours": "Contour Detection",
            "classifier": "Classifier Training",
            "general": "General"
        }
        
        current_tab_display = tab_display_names.get(current_tab, current_tab.title())
        
        # Check for contextual questions about current tab
        is_tab_question = any(phrase in user_input_lower for phrase in [
            "what is on this tab", "current tab", "this section", "this feature", 
            "what is general", "what is histogram", "what is texture", "what is kmeans",
            "how do i use", "how to use", "what does this do", "explain this"
        ])
        
        # Check if question is about Sehen Lernen features
        sehen_lernen_keywords = [
            # core features
            "histogram", "kmeans", "similarity", "shape", "texture", "lbp", "contour", "sift", "edge",
            # common specific terms
            "hog", "haralick", "glcm", "co-occurrence", "cooccurrence", "classifier", "training", "predict", "model",
            # general
            "image", "analysis", "feature", "parameter", "setting", "result", "sehen", "lernen", "general"
        ]
        
        is_sehen_lernen_question = any(keyword in user_input_lower for keyword in sehen_lernen_keywords)
        is_greeting = any(word in user_input_lower for word in ["hi", "hello", "hey", "what can you do", "who are you"])
        
        # If not about Sehen Lernen, not a greeting, and not about current tab, redirect
        if not is_sehen_lernen_question and not is_greeting and not is_tab_question:
            return "I'm Seher, your dedicated Sehen Lernen assistant! 🎯 I only help with image analysis features in this tool. Ask me about histograms, k-means clustering, similarity search, shape analysis, texture analysis, LBP, contours, or any other Sehen Lernen features!"
        
        # Greeting and introduction responses (updated for new tool)
        if is_greeting:
            greetings = [
                f"Hello! I'm Seher, your expert guide for Sehen Lernen! 🎯 I'm here to help you master this powerful new image analysis tool. I can see you're currently in the **{current_tab.title()}** section - ready to explore what makes this feature amazing?",
                f"Hey there! 👋 Seher here - I'm your dedicated assistant for navigating Sehen Lernen! This cutting-edge tool has incredible capabilities, and I'm here to guide you through each feature. Since you're working with **{current_tab.title()}**, shall we dive in?",
                f"Welcome! I'm Seher, and I'm passionate about helping you unlock Sehen Lernen's potential! 🔍 This innovative image analysis platform is packed with powerful features. I notice you're in the **{current_tab.title()}** section - let me show you what makes it brilliant!"
            ]
            return random.choice(greetings)
        
        # First, check for specific feature questions regardless of current tab
        feature_keywords = {
            "histogram": ["histogram", "hist"],
            "kmeans": ["k-means", "kmeans", "clustering", "cluster"],
            "similarity": ["similarity", "similar", "search"],
            "shape": ["shape", "geometry", "hog", "sift", "edge", "edges"],
            "texture": ["texture", "haralick", "glcm", "co-occurrence", "cooccurrence", "gray level"],
            "lbp": ["lbp", "local binary", "pattern"],
            "contours": ["contour", "contours", "outline", "boundary"],
            "classifier": ["classifier", "training", "predict", "model"],
        }
        
        # Special handling for specific HOG/orientation questions
        if any(term in user_input_lower for term in ["orientation bins", "orientations", "hog bins", "gradient bins"]):
            return "🧭 **Orientation Bins in HOG - Simple Explanation!**\n\n**Think of it like a compass with direction buckets:**\n\n🔹 **What they are:** Bins that count edges pointing in similar directions\n🔹 **Default (9 bins):** 0°, 20°, 40°, 60°, 80°, 100°, 120°, 140°, 160°\n🔹 **Why 9?** Good balance between detail and efficiency\n\n**How to change:**\n• **6 bins** = 30° each (faster, less detail)\n• **12 bins** = 15° each (slower, more detail)\n• **18 bins** = 10° each (very detailed)\n\n**Pro tip:** Start with 9, increase if you need finer direction sensitivity for complex shapes!"
        
        # Find which feature is being asked about
        asked_feature = None
        for feature, keywords in feature_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                asked_feature = feature
                break
        
        # Handle specific feature questions - much more flexible matching
        if asked_feature:
            # Check for "what" questions (what is, what does, etc.)
            if any(phrase in user_input_lower for phrase in ["what is", "what does", "what's", "explain", "tell me about"]):
                responses = {
                    "histogram": "Ah, histograms! 📊 Think of them as your image's DNA fingerprint. They reveal the 'mood' of your images - are they bright and cheerful, dark and moody, or perfectly balanced? In Sehen Lernen, histograms show exactly how light and dark pixels are distributed across your image. Perfect for quality assessment and understanding brightness patterns!",
                    "texture": "Texture analysis uses Haralick/GLCM and co-occurrence statistics to quantify surface patterns: contrast, energy, homogeneity, correlation. Great for materials, brush strokes, fabrics, and surface condition.",
                    "kmeans": "K-means clustering is pure magic! ✨ It groups similar colors and patterns together, like having a super-organized assistant who sorts pixels by similarity. Perfect for color analysis, object segmentation, and simplifying complex images into meaningful groups!",
                    "similarity": "Similarity search is like having photographic memory! 🧠 It finds images that 'feel' similar based on visual features like colors, textures, and shapes. Great for organizing large collections or finding reference images with similar characteristics!",
                    "shape": "📐 **HOG (Histogram of Oriented Gradients) Explained!**\n\n**What it does:** HOG captures the 'shape DNA' of objects by counting edge directions!\n\n🔹 **Orientation bins:** Like compass directions (0°, 45°, 90°, 135°, etc.)\n🔹 **Default:** Usually 9 bins = 20° each (covers 180°)\n🔹 **How to change:** Look for 'orientations' parameter - try 6, 9, or 12\n🔹 **Cells & Blocks:** Image divided into small cells, then grouped into blocks\n\n**Simple analogy:** Imagine drawing arrows on all edges, then counting how many point each direction! Perfect for detecting people, cars, or any objects with consistent shapes.",
                    "lbp": "Local Binary Patterns (LBP) are like microscopic texture vision! 🔬 They detect tiny local patterns invisible to the naked eye - ideal for analyzing fabric weaves, paper grain, or surface deterioration.",
                    "contours": "Contour extraction finds object boundaries using thresholding and OpenCV findContours. Useful for counting objects, measuring shapes, and generating bounding boxes.",
                    "classifier": "Classifier training lets you label images, extract features (HOG/LBP/CNN), train a model (SVM/KNN/LogReg), and then predict on holdout or new images.",
                }
                return responses.get(asked_feature, f"Great question about {asked_feature}! Let me explain this powerful feature...")
            
            # Check for "how" questions (how does, how to, how do, etc.)
            elif any(phrase in user_input_lower for phrase in ["how does", "how do", "how to", "how it work", "how work"]):
                how_responses = {
                    "histogram": "Here's how histograms work in Sehen Lernen! 📊 1) Choose your histogram type (Black & White or Colored), 2) Select a single image or analyze your entire dataset, 3) Click 'Generate Histogram' and watch the magic! The graph shows pixel distribution - peaks on the left = dark images, peaks on the right = bright images, peaks in center = well-balanced. You can download results as ZIP files too!",
                    "texture": "For texture analysis: 1) Choose distances and angles, 2) Select properties (contrast, energy, homogeneity, correlation), 3) Resize optionally, 4) Run and compare metrics across images.",
                    "kmeans": "K-means strategy: Start with k=3-5 clusters, adjust the random state for consistency, then experiment! Lower k=broader groups, higher k=detailed clusters. Watch how your image transforms with different k values!",
                    "shape": "📐 **HOG Step-by-Step Process:**\n\n**1. Parameter Setup:**\n🔹 **Orientations:** Set number of direction bins (try 9, 12, or 16)\n🔹 **Pixels per cell:** Usually 8x8 or 16x16 pixels\n🔹 **Cells per block:** Typically 2x2 cells for normalization\n\n**2. How it works:**\n• Calculates gradients (edge strength & direction)\n• Bins gradients by orientation (like sorting arrows by direction)\n• Creates histogram for each cell\n• Normalizes across blocks to handle lighting\n\n**3. Changing parameters:**\n• **More orientations** = finer direction detail\n• **Smaller cells** = more local detail\n• **Larger blocks** = more robust to noise\n\n**Perfect for:** Object detection, shape recognition, and feature matching!"
                }
                how_responses["shape"] = (
                    "For shape features: 1) Pick method (HOG/SIFT/FAST), 2) Choose image(s), 3) For HOG, set orientations/cell/block sizes. "
                    "4) Run extraction; download CSV or view visualization to inspect edges and structure."
                )
                how_responses["classifier"] = (
                    "Classifier training: 1) Label images in the table, 2) Choose feature type and model, 3) Train classifier, "
                    "4) Review metrics and predictions, 5) Use Predict to run inference on selected images."
                )
                return how_responses.get(asked_feature, f"Let me walk you through how {asked_feature} works in Sehen Lernen...")
            
            # General feature question (just mentions the feature without what/how)
            else:
                return f"You're asking about **{asked_feature}** - excellent choice! 🎯 In Sehen Lernen, this is a powerful feature for image analysis. Want me to explain what {asked_feature} does, or would you prefer to know how to use it? I can dive deep into the technical details or give you a quick overview!"
        
        # Handle "What is General" specifically
        if "what is general" in user_input_lower:
            return "It looks like you're asking about the 'General' context! 🎯 This usually means the chat system hasn't detected which specific feature tab you're currently using. To get the most helpful responses, please navigate to one of the specific analysis tabs like **Histogram Analysis**, **K-Means Clustering**, **Texture Analysis**, etc. Once you're in a specific tab, I can provide detailed guidance about that feature!"
        
        # Handle tab-specific context questions
        if any(phrase in user_input_lower for phrase in ["what is on this tab", "current tab", "this section", "this feature", "what tab", "am i currently exploring"]):
            tab_explanations = {
                "histogram": f"You're in the **Histogram Analysis** section! 📊 This is where you analyze the distribution of pixel intensities in your images. You can generate histograms for individual images or your entire dataset, choose between black & white or colored analysis, and understand your images' brightness characteristics.",
                "texture": f"You're in the **Texture Analysis** section! 🔍 Here you can analyze surface patterns and material properties using advanced algorithms. Perfect for identifying different materials, surface roughness, and micro-patterns in your images.",
                "kmeans": f"You're in the **K-Means Clustering** section! ✨ This feature groups similar colors and patterns together, perfect for color analysis and image segmentation.",
                "classifier": f"You're in the **Classifier Training** section! 🧠 Label images, choose a feature extractor, train a model, and run predictions on selected images.",
                "general": f"You're currently in a **General** context! 🎯 This means I haven't detected which specific feature you're using. To get detailed help, please navigate to one of the analysis tabs like Histogram Analysis, K-Means Clustering, Texture Analysis, etc. Once there, I can provide specific guidance!"
            }
            return tab_explanations.get(current_tab.lower(), f"You're currently in the **{current_tab_display}** section! This feature helps you analyze specific aspects of your images.")
        
        # Handle follow-up responses
        if any(phrase in user_input_lower for phrase in ["go ahead", "continue", "tell me more", "show me", "yes please", "sure", "ok"]):
            if current_tab.lower() == "histogram":
                return "Perfect! Let me break down histograms for you! 📊\n\n**What they show:** Histograms display how many pixels have each brightness level in your image. Think of it as a 'brightness census' of your photo!\n\n**How to read them:**\n- Left side (0-85) = Dark pixels 🌑\n- Middle (85-170) = Mid-tones 🌗 \n- Right side (170-255) = Bright pixels ☀️\n\n**In Sehen Lernen:** Choose 'Black and White' for overall brightness or 'Colored' to see individual RGB channels. Try both single images and your whole dataset to spot patterns!"
            else:
                return f"Great! Let me explain **{current_tab_display}** in detail! This feature is designed to help you analyze specific aspects of your images. What would you like to know - the technical details, how to use the interface, or how to interpret the results?"
        
        # Handle very short/unclear questions
        if len(user_input.strip()) <= 3 or user_input_lower in ["what is", "what", "how", "help", "what do you actually do", "who are you"]:
            return f"I'm ready to help with **{current_tab_display}**! 🎯 Ask me specific questions like 'What is histogram analysis?', 'How do I interpret the results?', or 'What parameters should I use?' I love diving deep into the technical details!"
        
        # Context-specific feature explanations
        feature_explanations = {
            "histogram": {
                "explain": "Histograms show the distribution of pixel intensities! 📊 Think of them as revealing your image's 'personality' - bright, dark, or balanced. Perfect for quality assessment and understanding your dataset's characteristics.",
                "what": "Histograms are visual graphs showing how many pixels have each brightness level. Left = dark pixels, right = bright pixels, peaks = most common brightness values!",
                "how": "In Sehen Lernen: Choose your histogram type (B&W or Colored), select single image or analyze all images, then interpret the mountain-like shape to understand your image's lighting characteristics!"
            },
            "kmeans": {
                "explain": "K-means clustering is pure magic! ✨ I think of it as teaching the computer to see like an artist - grouping similar colors and patterns together. In my experience with Sehen Lernen, it's perfect for simplifying complex images, finding dominant color palettes, or segmenting regions. The 'K' is how many groups you want - start with 3-5 and watch your image transform!",
                "what": "K-means is like having a super-organized assistant who sorts everything by similarity! 🎨 It groups pixels that 'look alike' into clusters. I use it constantly for color analysis, object segmentation, and even artistic effects. It's fascinating how it can reduce thousands of colors to just a few meaningful groups!",
                "how": "My k-means strategy: Start with k=3 for basic groupings, then experiment! Watch how different k values reveal different aspects of your images. Lower k = broader groups, higher k = more detailed clusters. The random state ensures reproducible results - very important for scientific work!"
            },
            "similarity": {
                "explain": "Similarity search is like having photographic memory! 🧠 I use it to find images that 'feel' similar - same lighting, composition, or subject matter. Sehen Lernen's algorithms can spot visual patterns that even trained eyes might miss. Perfect for organizing large collections or finding reference images!",
                "what": "It's visual pattern matching at its finest! The tool compares features like colors, textures, and shapes to find images that share visual DNA. I love using it to discover unexpected connections between seemingly different images.",
                "how": "Pro tip: Adjust your threshold carefully! Too low = only nearly identical images, too high = everything matches. I usually start at 0.7 and fine-tune from there. The algorithm choice depends on what similarities matter most to you!"
            }
        }
        
        # Check for specific feature questions
        if current_tab in feature_explanations:
            responses = feature_explanations[current_tab]
            if any(word in user_input_lower for word in ["what", "explain"]):
                return responses.get("explain", responses.get("what", ""))
            elif "how" in user_input_lower:
                return responses.get("how", responses.get("explain", ""))
        
        # Parameter and settings questions
        if "parameter" in user_input_lower or "setting" in user_input_lower:
            return f"Great question! 🎛️ As someone who's fine-tuned thousands of analyses, I always tell users: parameters are like seasoning in cooking - start light, then adjust! For {current_tab}, I recommend beginning with defaults, then tweaking based on your specific images. What particular setting is puzzling you?"
        
        # Results interpretation
        if "result" in user_input_lower or "interpret" in user_input_lower:
            return f"Interpreting results is where the real magic happens! ✨ In {current_tab}, I look for patterns and trends across your entire dataset, not just individual images. The key is understanding what the numbers tell you about your research question. Want me to walk you through reading your specific results?"
        
        # Recommendations
        if "best" in user_input_lower or "recommend" in user_input_lower:
            return f"From my experience with Sehen Lernen, the 'best' approach always depends on your goals! 🎯 For {current_tab}, I'd suggest starting simple, then building complexity. What kind of images are you working with? Knowing your dataset helps me give you targeted advice!"
        
        # Current feature context
        context_responses = {
            "histogram": "I can see you're exploring histograms - my favorite starting point for any image analysis! Want to know how to spot the telltale signs of different image types, or shall we dive into advanced histogram techniques?",
            "kmeans": "K-means clustering - excellent choice! It's incredibly powerful for color analysis and segmentation. Shall I share my favorite k-means tricks, or do you have specific questions about the clustering process?",
            "similarity": "Similarity search is fascinating! I love how it can find hidden connections between images. Want to explore different similarity algorithms, or shall we talk about optimizing your search parameters?",
            "texture": "Texture analysis - now we're getting into the really sophisticated stuff! It's amazing how much you can learn about materials and surfaces. Ready to explore some advanced texture techniques?",
            "shape": "Shape analysis combines the best of geometry and AI! I'm excited to show you how it can quantify what your eyes see. Shall we start with traditional methods or dive into the deep learning approaches?",
        }
        
        if current_tab in context_responses:
            return context_responses[current_tab]
        
        # Default engaging response
        return f"I'm here as your Sehen Lernen expert! 🚀 I've mastered every feature in this tool and love sharing knowledge. Since you're in the **{current_tab.title()}** section, I'm ready to dive deep into advanced techniques, troubleshoot tricky parameters, or explain complex results. What would you like to explore together?"
    
    def update_context(self, tab_name: str, available_fields: list = None):
        """Update the current context for context-aware responses"""
        st.session_state.seher_current_context = {
            "tab": tab_name.lower(),
            "fields": available_fields or [],
            "timestamp": datetime.now()
        }

# Global instance
seher_chat = SeherChatWindow()
