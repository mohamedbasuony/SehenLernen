# Fronted/components/seher_chat_island.py
import streamlit as st
import time
import random
from datetime import datetime
from typing import Optional, Dict, Any

class SeherChatIsland:
    """
    Independent Facebook/LinkedIn style chat island for Seher AI
    """
    
    def __init__(self):
        self.chat_key = "seher_island_chat"
        self.history_key = "seher_island_history"
        self.expanded_key = "seher_island_expanded"
        self.context_key = "seher_island_context"
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize all session state variables for the chat island"""
        if self.expanded_key not in st.session_state:
            st.session_state[self.expanded_key] = False
            
        if self.history_key not in st.session_state:
            st.session_state[self.history_key] = []
            
        if self.context_key not in st.session_state:
            st.session_state[self.context_key] = {"tab": "general", "details": []}
    
    def update_context(self, tab: str, details: list = None):
        """Update the current context for intelligent responses"""
        st.session_state[self.context_key] = {
            "tab": tab,
            "details": details or []
        }
    
    def render_chat_island(self):
        """Render the complete chat island UI"""
        # Ensure session state is initialized
        self._init_session_state()
        
        # Chat bubble or expanded window
        if st.session_state.get(self.expanded_key, False):
            self._render_expanded_chat()
        else:
            self._render_chat_bubble()
    
    def _get_chat_island_css_unused(self) -> str:
        """Get the CSS for the chat island styling"""
        return """
        <style>
        /* Chat Island Container */
        .chat-island {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        
        /* Chat Bubble */
        .chat-bubble {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            border: 3px solid white;
        }
        
        .chat-bubble:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        }
        
        .chat-bubble-icon {
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        
        /* Online Indicator */
        .online-indicator {
            position: absolute;
            bottom: 4px;
            right: 4px;
            width: 16px;
            height: 16px;
            background: #00c851;
            border: 2px solid white;
            border-radius: 50%;
            animation: pulse-online 2s infinite;
        }
        
        @keyframes pulse-online {
            0% { box-shadow: 0 0 0 0 rgba(0, 200, 81, 0.7); }
            70% { box-shadow: 0 0 0 6px rgba(0, 200, 81, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 200, 81, 0); }
        }
        
        /* Expanded Chat Window */
        .chat-window {
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid #e1e8ed;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        /* Chat Header */
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
        }
        
        .chat-header-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .chat-avatar {
            width: 36px;
            height: 36px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 16px;
        }
        
        .chat-status {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        
        .chat-name {
            font-weight: 600;
            font-size: 15px;
        }
        
        .chat-online {
            font-size: 12px;
            opacity: 0.9;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: #00c851;
            border-radius: 50%;
        }
        
        .chat-close {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        
        .chat-close:hover {
            background: rgba(255,255,255,0.1);
        }
        
        /* Chat Messages Area */
        .chat-messages {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        /* Message Bubbles */
        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .message.user {
            align-self: flex-end;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .message-bubble {
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .message.user .message-bubble {
            background: #667eea;
            color: white;
        }
        
        .message.assistant .message-bubble {
            background: white;
            color: #333;
            border: 1px solid #e1e8ed;
        }
        
        .message-time {
            font-size: 11px;
            color: #8899a6;
            margin-top: 4px;
            text-align: right;
        }
        
        .message.assistant .message-time {
            text-align: left;
        }
        
        /* Thinking Indicator */
        .thinking-message {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: white;
            border-radius: 18px;
            border: 1px solid #e1e8ed;
            max-width: 80%;
            align-self: flex-start;
        }
        
        .thinking-dots {
            display: flex;
            gap: 4px;
        }
        
        .thinking-dots span {
            width: 6px;
            height: 6px;
            background: #667eea;
            border-radius: 50%;
            animation: thinking 1.4s infinite;
        }
        
        .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes thinking {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        
        /* Chat Input */
        .chat-input-area {
            padding: 16px;
            background: white;
            border-top: 1px solid #e1e8ed;
        }
        
        .chat-input {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #e1e8ed;
            border-radius: 20px;
            font-size: 14px;
            outline: none;
            resize: none;
            font-family: inherit;
            transition: border-color 0.2s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
        }
        
        .chat-input::placeholder {
            color: #8899a6;
        }
        
        /* Utility Classes */
        .hidden {
            display: none !important;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .chat-island {
                bottom: 10px;
                right: 10px;
            }
            
            .chat-window {
                width: calc(100vw - 20px);
                height: 70vh;
                max-width: 350px;
            }
        }
        </style>
        """
    
    def _render_chat_bubble(self):
        """Render the collapsed chat bubble"""
        # Create a simple floating chat button
        st.markdown("""
        <style>
        .stButton > button[kind="primary"] {
            position: fixed !important;
            bottom: 20px !important;
            right: 20px !important;
            z-index: 1000 !important;
            border-radius: 50% !important;
            width: 60px !important;
            height: 60px !important;
            font-size: 24px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🤖", key="seher_bubble_btn", help="💬 Chat with Seher AI", 
                    type="primary"):
            st.session_state[self.expanded_key] = True
            st.rerun()
    
    def _render_expanded_chat(self):
        """Render the expanded chat window"""
        current_context = st.session_state.get(self.context_key, {"tab": "general", "details": []})
        current_tab = current_context.get("tab", "general").title()
        
        # Create the chat window container
        with st.container():
            # Chat header
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 16px; border-radius: 12px 12px 0 0; 
                           display: flex; align-items: center; gap: 12px;">
                    <div style="width: 36px; height: 36px; background: rgba(255,255,255,0.2); 
                               border-radius: 50%; display: flex; align-items: center; 
                               justify-content: center; font-weight: bold;">🤖</div>
                    <div>
                        <div style="font-weight: 600; font-size: 15px;">Seher AI</div>
                        <div style="font-size: 12px; opacity: 0.9; display: flex; align-items: center; gap: 6px;">
                            <span style="width: 8px; height: 8px; background: #00c851; border-radius: 50%;"></span>
                            Online • {current_tab}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("✕", key="seher_close_btn", help="Close chat"):
                    st.session_state[self.expanded_key] = False
                    st.rerun()
            
            # Messages area
            messages_container = st.container()
            with messages_container:
                chat_history = st.session_state.get(self.history_key, [])
                
                # Display messages
                for message in chat_history:
                    if message.get("thinking"):
                        st.markdown("""
                        <div style="display: flex; align-items: center; gap: 8px; 
                                   padding: 10px 14px; background: white; border-radius: 18px; 
                                   border: 1px solid #e1e8ed; max-width: 80%; margin: 8px 0;">
                            <span>🤔</span>
                            <span style="font-style: italic; color: #8899a6;">Seher is thinking...</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        role = message["role"]
                        content = message["content"]
                        timestamp = message.get("timestamp", datetime.now())
                        
                        if role == "user":
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin: 8px 0;">
                                <div style="background: #667eea; color: white; padding: 10px 14px; 
                                           border-radius: 18px; max-width: 80%; font-size: 14px;">
                                    {content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin: 8px 0;">
                                <div style="background: white; color: #333; padding: 10px 14px; 
                                           border-radius: 18px; border: 1px solid #e1e8ed; 
                                           max-width: 80%; font-size: 14px; line-height: 1.4;">
                                    {content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Chat input
            with st.form("seher_island_input_form", clear_on_submit=True):
                user_input = st.text_area(
                    "",
                    placeholder=f"Message Seher about {current_tab}...",
                    height=60,
                    key="seher_island_input",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    send_btn = st.form_submit_button("💬 Send", width='stretch', type="primary")
                with col2:
                    help_btn = st.form_submit_button("❓ Help", width='stretch')
                
                if send_btn and user_input.strip():
                    # Add user message
                    if self.history_key not in st.session_state:
                        st.session_state[self.history_key] = []
                    st.session_state[self.history_key].append({
                        "role": "user",
                        "content": user_input.strip(),
                        "timestamp": datetime.now()
                    })
                    
                    # Add thinking indicator
                    if self.history_key not in st.session_state:
                        st.session_state[self.history_key] = []
                    st.session_state[self.history_key].append({
                        "role": "assistant",
                        "content": "Thinking...",
                        "timestamp": datetime.now(),
                        "thinking": True
                    })
                    
                    st.rerun()
                
                elif help_btn:
                    help_message = f"What is {current_tab} and how do I use it?"
                    if self.history_key not in st.session_state:
                        st.session_state[self.history_key] = []
                    st.session_state[self.history_key].append({
                        "role": "user",
                        "content": help_message,
                        "timestamp": datetime.now()
                    })
                    if self.history_key not in st.session_state:
                        st.session_state[self.history_key] = []
                    st.session_state[self.history_key].append({
                        "role": "assistant",
                        "content": "Thinking...",
                        "timestamp": datetime.now(),
                        "thinking": True
                    })
                    st.rerun()
        
        # Process any pending responses
        self._process_pending_response()
    
    def _process_pending_response(self):
        """Process any pending AI responses"""
        chat_history = st.session_state.get(self.history_key, [])
        
        # Check if the last message is a thinking indicator
        if chat_history and chat_history[-1].get("thinking"):
            # Remove thinking indicator
            thinking_msg = chat_history.pop()
            
            # Find the user message that triggered this response
            user_message = None
            for msg in reversed(chat_history):
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break
            
            if user_message:
                # Generate response
                response = self._generate_response(user_message)
                
                # Simulate thinking delay
                time.sleep(random.uniform(1.0, 2.5))
                
                # Add AI response
                if self.history_key not in st.session_state:
                    st.session_state[self.history_key] = []
                st.session_state[self.history_key].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(),
                    "typing_delay": random.uniform(1.5, 3.0)
                })
                
                st.rerun()
    
    def _generate_response(self, user_input: str) -> str:
        """Generate intelligent responses based on user input and context"""
        current_context = st.session_state[self.context_key]
        current_tab = current_context.get("tab", "general").lower()
        user_input_lower = user_input.lower()
        
        # Map tab names to display names
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
        
        # Check for Sehen Lernen related questions
        sehen_lernen_keywords = [
            "histogram", "kmeans", "similarity", "shape", "texture", "lbp", "contour", 
            "sift", "edge", "hog", "haralick", "glcm", "classifier", "training", 
            "image", "analysis", "feature", "parameter", "sehen", "lernen", "orientation", "bins"
        ]
        
        is_sehen_lernen_question = any(keyword in user_input_lower for keyword in sehen_lernen_keywords)
        is_greeting = any(word in user_input_lower for word in ["hi", "hello", "hey", "what can you do", "who are you"])
        
        # Redirect non-Sehen Lernen questions
        if not is_sehen_lernen_question and not is_greeting:
            return "I'm Seher, your dedicated Sehen Lernen assistant! 🎯 I specialize in helping with image analysis features. Ask me about histograms, k-means clustering, similarity search, shape analysis, texture analysis, or any other Sehen Lernen features!"
        
        # Handle greetings
        if is_greeting:
            greetings = [
                f"Hello! I'm Seher, your expert guide for Sehen Lernen! 🎯 Ready to explore the **{current_tab_display}** features?",
                f"Hey there! 👋 I'm here to help you master Sehen Lernen's powerful image analysis tools. I see you're working with **{current_tab_display}** - let's dive in!",
                f"Welcome! I'm Seher, and I'm passionate about helping you unlock Sehen Lernen's potential! 🔍 Currently in **{current_tab_display}** - what would you like to know?"
            ]
            return random.choice(greetings)
        
        # Special handling for specific HOG/orientation questions
        if any(term in user_input_lower for term in ["orientation bins", "orientations", "hog bins", "gradient bins"]):
            return "🧭 **Orientation Bins in HOG - Simple Explanation!**\n\n**Think of it like a compass with direction buckets:**\n\n🔹 **What they are:** Bins that count edges pointing in similar directions\n🔹 **Default (9 bins):** 0°, 20°, 40°, 60°, 80°, 100°, 120°, 140°, 160°\n🔹 **Why 9?** Good balance between detail and efficiency\n\n**How to change:**\n• **6 bins** = 30° each (faster, less detail)\n• **12 bins** = 15° each (slower, more detail)\n• **18 bins** = 10° each (very detailed)\n\n**Pro tip:** Start with 9, increase if you need finer direction sensitivity for complex shapes!"
        
        # Feature keyword mapping for intelligent detection
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
        
        # Find which feature is being asked about
        asked_feature = None
        for feature, keywords in feature_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                asked_feature = feature
                break
        
        # Handle specific feature questions with intelligent matching
        if asked_feature:
            # Check for "what" questions
            if any(phrase in user_input_lower for phrase in ["what is", "what does", "what's", "explain", "tell me about"]):
                responses = {
                    "histogram": "📊 **Histogram Analysis Made Simple!**\n\nThink of histograms like a **popularity chart for brightness levels**:\n\n🔹 **What it shows:** How many pixels are dark, medium, or bright\n🔹 **X-axis:** Brightness scale (0=black → 255=white)\n🔹 **Y-axis:** How many pixels have that brightness\n🔹 **Peaks:** Most common brightness levels in your image\n\n**Quick Tips:**\n- Peaks on left = darker image 🌑\n- Peaks on right = brighter image ☀️\n- Multiple peaks = good contrast 📸\n- Flat line = poor contrast 😴",
                    "texture": "🌊 **Texture Analysis Made Simple!**\n\nLike **feeling fabric with your eyes** - analyzes surface patterns!\n\n🔹 **What it detects:** Smooth vs rough, patterns, grain\n🔹 **Methods:** Haralick/GLCM compares pixel neighborhoods\n🔹 **Results:** Texture 'fingerprint' of your image\n🔹 **Perfect for:** Material classification, quality control\n\n**Examples:** \n- Smooth: glass, water, skin\n- Rough: bark, fabric, concrete\n- Patterned: wood grain, brick walls",
                    "kmeans": "🎨 **K-Means Clustering Made Simple!**\n\nImagine **sorting M&Ms by color** - that's k-means!\n\n🔹 **What it does:** Groups similar pixels together\n🔹 **K number:** How many groups you want (like 5 main colors)\n🔹 **Result:** Your image simplified to just those main colors\n🔹 **Magic:** Finds the most important colors automatically!\n\n**Real example:** Photo with 1000 colors → simplified to 5 main colors\n**Use cases:** Color palettes, art styles, data compression",
                    "similarity": "🔍 **Similarity Search Made Simple!**\n\nLike **Google Images 'find similar'** but for your dataset!\n\n🔹 **How it works:** Upload one image, find lookalikes\n🔹 **Smart matching:** Compares shapes, colors, patterns\n🔹 **Results:** Shows most similar images ranked by match %\n🔹 **Perfect for:** Finding duplicates, style matching, quality control\n\n**Example:** Upload a red car → finds all red cars in your dataset!",
                    "shape": "📐 **HOG (Histogram of Oriented Gradients) Explained!**\n\n**What it does:** HOG captures the 'shape DNA' of objects by counting edge directions!\n\n🔹 **Orientation bins:** Like compass directions (0°, 45°, 90°, 135°, etc.)\n🔹 **Default:** Usually 9 bins = 20° each (covers 180°)\n🔹 **How to change:** Look for 'orientations' parameter - try 6, 9, or 12\n🔹 **Cells & Blocks:** Image divided into small cells, then grouped into blocks\n\n**Simple analogy:** Imagine drawing arrows on all edges, then counting how many point each direction! Perfect for detecting people, cars, or any objects with consistent shapes.",
                    "lbp": "🔬 **LBP (Local Binary Patterns) Made Simple!**\n\nLike having **microscopic texture vision**!\n\n🔹 **What it does:** Compares each pixel with its 8 neighbors\n🔹 **Creates patterns:** Binary codes for tiny texture details\n🔹 **Perfect for:** Fabric analysis, surface inspection, material classification\n🔹 **Advantages:** Works well even with lighting changes\n\n**Simple process:** Look at center pixel → compare with neighbors → create binary pattern → histogram of all patterns",
                    "contours": "📋 **Contour Detection Made Simple!**\n\n**Like tracing object outlines automatically!**\n\n🔹 **What it finds:** Boundaries and edges of objects\n🔹 **How it works:** Finds connected pixels with similar intensity\n🔹 **Results:** Precise object outlines and shapes\n🔹 **Perfect for:** Counting objects, measuring sizes, quality inspection\n\n**Applications:** Count pills, measure parts, detect defects",
                    "classifier": "🎯 **Classifier Training Made Simple!**\n\n**Teach the AI to recognize your images!**\n\n🔹 **Process:** Label examples → extract features → train model → predict new images\n🔹 **Features:** HOG, LBP, or deep learning features\n🔹 **Models:** SVM, Random Forest, or Neural Networks\n🔹 **Goal:** Automatic image classification\n\n**Example:** Train on 'good' vs 'defective' products → automatically sort new items"
                }
                return responses.get(asked_feature, f"Great question about {asked_feature}! Let me explain this powerful feature...")
            
            # Check for "how" questions
            elif any(phrase in user_input_lower for phrase in ["how does", "how do", "how to", "how it work", "how work"]):
                how_responses = {
                    "histogram": "📊 **How Histograms Work in Sehen Lernen:**\n\n**Step-by-step:**\n1️⃣ Choose histogram type (B&W or Colored)\n2️⃣ Select single image or entire dataset\n3️⃣ Click 'Generate Histogram'\n4️⃣ Analyze the results!\n\n**Reading the graph:**\n• **Left peaks** = Dark image 🌑\n• **Right peaks** = Bright image ☀️\n• **Center peaks** = Well-balanced 📸\n• **Multiple peaks** = Good contrast\n• **Flat distribution** = Poor contrast\n\n**Pro tip:** Download results as ZIP for detailed analysis!",
                    "texture": "🌊 **How Texture Analysis Works:**\n\n**Process:**\n1️⃣ Choose distances (1, 2, 3 pixels)\n2️⃣ Select angles (0°, 45°, 90°, 135°)\n3️⃣ Pick properties (contrast, energy, homogeneity, correlation)\n4️⃣ Optional: resize for faster processing\n5️⃣ Run analysis and compare metrics!\n\n**Understanding results:**\n• **High contrast** = Varied textures\n• **High energy** = Uniform patterns\n• **High homogeneity** = Smooth surfaces\n• **High correlation** = Linear structures",
                    "kmeans": "🎨 **How K-Means Works:**\n\n**Smart strategy:**\n1️⃣ Start with k=3-5 clusters\n2️⃣ Adjust random state for consistency\n3️⃣ Experiment with different k values\n4️⃣ Watch your image transform!\n\n**Parameter effects:**\n• **Lower k** = Broader color groups\n• **Higher k** = More detailed clusters\n• **Random state** = Reproducible results\n\n**Try this:** Same image with k=3, k=5, k=10 to see the difference!",
                    "shape": "📐 **HOG Step-by-Step Process:**\n\n**1. Parameter Setup:**\n🔹 **Orientations:** Set number of direction bins (try 9, 12, or 16)\n🔹 **Pixels per cell:** Usually 8x8 or 16x16 pixels\n� **Cells per block:** Typically 2x2 cells for normalization\n\n**2. How it works:**\n• Calculates gradients (edge strength & direction)\n• Bins gradients by orientation (like sorting arrows by direction)\n• Creates histogram for each cell\n• Normalizes across blocks to handle lighting\n\n**3. Changing parameters:**\n• **More orientations** = finer direction detail\n• **Smaller cells** = more local detail\n• **Larger blocks** = more robust to noise\n\n**Perfect for:** Object detection, shape recognition, and feature matching!"
                }
                return how_responses.get(asked_feature, f"Let me walk you through how {asked_feature} works in Sehen Lernen...")
        
        # Handle follow-up responses
        if any(phrase in user_input_lower for phrase in ["go ahead", "continue", "tell me more", "show me", "yes please", "sure", "ok"]):
            if current_tab.lower() == "histogram":
                return "Perfect! Let me break down histograms for you! 📊\n\n**What they show:** Histograms display how many pixels have each brightness level in your image. Think of it as a 'brightness census' of your photo!\n\n**How to read them:**\n- Left side (0-85) = Dark pixels 🌑\n- Middle (85-170) = Mid-tones 🌗 \n- Right side (170-255) = Bright pixels ☀️\n\n**In Sehen Lernen:** Choose 'Black and White' for overall brightness or 'Colored' to see individual RGB channels. Try both single images and your whole dataset to spot patterns!"
            else:
                return f"Great! Let me explain **{current_tab_display}** in detail! This feature is designed to help you analyze specific aspects of your images. What would you like to know - the technical details, how to use the interface, or how to interpret the results?"
        
        # Handle very short/unclear questions
        if len(user_input.strip()) <= 3 or user_input_lower in ["what is", "what", "how", "help", "what do you actually do", "who are you"]:
            return f"I'm ready to help with **{current_tab_display}**! 🎯 Ask me specific questions like 'What is histogram analysis?', 'How do I interpret the results?', or 'What parameters should I use?' I love diving deep into the technical details!"
        
        # Default helpful response
        return f"I'm here to help with **{current_tab_display}**! 🎯 Ask me specific questions like 'How does this work?', 'What parameters should I use?', or 'How do I interpret the results?' I love diving deep into the technical details!"

# Global instance for easy access
seher_island = SeherChatIsland()

def render_seher_island():
    """Render the Seher chat island - call this from your main app"""
    seher_island.render_chat_island()

def update_seher_context(tab: str, details: list = None):
    """Update Seher's context - call this when switching tabs"""
    seher_island.update_context(tab, details)