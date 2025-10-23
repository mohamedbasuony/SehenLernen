# Fronted/components/seher_chat_window.py
"""
Seher - AI Chat Window for explaining image processing features
Interactive chat interface for humanities users
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Tuple
import random

# Try to import AI text generation libraries
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SeherChatWindow:
    """
    Seher AI Chat Window - LinkedIn-style collapsible sidebar chat with context awareness
    """
    
    def __init__(self):
        # Initialize chat history in session state
        if 'seher_chat_history' not in st.session_state:
            st.session_state.seher_chat_history = []
            
        if 'seher_chat_open' not in st.session_state:
            st.session_state.seher_chat_open = False
            
        if 'seher_current_context' not in st.session_state:
            st.session_state.seher_current_context = {"tab": "general", "fields": []}
            
        # Initialize AI generator if available
        self.ai_enabled = self._initialize_ai()
        
        # Predefined knowledge base
        self.knowledge_base = {
            "histogram": {
                "basic": "A histogram shows how often different colors or brightness levels appear in your image. Think of it like a chart showing which colors are most common.",
                "detailed": "Histograms are like fingerprints for images - they tell us about the distribution of light and color. If you're studying historical paintings, a histogram can reveal if the artist preferred dark or bright tones, or used a limited color palette.",
                "examples": "For example, a portrait painted during the Renaissance might show peaks in flesh tones and dark backgrounds, while a modern abstract painting might have a more evenly distributed histogram."
            },
            "kmeans": {
                "basic": "K-means clustering groups similar colors together in your image. It's like sorting colored candies into separate bowls.",
                "detailed": "This technique helps identify the dominant color themes in artwork. For art historians, this could reveal an artist's signature palette or help authenticate works by comparing color usage patterns.",
                "examples": "If you're analyzing Van Gogh's paintings, k-means might show his characteristic blues and yellows clustered together, revealing his unique color relationships."
            },
            "shape": {
                "basic": "Shape features describe the outlines and forms in your image - like identifying circles, squares, or more complex shapes.",
                "detailed": "Shape analysis can help identify artistic styles, cultural symbols, or architectural elements across different time periods and cultures.",
                "examples": "In studying medieval manuscripts, shape analysis could help identify recurring decorative motifs or compare how different scribes drew similar symbols."
            },
            "similarity": {
                "basic": "Similarity search finds images that look alike based on their visual characteristics - colors, shapes, or patterns.",
                "detailed": "This is incredibly powerful for humanities research - you can find visual connections across vast collections that would take years to discover manually.",
                "examples": "You could find all paintings in a museum collection that use similar composition techniques, or discover how a particular artistic motif spread across different cultures."
            },
            "texture": {
                "basic": "Texture analysis looks at the surface patterns in images - whether something looks smooth, rough, striped, or dotted.",
                "detailed": "For humanities scholars, texture analysis can reveal artistic techniques, material properties, or degradation patterns in historical artifacts.",
                "examples": "In analyzing ancient textiles, texture analysis could help identify weaving patterns, or in studying paintings, reveal brushstroke techniques unique to specific artists."
            },
            "lbp": {
                "basic": "LBP (Local Binary Patterns) looks at tiny local patterns in images - like examining the weave of a fabric up close.",
                "detailed": "This technique is excellent for identifying surface textures and can help in authentication, condition assessment, or style analysis of artwork and artifacts.",
                "examples": "LBP analysis could help distinguish between original Renaissance frescoes and later restorations by analyzing the subtle texture differences in paint application."
            },
            "contours": {
                "basic": "Contour extraction finds the outlines and edges of objects in images - like tracing around shapes with a pencil.",
                "detailed": "For visual culture studies, contours help analyze composition, identify objects, and understand how artists create emphasis and focus.",
                "examples": "Contour analysis of historical portraits could reveal how different artists approached facial structure or how clothing styles evolved over time."
            },
            "sift": {
                "basic": "SIFT finds distinctive points and edges in images that remain consistent even when the image is rotated or scaled.",
                "detailed": "This is invaluable for matching details across different images or identifying specific elements that appear in multiple works.",
                "examples": "SIFT could help track how specific architectural details appear across different photographs of the same building taken over time, or identify recurring symbolic elements in different artworks."
            }
        }
        
    def _initialize_ai(self) -> bool:
        """Initialize AI text generation if available"""
        if not TRANSFORMERS_AVAILABLE:
            return False
            
        try:
            # Use a lightweight model for better performance
            self.generator = pipeline("text-generation", 
                                    model="gpt2",
                                    max_length=100,
                                    do_sample=True,
                                    temperature=0.7,
                                    pad_token_id=50256)
            return True
        except Exception as e:
            st.warning(f"AI text generation not available: {e}")
            return False
    
    def _generate_ai_response(self, user_message: str, context: str = "") -> str:
        """Generate AI response using the text generation model with current context"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general", "fields": []})
        current_tab = current_context.get("tab", "general")
        
        if not self.ai_enabled:
            return self._get_context_aware_fallback(user_message, current_tab)
            
        try:
            # Create a context-aware prompt
            context_info = f"Currently in {current_tab} tab. "
            if current_context.get("fields"):
                context_info += f"Available fields: {', '.join(current_context['fields'])}. "
            
            prompt = f"As an AI assistant for image analysis software, user is {context_info}User asks: {user_message} {context}. Respond helpfully:"
            
            # Generate response
            response = self.generator(prompt, max_length=150, num_return_sequences=1)
            generated_text = response[0]['generated_text']
            
            # Extract just the generated part (remove the prompt)
            ai_response = generated_text[len(prompt):].strip()
            
            # Clean up the response
            if ai_response:
                return ai_response
            else:
                return self._get_context_aware_fallback(user_message, current_tab)
                
        except Exception as e:
            return self._get_context_aware_fallback(user_message, current_tab)
    
    def _get_context_aware_fallback(self, user_message: str, current_tab: str) -> str:
        """Provide context-aware fallback responses when AI is not available"""
        user_lower = user_message.lower()
        
        # Context-specific responses based on current tab
        tab_contexts = {
            "histogram": "You're currently in the Histogram Analysis section. I can help explain histograms, color distributions, or how to interpret the charts.",
            "kmeans": "You're in the K-means Clustering section. I can explain how color clustering works or help with the cluster settings.",
            "shape": "You're in the Shape Features section. I can explain geometric properties, contours, or shape measurements.",
            "similarity": "You're in the Similarity Search section. I can help explain how image matching works or distance metrics.",
            "texture": "You're in the Texture Analysis section. I can explain texture patterns, Haralick features, or surface properties.",
            "lbp": "You're in the LBP Analysis section. I can explain Local Binary Patterns and texture recognition.",
            "contours": "You're in the Contour Extraction section. I can explain edge detection and object boundaries.",
            "sift": "You're in the SIFT Features section. I can explain keypoint detection and feature matching."
        }
        
        # Check if asking about current context
        if any(word in user_lower for word in ["this", "here", "current", "what is this"]):
            return tab_contexts.get(current_tab, "I can help explain any feature you're looking at. What would you like to know?")
        
        # Check for specific features mentioned
        for feature, explanations in self.knowledge_base.items():
            if feature in user_lower:
                return explanations["basic"]
        
        # Generic responses with current context
        if any(word in user_lower for word in ["what", "how", "why", "explain"]):
            context_hint = tab_contexts.get(current_tab, "")
            return f"{context_hint} What specifically would you like me to explain?"
        
        if any(word in user_lower for word in ["help", "start", "begin"]):
            return f"Welcome! I'm Seher, your AI guide. {tab_contexts.get(current_tab, 'I can explain any image analysis feature in simple terms.')}"
            
        return f"{tab_contexts.get(current_tab, 'I can help with any image analysis questions.')} What would you like to know?"
    
    def _get_detailed_explanation(self, topic: str) -> str:
        """Get detailed explanation for a specific topic"""
        topic_lower = topic.lower()
        
        for feature, explanations in self.knowledge_base.items():
            if feature in topic_lower:
                return f"{explanations['detailed']}\n\n{explanations['examples']}"
        
        return "I can provide detailed explanations about histograms, k-means clustering, shape analysis, similarity search, texture analysis, LBP, contours, and SIFT features. Which one interests you?"
    
    def render_chat_sidebar(self):
        """Render social media-style collapsible chat sidebar on the right"""
        # Add custom CSS for social media style chat
        st.markdown("""
        <style>
        /* Main chat container */
        .seher-chat-container {
            position: fixed;
            right: 0;
            bottom: 0;
            width: 350px;
            height: 500px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px 15px 0 0;
            box-shadow: -5px -5px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }
        
        /* Collapsed state */
        .seher-chat-collapsed {
            height: 60px !important;
            width: 300px !important;
        }
        
        /* Chat header */
        .seher-chat-header {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border-radius: 15px 15px 0 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .seher-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: white;
            font-weight: bold;
        }
        
        .seher-chat-title {
            flex: 1;
            margin-left: 12px;
            color: white;
            font-weight: 600;
        }
        
        .seher-status {
            font-size: 12px;
            color: rgba(255,255,255,0.8);
            margin-top: 2px;
        }
        
        .seher-toggle {
            color: white;
            font-size: 18px;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.2s;
        }
        
        .seher-toggle:hover {
            opacity: 1;
        }
        
        /* Chat content */
        .seher-chat-content {
            height: calc(100% - 120px);
            padding: 15px;
            overflow-y: auto;
            background: white;
        }
        
        /* Messages */
        .seher-message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .seher-message-user {
            justify-content: flex-end;
        }
        
        .seher-message-bubble {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .seher-message-user .seher-message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .seher-message-bot .seher-message-bubble {
            background: #f0f2f5;
            color: #333;
            border-bottom-left-radius: 4px;
            border: 1px solid #e4e6ea;
        }
        
        .seher-message-avatar {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: white;
            font-weight: bold;
            margin-right: 8px;
            flex-shrink: 0;
        }
        
        /* Typing indicator */
        .seher-typing {
            display: flex;
            align-items: center;
            padding: 10px 16px;
            background: #f0f2f5;
            border-radius: 18px;
            margin-left: 32px;
            max-width: fit-content;
        }
        
        .seher-typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #90949c;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .seher-typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .seher-typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        /* Input area */
        .seher-input-area {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 15px;
            border-top: 1px solid #e4e6ea;
        }
        
        /* Quick actions */
        .seher-quick-actions {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }
        
        .seher-quick-btn {
            padding: 6px 12px;
            background: #f0f2f5;
            border: 1px solid #e4e6ea;
            border-radius: 15px;
            font-size: 12px;
            color: #65676b;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .seher-quick-btn:hover {
            background: #e4e6ea;
        }
        
        /* Context indicator */
        .seher-context {
            font-size: 11px;
            color: #65676b;
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 10px;
            margin-left: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render the social chat widget on the right side
        self._render_social_chat_widget()
    

    
    def _generate_chat_content_html(self) -> str:
        """Generate HTML for chat messages"""
        html = '<div class="seher-chat-content" style="max-height: 300px; overflow-y: auto;">'
        
        # Welcome message if no history
        if not st.session_state.seher_chat_history:
            html += '''
            <div class="seher-message seher-message-bot">
                <div class="seher-message-avatar">🤖</div>
                <div class="seher-message-bubble">
                    Hi! I'm Seher, your AI assistant. I can see you're in the current section. Ask me anything! 👋
                </div>
            </div>
            '''
        
        # Display recent messages
        for message in st.session_state.seher_chat_history[-8:]:  # Show last 8 messages
            if message["role"] == "user":
                html += f'''
                <div class="seher-message seher-message-user">
                    <div class="seher-message-bubble">
                        {message['content']}
                    </div>
                </div>
                '''
            else:
                html += f'''
                <div class="seher-message seher-message-bot">
                    <div class="seher-message-avatar">🤖</div>
                    <div class="seher-message-bubble">
                        {message['content']}
                    </div>
                </div>
                '''
        
        # Show typing indicator if bot is "thinking"
        if st.session_state.get('seher_typing', False):
            html += '''
            <div class="seher-typing">
                <div class="seher-typing-dot"></div>
                <div class="seher-typing-dot"></div>
                <div class="seher-typing-dot"></div>
            </div>
            '''
        
        html += '</div>'
        return html
    
    def _generate_input_area_html(self) -> str:
        """Generate HTML for input area"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        return f'''
        <div class="seher-input-area">
            <div class="seher-quick-actions">
                <button class="seher-quick-btn" onclick="quickHelp('what')">❓ What is this?</button>
                <button class="seher-quick-btn" onclick="quickHelp('how')">🔧 How to use?</button>
                <button class="seher-quick-btn" onclick="quickHelp('explain')">📖 Explain more</button>
            </div>
        </div>
        '''
    
    def _handle_chat_input(self):
        """Handle chat input with typing simulation"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        # Input form
        with st.form("seher_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type a message...", 
                placeholder=f"Ask about {current_tab} features...",
                key="seher_chat_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                send_btn = st.form_submit_button("Send �", width='stretch')
            with col2:
                quick_help = st.form_submit_button("❓ Help", width='stretch')
            with col3:
                clear_btn = st.form_submit_button("🗑️ Clear", width='stretch')
            
            if send_btn and user_input.strip():
                self._process_user_input_with_typing(user_input.strip())
                st.rerun()
            elif quick_help:
                self._process_user_input_with_typing("What is this feature and how do I use it?")
                st.rerun()
            elif clear_btn:
                st.session_state.seher_chat_history = []
                st.rerun()
    
    def update_context(self, tab_name: str, available_fields: list = None):
        """Update the current context for context-aware responses"""
        st.session_state.seher_current_context = {
            "tab": tab_name.lower(),
            "fields": available_fields or [],
            "timestamp": datetime.now()
        }
    
    def get_context_hint(self, field_name: str = None) -> str:
        """Get contextual hint for specific field"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        field_hints = {
            "histogram": {
                "hist_type": "Choose between Black and White (grayscale) or Colored histogram analysis",
                "all_images": "Enable to generate histograms for all uploaded images at once",
                "selected_image": "Pick which specific image to analyze"
            },
            "kmeans": {
                "k_value": "Number of color clusters to group similar colors together",
                "all_images": "Apply k-means clustering to all images simultaneously"
            },
            "shape": {
                "shape_method": "Different algorithms for extracting geometric properties",
                "contour_method": "How to detect and measure object boundaries"
            },
            "similarity": {
                "similarity_method": "Algorithm used to compare images (color, texture, etc.)",
                "threshold": "How similar images need to be to match"
            }
        }
        
        if field_name and current_tab in field_hints and field_name in field_hints[current_tab]:
            return field_hints[current_tab][field_name]
        
        return f"This is part of the {current_tab} analysis. Ask me anything about it!"
    
    def _render_social_chat_widget(self):
        """Render a completely self-contained chat interface with no external leakage"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general").title()
        
        # Position the chat widget in bottom right corner using Streamlit components only
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # When collapsed, show only a compact toggle button
            if not st.session_state.seher_chat_open:
        <div style="
            position: fixed;
            right: 20px;
            bottom: 20px;
            width: 350px;
            height: {'60px' if chat_collapsed else '500px'};
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px 15px 0 0;
            box-shadow: -5px -5px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            overflow: hidden;
        ">
            <!-- Chat Header -->
            <div id="seher-header" style="
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                padding: 15px 20px;
                border-radius: 15px 15px 0 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: pointer;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <div style="
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    color: white;
                    font-weight: bold;
                ">🤖</div>
                <div style="flex: 1; margin-left: 12px; color: white;">
                    <div style="font-weight: 600;">Seher AI</div>
                    <div style="font-size: 12px; opacity: 0.8; margin-top: 2px;">Online • {current_tab}</div>
                </div>
                <div style="color: white; font-size: 18px; opacity: 0.8;">
                    {'▼' if chat_collapsed else '▲'}
                </div>
            </div>
        """
        
        # Add chat content if expanded
        if st.session_state.seher_chat_open:
            chat_html += """
            <!-- Chat Messages -->
            <div style="
                height: calc(100% - 120px);
                background: white;
                overflow-y: auto;
                padding: 15px;
            ">
            """
            
            # Display messages
            if not st.session_state.seher_chat_history:
                chat_html += f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 15px;">
                    <div style="
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 10px;
                        color: white;
                        font-weight: bold;
                        margin-right: 8px;
                        flex-shrink: 0;
                    ">🤖</div>
                    <div style="
                        background: #f0f2f5;
                        color: #333;
                        padding: 12px 16px;
                        border-radius: 18px;
                        border-bottom-left-radius: 4px;
                        font-size: 14px;
                        line-height: 1.4;
                        max-width: 80%;
                    ">
                        Hi! I'm Seher, your AI guide. I can see you're in the <strong>{current_tab}</strong> section. 
                        Ask me anything in natural language! 👋
                    </div>
                </div>
                """
            
            # Display chat history
            for message in st.session_state.seher_chat_history[-6:]:
                if message["role"] == "user":
                    chat_html += f"""
                    <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 15px;">
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 12px 16px;
                            border-radius: 18px;
                            border-bottom-right-radius: 4px;
                            font-size: 14px;
                            line-height: 1.4;
                            max-width: 80%;
                            word-wrap: break-word;
                        ">{message['content']}</div>
                    </div>
                    """
                else:
                    delay_info = ""
                    if message.get('typing_delay'):
                        delay_info = f"<small style='opacity: 0.6;'> ({message['typing_delay']:.1f}s)</small>"
                    
                    chat_html += f"""
                    <div style="display: flex; align-items: flex-start; margin-bottom: 15px;">
                        <div style="
                            width: 24px;
                            height: 24px;
                            border-radius: 50%;
                            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 10px;
                            color: white;
                            font-weight: bold;
                            margin-right: 8px;
                            flex-shrink: 0;
                        ">🤖</div>
                        <div style="
                            background: #f0f2f5;
                            color: #333;
                            padding: 12px 16px;
                            border-radius: 18px;
                            border-bottom-left-radius: 4px;
                            border: 1px solid #e4e6ea;
                            font-size: 14px;
                            line-height: 1.4;
                            max-width: 80%;
                            word-wrap: break-word;
                        ">{message['content']}{delay_info}</div>
                    </div>
                    """
            
            chat_html += "</div>"
        
        chat_html += "</div>"
        
        # Render the HTML
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Simple approach: Use Streamlit's native button styled as the header
        # Position it where the chat header would be
        col1, col2, col3, col4 = st.columns([2.5, 0.2, 0.2, 0.1])
        
        with col4:  # Position in bottom right
            # Style the button to look like the chat header
            css_styles = """
            <style>
            div[data-testid="stButton"] > button {
                position: fixed;
                right: 20px;
                bottom: 20px;
                width: 350px;
                height: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 15px 15px 0 0;
                color: white;
                font-weight: 600;
                z-index: 1000;
                box-shadow: -5px -5px 20px rgba(0,0,0,0.15);
            }
            </style>
            """
            st.markdown(css_styles, unsafe_allow_html=True)
            
            # The actual clickable button that looks like the header
            toggle_text = f"🤖 Seher AI {'▲' if st.session_state.seher_chat_open else '▼'}\nOnline • {current_tab}"
            
            if st.button(toggle_text, key="seher_clickable_header"):
                st.session_state.seher_chat_open = not st.session_state.seher_chat_open
                st.rerun()
        
        if st.button("�", key="seher_chat_toggle_btn_disabled", help="Toggle Seher Chat", type="secondary") and False:
            st.session_state.seher_chat_open = not st.session_state.seher_chat_open
            st.rerun()
        
            # Chat interface is now self-contained above - no floating input needed
        

    
    def _render_chat_messages(self):
        """Render chat messages in a clean format"""
        st.markdown("### Chat")
        
        # Display welcome message if no history
        if not st.session_state.seher_chat_history:
            current_context = st.session_state.get('seher_current_context', {"tab": "general"})
            current_tab = current_context.get("tab", "general").title()
            
            st.markdown(f"""
            <div style="
                background: #f0f2f5;
                padding: 10px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin-bottom: 10px;
            ">
                <strong>🤖 Seher:</strong> Hi! I'm your AI assistant for image analysis. 
                I can see you're in the <strong>{current_tab}</strong> section. 
                Feel free to ask me anything in natural language! 👋
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.seher_chat_history[-8:]:  # Show last 8 messages
            if message["role"] == "user":
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 10px;
                    margin: 5px 0 5px 20px;
                    text-align: right;
                ">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                delay_text = ""
                if message.get('typing_delay'):
                    delay_text = f"<small style='opacity: 0.7;'> ({message['typing_delay']:.1f}s)</small>"
                
                st.markdown(f"""
                <div style="
                    background: #f0f2f5;
                    padding: 8px 12px;
                    border-radius: 10px;
                    margin: 5px 20px 5px 0;
                    border-left: 3px solid #667eea;
                ">
                    <strong>🤖 Seher:</strong> {message['content']}{delay_text}
                </div>
                """, unsafe_allow_html=True)
    
    def _render_natural_input(self):
        """Render natural language input for free-form questions"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        st.markdown("---")
        
        # Natural language input
        with st.form("natural_chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask Seher anything in natural language:",
                placeholder=f"e.g., 'What does this {current_tab} feature do?', 'How do I interpret the results?', 'What are the best settings for my images?'",
                height=80,
                key="natural_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                send_btn = st.form_submit_button("Send Message 💬", width='stretch', type="primary")
            with col2:
                clear_btn = st.form_submit_button("Clear 🗑️", width='stretch')
            
            if send_btn and user_input.strip():
                self._process_natural_language_input(user_input.strip())
                st.rerun()
            elif clear_btn:
                st.session_state.seher_chat_history = []
                st.rerun()
        
        # Context help
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            padding: 8px;
            border-radius: 5px;
            font-size: 12px;
            color: #667eea;
            text-align: center;
            margin-top: 10px;
        ">
            💡 I understand natural language! Ask me about {current_tab} features, 
            how to use settings, interpret results, or anything else.
        </div>
        """, unsafe_allow_html=True)
    
    def _render_floating_input(self):
        """Render floating input area for the chat"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        # Create a floating input positioned at bottom right
        with st.container():
            # Position input in the corner with custom styling
            st.markdown("""
            <style>
            .seher-floating-input {
                position: fixed;
                right: 40px;
                bottom: 40px;
                width: 310px;
                background: white;
                border: 1px solid #e4e6ea;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 1001;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Input form
            with st.form("seher_floating_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Message Seher...",
                    placeholder=f"Ask anything about {current_tab} in natural language!",
                    height=60,
                    key="seher_floating_input"
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    send_btn = st.form_submit_button("💬 Send", width='stretch', type="primary")
                with col2:
                    clear_btn = st.form_submit_button("🗑️", width='stretch')
                
                if send_btn and user_input.strip():
                    self._process_natural_language_input(user_input.strip())
                    st.rerun()
                elif clear_btn:
                    st.session_state.seher_chat_history = []
                    st.rerun()
    
    def _render_chat_input_inline(self):
        """Render input area inline with the chat"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        
        # Input form
        with st.form("seher_inline_form", clear_on_submit=True):
            st.markdown("---")
            
            user_input = st.text_area(
                "Ask Seher anything:",
                placeholder=f"e.g., 'What does {current_tab} do?', 'How do I use this?', 'What are the best settings?'",
                height=60,
                key="seher_inline_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                send_btn = st.form_submit_button("💬 Send Message", width='stretch', type="primary")
            with col2:
                clear_btn = st.form_submit_button("🗑️ Clear", width='stretch')
            
            if send_btn and user_input.strip():
                self._process_natural_language_input(user_input.strip())
                st.rerun()
            elif clear_btn:
                st.session_state.seher_chat_history = []
                st.rerun()
        
        # Helper text
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            padding: 8px;
            border-radius: 8px;
            font-size: 12px;
            color: #667eea;
            text-align: center;
            margin: 10px 15px;
        ">
            💡 I understand natural language! Ask me about {current_tab} features, 
            how to use settings, interpret results, or anything else.
        </div>
        """, unsafe_allow_html=True)

    
    def _process_user_input_with_typing(self, user_input: str):
        """Process user input with realistic typing simulation"""
        import time
        import random
        
        # Add user message to history
        st.session_state.seher_chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Generate response based on input
        if "more" in user_input.lower() or "detail" in user_input.lower():
            response = self._get_detailed_explanation(user_input)
        else:
            response = self._generate_ai_response(user_input)
        
        # Simulate realistic typing delay based on response length
        response_length = len(response)
        base_delay = 1.0  # Minimum 1 second
        length_factor = response_length / 100  # Extra time based on length
        random_factor = random.uniform(0.5, 1.5)  # Add some randomness
        
        typing_delay = min(base_delay + length_factor * random_factor, 5.0)  # Max 5 seconds
        
        # Show typing indicator
        st.session_state.seher_typing = True
        
        # In a real implementation, we'd use async/await or threading
        # For Streamlit, we'll simulate with a progress indicator
        with st.spinner(f"🤖 Seher is typing... ({typing_delay:.1f}s)"):
            time.sleep(typing_delay)
        
        st.session_state.seher_typing = False
        
        # Add bot response to history
        st.session_state.seher_chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(),
            "typing_delay": typing_delay
        })
        
        # Limit chat history to prevent memory issues
        if len(st.session_state.seher_chat_history) > 20:
            st.session_state.seher_chat_history = st.session_state.seher_chat_history[-20:]
    
    def _process_user_input(self, user_input: str):
        """Process user input and generate response (legacy method for compatibility)"""
        # For backwards compatibility, call the typing version
        self._process_user_input_with_typing(user_input)
    
    def _process_natural_language_input(self, user_input: str):
        """Process natural language input with enhanced understanding"""
        import time
        import random
        
        # Add user message to history
        st.session_state.seher_chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Enhanced natural language processing
        response = self._generate_natural_response(user_input)
        
        # Simulate realistic typing delay
        response_length = len(response)
        base_delay = 1.0
        length_factor = response_length / 150  # Adjust based on response length
        random_factor = random.uniform(0.8, 1.2)
        typing_delay = min(base_delay + length_factor * random_factor, 5.0)
        
        # Show typing indicator with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(int(typing_delay * 10)):
            progress_bar.progress((i + 1) / (typing_delay * 10))
            status_text.text(f"🤖 Seher is thinking... {i/10:.1f}s")
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        # Add bot response to history
        st.session_state.seher_chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(),
            "typing_delay": typing_delay
        })
        
        # Limit chat history
        if len(st.session_state.seher_chat_history) > 20:
            st.session_state.seher_chat_history = st.session_state.seher_chat_history[-20:]
    
    def _generate_natural_response(self, user_input: str) -> str:
        """Generate natural language response with context awareness"""
        current_context = st.session_state.get('seher_current_context', {"tab": "general"})
        current_tab = current_context.get("tab", "general")
        user_lower = user_input.lower()
        
        # Enhanced natural language understanding
        if any(word in user_lower for word in ["what is", "what does", "explain", "tell me about"]):
            if current_tab in self.knowledge_base:
                feature_info = self.knowledge_base[current_tab]
                return f"{feature_info['basic']} \n\n{feature_info['detailed']}"
            else:
                return f"You're currently in the {current_tab} section. This feature helps analyze images using specific algorithms. What would you like to know specifically?"
        
        elif any(word in user_lower for word in ["how do i", "how to", "steps", "guide", "tutorial"]):
            return f"Here's how to use the {current_tab} feature: \n\n1. Select your image(s) from the uploaded collection\n2. Adjust the parameters based on your analysis needs\n3. Click the analysis button to process\n4. Review the results and download if needed\n\nWould you like me to explain any specific parameter or step in more detail?"
        
        elif any(word in user_lower for word in ["best", "optimal", "recommend", "should i", "which"]):
            recommendations = {
                "histogram": "For artwork analysis, colored histograms often provide more insight than grayscale. Try analyzing all images to compare patterns.",
                "kmeans": "Start with 3-5 clusters for most images. Use fewer clusters (2-3) for simple images, more (6-8) for complex scenes.",
                "shape": "HOG features work well for object detection, SIFT for matching, and traditional features for geometric analysis.",
                "similarity": "Color-based similarity works well for finding similar scenes, while texture-based similarity is better for material analysis.",
                "texture": "Haralick features are excellent for material analysis. Start with distance=1 and angle=0 for general texture analysis.",
                "lbp": "Use radius=3 and 24 points for detailed texture analysis. Smaller radius (1-2) for fine textures.",
                "contours": "For object counting, use threshold around 127. For detailed shape analysis, try different threshold values.",
                "sift": "SIFT is perfect for finding matching features between images. Great for authentication or similarity analysis."
            }
            return recommendations.get(current_tab, f"For {current_tab} analysis, experiment with different parameters to see what works best for your specific images. Would you like specific parameter recommendations?")
        
        elif any(word in user_lower for word in ["result", "output", "interpret", "meaning", "understand"]):
            return f"The {current_tab} results show the characteristics of your image data. Higher values typically indicate stronger presence of the analyzed features. Look for patterns, peaks, or distinctive values that differentiate your images. Would you like me to explain how to interpret specific aspects of the results?"
        
        elif any(word in user_lower for word in ["error", "not working", "failed", "problem"]):
            return f"If you're experiencing issues with {current_tab} analysis, try these steps: \n\n1. Check that your images are properly uploaded\n2. Verify parameter values are within valid ranges\n3. Ensure images are in supported formats (JPG, PNG)\n4. Try with a single image first to test\n\nWhat specific error or issue are you seeing?"
        
        elif any(word in user_lower for word in ["compare", "difference", "similar", "match"]):
            return f"Great question! The {current_tab} feature can help you compare images by analyzing their characteristics. You can process multiple images and compare their feature values, look for similar patterns, or identify outliers. Would you like me to explain how to set up a comparison analysis?"
        
        elif any(word in user_lower for word in ["example", "sample", "demo", "show me"]):
            examples = {
                "histogram": "Try uploading a portrait and a landscape photo. Notice how the histogram peaks differ - portraits often peak in flesh tones, landscapes in greens and blues.",
                "kmeans": "Upload an image with distinct colors (like a flower). Set k=3 to see how the algorithm groups the main color regions.",
                "similarity": "Upload several related images (like different views of the same building) to see how similarity scores compare.",
                "texture": "Compare a smooth surface (like skin) with a rough one (like tree bark) to see how texture values differ."
            }
            return examples.get(current_tab, f"Try experimenting with the {current_tab} feature on different types of images to see how the analysis changes. Upload a few test images and compare the results!")
        
        else:
            # Use the enhanced AI response or fallback
            return self._generate_ai_response(user_input, f"User is asking about {current_tab} in natural language")
    


# Global instance
seher_chat = SeherChatWindow()