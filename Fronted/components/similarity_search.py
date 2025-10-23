import streamlit as st
import pandas as pd
from PIL import Image
import base64
import io
from typing import List, Dict, Any
import time

from utils.api_client import (
    similarity_search, 
    precompute_similarity_features, 
    clear_similarity_cache,
    get_similarity_cache_stats,
    get_similarity_methods
)
from utils.api_client import _get_base_url

def render_similarity_search():
    """Render the similarity search interface."""
    st.title("🔍 Image Similarity Search")
    st.markdown("Find visually similar images using various feature extraction methods and distance metrics.")
    
    # Get available methods
    try:
        methods_info = get_similarity_methods()
        available_methods = methods_info.get("feature_methods", ["CNN", "HOG", "SIFT", "histogram"])
        available_metrics = methods_info.get("distance_metrics", ["cosine", "euclidean", "manhattan"])
        method_descriptions = methods_info.get("method_descriptions", {})
        metric_descriptions = methods_info.get("metric_descriptions", {})
    except Exception as e:
        st.error(f"Failed to get available methods: {e}")
        available_methods = ["CNN", "HOG", "SIFT", "histogram"]
        available_metrics = ["cosine", "euclidean", "manhattan"]
        method_descriptions = {}
        metric_descriptions = {}
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Search Parameters")
        
        # Feature extraction method with manuscript highlight
        st.markdown("### Feature Extraction Method")
        
        # Highlight manuscript method if available
        if "manuscript" in available_methods:
            st.success("📜 **RECOMMENDED FOR MANUSCRIPTS**: Use 'manuscript' method for ancient books, handwritten texts, and historical documents!")
        
        feature_method = st.selectbox(
            "Select Method",
            options=available_methods,
            help="Method used to extract features from images",
            index=available_methods.index("manuscript") if "manuscript" in available_methods else 0
        )
        
        if feature_method in method_descriptions:
            if feature_method == "manuscript":
                st.success(method_descriptions[feature_method])
            else:
                st.info(method_descriptions[feature_method])
        
        # Distance metric
        distance_metric = st.selectbox(
            "Distance Metric",
            options=available_metrics,
            help="Metric used to compute similarity between images"
        )
        
        if distance_metric in metric_descriptions:
            st.info(metric_descriptions[distance_metric])
        
        # Search parameters
        max_results = st.slider("Max Results", min_value=1, max_value=50, value=15)
        
        use_threshold = st.checkbox("Use Similarity Threshold", help="Uncheck to use automatic smart thresholds optimized for each method")
        threshold = None
        if use_threshold:
            # Set different default thresholds based on method
            if feature_method == "manuscript":
                default_threshold = 0.3
                st.info("📜 Recommended threshold for manuscripts: 0.2-0.4")
            elif feature_method == "HOG":
                default_threshold = 0.4
            elif feature_method == "SIFT":
                default_threshold = 0.2
            else:
                default_threshold = 0.5
                
            threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, 
                                value=default_threshold, step=0.01,
                                help=f"Higher values = more strict matching. Recommended: {default_threshold}")
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            # Image resizing
            st.subheader("Image Preprocessing")
            resize_width = st.number_input("Resize Width", min_value=32, max_value=512, value=224)
            resize_height = st.number_input("Resize Height", min_value=32, max_value=512, value=224)
            resize_dimensions = [resize_width, resize_height]
            
            # Method-specific parameters
            if feature_method == "HOG":
                st.subheader("HOG Parameters")
                hog_orientations = st.slider("Orientations", min_value=1, max_value=32, value=9)
                hog_ppc_h = st.number_input("Pixels per Cell (Height)", min_value=1, max_value=32, value=8)
                hog_ppc_w = st.number_input("Pixels per Cell (Width)", min_value=1, max_value=32, value=8)
                hog_cpb_h = st.number_input("Cells per Block (Height)", min_value=1, max_value=8, value=2)
                hog_cpb_w = st.number_input("Cells per Block (Width)", min_value=1, max_value=8, value=2)
                hog_pixels_per_cell = [hog_ppc_h, hog_ppc_w]
                hog_cells_per_block = [hog_cpb_h, hog_cpb_w]
            else:
                hog_orientations = None
                hog_pixels_per_cell = None
                hog_cells_per_block = None
            
            if feature_method == "histogram":
                st.subheader("Histogram Parameters")
                hist_bins = st.slider("Number of Bins", min_value=8, max_value=256, value=64)
                
                # Channel selection
                use_red = st.checkbox("Red Channel", value=True)
                use_green = st.checkbox("Green Channel", value=True)
                use_blue = st.checkbox("Blue Channel", value=True)
                
                hist_channels = []
                if use_red:
                    hist_channels.append(0)
                if use_green:
                    hist_channels.append(1)
                if use_blue:
                    hist_channels.append(2)
                
                if not hist_channels:
                    hist_channels = [0, 1, 2]  # Default to all channels
            else:
                hist_bins = None
                hist_channels = None
                
            if feature_method == "manuscript":
                st.subheader("📜 Manuscript Analysis Parameters")
                st.info("Optimized preprocessing for historical documents:")
                st.markdown("""
                - ✅ **CLAHE**: Contrast enhancement for faded texts
                - ✅ **Multi-scale HOG**: Fine & coarse text patterns
                - ✅ **LBP**: Local texture analysis for script styles
                - ✅ **Layout analysis**: Line and word structure
                - ✅ **Edge orientation**: Script direction patterns
                """)
                
                # For manuscript method, resize is optimized
                if resize_dimensions == [224, 224]:
                    resize_dimensions = [256, 256]  # Better for manuscript analysis
                    st.success("Using optimized 256x256 resolution for manuscripts")
        
        # Cache management
        with st.expander("Cache Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Cache"):
                    try:
                        result = clear_similarity_cache()
                        st.success(result.get("message", "Cache cleared"))
                    except Exception as e:
                        st.error(f"Failed to clear cache: {e}")
            
            with col2:
                if st.button("Show Cache Stats"):
                    try:
                        stats = get_similarity_cache_stats()
                        st.json(stats)
                    except Exception as e:
                        st.error(f"Failed to get cache stats: {e}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Image")
        
        # Query image selection
        query_type = st.radio(
            "Select Query Method",
            ["Use uploaded image by index", "Upload new query image"],
            key="query_type"
        )
        
        query_image_index = None
        query_image_base64 = None
        query_image_display = None
        
        if query_type == "Use uploaded image by index":
            query_image_index = st.number_input(
                "Image Index",
                min_value=0,
                value=0,
                help="Index of the uploaded image to use as query"
            )
            
            # Try to display the selected image
            try:
                # This is a simplified approach - in a real app, you might want to 
                # implement an endpoint to get image by index for preview
                st.info(f"Using uploaded image at index {query_image_index}")
            except Exception:
                st.warning("Cannot preview image. Make sure you have uploaded images first.")
        
        else:  # Upload new query image
            uploaded_query = st.file_uploader(
                "Upload Query Image",
                type=["png", "jpg", "jpeg", "tiff"],
                key="query_upload"
            )
            
            if uploaded_query is not None:
                # Display the uploaded image
                query_image_display = Image.open(uploaded_query)
                st.image(query_image_display, caption="Query Image", width='stretch')
                
                # Convert to base64
                buffer = io.BytesIO()
                query_image_display.save(buffer, format="PNG")
                query_image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Search button
        search_button = st.button("🔍 Search Similar Images", type="primary", width='stretch')
        
        # Precompute features section
        st.markdown("---")
        
        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            precompute_button = st.button("⚡ Precompute Features", help="Compute and cache features for faster searches")
        with col_clear:
            if st.button("🗑️ Clear"):
                if 'precompute_result' in st.session_state:
                    del st.session_state.precompute_result
                if 'show_precompute_details' in st.session_state:
                    del st.session_state.show_precompute_details
                st.rerun()
        
        # Show precomputed features in collapsible section if button is clicked
        if precompute_button:
            with st.spinner("Computing features for all images..."):
                try:
                    result = precompute_similarity_features(
                        feature_method=feature_method,
                        resize_dimensions=resize_dimensions,
                        hog_orientations=hog_orientations,
                        hog_pixels_per_cell=hog_pixels_per_cell,
                        hog_cells_per_block=hog_cells_per_block,
                        hist_bins=hist_bins,
                        hist_channels=hist_channels
                    )
                    
                    # Success message
                    st.success(f"✅ {result.get('message', 'Features precomputed successfully')}")
                    
                    # Store result in session state to show in search results column
                    st.session_state.precompute_result = result
                    
                    # Force a rerun to show the results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to precompute features: {e}")
                    st.session_state.precompute_result = None
    
    with col2:
        st.subheader("Search Results")
        
        # Show precompute results in the search results area
        if 'precompute_result' in st.session_state and st.session_state.precompute_result:
            result = st.session_state.precompute_result
            
            st.markdown("## 📊 Precomputed Feature Details")
            
            # Show basic statistics from the result
            st.markdown("### 📈 Processing Summary")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                processed_count = result.get('processed_count', 0)
                st.metric("Images Processed", processed_count)
            
            with col_b:
                failed_count = result.get('failed_count', 0)
                st.metric("Failed", failed_count, delta_color="inverse" if failed_count > 0 else "normal")
            
            with col_c:
                method_used = result.get('feature_method', 'CNN')
                st.metric("Method", method_used.upper())
            
            # Show cache statistics
            cache_stats = result.get('cache_stats', {})
            if cache_stats:
                st.markdown("### 💾 Cache Information")
                
                # Create a simple table of cache data
                cache_data = []
                
                # Try different field names that the backend might use
                total_cached = (cache_stats.get('total_cached_features') or 
                               cache_stats.get('total_images_cached') or
                               cache_stats.get('cached_images') or
                               len(cache_stats.get('cache_breakdown', {})))
                
                cache_size_mb = cache_stats.get('cache_size_mb', 0)
                
                if total_cached:
                    cache_data.append({"Property": "Total Cached Features", "Value": str(total_cached)})
                if cache_size_mb:
                    cache_data.append({"Property": "Cache Size (MB)", "Value": f"{cache_size_mb:.2f}"})
                
                # Add any other cache properties
                for key, value in cache_stats.items():
                    if key not in ['cache_breakdown']:
                        cache_data.append({"Property": key.replace('_', ' ').title(), "Value": str(value)})
                
                if cache_data:
                    df = pd.DataFrame(cache_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show raw response for debugging
            with st.expander("🔧 Complete API Response", expanded=False):
                st.json(result)
            
            st.markdown("---")
        
        if search_button:
            if query_type == "Use uploaded image by index":
                if query_image_index is None:
                    st.error("Please specify an image index")
                    return
            else:
                if query_image_base64 is None:
                    st.error("Please upload a query image")
                    return
            
            # Perform similarity search
            with st.spinner("Searching for similar images..."):
                try:
                    start_time = time.time()
                    
                    results = similarity_search(
                        query_image_index=query_image_index,
                        query_image_base64=query_image_base64,
                        feature_method=feature_method,
                        distance_metric=distance_metric,
                        max_results=max_results,
                        threshold=threshold,
                        resize_dimensions=resize_dimensions,
                        hog_orientations=hog_orientations,
                        hog_pixels_per_cell=hog_pixels_per_cell,
                        hog_cells_per_block=hog_cells_per_block,
                        hist_bins=hist_bins,
                        hist_channels=hist_channels
                    )
                    
                    search_time = time.time() - start_time
                    
                    similar_images = results.get("similar_images", [])
                    computation_time = results.get("computation_time", search_time)
                    
                    if not similar_images:
                        st.warning("No similar images found. Try adjusting the threshold or parameters.")
                    else:
                        st.success(f"Found {len(similar_images)} similar images in {computation_time:.2f}s")
                        
                        # Display results in a grid
                        cols_per_row = 3
                        for i in range(0, len(similar_images), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j, col in enumerate(cols):
                                if i + j < len(similar_images):
                                    img_data = similar_images[i + j]
                                    image_id = img_data["image_id"]
                                    similarity_score = img_data["similarity_score"]
                                    distance = img_data["distance"]
                                    
                                    with col:
                                        # Display placeholder or try to load image
                                        # Note: In a real implementation, you might want an endpoint 
                                        # to serve images by ID
                                        st.write(f"**{image_id}**")
                                        st.metric(
                                            "Similarity", 
                                            f"{similarity_score:.3f}",
                                            help=f"Distance: {distance:.3f}"
                                        )
                                        
                                        # Create a placeholder box for the image
                                        st.container()
                                        st.markdown(
                                            f"""
                                            <div style="
                                                border: 2px solid #ddd; 
                                                border-radius: 5px; 
                                                padding: 10px; 
                                                text-align: center; 
                                                background-color: #f9f9f9;
                                                height: 150px;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                            ">
                                                <strong>{image_id}</strong><br>
                                                <small>Similarity: {similarity_score:.3f}</small>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                        
                        # Show detailed results in expandable table
                        with st.expander("Detailed Results"):
                            results_df = pd.DataFrame(similar_images)
                            if not results_df.empty:
                                # Format similarity scores
                                results_df['similarity_score'] = results_df['similarity_score'].round(4)
                                results_df['distance'] = results_df['distance'].round(4)
                                st.dataframe(results_df, width='stretch')
                        
                        # Show search parameters used
                        with st.expander("Search Parameters Used"):
                            st.json({
                                "feature_method": feature_method,
                                "distance_metric": distance_metric,
                                "max_results": max_results,
                                "threshold": threshold,
                                "resize_dimensions": resize_dimensions,
                                "computation_time": f"{computation_time:.2f}s"
                            })
                
                except Exception as e:
                    st.error(f"Similarity search failed: {e}")
                    if "404" in str(e):
                        st.info("Make sure you have uploaded some images first using the 'Data Input' section.")
                    elif "500" in str(e):
                        st.info("Internal server error. Check if the backend is running and try again.")
    
    # Information section
    with st.expander("ℹ️ How to Use Similarity Search"):
        st.markdown("""
        **Similarity Search** allows you to find images that are visually similar to a query image.
        
        ### Steps:
        1. **Upload images** in the 'Data Input' section first
        2. **Choose a feature extraction method**:
           - **CNN**: Statistical features from image patches (good general purpose)
           - **HOG**: Histogram of Oriented Gradients (good for shapes and textures)
           - **SIFT**: Scale-Invariant Feature Transform (robust to scale/rotation)
           - **histogram**: Color distribution (good for color-based similarity)
           - **manuscript**: Specialized features for handwritten documents and text analysis
        
        3. **Select a distance metric**:
           - **cosine**: Measures angle between feature vectors
           - **euclidean**: Straight-line distance in feature space
           - **manhattan**: City-block distance
        
        4. **Choose your query image**:
           - Use an already uploaded image by index, or
           - Upload a new image as query
        
        5. **Adjust parameters** as needed and click search
        
        ### Tips:
        - **Precompute features** for faster searches on large datasets
        - **Adjust threshold** to filter results by minimum similarity
        - **Different methods work better for different types of similarity**
        - **Clear cache** if you upload new images or change parameters significantly
        """)
