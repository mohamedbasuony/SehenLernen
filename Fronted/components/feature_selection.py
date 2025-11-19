# Fronted/components/feature_selection.py
import io
import csv
import zipfile
import math
import base64
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper  # interactive cropper
import time
from typing import Optional

from utils.api_client import (
    generate_histogram,
    perform_kmeans,
    perform_single_image_kmeans,
    extract_shape_features,
    extract_haralick_texture,      # train/predict workflow (kept, just renamed in UI)
    replace_image,                 # persist cropped image to backend
    extract_haralick_features,     # table-style GLCM Haralick
    extract_lbp_features,          # NEW: LBP feature extraction
    extract_contours,              # NEW: Contour extraction
    extract_hog_features,          # NEW: HOG convenience call
    extract_sift_features,
    extract_edge_features,
    similarity_search,             # NEW: Similarity search
    get_similarity_methods,        # NEW: Get available methods
    precompute_similarity_features, # NEW: Precompute features
    extract_image_embedding,       # NEW: Image embedding extraction
    train_classifier,
    predict_classifier,
)

# Import Seher AI Chat Island (floating bubble only)
from .seher_smart_chat import update_chat_context

# ---------- Helpers ----------
def _get_image_id_for_index(idx: int) -> str | None:
    ids = st.session_state.get("uploaded_image_ids")
    if ids and 0 <= idx < len(ids):
        return ids[idx]
    return None


def _ensure_backend_sync(force_reupload: bool = False):
    """Ensure backend is synchronized with frontend images by re-uploading if needed."""
    try:
        from utils.api_client import get_current_image_ids, clear_all_backend_images, upload_images, get_all_images
        
        # Get current state
        frontend_images = st.session_state.get("images", [])
        backend_ids = get_current_image_ids()
        
        logging.info(f"Sync check: {len(frontend_images)} frontend images, {len(backend_ids)} backend IDs, force={force_reupload}")
        
        # If we have backend images but no frontend PIL images, fetch them from backend
        if not frontend_images and backend_ids:
            logging.info("No frontend images but backend has IDs - fetching from backend")
            try:
                frontend_images = get_all_images()
                st.session_state["images"] = frontend_images
                st.session_state["uploaded_image_ids"] = backend_ids
                logging.info(f"Fetched {len(frontend_images)} images from backend")
                return True
            except Exception as e:
                logging.error(f"Failed to fetch images from backend: {e}")
                st.session_state["classifier_sync_error"] = f"Failed to fetch images: {str(e)}"
                return False
        
        # If forced or counts don't match, clean backend and re-upload
        if force_reupload or len(frontend_images) != len(backend_ids):
            logging.info(f"Re-uploading: force={force_reupload}, mismatch={len(frontend_images) != len(backend_ids)}")
            
            if not frontend_images:
                st.session_state["uploaded_image_ids"] = []
                logging.warning("No frontend images to upload")
                st.session_state["classifier_sync_error"] = "No images in session state to upload"
                return False
            
            # Clear backend storage
            logging.info("Clearing backend storage...")
            clear_all_backend_images()
            logging.info("Backend cleared successfully")
            
            # Convert PIL images to uploadable format
            class FileWrapper:
                def __init__(self, name, data):
                    self.name = name
                    self.data = data
                    self.type = "image/png"
                def getvalue(self):
                    return self.data
            
            import time
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            image_files = []
            for i, img in enumerate(frontend_images):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                unique_filename = f"image_{timestamp}_{i+1}.png"
                file_wrapper = FileWrapper(unique_filename, img_bytes.getvalue())
                image_files.append(file_wrapper)
            
            logging.info(f"Prepared {len(image_files)} files for upload")
            
            # Upload images and get new IDs
            new_ids = upload_images(image_files)
            logging.info(f"Backend returned {len(new_ids)} IDs: {new_ids}")
            
            st.session_state["uploaded_image_ids"] = new_ids
            logging.info(f"Updated session state with {len(new_ids)} IDs")
            
            # Verify the upload was successful
            if len(new_ids) != len(frontend_images):
                error_msg = f"Upload mismatch: expected {len(frontend_images)} IDs, got {len(new_ids)}"
                logging.error(error_msg)
                st.session_state["classifier_sync_error"] = error_msg
                return False
            
            logging.info("Sync successful - all images uploaded and verified")
            return True
        else:
            # Counts match, just update session state
            st.session_state["uploaded_image_ids"] = backend_ids
            logging.info("Counts match - sync successful")
            return True
            
    except Exception as e:
        logging.error(f"Failed to sync backend: {e}", exc_info=True)
        st.session_state["classifier_sync_error"] = str(e)
        return False


def _init_state():
    st.session_state.setdefault("crop_active", False)
    st.session_state.setdefault("crop_index", None)
    st.session_state.setdefault("crop_aspect_label", "Free")
    st.session_state.setdefault("crop_realtime", True)
    st.session_state.setdefault("fullscreen_image", None)
    st.session_state.setdefault("fullscreen_section", None)


def _aspect_ratio_value(label: str):
    mapping = {"Free": None, "1:1 (Square)": (1, 1), "4:3": (4, 3), "16:9": (16, 9)}
    return mapping.get(label, None)


def _to_csv_bytes(columns, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")

def base64_to_bytes(b64_str: str) -> bytes:
    """Convert a base64‑encoded PNG string to raw bytes that Streamlit can display."""
    return base64.b64decode(b64_str)


def _ensure_classifier_table(image_ids: list[str]) -> pd.DataFrame:
    """
    Initialize or update the classifier label/split table stored in session_state.
    Ensures one row per image with columns: image_index, image_id, label, split.
    """
    df: Optional[pd.DataFrame] = st.session_state.get("classifier_training_df")

    if df is None:
        data = [
            {
                "image_index": idx,
                "image_id": image_id,
                "label": "",
                "split": "train",
            }
            for idx, image_id in enumerate(image_ids)
        ]
        df = pd.DataFrame(data)
    else:
        df = df.copy()
        if "image_index" not in df.columns:
            df["image_index"] = df.index
        df["image_index"] = df["image_index"].astype(int)
        df = df[df["image_index"] < len(image_ids)]
        existing_indices = set(df["image_index"].tolist())

        for idx, image_id in enumerate(image_ids):
            if idx in existing_indices:
                df.loc[df["image_index"] == idx, "image_id"] = image_id
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [{
                                "image_index": idx,
                                "image_id": image_id,
                                "label": "",
                                "split": "train",
                            }]
                        ),
                    ],
                    ignore_index=True,
                )

    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str)

    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].fillna("train").astype(str)

    df = df.sort_values("image_index").reset_index(drop=True)
    st.session_state["classifier_training_df"] = df
    return df


def _probabilities_to_string(probs: dict | None) -> str:
    if not probs:
        return ""
    return ", ".join(f"{label}: {value:.2f}" for label, value in probs.items())


def _render_metrics_section(metrics: dict | None, heading: str) -> None:
    if not metrics:
        return

    st.markdown(f"**{heading}**")
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        st.metric(label=f"{heading} Accuracy", value=f"{accuracy * 100:.2f}%")

    if metrics.get("confusion_matrix"):
        st.caption("Confusion Matrix")
        matrix_df = pd.DataFrame(metrics["confusion_matrix"])
        st.dataframe(matrix_df, width='stretch')

    if metrics.get("classification_report"):
        report_df = pd.DataFrame(metrics["classification_report"]).T
        st.caption("Classification Report")
        st.dataframe(report_df, width='stretch')

# ---------- Main Entry ----------
def render_feature_selection():
    import pandas as pd  # Ensure pandas is available in function scope
    st.markdown("<h1 style='text-align: center; font-size: 2.2rem; margin-bottom: 1.5rem; color: #2c3e50;'>Feature Selection</h1>", unsafe_allow_html=True)
    _init_state()

    if "images" not in st.session_state or not st.session_state["images"]:
        st.warning("Please upload images first.")
        return

    images = st.session_state["images"]

    # =========================
    # Fullscreen Image View
    # =========================
    if st.session_state.get("fullscreen_image"):
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>Fullscreen View</h2>", unsafe_allow_html=True)
        
        # Exit button at the top
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔙 Back to Feature Selection", key="exit_fullscreen", type="primary", width='stretch'):
                st.session_state["fullscreen_image"] = None
                st.session_state["fullscreen_section"] = None
                st.rerun()
        
        st.markdown("---")
        
        # Display the fullscreen image
        st.image(
            st.session_state["fullscreen_image"], 
            caption=f"Enlarged view - {st.session_state.get('fullscreen_section', 'Image')}", 
            use_column_width=True
        )
        
        # Another exit button at the bottom for convenience
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔙 Back to Feature Selection", key="exit_fullscreen_bottom", type="primary", width='stretch'):
                st.session_state["fullscreen_image"] = None
                st.session_state["fullscreen_section"] = None
                st.rerun()
        
        return  # Exit early to show only fullscreen view

    # =========================
    # Dedicated Crop Screen (centered)
    # =========================
    if st.session_state.get("crop_active") and st.session_state.get("crop_index") is not None:
        idx = int(st.session_state["crop_index"])
        if idx < 0 or idx >= len(images):
            st.error("Invalid image index for cropping.")
            st.session_state["crop_active"] = False
            st.session_state["crop_index"] = None
            st.rerun()

        st.subheader(f"Crop Image {idx+1}")
        with st.expander("Crop Options", expanded=False):
            st.session_state["crop_aspect_label"] = st.selectbox(
                "Aspect ratio",
                options=["Free", "1:1 (Square)", "4:3", "16:9"],
                index=["Free", "1:1 (Square)", "4:3", "16:9"].index(st.session_state["crop_aspect_label"]),
                key="crop_aspect_label_select",
            )
            st.session_state["crop_realtime"] = st.checkbox(
                "Realtime update",
                value=st.session_state["crop_realtime"],
                key="crop_realtime_checkbox",
            )

        img = images[idx]
        aspect = _aspect_ratio_value(st.session_state["crop_aspect_label"])

        # Centered container for cropper
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            cropped_raw = st_cropper(
                img,
                aspect_ratio=aspect,
                realtime_update=st.session_state["crop_realtime"],
                return_type="image",
                key=f"cropper_full_{idx}",
            )

        # Normalize to PIL.Image (no external preview)
        final_crop = None
        if isinstance(cropped_raw, Image.Image):
            final_crop = cropped_raw
        elif isinstance(cropped_raw, np.ndarray):
            try:
                final_crop = Image.fromarray(cropped_raw)
            except Exception:
                final_crop = None
        elif cropped_raw is not None:
            try:
                final_crop = Image.fromarray(np.array(cropped_raw))
            except Exception:
                final_crop = None

        # Centered buttons directly under cropper
        with center_col:
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                if st.button("✖️ Cancel", key="crop_cancel"):
                    st.session_state["crop_active"] = False
                    st.session_state["crop_index"] = None
                    st.rerun()
            with btn_cols[1]:
                if st.button("✅ Crop & Save", key="crop_confirm"):
                    if final_crop is None:
                        st.error("No cropped image yet. Please adjust the crop box, then try again.")
                    else:
                        try:
                            # 1) Replace locally so ALL downstream steps use cropped image
                            st.session_state["images"][idx] = final_crop

                            # 2) Persist to backend so processing endpoints read the cropped file
                            image_id = _get_image_id_for_index(idx)
                            if image_id:
                                replace_image(image_id, final_crop)
                                st.success(f"Image {idx+1} cropped and saved.")
                            else:
                                st.warning("Could not map image to backend image_id; saved locally only.")
                        except Exception as e:
                            st.error(f"Failed to crop image {idx+1}: {e}")
                        finally:
                            # Exit crop screen back to tabs
                            st.session_state["crop_active"] = False
                            st.session_state["crop_index"] = None
                            st.rerun()

        # Stop rendering further UI on the crop screen
        return

    # =========================
    # Normal Tabs (when not cropping)
    # =========================
    # ADD: New tabs including Similarity Search
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "Histogram Analysis",
            "k-means Clustering", 
            "Shape Features",
            "Similarity Search",             # NEW - Added here
            "Haralick Texture",
            "Local Binary Patterns (LBP)",
            "Contour Extraction",
            "SIFT & Edge Detection",
            "Classifier Training",
        ]
    )

    # --- Histogram Analysis ---
    with tab1:
        # Update context for Seher AI as soon as this tab is shown
        update_chat_context("histogram", ["hist_type", "all_images", "selected_image"])
        st.subheader("📊 Histogram Analysis")
        
        hist_type = st.radio("Histogram Type", options=["Black and White", "Colored"], key="hist_type")
        all_images = st.checkbox("Generate histogram for all images", key="hist_all")
        selected_index = st.selectbox(
            "Select image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="hist_image_idx"
        )

        if all_images:
            st.markdown("### Selected Images")
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img, caption=f"Image {idx+1}", width=150)
            st.info("Cropping works on a single selected image. Uncheck 'Generate histogram for all images' to crop.")
        else:
            img = images[selected_index]
            st.image(img, caption=f"Preview: Image {selected_index+1}", width=250)

            # The ONLY Crop button — takes you to the centered crop screen
            if st.button(f"Crop {selected_index+1}", key="btn_crop_under_hist"):
                st.session_state["crop_active"] = True
                st.session_state["crop_index"] = int(selected_index)
                st.rerun()

        if st.button("Generate Histogram", key="btn_histogram"):
            params = {"hist_type": hist_type, "image_index": selected_index, "all_images": all_images}
            hist_list = generate_histogram(params)
            if hist_list:
                st.session_state["histogram_images"] = hist_list
                st.success(f"Generated {len(hist_list)} histogram{'s' if len(hist_list) > 1 else ''}!")
            else:
                st.warning("No histograms were generated.")

        if st.session_state.get("histogram_images"):
            st.subheader("Generated Histograms")
            cols = st.columns(4)
            for i, img_bytes in enumerate(st.session_state["histogram_images"]):
                with cols[i % 4]:
                    st.image(img_bytes, caption=f"Histogram {i+1}", width=200)
                    # Enlarge button (existing)
                    if st.button(f"Enlarge {i+1}", key=f"enlarge_hist_{i}"):
                        st.session_state["fullscreen_image"] = img_bytes
                        st.session_state["fullscreen_section"] = "histogram"
                        st.rerun()
                    # NEW: Contours button next to Enlarge
                    if st.button(f"Contours {i+1}", key=f"contours_hist_{i}"):
                        with st.spinner("Extracting contours..."):
                            try:
                                # Map histogram index back to image index
                                # If 'all images' was used, i maps directly.
                                # Otherwise, use the selected index.
                                img_idx = i if st.session_state.get("hist_all") else st.session_state.get("hist_image_idx", 0)
                                params = {
                                    "image_index": int(img_idx),
                                    "mode": "RETR_EXTERNAL",
                                    "method": "CHAIN_APPROX_SIMPLE",
                                    "min_area": 10,
                                    "return_bounding_boxes": True,
                                    "return_hierarchy": False,
                                }
                                result = extract_contours(params)
                                if result.get("visualization_bytes"):
                                    st.image(result["visualization_bytes"], caption="Contours Overlay", width=200)
                                # Show quick stats
                                areas = result.get("areas", [])
                                bbs = result.get("bounding_boxes", [])
                                if areas:
                                    st.write(f"Contours: {len(areas)}")
                                if bbs:
                                    st.write("Sample bounding box:", bbs[0])
                            except Exception as e:
                                st.error(f"Contour extraction failed: {e}")

            if st.button("Download All Histograms", key="download_all_histograms"):
                zip_data = create_histogram_zip(st.session_state["histogram_images"])
                st.session_state["histogram_zip"] = zip_data
                st.success("Histograms ZIP created. Click below to download.")

            if "histogram_zip" in st.session_state:
                st.download_button(
                    label="Download Histograms ZIP",
                    data=st.session_state["histogram_zip"],
                    file_name="histograms.zip",
                    mime="application/zip",
                )

        # -------------------------------
        # NEW: HOG within Histogram Tab
        # -------------------------------
        st.divider()
        st.markdown("### HOG (Histogram of Oriented Gradients)")
        st.caption(
            "HOG summarizes local gradient directions in small cells to capture object structure and edges. "
            "It’s commonly used for detection and classification. Configure the parameters and compute HOG "
            "for the selected image above."
        )
        col_hl, col_hr = st.columns(2)
        with col_hl:
            hog_orient = st.number_input("Orientation bins", min_value=3, max_value=18, value=9, step=1, key="hog_orient")
            ppc_x = st.number_input("Pixels per cell — X", min_value=4, max_value=64, value=8, step=1, key="hog_ppc_x")
            ppc_y = st.number_input("Pixels per cell — Y", min_value=4, max_value=64, value=8, step=1, key="hog_ppc_y")
        with col_hr:
            cpb_x = st.number_input("Cells per block — X", min_value=1, max_value=8, value=2, step=1, key="hog_cpb_x")
            cpb_y = st.number_input("Cells per block — Y", min_value=1, max_value=8, value=2, step=1, key="hog_cpb_y")
            use_resize = st.checkbox("Resize before HOG", value=False, key="hog_use_resize")
            if use_resize:
                hog_w = st.number_input("Resize width", min_value=32, max_value=2048, value=128, step=16, key="hog_w")
                hog_h = st.number_input("Resize height", min_value=32, max_value=2048, value=64, step=16, key="hog_h")
            else:
                hog_w = None
                hog_h = None

        if st.button("Compute HOG for Selected Image", key="btn_hog_compute"):
            with st.spinner("Computing HOG..."):
                try:
                    # NOTE: The backend currently uses default HOG params.
                    # We pass the UI params for forward-compatibility. If the backend
                    # ignores them, results reflect its defaults until we wire it up.
                    result = extract_hog_features(
                        image_index=int(selected_index),
                        orientations=int(hog_orient),
                        pixels_per_cell=(int(ppc_x), int(ppc_y)),
                        cells_per_block=(int(cpb_x), int(cpb_y)),
                        resize_width=int(hog_w) if hog_w else None,
                        resize_height=int(hog_h) if hog_h else None,
                        visualize=True,
                    )
                    feats = result.get("features", []) or []
                    st.success(f"HOG feature vector length: {len(feats)}")
                    if result.get("visualization") is not None:
                        st.image(result["visualization"], caption="HOG Visualization", width=400)
                        if st.button("Enlarge HOG Visualization", key="enlarge_hog_viz"):
                            st.session_state["fullscreen_image"] = result["visualization"]
                            st.session_state["fullscreen_section"] = "HOG Visualization"
                            st.rerun()

                    # Download CSV for the single vector
                    cols = [f"f{i}" for i in range(len(feats))]
                    csv_bytes = _to_csv_bytes(["image_index"] + cols, [[int(selected_index)] + feats])
                    st.download_button(
                        "Download HOG Features (CSV)",
                        data=csv_bytes,
                        file_name=f"hog_image_{int(selected_index)+1}.csv",
                        mime="text/csv",
                        key="btn_dl_hog_csv",
                    )
                except Exception as e:
                    st.error(f"HOG computation failed: {e}")

    # --- k-means Clustering ---
    with tab2:
        # Update context for Seher AI
        update_chat_context("kmeans", ["n_clusters", "random_state", "image_selection", "clustering_mode"])
        st.subheader("🎨 K-means Clustering")
        
        # Clustering mode selection
        clustering_mode = st.radio(
            "Clustering Mode",
            options=["Multi-Image Clustering", "Single Image Segmentation"],
            help="Multi-Image: Cluster multiple images based on their features. Single Image: Segment one image by color regions.",
            key="kmeans_mode"
        )
        
        cluster_count = st.number_input("Number of Clusters", min_value=2, max_value=10, value=2, key="kmeans_k")
        random_state = st.number_input("Random Seed", min_value=0, max_value=100, value=42, key="kmeans_rs")

        if clustering_mode == "Multi-Image Clustering":
            st.markdown("### Image Selection")
            all_images_checkbox = st.checkbox("Select all images", key="kmeans_all_images")

            if all_images_checkbox:
                selected_indices = list(range(len(images)))
                st.info(f"All {len(images)} images selected")
            else:
                selected_indices = st.multiselect(
                    "Select images for clustering",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="kmeans_image_indices",
                )

            if st.button("Perform Multi-Image K-means", key="btn_kmeans"):
                if not images:
                    st.error("No images have been uploaded. Please upload images first.")
                elif not selected_indices and not all_images_checkbox:
                    st.error("Select at least one image or check 'Select all images'.")
                elif len(selected_indices) < 2 and not all_images_checkbox:
                    st.error("Multi-image clustering requires at least 2 images. Select more images or use Single Image Segmentation mode.")
                elif all_images_checkbox and len(images) < 2:
                    st.error("Multi-image clustering requires at least 2 images. Upload more images or use Single Image Segmentation mode.")
                else:
                    try:
                        params = {
                            "n_clusters": cluster_count,
                            "random_state": random_state,
                            "selected_images": selected_indices if not all_images_checkbox else [],
                            "use_all_images": all_images_checkbox,
                        }
                        plot_bytes, assignments = perform_kmeans(params)
                        st.session_state["kmeans_plot"] = plot_bytes
                        st.session_state["kmeans_assignments"] = assignments
                        st.success("K-means clustering completed!")
                    except Exception as e:
                        st.error(f"Error performing multi-image K-means clustering: {str(e)}")
        
        # Display K-means results if available
        if st.session_state.get("kmeans_plot"):
            st.subheader("K-means Clustering Results")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(st.session_state["kmeans_plot"], caption="Multi-Image K-means Clustering Graph", width=600)
            with col2:
                if st.button("📊 Enlarge Graph", key="enlarge_kmeans_plot", use_container_width=True):
                    st.session_state["fullscreen_image"] = st.session_state["kmeans_plot"]
                    st.session_state["fullscreen_section"] = "K-means Clustering"
                    st.rerun()
            
            # Display cluster assignments
            st.subheader("Cluster Assignments")
            assignments = st.session_state.get("kmeans_assignments", [])
            
            if "metadata_df" in st.session_state and "image_id_col" in st.session_state:
                metadata = st.session_state["metadata_df"]
                id_col = st.session_state["image_id_col"]
                image_ids = metadata[id_col].tolist()
                
                # Create a nice table view
                assignment_data = []
                for idx, label in enumerate(assignments):
                    image_name = image_ids[idx] if idx < len(image_ids) else f"Image {idx+1}"
                    assignment_data.append({"Image": image_name, "Cluster": label})
                
                df_assignments = pd.DataFrame(assignment_data)
                st.dataframe(df_assignments, use_container_width=True)
            else:
                # Simple text display
                assignment_data = []
                for idx, label in enumerate(assignments):
                    assignment_data.append({"Image": f"Image {idx+1}", "Cluster": label})
                
                df_assignments = pd.DataFrame(assignment_data)
                st.dataframe(df_assignments, use_container_width=True)
        
        else:  # Single Image Segmentation
            st.markdown("### Image Selection")
            selected_idx = st.selectbox(
                "Select image for segmentation",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="kmeans_single_image_idx"
            )
            
            max_pixels = st.number_input(
                "Max Pixels to Process", 
                min_value=1000, 
                max_value=50000, 
                value=10000, 
                step=1000,
                help="Larger values give better quality but slower processing",
                key="kmeans_max_pixels"
            )
            
            if st.button("Perform Single Image Segmentation", key="btn_single_kmeans"):
                if not images:
                    st.error("No images have been uploaded. Please upload images first.")
                else:
                    try:
                        params = {
                            "image_index": selected_idx,
                            "n_clusters": cluster_count,
                            "random_state": random_state,
                            "max_pixels": max_pixels,
                        }
                        segmented_bytes, plot_bytes = perform_single_image_kmeans(params)
                        st.session_state["kmeans_segmentation_plot"] = plot_bytes
                        st.session_state["kmeans_segmented_image"] = segmented_bytes
                        st.session_state["kmeans_cluster_count"] = cluster_count
                        st.success("K-means segmentation completed!")
                        
                    except Exception as e:
                        st.error(f"Error performing single image segmentation: {str(e)}")
        
        # Display segmentation results if available
        if st.session_state.get("kmeans_segmentation_plot"):
            st.subheader("K-means Segmentation Results")
            
            # Display comparison plot with enlarge button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(st.session_state["kmeans_segmentation_plot"], caption="Original vs Segmented Comparison", width=600)
            with col2:
                if st.button("📊 Enlarge Comparison", key="enlarge_segmentation_plot", use_container_width=True):
                    st.session_state["fullscreen_image"] = st.session_state["kmeans_segmentation_plot"]
                    st.session_state["fullscreen_section"] = "K-means Segmentation"
                    st.rerun()
            
            # Display segmented image separately
            st.subheader("Segmented Image (Color Quantization)")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(st.session_state["kmeans_segmented_image"], 
                        caption=f"Color Segmentation (k={st.session_state.get('kmeans_cluster_count', 2)})", 
                        width=600)
            with col2:
                if st.button("📊 Enlarge Segmented", key="enlarge_segmented_image", use_container_width=True):
                    st.session_state["fullscreen_image"] = st.session_state["kmeans_segmented_image"]
                    st.session_state["fullscreen_section"] = "Segmented Image"
                    st.rerun()
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download Comparison",
                    data=st.session_state["kmeans_segmentation_plot"],
                    file_name=f"kmeans_comparison_k{st.session_state.get('kmeans_cluster_count', 2)}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="⬇️ Download Segmented Image",
                    data=st.session_state["kmeans_segmented_image"],
                    file_name=f"kmeans_segmented_k{st.session_state.get('kmeans_cluster_count', 2)}.png",
                    mime="image/png",
                    use_container_width=True
                )

    # --- Shape Features ---
    with tab3:
        # Update context for Seher AI
        update_chat_context("shape", ["method", "selected_image"])
        st.subheader("🔷 Shape Feature Extraction")
        
        # Create sub-tabs for different feature types
        shape_subtab1, shape_subtab2 = st.tabs(["Traditional Features", "Deep Learning Embeddings"])
        
        # Traditional shape features (HOG, SIFT, FAST)
        with shape_subtab1:
            shape_methods = ["HOG", "SIFT", "FAST"]
            shape_method = st.selectbox("Method", options=shape_methods, key="shape_method")
            selected_idx = st.selectbox(
                "Select Image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="shape_img_idx"
            )

            if st.button("Extract Shape Features", key="btn_shape"):
                result = extract_shape_features({"method": shape_method, "image_index": selected_idx})
                st.write(f"{shape_method} Features:")
                st.write(result.get("features"))
                viz = result.get("visualization")
                if viz:
                    st.image(viz, caption=f"{shape_method} Visualization", width=400)
                    if st.button(f"Enlarge {shape_method} Visualization", key=f"enlarge_shape_{shape_method.lower()}"):
                        st.session_state["fullscreen_image"] = viz
                        st.session_state["fullscreen_section"] = f"{shape_method} Visualization"
                        st.rerun()
        
        # Deep learning embeddings
        with shape_subtab2:
            st.markdown("### Image Embedding Extraction")
            st.markdown("Extract fixed-length feature vectors using deep neural networks (ResNet, MobileNet).")
            
            # Model selection
            model_options = {
                "ResNet-50 (2048D)": "resnet50",
                "ResNet-18 (512D)": "resnet18", 
                "MobileNet-v2 (1280D)": "mobilenet_v2"
            }
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                key="embedding_model",
                help="Choose a pretrained model. Higher dimensions capture more information but require more storage."
            )
            selected_model = model_options[selected_model_name]
            
            # Image selection
            embed_mode = st.radio(
                "Process",
                options=["Single Image", "All Images"],
                key="embed_mode",
                horizontal=True
            )
            
            if embed_mode == "Single Image":
                embed_idx = st.selectbox(
                    "Select Image",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="embed_img_idx"
                )
                use_all = False
                image_indices = [embed_idx]
            else:
                use_all = True
                image_indices = None
                st.info(f"Will process all {len(images)} images")
            
            # Extract button
            if st.button("🎯 Extract Embeddings", key="btn_embedding", type="primary"):
                with st.spinner(f"Extracting embeddings using {selected_model_name}..."):
                    try:
                        params = {
                            "image_indices": image_indices,
                            "use_all_images": use_all,
                            "model_name": selected_model
                        }
                        result = extract_image_embedding(params)
                        
                        # Display results
                        st.success(f"✅ Extracted embeddings for {result['num_images']} image(s)")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", result['model_name'])
                        with col2:
                            st.metric("Embedding Dimension", result['embedding_dim'])
                        with col3:
                            st.metric("Images Processed", result['num_images'])
                        
                        # Show embeddings
                        st.markdown("#### Embedding Vectors")
                        
                        embeddings = result['embeddings']
                        image_ids = result['image_ids']
                        
                        # Display in expandable sections
                        for i, (emb, img_id) in enumerate(zip(embeddings, image_ids)):
                            with st.expander(f"Image {i+1} ({img_id}) - {len(emb)}D vector"):
                                # Show first 10 values as preview
                                st.write(f"**Preview (first 10 values):** {emb[:10]}")
                                st.write(f"**Shape:** {len(emb)} dimensions")
                                
                                # Show statistics
                                emb_array = np.array(emb)
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Mean", f"{emb_array.mean():.4f}")
                                with col_b:
                                    st.metric("Std Dev", f"{emb_array.std():.4f}")
                                with col_c:
                                    st.metric("L2 Norm", f"{np.linalg.norm(emb_array):.4f}")
                                
                                # Full vector as CSV download
                                csv_data = ",".join(map(str, emb))
                                st.download_button(
                                    label="📥 Download Full Vector (CSV)",
                                    data=csv_data,
                                    file_name=f"embedding_{img_id}.csv",
                                    mime="text/csv",
                                    key=f"download_emb_{i}"
                                )
                        
                        # Download all embeddings as CSV
                        if len(embeddings) > 1:
                            st.markdown("---")
                            # Create CSV with all embeddings
                            csv_rows = []
                            csv_rows.append(["image_id"] + [f"dim_{j}" for j in range(len(embeddings[0]))])
                            for img_id, emb in zip(image_ids, embeddings):
                                csv_rows.append([img_id] + emb)
                            
                            csv_buffer = io.StringIO()
                            import csv
                            writer = csv.writer(csv_buffer)
                            writer.writerows(csv_rows)
                            
                            st.download_button(
                                label="📥 Download All Embeddings (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"all_embeddings_{selected_model}.csv",
                                mime="text/csv",
                                key="download_all_emb"
                            )
                            
                    except Exception as e:
                        st.error(f"Failed to extract embeddings: {str(e)}")
                        if "PyTorch is not installed" in str(e):
                            st.info("💡 Please install PyTorch in the backend: `pip install torch torchvision`")

    # --- Similarity Search ---
    with tab4:
        st.subheader("🔍 Image Similarity Search")
        
        if not images:
            st.warning("Please upload some images first in the Data Input section.")
            return
        
        # Initialize uploaded_ids from session state or use a placeholder
        uploaded_ids = st.session_state.get("uploaded_image_ids", [])
        
        # Silently attempt to ensure sync only if there's a clear mismatch
        if len(uploaded_ids) != len(images):
            try:
                from utils.api_client import get_current_image_ids
                backend_ids = get_current_image_ids()
                if len(backend_ids) == len(images):
                    st.session_state["uploaded_image_ids"] = backend_ids
                    uploaded_ids = backend_ids
            except:
                # If sync fails, just proceed - similarity search can still work with indices
                pass
        
        # Get available methods
        try:
            methods_info = get_similarity_methods()
            available_methods = methods_info.get("feature_methods", ["CNN", "HOG", "SIFT", "histogram"])
            available_metrics = methods_info.get("distance_metrics", ["cosine", "euclidean", "manhattan"])
        except Exception:
            available_methods = ["CNN", "HOG", "SIFT", "histogram"]
            available_metrics = ["cosine", "euclidean", "manhattan"]
        
        # Explanation of similarity scoring
        with st.expander("ℹ️ How Similarity Scores Are Calculated", expanded=False):
            st.markdown("""
            ### 🧠 **Feature Extraction Methods**
            
            **1. CNN (Convolutional Neural Network) Features:**
            - Divides image into 16×16 pixel patches
            - Calculates statistical features: mean, std, min, max, median for each patch
            - Good for: Overall image structure and texture patterns
            
            **2. HOG (Histogram of Oriented Gradients):**
            - Analyzes edge directions and gradients in image regions  
            - Parameters: orientations (default: 9), pixels per cell (8×8), cells per block (2×2)
            - Good for: Shape detection and object recognition
            
            **3. SIFT (Scale-Invariant Feature Transform):**
            - Detects distinctive keypoints and creates descriptors
            - Aggregates up to 100 keypoint descriptors into a single feature vector
            - Good for: Finding similar objects regardless of rotation/scale
            
            **4. Histogram Features:**
            - Analyzes color distribution across RGB channels
            - Creates normalized histograms with 64 bins per channel by default
            - Good for: Color-based similarity comparison
            
            ### 📏 **Distance Metrics & Score Calculation**
            
            **Cosine Similarity:**
            - Measures angle between feature vectors (0° = identical, 90° = orthogonal)
            - Score = (cosine_similarity + 1) ÷ 2  → Range: 0.0 to 1.0
            - **1.0 = identical images, 0.0 = completely different**
            
            **Euclidean Distance:**  
            - Measures straight-line distance between feature points
            - Score = exp(-distance ÷ normalization_factor)  → Range: 0.0 to 1.0
            - **1.0 = identical, lower = more different**
            
            **Manhattan Distance:**
            - Measures sum of absolute differences between features  
            - Score = exp(-distance ÷ normalization_factor)  → Range: 0.0 to 1.0
            - **1.0 = identical, lower = more different**
            
            ### 🎯 **Score Interpretation**
            - **0.9 - 1.0:** Very similar (near identical)
            - **0.7 - 0.9:** Quite similar  
            - **0.5 - 0.7:** Moderately similar
            - **0.3 - 0.5:** Somewhat different
            - **0.0 - 0.3:** Very different
            """)
            
            st.info("💡 **Tip:** Different feature methods work better for different types of images. Try multiple methods to find what works best for your data!")
        
        # Comparison mode selection
        st.markdown("### 🎯 What do you want to compare?")
        comparison_mode = st.radio(
            "Choose comparison mode:",
            ["Compare one image against all others", "Compare two specific images"],
            key="sim_comparison_mode"
        )
        
        if comparison_mode == "Compare one image against all others":
            st.markdown("**Mode: Single vs All** - Find images similar to your query image")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### 📷 Query Image (What to search for)")
                
                # Query image selector with preview
                query_idx = st.selectbox(
                    "Select query image:",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="sim_query_idx",
                    help="This image will be compared against all other images"
                )
                
                st.image(images[query_idx], caption=f"🔍 Query: Image {query_idx+1}", width=200)
                st.info(f"Searching for images similar to Image {query_idx+1}")
                
                # Search parameters
                st.markdown("##### ⚙️ Search Settings")
                
                # Feature method with descriptions
                method_options = {
                    "CNN": "CNN (texture patterns)",
                    "HOG": "HOG (shapes & edges)",
                    "SIFT": "SIFT (keypoint features)",
                    "histogram": "Histogram (color distribution)"
                }
                method_display = [method_options.get(method, method) for method in available_methods]
                selected_method_idx = st.selectbox("Feature Method:", range(len(available_methods)), 
                                                   format_func=lambda x: method_display[x], key="sim_method")
                feature_method = available_methods[selected_method_idx]
                
                # Distance metric with descriptions
                metric_options = {
                    "cosine": "Cosine (angle similarity)",
                    "euclidean": "Euclidean (straight distance)",
                    "manhattan": "Manhattan (grid distance)"
                }
                metric_display = [metric_options.get(metric, metric) for metric in available_metrics]
                selected_metric_idx = st.selectbox("Distance Metric:", range(len(available_metrics)),
                                                   format_func=lambda x: metric_display[x], key="sim_metric")
                distance_metric = available_metrics[selected_metric_idx]
                max_results = st.slider("Max Results:", 1, min(20, len(images)-1), 5, key="sim_max_results")
                
                # Advanced parameters
                with st.expander("🔧 Advanced Parameters"):
                    use_threshold = st.checkbox("Use similarity threshold", key="sim_use_threshold")
                    threshold = st.slider("Threshold:", 0.0, 1.0, 0.5, 0.01, key="sim_threshold") if use_threshold else None
                    
                    if feature_method == "HOG":
                        st.markdown("**HOG Parameters:**")
                        hog_orientations = st.slider("Orientations:", 1, 32, 9, key="sim_hog_orient")
                    else:
                        hog_orientations = None
                    
                    if feature_method == "histogram":
                        st.markdown("**Histogram Parameters:**")
                        hist_bins = st.slider("Bins:", 8, 256, 64, key="sim_hist_bins")
                    else:
                        hist_bins = None
            
            with col2:
                st.markdown("##### 📊 Search Results")
                
                # Performance optimization
                if st.button("🚀 Precompute Features (Recommended)", key="sim_precompute"):
                    with st.spinner("Precomputing features for faster searches..."):
                        try:
                            result = precompute_similarity_features(feature_method=feature_method)
                            st.success(f"✅ {result.get('message', 'Features precomputed successfully!')}")
                        except Exception as e:
                            st.error(f"❌ Precompute failed: {e}")
                
                # Main search button
                if st.button("🔍 Find Similar Images", type="primary", key="sim_search_btn", width='stretch'):
                    with st.spinner(f"Searching for images similar to Image {query_idx+1}..."):
                        try:
                            # Build search parameters
                            search_params = {
                                "query_image_index": query_idx,
                                "feature_method": feature_method,
                                "distance_metric": distance_metric,
                                "max_results": max_results,
                            }
                            
                            if use_threshold and threshold is not None:
                                search_params["threshold"] = threshold
                            if feature_method == "HOG" and hog_orientations:
                                search_params["hog_orientations"] = hog_orientations
                            if feature_method == "histogram" and hist_bins:
                                search_params["hist_bins"] = hist_bins
                            
                            # Perform search
                            results = similarity_search(**search_params)
                            similar_images = results.get("similar_images", [])
                            
                            if not similar_images:
                                st.warning("No similar images found. Try adjusting parameters or threshold.")
                            else:
                                st.success(f"🎉 Found {len(similar_images)} similar images!")
                                
                                # Build image_id to index mapping
                                id_to_index = {img_id: idx for idx, img_id in enumerate(st.session_state.uploaded_image_ids)}
                                # Display results
                                for i, img_data in enumerate(similar_images[:6]):
                                    image_id = img_data["image_id"]
                                    similarity_score = img_data["similarity_score"]
                                    image_idx = id_to_index.get(image_id)
                                    result_col1, result_col2 = st.columns([1, 2])
                                    with result_col1:
                                        if image_idx is not None and image_idx < len(images):
                                            st.image(images[image_idx], width=80)
                                        else:
                                            st.write("📷")
                                    with result_col2:
                                        if image_idx is not None:
                                            st.write(f"**Image {image_idx + 1}** ({image_id})")
                                        else:
                                            st.write(f"**{image_id}**")
                                        st.metric("Similarity", f"{similarity_score:.3f}", help="Higher = more similar")
                                    if i < len(similar_images) - 1:
                                        st.markdown("---")
                                
                                # Detailed results
                                with st.expander(f"📋 All {len(similar_images)} Results"):
                                    import pandas as pd
                                    df = pd.DataFrame(similar_images)
                                    df['similarity_score'] = df['similarity_score'].round(4)
                                    df['distance'] = df['distance'].round(4)
                                    st.dataframe(df, width='stretch')
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "list index out of range" in error_msg:
                                st.error("❌ Image synchronization error detected.")
                                st.info("💡 The images and their IDs are not properly synchronized. Try re-uploading your images.")
                                st.info("🔄 Or refresh the page and upload images again.")
                            elif "422" in error_msg or "Unprocessable Entity" in error_msg:
                                st.error("❌ Invalid request parameters. Please check your search settings.")
                                st.info("💡 Try adjusting feature method parameters or contact support.")
                            elif "400" in error_msg or "query_image_index" in error_msg:
                                st.error("❌ Invalid query image selection.")
                                st.info("💡 Make sure you have uploaded images and selected a valid query image.")
                            elif "404" in error_msg:
                                st.error("❌ Similarity search service not available.")
                                st.info("💡 Check if the backend server is running on http://localhost:8000")
                            elif "500" in error_msg:
                                st.error("❌ Internal server error occurred.")
                                st.info("💡 Try precomputing features first or check backend logs.")
                            else:
                                st.error(f"❌ Search failed: {error_msg}")
                                st.info("💡 Please try again or check your connection to the backend.")
        
        else:  # Pairwise comparison
            st.markdown("**Mode: Pairwise Comparison** - Compare two specific images directly")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("##### 📷 First Image")
                img1_idx = st.selectbox(
                    "Select first image:",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="sim_img1_idx"
                )
                st.image(images[img1_idx], caption=f"Image {img1_idx+1}", width=150)
            
            with col2:
                st.markdown("##### 📷 Second Image")
                available_indices = [i for i in range(len(images)) if i != img1_idx]
                if available_indices:
                    img2_idx = st.selectbox(
                        "Select second image:",
                        options=available_indices,
                        format_func=lambda x: f"Image {x+1}",
                        key="sim_img2_idx"
                    )
                    st.image(images[img2_idx], caption=f"Image {img2_idx+1}", width=150)
                else:
                    st.warning("Need at least 2 images for pairwise comparison")
                    img2_idx = None
            
            with col3:
                st.markdown("##### ⚙️ Comparison Settings")
                
                # Feature method with descriptions for pairwise
                method_options = {
                    "CNN": "CNN (texture patterns)",
                    "HOG": "HOG (shapes & edges)", 
                    "SIFT": "SIFT (keypoint features)",
                    "histogram": "Histogram (color distribution)"
                }
                method_display = [method_options.get(method, method) for method in available_methods]
                selected_method_pair_idx = st.selectbox("Feature Method:", range(len(available_methods)),
                                                        format_func=lambda x: method_display[x], key="sim_method_pair")
                feature_method_pair = available_methods[selected_method_pair_idx]
                
                # Distance metric with descriptions for pairwise
                metric_options = {
                    "cosine": "Cosine (angle similarity)",
                    "euclidean": "Euclidean (straight distance)",
                    "manhattan": "Manhattan (grid distance)"
                }
                metric_display = [metric_options.get(metric, metric) for metric in available_metrics]
                selected_metric_pair_idx = st.selectbox("Distance Metric:", range(len(available_metrics)),
                                                        format_func=lambda x: metric_display[x], key="sim_metric_pair") 
                distance_metric_pair = available_metrics[selected_metric_pair_idx]
                
                if img2_idx is not None and st.button("🔍 Compare Images", type="primary", key="sim_compare_btn"):
                    with st.spinner(f"Comparing Image {img1_idx+1} vs Image {img2_idx+1}..."):
                        try:
                            # Use the first image as query and search for the second
                            search_params = {
                                "query_image_index": img1_idx,
                                "feature_method": feature_method_pair,
                                "distance_metric": distance_metric_pair,
                                "max_results": len(images),
                            }
                            
                            results = similarity_search(**search_params)
                            similar_images = results.get("similar_images", [])
                            
                            # Find the second image in results (IDs should now be synchronized)
                            uploaded_ids = st.session_state.get("uploaded_image_ids", [])
                            if img2_idx < len(uploaded_ids):
                                img2_id = uploaded_ids[img2_idx]
                                similarity_result = next((img for img in similar_images if img["image_id"] == img2_id), None)
                                
                                if similarity_result:
                                    similarity_score = similarity_result["similarity_score"]
                                    st.metric("🎯 Similarity Score", f"{similarity_score:.4f}", help="0 = completely different, 1 = identical")
                                    
                                    if similarity_score > 0.8:
                                        st.success(f"🎉 Very similar images! (Score: {similarity_score:.3f})")
                                    elif similarity_score > 0.6:
                                        st.info(f"🤔 Moderately similar images. (Score: {similarity_score:.3f})")
                                    else:
                                        st.warning(f"🤷 Images are quite different. (Score: {similarity_score:.3f})")
                                else:
                                    st.error("❌ Could not find the second image in similarity results.")
                                    st.info("💡 This might be a backend issue. Try refreshing or re-uploading images.")
                            else:
                                st.info("🔄 Images are being synchronized. Please try the comparison again.")
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "list index out of range" in error_msg:
                                st.error("❌ Image synchronization error detected.")
                                st.info("💡 The images and their IDs are not properly synchronized. Try re-uploading your images.")
                                st.info("🔄 Or refresh the page and upload images again.")
                            elif "422" in error_msg or "Unprocessable Entity" in error_msg:
                                st.error("❌ Invalid comparison parameters.")
                                st.info("💡 Try adjusting feature method parameters or contact support.")
                            elif "400" in error_msg or "query_image_index" in error_msg:
                                st.error("❌ Invalid image selection for comparison.")
                                st.info("💡 Make sure you have selected two different valid images.")
                            elif "404" in error_msg:
                                st.error("❌ Similarity search service not available.")
                                st.info("💡 Check if the backend server is running on http://localhost:8000")
                            elif "500" in error_msg:
                                st.error("❌ Internal server error during comparison.")
                                st.info("💡 Try precomputing features first or check backend logs.")
                            else:
                                st.error(f"❌ Comparison failed: {error_msg}")
                                st.info("💡 Please try again or check your connection to the backend.")
        
        # Help section
        st.markdown("---")
        with st.expander("ℹ️ How to Use Similarity Search"):
            st.markdown("""
            **🎯 Comparison Modes:**
            - **Single vs All**: Select one query image and find all similar images in your dataset
            - **Pairwise**: Compare exactly two images to see how similar they are
            
            **🧠 Feature Methods:**
            - **CNN**: Best general-purpose method, works well for most images
            - **HOG**: Good for shapes, objects, and structural patterns
            - **SIFT**: Best for images with distinctive keypoints, robust to rotation/scaling
            - **Histogram**: Good for color-based similarity
            
            **📊 Distance Metrics:**
            - **Cosine**: Measures angle between feature vectors (recommended)
            - **Euclidean**: Direct distance in feature space
            - **Manhattan**: City-block distance, less sensitive to outliers
            
            **💡 Tips:**
            - Use "Precompute Features" for faster searches when you have many images
            - Try different feature methods to find what works best for your images
            - Adjust similarity threshold to filter results (higher = more strict)
            """)

    # --- Haralick Texture ---
    with tab5:
        st.subheader("🧵 Haralick Texture Tools")
        
        st.caption(
            "Compute Haralick texture features directly from the images you uploaded above. "
            "Choose distances, angles, quantization levels, and (optionally) resize for faster or consistent results."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            # Select images
            use_all = st.checkbox("Use all images", value=True, key="har_use_all")
            if use_all:
                image_indices = list(range(len(images)))
            else:
                image_indices = st.multiselect(
                    "Select images",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="har_img_indices",
                )
            # Levels
            levels = st.selectbox("Quantization levels", [16, 32, 64, 128, 256], index=4, key="har_levels")
            # Distances
            distances = st.multiselect("Distances (pixels)", [1, 2, 3, 5], default=[1, 2], key="har_distances")
        with col_b:
            # Angles (radians)
            angle_map = {
                "0°": 0.0,
                "45°": math.pi / 4,
                "90°": math.pi / 2,
                "135°": 3 * math.pi / 4,
            }
            angle_labels = list(angle_map.keys())
            angles_sel = st.multiselect("Angles", angle_labels, default=angle_labels, key="har_angles_lbls")
            angles = [angle_map[a] for a in angles_sel]
            # Resize
            use_resize = st.checkbox("Resize before analysis", value=False, key="har_use_resize")
            if use_resize:
                resize_width = st.number_input("Resize width", min_value=32, max_value=2048, value=256, step=16, key="har_w")
                resize_height = st.number_input("Resize height", min_value=32, max_value=2048, value=256, step=16, key="har_h")
            else:
                resize_width = None
                resize_height = None

        props = st.multiselect(
            "Properties",
            ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
            default=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
            key="har_props",
        )
        avg = st.checkbox("Average across all distances/angles", value=True, key="har_avg")

        if st.button("Compute Haralick (GLCM)", key="btn_haralick_table"):
            if not image_indices:
                st.error("Please select at least one image.")
            else:
                with st.spinner("Computing Haralick features..."):
                    try:
                        payload = {
                            "image_indices": image_indices,
                            "levels": int(levels),
                            "distances": distances if distances else [1],
                            "angles": angles if angles else [0.0],
                            "resize_width": int(resize_width) if resize_width else None,
                            "resize_height": int(resize_height) if resize_height else None,
                            "average_over_angles": bool(avg),
                            "properties": props,
                        }
                        result = extract_haralick_features(payload)
                        cols = result.get("columns", [])
                        rows = result.get("rows", [])
                        if cols and rows:
                            st.success(f"Computed features for {len(rows)} image(s).")
                            st.dataframe(
                                {cols[i]: [row[i] for row in rows] for i in range(len(cols))},
                                width='stretch',
                            )
                            csv_bytes = _to_csv_bytes(cols, rows)
                            st.download_button(
                                "Download CSV",
                                data=csv_bytes,
                                file_name="haralick_features.csv",
                                mime="text/csv",
                                key="btn_dl_haralick_csv",
                            )
                        else:
                            st.warning("No features returned.")
                    except Exception as e:
                        st.error(f"Haralick computation failed: {e}")

        st.divider()

        # --- Train & Predict workflow (formerly "legacy") ---
        st.markdown("##### Train & Predict from Labeled Images")
        st.caption(
            "Train a quick classifier using Haralick features:\n"
            "1) Upload training images, 2) Upload a CSV mapping filenames to labels, 3) Upload test images to classify."
        )
        st.caption(
            "CSV example:\n"
            "`filename,label`  →  `img1.jpg,classA`  `img2.jpg,classB`"
        )

        train_imgs = st.file_uploader(
            "1) Upload Training Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_train_imgs"
        )
        train_csv = st.file_uploader("2) Upload Training Labels (CSV)", type="csv", key="har_train_csv")
        test_imgs = st.file_uploader(
            "3) Upload Test Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_test_imgs"
        )

        # Keep key the same; only label changes
        if st.button("Train & Predict", key="btn_haralick"):
            try:
                labels, preds = extract_haralick_texture(
                    {"train_images": train_imgs, "train_labels": train_csv, "test_images": test_imgs}
                )
                
                st.success(f"✅ Successfully trained classifier with {len(labels)} training samples")
                
                st.subheader("📊 Prediction Results:")
                if preds:
                    for i, prediction in enumerate(preds):
                        # Predictions now come as "filename: predicted_class"
                        st.write(f"🔍 {prediction}")
                else:
                    st.warning("No predictions were generated. Please check your test images.")
                    
            except Exception as e:
                st.error(f"❌ Haralick prediction failed: {str(e)}")
                st.info("💡 Make sure your CSV has the correct format: filename,label")

    # --- Local Binary Patterns (LBP) ---
    with tab6:
        st.subheader("🔢 Local Binary Patterns (LBP)")

        st.caption(
            "Compute LBP texture histograms. Choose parameters below and run on one, multiple, or all images. "
            "In single-image mode, a visualization of the LBP-coded image may be shown if available."
        )

        col_l, col_r = st.columns(2)
        with col_l:
            use_all_lbp = st.checkbox("Use all images", value=False, key="lbp_use_all")
            if use_all_lbp:
                image_indices_lbp = list(range(len(images)))
                st.info(f"All {len(images)} images will be processed.")
            else:
                image_indices_lbp = st.multiselect(
                    "Select images",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="lbp_img_indices",
                )

        with col_r:
            radius = st.number_input("Radius", min_value=1, max_value=16, value=2, step=1, key="lbp_radius")
            neighbors = st.selectbox("Number of Neighbors", [8, 16, 24], index=1, key="lbp_neighbors")
            method = st.selectbox("Method", ["default", "ror", "uniform", "var"], index=2, key="lbp_method")
            normalize = st.checkbox("Normalize histogram", value=True, key="lbp_normalize")

        if st.button("Compute LBP", key="btn_lbp_compute"):
            # Validate selection when not "all"
            if not use_all_lbp and not image_indices_lbp:
                st.error("Please select at least one image or enable 'Use all images'.")
            else:
                with st.spinner("Computing LBP features..."):
                    try:
                        payload = {
                            "image_indices": image_indices_lbp,
                            "use_all_images": bool(use_all_lbp),
                            "radius": int(radius),
                            "num_neighbors": int(neighbors),
                            "method": method,
                            "normalize": bool(normalize),
                        }
                        result = extract_lbp_features(payload)

                        # Two response modes supported by backend:
                        if isinstance(result, dict) and result.get("mode") == "single":
                            # Single image: show histogram values and optional visualization
                            image_id = result.get("image_id", "image")
                            bins = int(result.get("bins", 0))
                            hist = result.get("histogram", [])
                            st.success(f"LBP histogram computed for {image_id} ({bins} bins).")
                            st.write(hist)

                            # Optional LBP-coded visualization
                            lbp_img_bytes = result.get("lbp_image_bytes")
                            if lbp_img_bytes:
                                st.image(lbp_img_bytes, caption="LBP-coded image", width=400)
                                if st.button("Enlarge LBP Image", key="enlarge_lbp_single"):
                                    st.session_state["fullscreen_image"] = lbp_img_bytes
                                    st.session_state["fullscreen_section"] = "LBP-coded Image"
                                    st.rerun()

                            # Download CSV
                            cols = [f"bin_{i}" for i in range(len(hist))]
                            csv_bytes = _to_csv_bytes(["image_id"] + cols, [[image_id] + hist])
                            st.download_button(
                                "Download LBP CSV (single image)",
                                data=csv_bytes,
                                file_name=f"lbp_{image_id}.csv",
                                mime="text/csv",
                                key="btn_dl_lbp_single",
                            )

                        elif isinstance(result, dict) and result.get("mode") == "multi":
                            # Multi/all images: tabular output
                            cols = result.get("columns", [])
                            rows = result.get("rows", [])
                            if cols and rows:
                                st.success(f"Computed LBP histograms for {len(rows)} image(s).")
                                st.dataframe(
                                    {cols[i]: [row[i] for row in rows] for i in range(len(cols))},
                                    width='stretch',
                                )
                                csv_bytes = _to_csv_bytes(cols, rows)
                                st.download_button(
                                    "Download LBP CSV",
                                    data=csv_bytes,
                                    file_name="lbp_features.csv",
                                    mime="text/csv",
                                    key="btn_dl_lbp_multi",
                                )
                            else:
                                st.warning("No LBP features returned.")
                        else:
                            st.warning("Unexpected LBP response format.")
                    except Exception as e:
                        st.error(f"LBP computation failed: {e}")

    # --- NEW: Contour Extraction (dedicated tab) ---
    with tab7:
        st.subheader("⭕ Contour Extraction")
        st.caption(
            "Extract contours (closed shapes or outlines) from a binary or grayscale version of your image using "
            "OpenCV’s `findContours`. This is useful for shape analysis, counting objects, generating bounding boxes, "
            "and creating polygonal outlines."
        )
        st.markdown(
            "**How it works:** The image is converted to grayscale and thresholded to binary. "
            "Contours are then detected and (optionally) simplified using the selected approximation method."
        )

        selected_idx = st.selectbox(
            "Select Image",
            options=list(range(len(images))),
            format_func=lambda x: f"Image {x+1}",
            key="contour_img_idx",
        )
        mode = st.selectbox(
            "Contour Retrieval Mode",
            ["RETR_EXTERNAL", "RETR_LIST", "RETR_TREE", "RETR_CCOMP"],
            index=0,
            key="contour_mode",
        )
        method = st.selectbox(
            "Contour Approximation Method",
            ["CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE"],
            index=0,
            key="contour_method",
        )
        min_area = st.number_input("Minimum contour area (filter small noise)", min_value=0, value=10, key="contour_min_area")
        return_bb = st.checkbox("Return bounding boxes", value=True, key="contour_return_bb")
        return_hier = st.checkbox("Return hierarchy (OpenCV format)", value=False, key="contour_return_hier")

        # Binarization method selection
        st.markdown("### Binarization Method")
        binarization_method = st.radio(
            "How to convert image to binary (black & white):",
            ["FIXED", "OTSU", "ADAPTIVE", "CANNY"],
            help="""
            - **FIXED**: Simple threshold at value you set (fast but less accurate)
            - **OTSU**: Automatic threshold (good for most cases)
            - **ADAPTIVE**: Local thresholding (better with uneven lighting)
            - **CANNY**: Edge detection (finds object boundaries)
            """,
            key="contour_binarization_method"
        )
        
        # Threshold controls based on method
        if binarization_method == "FIXED":
            threshold_value = st.slider(
                "Threshold Value (0=black, 255=white)", 
                min_value=0, 
                max_value=255, 
                value=127,
                help="Pixels above this value become white, below become black",
                key="contour_threshold_value"
            )
        elif binarization_method == "CANNY":
            col1, col2 = st.columns(2)
            with col1:
                canny_low = st.slider(
                    "Canny Low Threshold",
                    min_value=0,
                    max_value=200,
                    value=50,
                    key="contour_canny_low"
                )
            with col2:
                canny_high = st.slider(
                    "Canny High Threshold",
                    min_value=50,
                    max_value=500,
                    value=150,
                    key="contour_canny_high"
                )
        else:
            threshold_value = 127  # Not used for OTSU or ADAPTIVE
            canny_low = 50
            canny_high = 150

        if st.button("Run Contour Extraction", key="btn_contour_extract"):
            with st.spinner("Extracting contours..."):
                try:
                    params = {
                        "image_index": int(selected_idx),
                        "mode": mode,
                        "method": method,
                        "min_area": int(min_area),
                        "return_bounding_boxes": bool(return_bb),
                        "return_hierarchy": bool(return_hier),
                        "binarization_method": binarization_method,
                        "threshold_value": int(threshold_value) if binarization_method == "FIXED" else 127,
                        "canny_low": int(canny_low) if binarization_method == "CANNY" else 50,
                        "canny_high": int(canny_high) if binarization_method == "CANNY" else 150,
                    }
                    result = extract_contours(params)
                    st.session_state["contour_result"] = result
                    st.success("Contour extraction completed!")

                    # Preview overlay
                    if result.get("visualization_bytes"):
                        st.image(
                            result["visualization_bytes"],
                            caption="Contours overlay",
                            width=400,
                        )

                    # Details
                    areas = result.get("areas", [])
                    bbs = result.get("bounding_boxes", [])
                    st.write(f"Detected contours: **{len(areas)}**")
                    if bbs:
                        st.write("Bounding boxes (x, y, w, h):")
                        st.write(bbs[: min(10, len(bbs))])  # show first few

                    if return_hier:
                        st.write("Hierarchy (OpenCV):")
                        st.write(result.get("hierarchy"))
                except Exception as e:
                    st.error(f"Contour extraction failed: {e}")

    # ----------------------------------------------------------------------
#   Tab 8 – SIFT & Edge Detection
# ----------------------------------------------------------------------
    with tab8:
        st.subheader("🔍 SIFT & Edge Detection")
        st.caption(
            """
            *SIFT* extracts scale‑invariant key‑points and 128‑dimensional descriptors.  
            *Edge detection* (Canny or Sobel) highlights gradients in the image.  
            Choose a single image, a custom list, or run on **all** uploaded images.
            """
        )

        # ---------- Image selection ----------
        col_sel, col_opt = st.columns([2, 1])

        with col_sel:
            # 1️⃣  Single‑image selector (default)
            single_idx = st.selectbox(
                "Select Image (single)",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="sift_single_idx",
            )

            # 2️⃣  Multi‑select (optional)
            multi_idxs = st.multiselect(
                "Or select several images",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="sift_multi_idxs",
            )

            # 3️⃣  “All images” toggle
            use_all = st.checkbox("Run on **all** images", value=False, key="sift_use_all")

        # ---------- Edge‑detection parameters ----------
        with col_opt:
            edge_method = st.radio(
                "Edge‑Detection Method",
                options=["canny", "sobel"],
                index=0,
                key="edge_method",
            )
            if edge_method == "canny":
                low_thr = st.number_input(
                    "Low threshold", min_value=0, max_value=255, value=100, key="canny_low"
                )
                high_thr = st.number_input(
                    "High threshold", min_value=0, max_value=255, value=200, key="canny_high"
                )
            else:  # sobel
                sobel_ks = st.selectbox(
                    "Sobel kernel size",
                    options=[1, 3, 5, 7],
                    index=1,
                    key="sobel_ksize",
                )

        # ---------- Helper to build the payload ----------
        def _build_payload() -> dict:
            """
            Returns a dict that matches the Pydantic model `FeatureBaseRequest`.
            Priority (top → bottom):
                1. use_all → {"all_images": True}
                2. multi_idxs → {"image_indices": [...]}
                3. single_idx → {"image_index": …}
            """
            if use_all:
                return {"all_images": True}
            if multi_idxs:
                return {"image_indices": list(multi_idxs)}
            return {"image_index": int(single_idx)}

        # ---------- Buttons ----------
        col_btn1, col_btn2 = st.columns(2)

        # ---- SIFT ----
        with col_btn1:
            if st.button("Run SIFT", key="btn_sift"):
                payload = _build_payload()
                with st.spinner("Extracting SIFT key‑points…"):
                    try:
                        result = extract_sift_features(payload)

                        # ---- Visualisation (may be None for multi‑image requests) ----
                        viz = result.get("visualization")
                        if viz:
                            st.image(
                                viz,
                                caption="SIFT key‑points (visualisation)",
                                width='stretch',
                            )
                        else:
                            st.info("No visualisation returned (you asked for several images).")

                        # ---- Numeric descriptors (CSV download) ----
                        feats = result.get("features", [])
                        if feats:
                            st.success(f"Extracted **{len(feats)}** SIFT descriptors.")
                            cols = [f"d{i}" for i in range(128)]
                            csv_bytes = _to_csv_bytes(cols, feats)
                            st.download_button(
                                label="Download SIFT descriptors (CSV)",
                                data=csv_bytes,
                                file_name="sift_descriptors.csv",
                                mime="text/csv",
                                key="dl_sift_csv",
                            )
                        else:
                            st.warning("No SIFT descriptors were found.")
                    except Exception as e:
                        st.error(f"SIFT extraction failed: {e}")

        # ---- Edge detection ----
        with col_btn2:
            if st.button("Run Edge Detection", key="btn_edge"):
                payload = _build_payload()
                with st.spinner("Running edge detection…"):
                    try:
                        result = extract_edge_features(
                            payload,
                            method=edge_method,
                            low_thresh=low_thr if edge_method == "canny" else None,
                            high_thresh=high_thr if edge_method == "canny" else None,
                            sobel_ksize=sobel_ks if edge_method == "sobel" else None,
                        )

                        # ---- Show the FIRST edge map ----
                        edge_imgs = result.get("edge_images", [])
                        if edge_imgs:
                            st.image(
                                edge_imgs[0],
                                caption=f"{edge_method.title()} edge map",
                                width='stretch',
                            )
                        else:
                            st.warning("Backend returned no edge images.")

                        # ---- Generate CSV for ALL processed images ----
                        all_matrices = result.get("edges_matrices", [])
                        num_images = len(all_matrices)
                        
                        if num_images > 0:
                            try:
                                st.success(f"Processed {num_images} image{'s' if num_images > 1 else ''}.")
                                
                                # Prepare CSV data
                                csv_rows = []
                                col_names = ["image_id", "row", "col", "value"]
                                
                                for img_idx, matrix in enumerate(all_matrices):
                                    img_id = f"img_{img_idx}"
                                    for row_idx, row in enumerate(matrix):
                                        for col_idx, value in enumerate(row):
                                            csv_rows.append([
                                                img_id,
                                                row_idx,
                                                col_idx,
                                                float(value)
                                            ])
                                
                                # Generate CSV
                                csv_bytes = _to_csv_bytes(col_names, csv_rows)
                                
                                # Download button
                                st.download_button(
                                    label=f"Download edge matrices for {num_images} image{'s' if num_images > 1 else ''} (CSV)",
                                    data=csv_bytes,
                                    file_name=f"edge_matrices_{num_images}_images.csv",
                                    mime="text/csv",
                                    key=f"dl_edge_matrices_{int(time.time())}",
                                )
                                
                                # Optional: Show matrix statistics
                                if num_images == 1:
                                    matrix = all_matrices[0]
                                    st.write(f"Matrix shape: {len(matrix)} × {len(matrix[0])}")
                                else:
                                    st.write(f"Total data points: {len(csv_rows)}")
                                    
                            except Exception as e:
                                st.error(f"CSV generation failed: {str(e)}")
                                st.write("Debug info:")
                                st.write(f"Number of images: {num_images}")
                                if num_images > 0:
                                    st.write(f"First matrix shape: {len(all_matrices[0])} × {len(all_matrices[0][0])}")
                        else:
                            st.info("No gradient matrices were generated.")
                            
                    except Exception as e:
                        st.error(f"Edge detection failed: {e}")

        # --- Navigation ---
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("Next: Statistical Analysis", key="next_stats"):
                st.session_state["active_section"] = "Statistics Analysis"

    # --- Classifier Training ---
    with tab9:
        st.subheader("Classifier Training")
        st.caption(
            "Label your uploaded images, pick a feature representation, and train a classical ML classifier. "
            "Use the holdout split to benchmark the model, or run additional predictions once training completes."
        )

        st.markdown(
            """
            **Quick guide**

            1. Review each image row below and type in a label for every entry you want to use for training.
            2. Decide whether each image should be part of the training group or kept aside for testing.
            3. Pick a feature recipe (HOG, LBP, or CNN embeddings) and the classifier type, then press **Train classifier**.
            4. Once the model is trained you can run extra predictions on any subset of images with a single click.
            """
        )

        image_ids: list[str] = st.session_state.get("uploaded_image_ids", [])

        # Debug: show current state
        st.caption(f"🔍 Debug: {len(images)} images in session state, {len(image_ids)} image IDs from backend")

        if len(image_ids) != len(images):
            with st.spinner("Aligning the uploaded images with the training workspace..."):
                sync_ok = _ensure_backend_sync(force_reupload=False)  # Try without force first
            
            # Re-fetch state after sync
            image_ids = st.session_state.get("uploaded_image_ids", [])
            images = st.session_state.get("images", [])
            
            # Double-check backend state
            from utils.api_client import get_current_image_ids
            actual_backend_ids = get_current_image_ids()
            
            sync_error = st.session_state.pop("classifier_sync_error", None)
            
            st.caption(f"🔍 After sync: {len(images)} images, {len(image_ids)} IDs in session, {len(actual_backend_ids)} actual backend IDs, sync_ok={sync_ok}")
            
            if not sync_ok or len(image_ids) != len(images) or len(actual_backend_ids) != len(images):
                st.error(
                    "We could not align the uploaded images with the training workspace. "
                    "Please refresh the dataset so every image has a matching identifier."
                )
                if sync_error:
                    if "Connection refused" in sync_error or "Failed to establish a new connection" in sync_error:
                        st.info(
                            "The analysis server is not responding. Please start the FastAPI backend "
                            "(run `uvicorn app.main:app --reload` inside the `Backend` folder) and try again."
                        )
                    else:
                        st.caption(f"Technical detail: {sync_error}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Refresh training dataset", key="classifier_resync_btn"):
                        st.session_state.pop("classifier_training_df", None)
                        st.session_state.pop("uploaded_image_ids", None)
                        st.session_state.pop("classifier_sync_error", None)
                        st.session_state.pop("images", None)
                        st.rerun()
                with col2:
                    if st.button("Force re-upload images", key="classifier_force_reupload_btn"):
                        _ensure_backend_sync(force_reupload=True)
                        st.rerun()
                st.info(
                    "Tip: If images were recently added or replaced, visit the **Data Input** section and upload them again, "
                    "then return to this tab."
                )
                return

        st.markdown("#### 1. Label Your Dataset")
        classifier_table: pd.DataFrame = _ensure_classifier_table(image_ids).copy()
        classifier_table["label"] = classifier_table["label"].fillna("").astype(str)
        classifier_table["split"] = classifier_table["split"].fillna("train").astype(str)

        edited_df = st.data_editor(
            classifier_table,
            column_config={
                "label": st.column_config.TextColumn("Label", help="Assign a class label (required for training rows)."),
                "split": st.column_config.SelectboxColumn(
                    "Split",
                    options=["train", "test"],
                    help="Rows marked as 'train' are used for fitting. 'test' rows are held out for evaluation.",
                ),
                "image_index": st.column_config.NumberColumn("Image Index", format="%d"),
                "image_id": st.column_config.TextColumn("Image ID"),
            },
            disabled=["image_index", "image_id"],
            hide_index=True,
            width='stretch',
            key="classifier_training_editor",
        )

        # Normalise edited values and persist them in session_state
        if isinstance(edited_df, pd.DataFrame):
            classifier_table = edited_df.copy()
        else:
            classifier_table = pd.DataFrame(edited_df)

        classifier_table["image_index"] = classifier_table["image_index"].astype(int)
        classifier_table["label"] = classifier_table["label"].fillna("").astype(str)
        classifier_table["split"] = classifier_table["split"].fillna("train").astype(str).str.lower()

        st.session_state["classifier_training_df"] = classifier_table

        classifier_table["label"] = classifier_table["label"].str.strip()
        train_rows = classifier_table[(classifier_table["split"] == "train") & (classifier_table["label"] != "")]
        test_rows = classifier_table[classifier_table["split"] == "test"]

        col_a, col_b = st.columns(2)
        col_a.metric("Labeled training images", len(train_rows))
        distinct_labels = sorted(train_rows["label"].unique())
        col_b.metric("Unique labels", len(distinct_labels))

        if len(distinct_labels) < 2:
            st.info("Provide at least two different labels across your training images.")

        st.markdown("#### 2. Feature Extraction")
        feature_type = st.selectbox(
            "Feature representation",
            options=["hog", "lbp", "embedding"],
            format_func=lambda x: {
                "hog": "HOG (Histogram of Oriented Gradients)",
                "lbp": "Local Binary Pattern Histogram",
                "embedding": "CNN Embeddings (ResNet/MobileNet)",
            }[x],
            index=0,
            key="classifier_feature_type",
        )

        hog_payload: dict | None = None
        lbp_payload: dict | None = None
        embedding_model: str | None = None

        if feature_type == "hog":
            with st.expander("HOG options", expanded=False):
                hog_orient = st.slider("Orientations", min_value=1, max_value=24, value=9, key="hog_orientations")
                cell_col1, cell_col2 = st.columns(2)
                with cell_col1:
                    hog_cell_h = st.number_input(
                        "Pixels per cell (height)",
                        min_value=4,
                        max_value=128,
                        value=8,
                        step=1,
                        key="hog_cell_h",
                    )
                with cell_col2:
                    hog_cell_w = st.number_input(
                        "Pixels per cell (width)",
                        min_value=4,
                        max_value=128,
                        value=8,
                        step=1,
                        key="hog_cell_w",
                    )
                block_col1, block_col2 = st.columns(2)
                with block_col1:
                    hog_block_y = st.number_input(
                        "Cells per block (rows)",
                        min_value=1,
                        max_value=8,
                        value=2,
                        step=1,
                        key="hog_block_y",
                    )
                with block_col2:
                    hog_block_x = st.number_input(
                        "Cells per block (columns)",
                        min_value=1,
                        max_value=8,
                        value=2,
                        step=1,
                        key="hog_block_x",
                    )
                resize_col1, resize_col2 = st.columns(2)
                with resize_col1:
                    hog_resize_w = st.number_input(
                        "Resize width (pixels)",
                        min_value=32,
                        max_value=1024,
                        value=128,
                        step=16,
                        key="hog_resize_width",
                    )
                with resize_col2:
                    hog_resize_h = st.number_input(
                        "Resize height (pixels)",
                        min_value=32,
                        max_value=1024,
                        value=128,
                        step=16,
                        key="hog_resize_height",
                    )
                hog_block_norm = st.selectbox(
                    "Block normalisation",
                    options=["L2-Hys", "L1", "L1-sqrt", "L2"],
                    index=0,
                    key="hog_block_norm",
                )

            hog_payload = {
                "orientations": int(hog_orient),
                "pixels_per_cell": [int(hog_cell_h), int(hog_cell_w)],
                "cells_per_block": [int(hog_block_y), int(hog_block_x)],
                "resize_width": int(hog_resize_w),
                "resize_height": int(hog_resize_h),
                "block_norm": hog_block_norm,
            }

        elif feature_type == "lbp":
            with st.expander("LBP options", expanded=False):
                lbp_radius = st.slider("Radius", min_value=1, max_value=16, value=1, key="lbp_radius")
                lbp_neighbors = st.slider(
                    "Number of sampling points",
                    min_value=4,
                    max_value=32,
                    value=8,
                    step=1,
                    key="lbp_neighbors",
                )
                lbp_method = st.selectbox(
                    "LBP method",
                    options=["default", "ror", "uniform", "var"],
                    index=2,
                    key="lbp_method",
                )
                lbp_normalize = st.checkbox("Normalize histogram", value=True, key="lbp_normalize")

            lbp_payload = {
                "radius": int(lbp_radius),
                "num_neighbors": int(lbp_neighbors),
                "method": lbp_method,
                "normalize": bool(lbp_normalize),
            }

        else:  # embedding
            embedding_model = st.selectbox(
                "Embedding backbone",
                options=["resnet50", "resnet18", "mobilenet_v2"],
                format_func=lambda x: {
                    "resnet50": "ResNet-50 (2048-d)",
                    "resnet18": "ResNet-18 (512-d)",
                    "mobilenet_v2": "MobileNet V2 (1280-d)",
                }[x],
                index=0,
                key="classifier_embedding_model",
            )

        st.markdown("#### 3. Classifier Configuration")
        classifier_type = st.selectbox(
            "Classifier",
            options=["svm", "knn", "logistic"],
            format_func=lambda x: {
                "svm": "Support Vector Machine",
                "knn": "k-Nearest Neighbours",
                "logistic": "Logistic Regression",
            }[x],
            index=0,
            key="classifier_type",
        )

        hyperparameters: dict[str, object] = {}
        if classifier_type == "svm":
            with st.expander("SVM hyperparameters", expanded=False):
                svm_kernel = st.selectbox(
                    "Kernel",
                    options=["rbf", "linear", "poly", "sigmoid"],
                    index=0,
                    key="svm_kernel",
                )
                svm_c = st.number_input("C (regularisation)", min_value=0.01, max_value=1000.0, value=1.0, step=0.5, key="svm_c")
                svm_gamma = st.selectbox(
                    "Gamma",
                    options=["scale", "auto"],
                    index=0,
                    key="svm_gamma",
                )
            hyperparameters = {"kernel": svm_kernel, "C": float(svm_c), "gamma": svm_gamma}

        elif classifier_type == "knn":
            with st.expander("k-NN hyperparameters", expanded=False):
                knn_neighbors = st.slider("Number of neighbours", min_value=1, max_value=25, value=5, key="knn_neighbors")
                knn_weights = st.selectbox(
                    "Weighting",
                    options=["uniform", "distance"],
                    index=0,
                    key="knn_weights",
                )
            hyperparameters = {"n_neighbors": int(knn_neighbors), "weights": knn_weights}

        else:  # logistic regression
            with st.expander("Logistic regression hyperparameters", expanded=False):
                log_reg_c = st.number_input(
                    "Inverse regularisation strength (C)",
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0,
                    step=0.5,
                    key="log_reg_c",
                )
                log_reg_iter = st.number_input(
                    "Max iterations",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100,
                    key="log_reg_max_iter",
                )
            hyperparameters = {"C": float(log_reg_c), "max_iter": int(log_reg_iter)}

        st.markdown("#### 4. Train & Evaluate")
        config_col1, config_col2, config_col3 = st.columns(3)
        with config_col1:
            test_size = st.slider(
                "Validation split size",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="classifier_validation_split",
            )
        with config_col2:
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=10_000,
                value=42,
                step=1,
                key="classifier_random_state",
            )
        with config_col3:
            return_probabilities = st.checkbox(
                "Return probabilities",
                value=True,
                key="classifier_return_probabilities",
            )

        train_button_disabled = len(train_rows) < 2 or len(distinct_labels) < 2

        if st.button("Train classifier", type="primary", disabled=train_button_disabled, key="classifier_train_btn"):
            if train_button_disabled:
                st.warning("Select at least two training images with different labels.")
            else:
                training_samples = [
                    {"image_index": int(row.image_index), "label": str(row.label)}
                    for row in train_rows.itertuples()
                ]
                test_samples = []
                for row in test_rows.itertuples():
                    sample = {"image_index": int(row.image_index)}
                    if isinstance(row.label, str) and row.label.strip():
                        sample["label"] = row.label.strip()
                    test_samples.append(sample)

                payload: dict[str, object] = {
                    "feature_type": feature_type,
                    "classifier_type": classifier_type,
                    "training_samples": training_samples,
                    "test_size": float(test_size),
                    "random_state": int(random_state),
                    "return_probabilities": bool(return_probabilities),
                }

                if hyperparameters:
                    payload["hyperparameters"] = hyperparameters
                if feature_type == "hog" and hog_payload:
                    payload["hog_options"] = hog_payload
                if feature_type == "lbp" and lbp_payload:
                    payload["lbp_options"] = lbp_payload
                if feature_type == "embedding" and embedding_model:
                    payload["embedding_model"] = embedding_model
                if test_samples:
                    payload["test_samples"] = test_samples

                try:
                    with st.spinner("Training classifier..."):
                        result = train_classifier(payload)
                    st.session_state["classifier_training_result"] = result
                    st.session_state["trained_model_id"] = result.get("model_id")
                    st.success("Model trained successfully.")
                except Exception as e:
                    error_msg = str(e)
                    response_obj = getattr(e, "response", None)
                    if response_obj is not None:
                        try:
                            error_msg = response_obj.json().get("detail", error_msg)
                        except Exception:
                            pass
                    st.error(f"Training failed: {error_msg}")

        latest_training = st.session_state.get("classifier_training_result")
        if latest_training:
            st.markdown("#### Training Summary")
            st.write(f"Model ID: `{latest_training.get('model_id')}`")
            st.write(
                f"Feature type: **{latest_training.get('feature_type', '').upper()}**, "
                f"Classifier: **{latest_training.get('classifier_type', '').upper()}**"
            )
            st.write(f"Training samples: {latest_training.get('num_training_samples', 0)} "
                     f"| Feature length: {latest_training.get('feature_vector_length')}")

            _render_metrics_section(latest_training.get("train_metrics"), "Training Metrics")
            _render_metrics_section(latest_training.get("validation_metrics"), "Validation Metrics")
            _render_metrics_section(latest_training.get("test_metrics"), "Test Metrics")

            test_predictions = latest_training.get("test_predictions") or []
            if test_predictions:
                pred_rows = []
                for entry in test_predictions:
                    pred_rows.append(
                        {
                            "Image": f"Image {int(entry.get('image_index', -1)) + 1}",
                            "Image ID": entry.get("image_id", ""),
                            "Prediction": entry.get("prediction", ""),
                            "Probabilities": _probabilities_to_string(entry.get("probabilities")),
                            "Actual Label": entry.get("actual_label") or "",
                        }
                    )
                st.caption("Holdout predictions")
                st.dataframe(pd.DataFrame(pred_rows), width='stretch')

        trained_model_id = st.session_state.get("trained_model_id")
        if trained_model_id:
            st.divider()
            st.markdown("#### Run Additional Predictions")
            default_indices = test_rows["image_index"].tolist()
            prediction_indices = st.multiselect(
                "Select images for inference",
                options=list(range(len(images))),
                default=default_indices,
                format_func=lambda x: f"Image {x+1}",
                key="prediction_indices",
            )
            include_probs = st.checkbox(
                "Return probabilities for inference",
                value=True,
                key="prediction_return_probabilities",
            )

            if st.button("Predict with trained model", key="predict_with_trained_model"):
                if not prediction_indices:
                    st.warning("Select at least one image to run predictions.")
                else:
                    samples_payload = []
                    for idx in prediction_indices:
                        sample = {"image_index": int(idx)}
                        match = classifier_table[classifier_table["image_index"] == idx]
                        if not match.empty:
                            label_value = str(match.iloc[0]["label"]).strip()
                            if label_value:
                                sample["label"] = label_value
                        samples_payload.append(sample)

                    predict_payload = {
                        "model_id": trained_model_id,
                        "samples": samples_payload,
                        "return_probabilities": bool(include_probs),
                    }

                    try:
                        with st.spinner("Running inference..."):
                            prediction_result = predict_classifier(predict_payload)
                        st.session_state["latest_prediction_result"] = prediction_result
                        st.success("Predictions generated.")
                    except Exception as e:
                        error_msg = str(e)
                        response_obj = getattr(e, "response", None)
                        if response_obj is not None:
                            try:
                                error_msg = response_obj.json().get("detail", error_msg)
                            except Exception:
                                pass
                        st.error(f"Prediction failed: {error_msg}")

        latest_prediction = st.session_state.get("latest_prediction_result")
        if latest_prediction:
            st.markdown("#### Prediction Results")
            st.write(f"Model ID: `{latest_prediction.get('model_id')}`")
            prediction_rows = []
            for entry in latest_prediction.get("predictions", []):
                prediction_rows.append(
                    {
                        "Image": f"Image {int(entry.get('image_index', -1)) + 1}",
                        "Image ID": entry.get("image_id", ""),
                        "Prediction": entry.get("prediction", ""),
                        "Probabilities": _probabilities_to_string(entry.get("probabilities")),
                        "Actual Label": entry.get("actual_label") or "",
                    }
                )
            if prediction_rows:
                st.dataframe(pd.DataFrame(prediction_rows), width='stretch')
            _render_metrics_section(latest_prediction.get("metrics"), "Prediction Metrics")
    
    # =========================
    # Seher AI Social Chat Widget
    # =========================


# ---------- Utility ----------
def create_histogram_zip(hist_list):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_bytes in enumerate(hist_list):
            zip_file.writestr(f"histogram_{i+1}.png", img_bytes)
    return zip_buffer.getvalue()


if __name__ == "__main__":
    render_feature_selection()
