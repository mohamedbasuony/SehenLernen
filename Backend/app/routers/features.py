# backend/app/routers/features.py

import logging
import base64
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.feature_service import (
    generate_histogram_service,
    perform_kmeans_service,
    perform_single_image_kmeans_service,
    extract_shape_service,
    extract_haralick_service,          # legacy train/predict demo (multipart)
    extract_haralick_features_service,  # table-style Haralick extraction
    compute_lbp_service,                # LBP service
    extract_contours_service,           # Contour extraction
    extract_sift_service,               # SIFT endpoint (multi/single)
    extract_edges_service,              # Edge detection endpoint
    train_classifier_service,
    predict_classifier_service,
)
from app.services import data_service

from app.models.requests import (
    HaralickExtractRequest,
    LBPRequest,
    ContourRequest,                     # contour request model
    ShapeRequest,                       # enhanced ShapeRequest (HOG + FAST options)
    FeatureBaseRequest,                 # base selector for single/multi/all image ops
    SiftResponse,                       # (imported for consistency; response returned as dict)
    EdgeResponse,                       # (imported for consistency; response returned as dict)
    ClassifierTrainingRequest,
    ClassifierPredictionRequest,
)

router = APIRouter()


# ---- Request Models (legacy inline ones kept for backwards compat) ----
class HistogramRequest(BaseModel):
    hist_type: str
    image_index: int
    all_images: bool


class KMeansRequest(BaseModel):
    n_clusters: int
    random_state: int
    selected_images: List[int] = []
    use_all_images: bool = False


class SingleImageKMeansRequest(BaseModel):
    image_index: int
    n_clusters: int
    random_state: int = 42
    max_pixels: int = 10000


# -------------------------------
# Endpoints
# -------------------------------
@router.post("/histogram")
async def histogram(request: HistogramRequest, session_id: Optional[str] = Query(None)):
    """
    Generate one or more histograms for the uploaded images (color or grayscale).
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            b64_list = generate_histogram_service(
                hist_type=request.hist_type,
                image_index=request.image_index,
                all_images=request.all_images
            )
            return {"histograms": b64_list}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception:
        logging.exception("Failed to generate histogram")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error generating histogram"}
        )


@router.post("/kmeans")
def kmeans(request: KMeansRequest, session_id: Optional[str] = Query(None)):
    """
    Perform K-means clustering on selected images (color-histogram features + PCA2 for plotting).
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            plot_b64, assignments = perform_kmeans_service(
                n_clusters=request.n_clusters,
                random_state=request.random_state,
                selected_images=request.selected_images,
                use_all_images=request.use_all_images
            )
            return {"plot": plot_b64, "assignments": assignments}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to do the clustering K-means")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kmeans-single-image")
def kmeans_single_image(request: SingleImageKMeansRequest, session_id: Optional[str] = Query(None)):
    """
    Perform K-means clustering on pixels within a single image to segment by color.
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            segmented_b64, plot_b64 = perform_single_image_kmeans_service(
                image_index=request.image_index,
                n_clusters=request.n_clusters,
                random_state=request.random_state,
                max_pixels=request.max_pixels
            )
            return {"segmented_image": segmented_b64, "comparison_plot": plot_b64}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to do single-image K-means clustering")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shape")
def shape_features(request: ShapeRequest, session_id: Optional[str] = Query(None)):
    """
    Extract shape/structure features from a single image.

    Supported methods:
      - "HOG": optional parameters:
          orientations, pixels_per_cell [h,w], cells_per_block [y,x],
          resize_width, resize_height, visualize
      - "SIFT": ignores HOG/FAST extras
      - "FAST": optional parameters:
          fast_threshold, fast_nonmax, fast_type ("TYPE_9_16" | "TYPE_7_12" | "TYPE_5_8")
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            features, viz_b64 = extract_shape_service(
            method=request.method,
            image_index=request.image_index,
            # HOG params (service ignores when method != "HOG")
            orientations=getattr(request, "orientations", None),
            pixels_per_cell=getattr(request, "pixels_per_cell", None),
            cells_per_block=getattr(request, "cells_per_block", None),
            resize_width=getattr(request, "resize_width", None),
            resize_height=getattr(request, "resize_height", None),
            visualize=getattr(request, "visualize", None),
            # FAST params (service ignores when method != "FAST")
            fast_threshold=getattr(request, "fast_threshold", None),
            fast_nonmax=getattr(request, "fast_nonmax", None),
            fast_type=getattr(request, "fast_type", None),
        )
            response: Dict[str, Any] = {"features": features}
            if viz_b64:
                response["visualization"] = viz_b64
            return response
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception:
        logging.exception("Failed to extract shape features")
        raise HTTPException(status_code=500, detail="Internal server error extracting shape features")


@router.post("/haralick")
async def haralick(
    train_images: List[UploadFile] = File(...),
    train_labels: UploadFile = File(...),
    test_images: List[UploadFile] = File(...),
    session_id: Optional[str] = Query(None)
):
    """
    Legacy demo: Extract a simple Haralick feature from training images, train a classifier,
    and predict labels for test images. (multipart upload)
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            labels, predictions = await extract_haralick_service(
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images
            )
            return {"labels": labels, "predictions": predictions}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception:
        logging.exception("Failed to extract Haralick texture features (legacy)")
        raise HTTPException(status_code=500, detail="Internal server error extracting Haralick features")


# ---- Haralick extraction (table) ----
@router.post("/haralick/extract")
def haralick_extract(req: HaralickExtractRequest, session_id: Optional[str] = Query(None)):
    """
    Compute GLCM (Haralick) properties for a set of uploaded images; returns table-like JSON.
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = extract_haralick_features_service(
                image_indices=req.image_indices,
                levels=req.levels,
                distances=req.distances,
                angles=req.angles,
                resize_width=req.resize_width,
                resize_height=req.resize_height,
                average_over_angles=req.average_over_angles,
                properties=req.properties
            )
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to extract Haralick table features")
        raise HTTPException(status_code=400, detail=str(e))


# ---- LBP extraction ----
@router.post("/lbp")
def lbp_extract(req: LBPRequest, session_id: Optional[str] = Query(None)):
    """
    Compute LBP histograms for one, multiple, or all images.
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = compute_lbp_service(
                image_indices=req.image_indices,
                use_all_images=req.use_all_images,
                radius=req.radius,
                num_neighbors=req.num_neighbors,
                method=req.method,
                normalize=req.normalize,
            )
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to compute LBP features")
        raise HTTPException(status_code=400, detail=str(e))


# ---- Contour extraction ----
@router.post("/contours")
def contour_extract(req: ContourRequest, session_id: Optional[str] = Query(None)):
    """
    Extract contours from a binary/grayscale version of the selected image.
    Uses OpenCV findContours to detect closed shapes and outlines.
    Returns contour points, areas, bounding boxes, and visualization.
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = extract_contours_service(
                image_index=req.image_index,
                mode=req.mode,
                method=req.method,
                min_area=req.min_area or 10,
                return_bounding_boxes=req.return_bounding_boxes,
                return_hierarchy=req.return_hierarchy,
                binarization_method=req.binarization_method,
                threshold_value=req.threshold_value,
                canny_low=req.canny_low,
                canny_high=req.canny_high,
            )
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to extract contours")
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------------------------------------------------
# SIFT endpoint
# ----------------------------------------------------------------------
@router.post("/sift")
def sift(request: FeatureBaseRequest, session_id: Optional[str] = Query(None)):
    """
    Run OpenCV SIFT on one or more images (single-image returns a visualization).
    Response:
        {
            "features": [[float, ...], ...],
            "visualization": base64_str | null
        }
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            feats, viz_bytes = extract_sift_service(
                image_index=request.image_index,
                all_images=request.all_images,
                image_indices=request.image_indices,
            )
            viz_b64 = base64.b64encode(viz_bytes).decode() if viz_bytes else None
            return {"features": feats, "visualization": viz_b64}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("SIFT extraction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Edge-detection endpoint (Canny by default)
# ----------------------------------------------------------------------
@router.post("/edges")
def edges(
    request: FeatureBaseRequest,
    session_id: Optional[str] = Query(None),
    method: str = "canny",
    low_thresh: int = 100,
    high_thresh: int = 200,
    sobel_ksize: int = 3,
):
    """
    Apply edge detection (Canny or Sobel) to one/multiple/all images selected via FeatureBaseRequest.
    Query params can adjust the method and thresholds.

    Response:
        {
            "edge_images": [base64_str, ...],   # one per processed image
            "edges_matrices": [                 # ALL gradient matrices
                [[float, ...], ...],            # for image 0
                [[float, ...], ...],            # for image 1
                ...
            ]
        }
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            edge_imgs_b64, all_matrices = extract_edges_service(
                image_index=request.image_index,
                all_images=request.all_images,
                image_indices=request.image_indices,
                method=method,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
                sobel_ksize=sobel_ksize,
            )
            return {"edge_images": edge_imgs_b64, "edges_matrices": all_matrices}
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Edge detection failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Image Embedding Extraction ----
class EmbeddingRequest(BaseModel):
    image_indices: Optional[List[int]] = None
    use_all_images: bool = False
    model_name: str = "resnet50"  # resnet50, resnet18, mobilenet_v2
    layer: Optional[str] = None


@router.post("/embedding")
def extract_embedding(request: EmbeddingRequest, session_id: Optional[str] = Query(None)):
    """
    Extract deep learning embeddings from images.
    
    Supported models:
      - resnet50: 2048-dimensional embeddings (default)
      - resnet18: 512-dimensional embeddings
      - mobilenet_v2: 1280-dimensional embeddings
    
    Returns embeddings as a list of vectors along with metadata.
    """
    from app.services.feature_service import extract_embedding_service
    
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = extract_embedding_service(
                image_indices=request.image_indices,
                use_all_images=request.use_all_images,
                model_name=request.model_name,
                layer=request.layer
            )
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except Exception as e:
        logging.exception("Failed to extract embeddings")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-classifier")
def train_classifier(request: ClassifierTrainingRequest, session_id: Optional[str] = Query(None)):
    """
    Train a traditional ML classifier (SVM, k-NN, Logistic Regression) on features
    derived from uploaded images (HOG, LBP, or deep embeddings).
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = train_classifier_service(request)
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logging.exception("Failed to train classifier")
        raise HTTPException(status_code=500, detail="Internal server error during classifier training")


@router.post("/predict-classifier")
def predict_classifier(request: ClassifierPredictionRequest, session_id: Optional[str] = Query(None)):
    """
    Run inference using a previously trained classifier and optional ground-truth labels.
    """
    try:
        token = data_service.CURRENT_SESSION_ID.set(session_id)
        try:
            result = predict_classifier_service(request)
            return result
        finally:
            data_service.CURRENT_SESSION_ID.reset(token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logging.exception("Failed to run classifier prediction")
        raise HTTPException(status_code=500, detail="Internal server error during classifier prediction")
