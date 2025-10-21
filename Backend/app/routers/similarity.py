# backend/app/routers/similarity.py

from fastapi import APIRouter, HTTPException
import logging
from typing import Optional, Tuple

import numpy as np
from app.services import data_service
from app.services.similarity_service import SimilarityService
from app.models.requests import SimilaritySearchRequest, AngleComparisonRequest
from app.models.responses import SimilaritySearchResponse, AngleComparisonResponse

router = APIRouter()

@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest):
    """
    Perform similarity search to find images visually similar to a query image.
    
    The query image can be provided either by:
    1. Index of an already uploaded image (query_image_index)
    2. Base64 encoded image data (query_image_base64)
    
    Returns a list of similar images with similarity scores and distances.
    """
    try:
        # Validate input
        if request.query_image_index is None and not request.query_image_base64:
            raise HTTPException(
                status_code=400, 
                detail="Either query_image_index or query_image_base64 must be provided"
            )
        
        # Prepare feature extraction parameters
        feature_params = {
            'resize_dimensions': request.resize_dimensions or [224, 224]
        }
        
        # Add method-specific parameters
        if request.feature_method == "HOG":
            feature_params.update({
                'hog_orientations': request.hog_orientations or 9,
                'hog_pixels_per_cell': request.hog_pixels_per_cell or [8, 8],
                'hog_cells_per_block': request.hog_cells_per_block or [2, 2]
            })
        elif request.feature_method == "histogram":
            feature_params.update({
                'hist_bins': request.hist_bins or 64,
                'hist_channels': request.hist_channels or [0, 1, 2]
            })
        
        # Perform similarity search
        result = SimilarityService.search_similar_images(
            query_image_index=request.query_image_index,
            query_image_base64=request.query_image_base64,
            feature_method=request.feature_method,
            distance_metric=request.distance_metric,
            max_results=request.max_results,
            threshold=request.threshold,
            **feature_params
        )
        
        return SimilaritySearchResponse(
            similar_images=result["similar_images"],
            query_features=result.get("query_features"),
            computation_time=result.get("computation_time")
        )
        
    except ValueError as e:
        logging.error(f"Invalid request for similarity search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Similarity search failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/angle-comparison", response_model=AngleComparisonResponse)
async def angle_comparison(request: AngleComparisonRequest):
    """
    Compare two character or word images by analysing their dominant stroke orientations.
    Returns an angle deviation score alongside supporting orientation statistics.
    """
    try:
        def load_image(source_index: Optional[int], source_b64: Optional[str], label: str) -> Tuple[np.ndarray, str]:
            if source_index is not None:
                image_ids = data_service.get_all_image_ids()
                if source_index < 0 or source_index >= len(image_ids):
                    raise ValueError(f"{label} index {source_index} is out of range (0-{len(image_ids) - 1}).")
                image_id = image_ids[source_index]
                image_array = SimilarityService._load_image_by_id(image_id)
                return image_array, image_id
            if source_b64:
                image_array = SimilarityService._load_image_from_base64(source_b64)
                return image_array, f"{label}_upload"
            raise ValueError(f"{label} image must be provided via index or base64 data.")
        image_a, image_a_ref = load_image(request.image_a_index, request.image_a_base64, "First")
        image_b, image_b_ref = load_image(request.image_b_index, request.image_b_base64, "Second")
        
        resize_dims = tuple(request.resize_dimensions or [256, 256])
        result = SimilarityService.compare_angle_orientation(
            image_a,
            image_b,
            resize_dims=resize_dims,
            num_bins=request.num_bins,
            blur_kernel_size=request.blur_kernel_size,
            canny_threshold1=request.canny_threshold1,
            canny_threshold2=request.canny_threshold2,
            gradient_threshold=request.gradient_threshold
        )
        
        if not request.return_histograms:
            result.pop("orientation_histogram_a", None)
            result.pop("orientation_histogram_b", None)
            result.pop("bin_edges", None)
        
        result.update({
            "image_a_reference": image_a_ref,
            "image_b_reference": image_b_ref
        })
        
        return AngleComparisonResponse(**result)
    
    except ValueError as e:
        logging.error(f"Invalid angle comparison request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Angle comparison failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/precompute-features")
async def precompute_features(
    feature_method: str = "CNN",
    resize_dimensions: list = None,
    hog_orientations: int = 9,
    hog_pixels_per_cell: list = None,
    hog_cells_per_block: list = None,
    hist_bins: int = 64,
    hist_channels: list = None
):
    """
    Precompute features for all uploaded images to speed up future similarity searches.
    This is optional but recommended for better performance when dealing with large datasets.
    """
    try:
        from app.services import data_service
        
        # Set default parameters
        if resize_dimensions is None:
            resize_dimensions = [224, 224]
        if hog_pixels_per_cell is None:
            hog_pixels_per_cell = [8, 8]
        if hog_cells_per_block is None:
            hog_cells_per_block = [2, 2]
        if hist_channels is None:
            hist_channels = [0, 1, 2]
        
        # Prepare feature extraction parameters
        feature_params = {
            'resize_dimensions': resize_dimensions
        }
        
        if feature_method == "HOG":
            feature_params.update({
                'hog_orientations': hog_orientations,
                'hog_pixels_per_cell': hog_pixels_per_cell,
                'hog_cells_per_block': hog_cells_per_block
            })
        elif feature_method == "histogram":
            feature_params.update({
                'hist_bins': hist_bins,
                'hist_channels': hist_channels
            })
        
        # Get all image IDs
        all_image_ids = data_service.get_all_image_ids()
        
        if not all_image_ids:
            return {"message": "No images found to process", "processed_count": 0}
        
        processed_count = 0
        failed_count = 0
        
        # Process each image
        for image_id in all_image_ids:
            try:
                # Check if already cached
                cache_key = SimilarityService.get_cache_key(image_id, feature_method, **feature_params)
                
                from app.services.similarity_service import feature_cache
                if image_id not in feature_cache or cache_key not in feature_cache[image_id]:
                    # Load and extract features
                    image = SimilarityService._load_image_by_id(image_id)
                    features = SimilarityService.extract_features(
                        image,
                        method=feature_method,
                        **feature_params
                    )
                    
                    # Cache the features
                    if image_id not in feature_cache:
                        feature_cache[image_id] = {}
                    feature_cache[image_id][cache_key] = features
                
                processed_count += 1
                
            except Exception as e:
                logging.warning(f"Failed to precompute features for image {image_id}: {e}")
                failed_count += 1
                continue
        
        return {
            "message": f"Successfully precomputed features for {processed_count} images",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "feature_method": feature_method,
            "cache_stats": SimilarityService.get_cache_stats()
        }
        
    except Exception as e:
        logging.exception("Feature precomputation failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/cache")
async def clear_feature_cache():
    """Clear the feature cache to free up memory."""
    try:
        SimilarityService.clear_feature_cache()
        return {"message": "Feature cache cleared successfully"}
    except Exception as e:
        logging.exception("Failed to clear feature cache")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get statistics about the current feature cache."""
    try:
        stats = SimilarityService.get_cache_stats()
        return {
            "cache_stats": stats,
            "message": f"Cache contains features for {stats['cached_images']} images"
        }
    except Exception as e:
        logging.exception("Failed to get cache stats")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/methods")
async def get_available_methods():
    """Get list of available feature extraction methods and distance metrics."""
    return {
        "feature_methods": ["CNN", "HOG", "SIFT", "histogram"],
        "distance_metrics": ["cosine", "euclidean", "manhattan"],
        "method_descriptions": {
            "CNN": "Convolutional Neural Network-like features using statistical patches",
            "HOG": "Histogram of Oriented Gradients - good for shape and texture",
            "SIFT": "Scale-Invariant Feature Transform - robust to scale and rotation",
            "histogram": "Color histogram features - good for color-based similarity"
        },
        "metric_descriptions": {
            "cosine": "Cosine similarity - measures angle between feature vectors",
            "euclidean": "Euclidean distance - measures straight-line distance",
            "manhattan": "Manhattan distance - measures city-block distance"
        },
        "specialized_methods": {
            "angle_orientation": {
                "endpoint": "/similarity/angle-comparison",
                "description": "Analyse character stroke directions and return an angle deviation score for two images.",
                "inputs": [
                    "First image: index or base64",
                    "Second image: index or base64"
                ],
                "outputs": [
                    "angle_deviation_score",
                    "mean_orientation_difference",
                    "optional histograms"
                ]
            }
        }
    }
