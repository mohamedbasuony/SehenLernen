# backend/app/services/similarity_service.py

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Computer vision and ML libraries
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from skimage.feature import hog
from skimage import exposure

# Import existing services and utilities
from app.services import data_service
from app.utils.image_utils import base64_to_bytes

# Feature cache to avoid recomputing features for the same images
feature_cache: Dict[str, Dict[str, np.ndarray]] = {}

class SimilarityService:
    """Service for image similarity search using various feature extraction methods."""
    
    @staticmethod
    def _load_image_by_id(image_id: str) -> np.ndarray:
        """Load image by ID and return as numpy array."""
        try:
            image_bytes = data_service.load_image(image_id)
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logging.error(f"Failed to load image {image_id}: {e}")
            raise
    
    @staticmethod
    def _load_image_from_base64(base64_data: str) -> np.ndarray:
        """Load image from base64 string and return as numpy array."""
        try:
            image_bytes = base64_to_bytes(base64_data)
            image = Image.open(BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logging.error(f"Failed to load image from base64: {e}")
            raise
    
    @staticmethod
    def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target dimensions."""
        height, width = target_size
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def extract_cnn_features(image: np.ndarray, resize_dims: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract CNN-like features using a simple approach.
        For a proper CNN, you'd use a pre-trained model like ResNet/VGG.
        This is a simplified version using statistical features from patches.
        """
        # Resize image
        image_resized = SimilarityService._resize_image(image, resize_dims)
        
        # Convert to grayscale for feature extraction
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        # Extract statistical features from patches
        patch_size = 16
        features = []
        
        h, w = gray.shape
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                
                # Calculate statistical features for each patch
                features.extend([
                    np.mean(patch),
                    np.std(patch),
                    np.min(patch),
                    np.max(patch),
                    np.median(patch)
                ])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_hog_features(
        image: np.ndarray, 
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        resize_dims: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        # Resize and convert to grayscale
        image_resized = SimilarityService._resize_image(image, resize_dims)
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        # Extract HOG features
        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True
        )
        
        return features.astype(np.float32)
    
    @staticmethod
    def extract_sift_features(image: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Extract SIFT features and aggregate them."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=max_features)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no features found
            return np.zeros(128, dtype=np.float32)
        
        # Aggregate descriptors (mean of all descriptors)
        aggregated = np.mean(descriptors, axis=0)
        return aggregated.astype(np.float32)
    
    @staticmethod
    def extract_histogram_features(
        image: np.ndarray,
        bins: int = 64,
        channels: List[int] = [0, 1, 2]
    ) -> np.ndarray:
        """Extract color histogram features."""
        features = []
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            features.extend(hist.flatten())
        else:
            # Color image - extract histogram for each specified channel
            for channel in channels:
                if channel < image.shape[2]:
                    hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
                    features.extend(hist.flatten())
        
        # Normalize histogram
        features = np.array(features, dtype=np.float32)
        if np.sum(features) > 0:
            features = features / np.sum(features)
        
        return features
    
    @staticmethod
    def _compute_axial_mean_orientation(angles_deg: np.ndarray, weights: np.ndarray) -> Optional[float]:
        """
        Compute the mean orientation for axial (0-180°) data using circular statistics.
        Returns None if angles cannot be determined (e.g., empty input).
        """
        if angles_deg.size == 0 or weights.size == 0:
            return None
        
        # Double the angles for axial data (periodicity of 180°) before averaging
        doubled_angles_rad = np.deg2rad(angles_deg) * 2.0
        weighted_sin = np.sum(np.sin(doubled_angles_rad) * weights)
        weighted_cos = np.sum(np.cos(doubled_angles_rad) * weights)
        
        if np.isclose(weighted_sin, 0.0) and np.isclose(weighted_cos, 0.0):
            return None
        
        mean_angle_rad = 0.5 * np.arctan2(weighted_sin, weighted_cos)
        mean_angle_deg = np.rad2deg(mean_angle_rad) % 180.0
        return float(mean_angle_deg)
    
    @staticmethod
    def _angular_difference(angle_a: Optional[float], angle_b: Optional[float]) -> Optional[float]:
        """Return the smallest difference between two orientations, accounting for axial symmetry."""
        if angle_a is None or angle_b is None:
            return None
        diff = abs(angle_a - angle_b)
        # Axial data repeats every 180°
        if diff > 90.0:
            diff = 180.0 - diff
        return float(diff)
    
    @staticmethod
    def extract_angle_orientation_features(
        image: np.ndarray,
        resize_dims: Tuple[int, int] = (256, 256),
        num_bins: int = 36,
        blur_kernel_size: int = 5,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        gradient_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Extract orientation features from an image by analysing gradient directions along prominent strokes.
        
        Returns a dictionary containing:
            - histogram: normalized orientation histogram (numpy array)
            - bin_edges: histogram bin edges (numpy array)
            - mean_orientation: weighted axial mean orientation in degrees
            - dominant_orientation: orientation of the most populated bin
            - sample_count: number of pixels contributing to the histogram
        """
        # Resize and convert to grayscale
        image_resized = SimilarityService._resize_image(image, resize_dims)
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        # Optionally smooth noise
        if blur_kernel_size and blur_kernel_size > 1:
            # Kernel size must be odd for Gaussian blur
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            gray_blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        else:
            gray_blurred = gray
        
        # Detect edges and gradients
        edges = cv2.Canny(gray_blurred, canny_threshold1, canny_threshold2)
        sobel_x = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        angles = np.rad2deg(np.arctan2(sobel_y, sobel_x)) % 180.0  # Orientation is axial
        
        # Focus on edge pixels with meaningful gradient magnitude
        mask = edges > 0
        if gradient_threshold > 0.0:
            mask = np.logical_and(mask, magnitude > gradient_threshold)
        else:
            mask = np.logical_and(mask, magnitude > 0.0)
        
        masked_angles = angles[mask]
        masked_magnitudes = magnitude[mask]
        sample_count = int(masked_angles.size)
        
        if sample_count == 0:
            return {
                "histogram": np.zeros(num_bins, dtype=np.float32),
                "bin_edges": np.linspace(0.0, 180.0, num_bins + 1, dtype=np.float32),
                "mean_orientation": None,
                "dominant_orientation": None,
                "sample_count": 0
            }
        
        hist, bin_edges = np.histogram(
            masked_angles,
            bins=num_bins,
            range=(0.0, 180.0),
            weights=masked_magnitudes
        )
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist_normalized = (hist / hist_sum).astype(np.float32)
        else:
            hist_normalized = np.zeros_like(hist, dtype=np.float32)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        dominant_idx = int(np.argmax(hist_normalized)) if hist_normalized.size else 0
        dominant_orientation = float(bin_centers[dominant_idx]) if hist_sum > 0 else None
        
        mean_orientation = SimilarityService._compute_axial_mean_orientation(
            masked_angles,
            masked_magnitudes
        )
        
        return {
            "histogram": hist_normalized,
            "bin_edges": bin_edges.astype(np.float32),
            "mean_orientation": mean_orientation,
            "dominant_orientation": dominant_orientation,
            "sample_count": sample_count
        }
    
    @staticmethod
    def compare_angle_orientation(
        image_a: np.ndarray,
        image_b: np.ndarray,
        resize_dims: Tuple[int, int] = (256, 256),
        num_bins: int = 36,
        blur_kernel_size: int = 5,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        gradient_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Compare orientation distributions between two images and compute a deviation score.
        
        Returns:
            Dict with angle_deviation_score (0-1, higher means larger deviation) and
            supporting statistics for each image.
        """
        features_a = SimilarityService.extract_angle_orientation_features(
            image_a,
            resize_dims=resize_dims,
            num_bins=num_bins,
            blur_kernel_size=blur_kernel_size,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            gradient_threshold=gradient_threshold
        )
        features_b = SimilarityService.extract_angle_orientation_features(
            image_b,
            resize_dims=resize_dims,
            num_bins=num_bins,
            blur_kernel_size=blur_kernel_size,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            gradient_threshold=gradient_threshold
        )
        
        hist_a = features_a["histogram"]
        hist_b = features_b["histogram"]
        samples_a = features_a.get("sample_count", 0)
        samples_b = features_b.get("sample_count", 0)
        
        if samples_a == 0 or samples_b == 0:
            angle_deviation = None
        else:
            # L1 distance between normalized histograms, scaled to 0-1
            angle_deviation = float(np.clip(np.sum(np.abs(hist_a - hist_b)) * 0.5, 0.0, 1.0))
        
        mean_diff = SimilarityService._angular_difference(
            features_a.get("mean_orientation"),
            features_b.get("mean_orientation")
        )
        
        return {
            "angle_deviation_score": angle_deviation,
            "mean_orientation_a": features_a.get("mean_orientation"),
            "mean_orientation_b": features_b.get("mean_orientation"),
            "mean_orientation_difference": mean_diff,
            "dominant_orientation_a": features_a.get("dominant_orientation"),
            "dominant_orientation_b": features_b.get("dominant_orientation"),
            "orientation_histogram_a": hist_a.tolist() if hist_a is not None else None,
            "orientation_histogram_b": hist_b.tolist() if hist_b is not None else None,
            "bin_edges": features_a.get("bin_edges").tolist() if features_a.get("bin_edges") is not None else None,
            "samples_a": samples_a,
            "samples_b": samples_b
        }
    
    @staticmethod
    def extract_features(
        image: np.ndarray,
        method: str = "CNN",
        **kwargs
    ) -> np.ndarray:
        """Extract features using the specified method."""
        try:
            if method == "CNN":
                resize_dims = kwargs.get('resize_dimensions', [224, 224])
                return SimilarityService.extract_cnn_features(image, tuple(resize_dims))
            
            elif method == "HOG":
                return SimilarityService.extract_hog_features(
                    image,
                    orientations=kwargs.get('hog_orientations', 9),
                    pixels_per_cell=tuple(kwargs.get('hog_pixels_per_cell', [8, 8])),
                    cells_per_block=tuple(kwargs.get('hog_cells_per_block', [2, 2])),
                    resize_dims=tuple(kwargs.get('resize_dimensions', [128, 128]))
                )
            
            elif method == "SIFT":
                return SimilarityService.extract_sift_features(image)
            
            elif method == "histogram":
                return SimilarityService.extract_histogram_features(
                    image,
                    bins=kwargs.get('hist_bins', 64),
                    channels=kwargs.get('hist_channels', [0, 1, 2])
                )
            
            else:
                raise ValueError(f"Unknown feature extraction method: {method}")
                
        except Exception as e:
            logging.error(f"Feature extraction failed for method {method}: {e}")
            raise
    
    @staticmethod
    def compute_similarity(
        query_features: np.ndarray,
        target_features: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two feature vectors."""
        try:
            # Ensure features are 2D for sklearn
            if query_features.ndim == 1:
                query_features = query_features.reshape(1, -1)
            if target_features.ndim == 1:
                target_features = target_features.reshape(1, -1)
            
            if metric == "cosine":
                # Cosine similarity returns values from -1 to 1, we convert to 0-1
                similarity = cosine_similarity(query_features, target_features)[0, 0]
                return float((similarity + 1) / 2)  # Convert to 0-1 range
            
            elif metric == "euclidean":
                # Convert euclidean distance to similarity (lower distance = higher similarity)
                distance = euclidean_distances(query_features, target_features)[0, 0]
                # Normalize to 0-1 range using exponential decay
                similarity = np.exp(-distance / (np.linalg.norm(query_features) + np.linalg.norm(target_features)))
                return float(similarity)
            
            elif metric == "manhattan":
                # Convert manhattan distance to similarity
                distance = manhattan_distances(query_features, target_features)[0, 0]
                # Normalize to 0-1 range using exponential decay
                similarity = np.exp(-distance / (np.sum(np.abs(query_features)) + np.sum(np.abs(target_features))))
                return float(similarity)
            
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
                
        except Exception as e:
            logging.error(f"Similarity computation failed: {e}")
            return 0.0
    
    @staticmethod
    def get_cache_key(image_id: str, method: str, **kwargs) -> str:
        """Generate cache key for features."""
        # Create a hash-like key from method and parameters
        params_str = "_".join([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        return f"{image_id}_{method}_{params_str}"
    
    @staticmethod
    def search_similar_images(
        query_image_index: Optional[int] = None,
        query_image_base64: Optional[str] = None,
        feature_method: str = "CNN",
        distance_metric: str = "cosine",
        max_results: int = 10,
        threshold: Optional[float] = None,
        **feature_params
    ) -> Dict[str, Any]:
        """
        Perform similarity search to find images similar to the query.
        
        Returns:
            Dict containing similar_images list, computation_time, and optional query_features
        """
        start_time = time.time()
        
        try:
            # Load query image
            if query_image_index is not None:
                # Get image by index from uploaded images
                image_ids = data_service.get_all_image_ids()
                if query_image_index >= len(image_ids):
                    raise ValueError(f"Query image index {query_image_index} out of range")
                query_image_id = image_ids[query_image_index]
                query_image = SimilarityService._load_image_by_id(query_image_id)
            elif query_image_base64:
                query_image = SimilarityService._load_image_from_base64(query_image_base64)
                query_image_id = "query_upload"
            else:
                raise ValueError("Either query_image_index or query_image_base64 must be provided")
            
            # Extract query features
            query_features = SimilarityService.extract_features(
                query_image, 
                method=feature_method,
                **feature_params
            )
            
            # Get all dataset images
            all_image_ids = data_service.get_all_image_ids()
            
            # If query is from dataset, remove it from comparison
            if query_image_index is not None and query_image_id in all_image_ids:
                comparison_image_ids = [img_id for img_id in all_image_ids if img_id != query_image_id]
            else:
                comparison_image_ids = all_image_ids
            
            similar_images = []
            
            # Compare with each image in the dataset
            for image_id in comparison_image_ids:
                try:
                    # Check cache first
                    cache_key = SimilarityService.get_cache_key(image_id, feature_method, **feature_params)
                    
                    if image_id in feature_cache and cache_key in feature_cache[image_id]:
                        target_features = feature_cache[image_id][cache_key]
                    else:
                        # Extract features and cache them
                        target_image = SimilarityService._load_image_by_id(image_id)
                        target_features = SimilarityService.extract_features(
                            target_image,
                            method=feature_method,
                            **feature_params
                        )
                        
                        # Cache the features
                        if image_id not in feature_cache:
                            feature_cache[image_id] = {}
                        feature_cache[image_id][cache_key] = target_features
                    
                    # Compute similarity
                    similarity_score = SimilarityService.compute_similarity(
                        query_features,
                        target_features,
                        metric=distance_metric
                    )
                    
                    # Apply threshold filter if specified
                    if threshold is None or similarity_score >= threshold:
                        similar_images.append({
                            "image_id": image_id,
                            "similarity_score": similarity_score,
                            "distance": 1.0 - similarity_score  # Convert similarity to distance
                        })
                
                except Exception as e:
                    logging.warning(f"Failed to process image {image_id}: {e}")
                    continue
            
            # Sort by similarity score (descending)
            similar_images.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Limit results
            similar_images = similar_images[:max_results]
            
            computation_time = time.time() - start_time
            
            return {
                "similar_images": similar_images,
                "query_features": query_features.tolist() if query_features is not None else None,
                "computation_time": computation_time
            }
            
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            raise
    
    @staticmethod
    def clear_feature_cache():
        """Clear the feature cache."""
        global feature_cache
        feature_cache.clear()
        logging.info("Feature cache cleared")
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get statistics about the feature cache."""
        return {
            "cached_images": len(feature_cache),
            "total_cached_features": sum(len(features) for features in feature_cache.values())
        }
