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
import scipy.stats
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
        Extract more discriminative CNN-like features using enhanced statistical analysis.
        Improved version with better discrimination between different image types.
        """
        # Resize image
        image_resized = SimilarityService._resize_image(image, resize_dims)
        
        # Work with both grayscale and color for better discrimination
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            # Keep color information for better discrimination
            color_channels = cv2.split(image_resized)
        else:
            gray = image_resized
            color_channels = [gray, gray, gray]
        
        features = []
        
        # 1. Multi-scale patch analysis for better discrimination
        patch_sizes = [8, 16, 32]  # Multiple scales
        
        for patch_size in patch_sizes:
            h, w = gray.shape
            patch_features = []
            
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    
                    # Enhanced statistical features for better discrimination
                    patch_features.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.var(patch),  # Variance for texture analysis
                        np.percentile(patch, 25),  # 1st quartile
                        np.percentile(patch, 75),  # 3rd quartile
                        np.sum(patch > np.mean(patch)) / patch.size,  # Threshold density
                    ])
            
            # Aggregate patch features with statistics for scale
            if patch_features:
                patch_array = np.array(patch_features)
                features.extend([
                    np.mean(patch_array),
                    np.std(patch_array),
                    np.min(patch_array),
                    np.max(patch_array),
                    np.percentile(patch_array, 10),
                    np.percentile(patch_array, 90)
                ])
        
        # 2. Global image statistics for overall discrimination
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.entropy(gray.flatten()) if hasattr(np, 'entropy') else np.std(gray),
            # Edge density
            cv2.Canny(gray, 50, 150).sum() / gray.size,
        ])
        
        # 3. Color distribution features for better discrimination
        for channel in color_channels:
            hist, _ = np.histogram(channel, bins=16, range=(0, 256))
            hist = hist.astype(float) / hist.sum()  # Normalize
            features.extend(hist)
        
        # 4. Spatial frequency analysis
        # High-frequency content analysis for texture discrimination
        kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = cv2.filter2D(gray, -1, kernel_laplacian)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            np.var(laplacian)
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
        """Extract multi-scale HOG features for better discrimination."""
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = []
        
        # Multi-scale HOG extraction for better discrimination
        scales = [(64, 64), (128, 128), (256, 256)]  # Different scales
        
        for width, height in scales:
            resized = cv2.resize(gray, (width, height))
            
            # Extract HOG features with different parameters for each scale
            if width == 64:
                # Fine-grained features
                hog_feat = hog(
                    resized,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1),
                    visualize=False,
                    feature_vector=True
                )
            elif width == 128:
                # Medium-grained features (original scale)
                hog_feat = hog(
                    resized,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=False,
                    feature_vector=True
                )
            else:  # 256x256
                # Coarse-grained features
                hog_feat = hog(
                    resized,
                    orientations=12,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                    visualize=False,
                    feature_vector=True
                )
            
            features.extend(hog_feat)
        
        # Add global image statistics for better discrimination
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            gray.shape[0] / gray.shape[1],  # Aspect ratio
        ])
        
        # Add gradient statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.std(grad_x),
            np.mean(np.abs(grad_y)),
            np.std(grad_y),
        ])
        
        # Add edge density information for better texture discrimination
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_sift_features(image: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Extract SIFT features with improved aggregation for better discrimination."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Initialize SIFT detector with better parameters for discrimination
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.04, edgeThreshold=10)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no features found
            return np.zeros(256, dtype=np.float32)  # Larger feature vector for better discrimination
        
        # Improved aggregation for better discrimination
        features = []
        
        # 1. Basic statistics
        features.extend([
            np.mean(descriptors, axis=0).mean(),
            np.std(descriptors, axis=0).mean(),
            np.min(descriptors, axis=0).mean(),
            np.max(descriptors, axis=0).mean(),
        ])
        
        # 2. Descriptor distribution analysis
        features.extend([
            np.mean(descriptors),
            np.std(descriptors),
            np.var(descriptors),
            len(descriptors) / max_features,  # Keypoint density
        ])
        
        # 3. Keypoint spatial distribution for better discrimination
        if len(keypoints) > 0:
            kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            features.extend([
                np.mean(kp_coords[:, 0]),  # Mean X coordinate
                np.mean(kp_coords[:, 1]),  # Mean Y coordinate
                np.std(kp_coords[:, 0]),   # X spread
                np.std(kp_coords[:, 1]),   # Y spread
            ])
            
            # Keypoint response and scale statistics
            responses = np.array([kp.response for kp in keypoints])
            scales = np.array([kp.size for kp in keypoints])
            
            features.extend([
                np.mean(responses),
                np.std(responses),
                np.mean(scales),
                np.std(scales),
            ])
        else:
            features.extend([0] * 8)  # Padding for missing keypoints
        
        # 4. Quantized descriptor histogram for pattern analysis
        # Create a histogram of descriptor values for better discrimination
        hist_bins = 32
        desc_flat = descriptors.flatten()
        hist, _ = np.histogram(desc_flat, bins=hist_bins, range=(0, 255))
        hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)
        
        # 5. Top descriptors analysis (most distinctive features)
        if len(descriptors) > 10:
            # Select top 10 most distinctive descriptors based on variance
            desc_vars = np.var(descriptors, axis=1)
            top_indices = np.argsort(desc_vars)[-10:]
            top_descriptors = descriptors[top_indices]
            features.extend(np.mean(top_descriptors, axis=0))
        else:
            # Pad with mean if not enough descriptors
            features.extend(np.mean(descriptors, axis=0))
        
        # Ensure consistent output size
        feature_array = np.array(features, dtype=np.float32)
        
        # Pad or truncate to exactly 256 features
        if len(feature_array) < 256:
            padding = np.zeros(256 - len(feature_array), dtype=np.float32)
            feature_array = np.concatenate([feature_array, padding])
        elif len(feature_array) > 256:
            feature_array = feature_array[:256]
        
        return feature_array
    
    @staticmethod
    def extract_manuscript_features(
        image: np.ndarray,
        resize_dims: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Extract features specifically optimized for manuscripts and ancient books.
        Combines texture analysis, edge patterns, and layout features.
        """
        # Resize image
        image_resized = SimilarityService._resize_image(image, resize_dims)
        
        # Convert to grayscale for analysis
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_resized
        
        features = []
        
        # 1. Enhanced contrast and preprocessing for old documents
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Text/Script texture features using Local Binary Patterns
        from skimage.feature import local_binary_pattern
        lbp_radius = 3
        lbp_points = 8 * lbp_radius
        lbp = local_binary_pattern(enhanced, lbp_points, lbp_radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_points + 2, range=(0, lbp_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        features.extend(lbp_hist)
        
        # 3. Multi-scale HOG for script patterns (optimized for text)
        # Fine-scale HOG for character strokes
        hog_fine = hog(enhanced, orientations=18, pixels_per_cell=(4, 4), 
                      cells_per_block=(2, 2), visualize=False, feature_vector=True)
        features.extend(hog_fine)
        
        # Coarse-scale HOG for word/line patterns  
        hog_coarse = hog(enhanced, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=False, feature_vector=True)
        features.extend(hog_coarse)
        
        # 4. Layout and structure features
        # Horizontal and vertical projection profiles for line detection
        h_profile = np.sum(enhanced, axis=1)  # Horizontal projection
        v_profile = np.sum(enhanced, axis=0)  # Vertical projection
        
        # Downsample profiles for feature vector
        h_profile_bins = np.histogram(h_profile, bins=32)[0].astype(float)
        v_profile_bins = np.histogram(v_profile, bins=32)[0].astype(float)
        h_profile_bins /= (h_profile_bins.sum() + 1e-7)
        v_profile_bins /= (v_profile_bins.sum() + 1e-7)
        features.extend(h_profile_bins)
        features.extend(v_profile_bins)
        
        # 5. Edge orientation analysis for script characteristics
        # Compute gradients
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Threshold by magnitude and create orientation histogram
        strong_edges = magnitude > np.percentile(magnitude, 75)
        edge_orientations = orientation[strong_edges]
        
        # Convert to degrees and create histogram
        edge_orientations_deg = np.degrees(edge_orientations) % 180
        orientation_hist, _ = np.histogram(edge_orientations_deg, bins=18, range=(0, 180))
        orientation_hist = orientation_hist.astype(float)
        orientation_hist /= (orientation_hist.sum() + 1e-7)
        features.extend(orientation_hist)
        
        # 6. Texture regularity features
        # Measure of text regularity vs. decorative elements
        # Standard deviation of pixel intensities in patches
        patch_size = 16
        texture_variance = []
        h, w = enhanced.shape
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = enhanced[y:y+patch_size, x:x+patch_size]
                texture_variance.append(np.std(patch))
        
        # Statistics of texture variance across patches
        if texture_variance:
            texture_stats = [
                np.mean(texture_variance),
                np.std(texture_variance),
                np.percentile(texture_variance, 25),
                np.percentile(texture_variance, 75)
            ]
            features.extend(texture_stats)
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_histogram_features(
        image: np.ndarray,
        bins: int = 64,
        channels: List[int] = [0, 1, 2]
    ) -> np.ndarray:
        """Extract enhanced histogram features for better discrimination."""
        features = []
        
        # Convert to different color spaces for comprehensive analysis
        if len(image.shape) == 3:
            # RGB histograms
            for channel in range(3):
                hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
                features.extend(hist.flatten())
            
            # HSV histograms for better color discrimination
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for channel in range(3):
                if channel == 0:  # Hue channel (0-180)
                    hist = cv2.calcHist([hsv], [channel], None, [bins//2], [0, 180])
                else:  # Saturation and Value channels (0-255)
                    hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
                features.extend(hist.flatten())
            
            # LAB color space for perceptual uniformity
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            for channel in range(3):
                hist = cv2.calcHist([lab], [channel], None, [bins], [0, 256])
                features.extend(hist.flatten())
            
            # Add color moments for each channel (RGB)
            for channel in range(3):
                channel_data = image[:, :, channel].flatten()
                # First moment (mean)
                features.append(np.mean(channel_data))
                # Second moment (variance)
                features.append(np.var(channel_data))
                # Third moment (skewness)
                features.append(scipy.stats.skew(channel_data))
                # Fourth moment (kurtosis)
                features.append(scipy.stats.kurtosis(channel_data))
            
        else:
            # Grayscale image - more detailed analysis
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            features.extend(hist.flatten())
            
            # Add texture-based histogram features
            # Local Binary Pattern histogram
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=bins//4, range=(0, 10))
            features.extend(lbp_hist)
            
            # Gradient magnitude histogram
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            mag_hist, _ = np.histogram(magnitude.flatten(), bins=bins//2, range=(0, 255))
            features.extend(mag_hist)
        
        # Add global statistics for better discrimination
        flat_image = image.flatten() if len(image.shape) == 2 else image.reshape(-1, image.shape[-1])
        
        if len(image.shape) == 2:
            features.extend([
                np.mean(flat_image),
                np.std(flat_image),
                np.min(flat_image),
                np.max(flat_image),
                np.median(flat_image),
                scipy.stats.entropy(np.histogram(flat_image, bins=256)[0] + 1e-7),  # Image entropy
            ])
        else:
            # Multi-channel statistics
            for channel in range(image.shape[-1]):
                channel_data = flat_image[:, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                ])
        
        # Normalize features
        features = np.array(features, dtype=np.float32)
        
        # L2 normalization for better discrimination
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
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
            
            elif method == "manuscript":
                return SimilarityService.extract_manuscript_features(
                    image,
                    resize_dims=tuple(kwargs.get('resize_dimensions', [256, 256]))
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
                    
                    # Apply STRICT intelligent thresholds based on feature method and score distribution
                    should_include = False
                    if threshold is None:
                        # Use much stricter method-specific thresholds based on analysis
                        if feature_method == "manuscript":
                            should_include = similarity_score >= 0.85  # Only very similar manuscript features
                        elif feature_method == "HOG":
                            should_include = similarity_score >= 0.8   # Only strong HOG matches
                        elif feature_method == "SIFT":
                            should_include = similarity_score >= 0.95  # SIFT has poor discrimination, use high threshold
                        elif feature_method == "histogram":
                            should_include = similarity_score >= 0.75  # Histogram shows good discrimination
                        elif feature_method == "CNN":
                            should_include = similarity_score >= 0.98  # CNN has very poor discrimination
                        else:
                            should_include = similarity_score >= 0.8   # Default strict threshold
                    else:
                        should_include = similarity_score >= threshold
                    
                    if should_include:
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
