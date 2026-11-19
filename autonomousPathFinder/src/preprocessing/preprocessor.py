"""
Image Preprocessing Module for AutoNav AI.
Handles image loading, resizing, normalization, and quality enhancement using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocesses input images for SegFormer and YOLO deep learning models."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640), max_dimension: int = 1280):
        self.target_size = target_size
        self.max_dimension = max_dimension

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Preprocesses a raw BGR image.
        
        Args:
            image: Input image as BGR numpy array.
            
        Returns:
            processed_bgr: Resized and enhanced BGR image.
            rgb_image: Resized RGB image.
            scale_factor: Scale factor applied to original image.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        h, w = image.shape[:2]
        scale_factor = 1.0

        # Downscale large images for efficiency while maintaining aspect ratio
        if max(h, w) > self.max_dimension:
            scale_factor = self.max_dimension / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Contrast enhancement via CLAHE on L-channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Convert to RGB for deep learning models
        rgb_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        return enhanced_bgr, rgb_image, scale_factor
