"""
Terrain Segmentation Module for AutoNav AI using SegFormer-B0.
Performs semantic segmentation on outdoor scenes and classifies pixels into
terrain categories (Grass, Soil, Gravel/Rock, Path, Vegetation, Water, Sky, etc.).
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class TerrainSegmenter:
    """Deep-learning semantic segmentation for outdoor terrain using SegFormer-B0."""

    # High-level outdoor navigation categories
    TERRAIN_CLASSES = {
        "path": {"name": "Road / Path", "color": (50, 200, 255), "traversability": 1.0},
        "grass": {"name": "Grass", "color": (40, 200, 60), "traversability": 0.85},
        "soil": {"name": "Soil / Earth", "color": (190, 140, 70), "traversability": 0.80},
        "vegetation": {"name": "Vegetation / Trees", "color": (20, 110, 30), "traversability": 0.10},
        "rock": {"name": "Rocks / Rough", "color": (150, 60, 60), "traversability": 0.20},
        "water": {"name": "Water / Hazard", "color": (30, 90, 240), "traversability": 0.0},
        "sky": {"name": "Sky", "color": (135, 206, 235), "traversability": 0.0},
        "obstacle_struct": {"name": "Structure / Wall", "color": (160, 40, 160), "traversability": 0.0},
        "other": {"name": "Other", "color": (120, 120, 120), "traversability": 0.30}
    }

    # ADE20K class IDs mapped to our terrain categories
    ADE20K_MAPPING = {
        # Path / Road
        6: "path", 11: "path", 52: "path", 29: "path", 91: "path",
        # Grass
        9: "grass",
        # Soil / Earth
        13: "soil", 94: "soil", 46: "soil",
        # Trees / Vegetation
        4: "vegetation", 17: "vegetation", 66: "vegetation", 72: "vegetation",
        # Rocks
        34: "rock", 68: "rock",
        # Water
        21: "water", 26: "water", 60: "water", 113: "water", 128: "water",
        # Sky
        2: "sky",
        # Structures / Obstacles
        0: "obstacle_struct", 1: "obstacle_struct", 7: "obstacle_struct", 12: "obstacle_struct",
        15: "obstacle_struct", 20: "obstacle_struct", 32: "obstacle_struct"
    }

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_dir is None:
            # Default local path inside autonomousPathFinder/models/segformer-b0
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, "models", "segformer-b0")

        self.model_dir = model_dir
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads SegFormer-B0 from local directory or falls back gracefully."""
        try:
            print(f"[TerrainSegmenter] Loading SegFormer-B0 from {self.model_dir} on {self.device}...")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("[TerrainSegmenter] Model loaded successfully.")
        except Exception as e:
            print(f"[TerrainSegmenter] Warning: Failed to load local SegFormer ({e}). Will attempt HuggingFace or fallback.")
            try:
                hub_model = "nvidia/segformer-b0-finetuned-ade-512-512"
                self.processor = SegformerImageProcessor.from_pretrained(hub_model)
                self.model = SegformerForSemanticSegmentation.from_pretrained(hub_model)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e2:
                print(f"[TerrainSegmenter] Offline fallback: Using OpenCV terrain segmentation ({e2}).")
                self.model = None

    def segment(self, rgb_image: np.ndarray) -> Dict:
        """
        Runs semantic segmentation on the input RGB image.
        
        Args:
            rgb_image: Input image in RGB format (H, W, 3).
            
        Returns:
            Dictionary containing:
                - 'class_map': 2D numpy array of categorized class indices.
                - 'color_mask': RGB visualization of the segmentation map.
                - 'traversability_map': 2D float array (0.0 to 1.0) of terrain suitability.
                - 'class_distribution': Percentage distribution of detected terrain types.
        """
        orig_h, orig_w = rgb_image.shape[:2]

        if self.model is not None:
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Interpolate logits to original image dimensions
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            pred_classes = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        else:
            # Heuristic fallback using HSV / color features if model unavailable
            pred_classes = self._fallback_segmentation(rgb_image)

        # Map raw ADE20K classes to high-level terrain categories & compute masks
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        traversability_map = np.zeros((orig_h, orig_w), dtype=np.float32)
        category_map = np.zeros((orig_h, orig_w), dtype=object)

        unique_classes, counts = np.unique(pred_classes, return_counts=True)
        total_pixels = orig_h * orig_w
        class_dist = {}

        # Default background values
        color_mask[:] = self.TERRAIN_CLASSES["other"]["color"]
        traversability_map[:] = self.TERRAIN_CLASSES["other"]["traversability"]

        for cls_id in unique_classes:
            cat_key = self.ADE20K_MAPPING.get(int(cls_id), "other")
            info = self.TERRAIN_CLASSES[cat_key]
            mask = (pred_classes == cls_id)

            color_mask[mask] = info["color"]
            traversability_map[mask] = info["traversability"]

            cat_name = info["name"]
            pixel_ratio = float(np.sum(mask)) / total_pixels * 100
            class_dist[cat_name] = class_dist.get(cat_name, 0.0) + pixel_ratio

        # Sort distribution by area
        class_dist = {k: round(v, 2) for k, v in sorted(class_dist.items(), key=lambda item: item[1], reverse=True) if v > 0.5}

        return {
            "pred_classes": pred_classes,
            "color_mask": color_mask,
            "traversability_map": traversability_map,
            "class_distribution": class_dist
        }

    def _fallback_segmentation(self, rgb_image: np.ndarray) -> np.ndarray:
        """Heuristic color-based segmentation fallback when deep learning model is disabled."""
        h, w = rgb_image.shape[:2]
        pred = np.zeros((h, w), dtype=np.int32)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # Upper region likely sky
        sky_mask = np.zeros((h, w), dtype=bool)
        sky_mask[:int(h * 0.4), :] = (hsv[:int(h * 0.4), :, 0] > 90) & (hsv[:int(h * 0.4), :, 0] < 130)
        pred[sky_mask] = 2  # Sky

        # Green hues -> Grass / Vegetation
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
        pred[green_mask] = 9  # Grass

        # Brown / earthy hues -> Earth / Soil
        soil_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] > 30)
        pred[soil_mask] = 13  # Earth

        # Gray low saturation road / path
        road_mask = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 70) & (~sky_mask)
        pred[road_mask] = 6  # Road

        return pred
