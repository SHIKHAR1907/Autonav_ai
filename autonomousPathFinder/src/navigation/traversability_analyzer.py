"""
Traversability Analysis Module for AutoNav AI.
Combines semantic terrain classifications and YOLO obstacle detections
into a unified spatial traversability costmap for ground vehicle navigation.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class TraversabilityAnalyzer:
    """Fuses terrain semantics and obstacle footprints into a traversability costmap."""

    def __init__(self, obstacle_inflation_radius: int = 15):
        """
        Args:
            obstacle_inflation_radius: Pixel dilation radius around obstacles for safety margin.
        """
        self.inflation_radius = obstacle_inflation_radius

    def analyze(self, terrain_traversability: np.ndarray, obstacle_mask: np.ndarray) -> Dict:
        """
        Calculates unified traversability map.
        
        Args:
            terrain_traversability: 2D float array (0.0 to 1.0) from SegFormer.
            obstacle_mask: 2D boolean array from YOLO detections.
            
        Returns:
            Dictionary containing:
                - 'costmap': 2D float array (0.0 = lethal obstacle, 1.0 = optimal path).
                - 'color_costmap': RGB visualization (Green = Safe, Yellow = Caution, Red = Hazard).
                - 'overall_traversability_score': Mean traversability percentage in drive zone.
        """
        h, w = terrain_traversability.shape[:2]

        # Inflate obstacle footprint for robotic collision safety margin
        if self.inflation_radius > 0 and np.any(obstacle_mask):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.inflation_radius * 2 + 1, self.inflation_radius * 2 + 1)
            )
            inflated_obstacles = cv2.dilate(obstacle_mask.astype(np.uint8), kernel).astype(bool)
        else:
            inflated_obstacles = obstacle_mask

        # Combine terrain traversability and obstacles
        # Obstacles zero-out traversability
        costmap = np.copy(terrain_traversability)
        costmap[inflated_obstacles] = 0.0

        # Construct RGB visual costmap
        # High traversability (> 0.7) -> Green
        # Medium traversability (0.35 - 0.7) -> Yellow/Orange
        # Low / Blocked (< 0.35) -> Red
        color_costmap = np.zeros((h, w, 3), dtype=np.uint8)

        # Vectorized color mapping
        # Safe zone: Green
        safe_mask = costmap >= 0.70
        color_costmap[safe_mask] = [30, 210, 60]

        # Caution zone: Amber / Yellow
        caution_mask = (costmap >= 0.35) & (costmap < 0.70)
        color_costmap[caution_mask] = [255, 185, 20]

        # Hazard / Non-traversable: Red
        hazard_mask = (costmap < 0.35) & (costmap > 0.0)
        color_costmap[hazard_mask] = [220, 40, 40]

        # Solid obstacle footprint: Bright Red
        color_costmap[inflated_obstacles] = [255, 0, 0]

        # Ground drive zone is bottom 55% of image
        drive_zone = costmap[int(h * 0.45):, :]
        overall_score = float(np.mean(drive_zone) * 100) if drive_zone.size > 0 else 0.0

        return {
            "costmap": costmap,
            "color_costmap": color_costmap,
            "overall_score": round(overall_score, 1)
        }
