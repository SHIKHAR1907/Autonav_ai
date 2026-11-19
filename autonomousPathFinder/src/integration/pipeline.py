"""
End-to-End Pipeline Integration Module for AutoNav AI.
Connects Preprocessor, SegFormer Terrain Segmenter, YOLO Obstacle Detector,
Traversability Costmap Analyzer, and Direction Planner into a unified pipeline.
"""

import time
import numpy as np
from typing import Dict, Optional

from ..preprocessing.preprocessor import ImagePreprocessor
from ..segmentation.terrain_segmenter import TerrainSegmenter
from ..obstacle_detection.yolo_detector import YOLODetector
from ..navigation.traversability_analyzer import TraversabilityAnalyzer
from ..navigation.direction_planner import DirectionPlanner


class AutoNavPipeline:
    """Master pipeline for autonomous outdoor navigation assistance."""

    def __init__(
        self,
        segformer_dir: Optional[str] = None,
        yolo_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        print("[AutoNavPipeline] Initializing AutoNav AI System Modules...")
        self.preprocessor = ImagePreprocessor()
        self.segmenter = TerrainSegmenter(model_dir=segformer_dir, device=device)
        self.detector = YOLODetector(model_path=yolo_path)
        self.traversability_analyzer = TraversabilityAnalyzer(obstacle_inflation_radius=15)
        self.direction_planner = DirectionPlanner()
        print("[AutoNavPipeline] Initialization Complete. Ready for inference.")

    def process(self, image_bgr: np.ndarray) -> Dict:
        """
        Executes full vision-based navigation analysis on input image.
        
        Args:
            image_bgr: Raw input BGR image (from camera or file).
            
        Returns:
            Dictionary containing outputs from all sub-modules:
                - 'original_rgb': Preprocessed RGB image.
                - 'segmentation': Dict with class_map, color_mask, and class_distribution.
                - 'detection': Dict with boxes, classes, scores, annotated_image.
                - 'traversability': Dict with costmap, color_costmap, overall_score.
                - 'navigation': Dict with direction, action_label, sector_scores, hud_overlay.
                - 'latencies': Latency in ms for each pipeline stage.
        """
        timings = {}
        t0 = time.time()

        # 1. Preprocessing
        enhanced_bgr, rgb_image, scale = self.preprocessor.preprocess(image_bgr)
        timings["preprocessor_ms"] = round((time.time() - t0) * 1000, 1)

        # 2. Terrain Segmentation (SegFormer)
        t_seg = time.time()
        seg_res = self.segmenter.segment(rgb_image)
        timings["segmentation_ms"] = round((time.time() - t_seg) * 1000, 1)

        # 3. Obstacle Detection (YOLO)
        t_det = time.time()
        det_res = self.detector.detect(enhanced_bgr)
        timings["detection_ms"] = round((time.time() - t_det) * 1000, 1)

        # 4. Traversability Costmap Fusion
        t_trav = time.time()
        trav_res = self.traversability_analyzer.analyze(
            terrain_traversability=seg_res["traversability_map"],
            obstacle_mask=det_res["obstacle_mask"]
        )
        timings["traversability_ms"] = round((time.time() - t_trav) * 1000, 1)

        # 5. Direction & Steering HUD Planner
        t_nav = time.time()
        nav_res = self.direction_planner.plan(
            costmap=trav_res["costmap"],
            base_rgb_image=rgb_image
        )
        timings["navigation_ms"] = round((time.time() - t_nav) * 1000, 1)

        timings["total_ms"] = round((time.time() - t0) * 1000, 1)

        return {
            "original_rgb": rgb_image,
            "segmentation": seg_res,
            "detection": det_res,
            "traversability": trav_res,
            "navigation": nav_res,
            "latencies": timings
        }
