"""
Obstacle Detection Module for AutoNav AI using YOLO Nano (YOLOv8n).
Detects objects and potential navigational hazards in outdoor environments,
returning bounding boxes, confidences, classifications, and obstacle masks.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class YOLODetector:
    """YOLO Nano object detector for real-time hazard identification."""

    # Key outdoor obstacles prioritized for UGV avoidance
    OBSTACLE_CLASSES = {
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "horse", "cow", "bench", "potted plant", "traffic light",
        "stop sign", "chair", "fire hydrant"
    }

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "yolov8n.pt")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads YOLOv8n weights."""
        try:
            print(f"[YOLODetector] Loading YOLOv8n from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("[YOLODetector] Model loaded successfully.")
        except Exception as e:
            print(f"[YOLODetector] Warning: Failed to load local YOLO model ({e}). Fallback to standard weights or heuristic.")
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception as e2:
                print(f"[YOLODetector] Offline fallback to contour detector: {e2}")
                self.model = None

    def detect(self, image_bgr: np.ndarray) -> Dict:
        """
        Runs object detection on input BGR image.
        
        Args:
            image_bgr: Input image in BGR format.
            
        Returns:
            Dictionary containing:
                - 'boxes': List of [x1, y1, x2, y2]
                - 'scores': List of float confidence scores
                - 'classes': List of str class names
                - 'obstacle_mask': 2D boolean mask indicating obstacle footprint
                - 'annotated_image': BGR image with drawn bounding boxes and labels
        """
        h, w = image_bgr.shape[:2]
        obstacle_mask = np.zeros((h, w), dtype=bool)
        annotated_image = image_bgr.copy()

        boxes = []
        scores = []
        class_names = []

        if self.model is not None:
            results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)

            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names.get(cls_id, f"obj_{cls_id}")

                    x1, y1, x2, y2 = coords
                    # Clip coordinates to image boundary
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    boxes.append([x1, y1, x2, y2])
                    scores.append(round(conf, 2))
                    class_names.append(label)

                    # Mark obstacle footprint
                    obstacle_mask[y1:y2, x1:x2] = True

                    # Color code box by risk (lower ground objects in center are critical)
                    is_center = (x1 + x2) // 2 > w // 3 and (x1 + x2) // 2 < 2 * w // 3
                    box_color = (0, 0, 255) if is_center else (0, 165, 255)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(
                        annotated_image,
                        text,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
        else:
            # Fallback contour detector if YOLO weights are unavailable
            boxes, scores, class_names, obstacle_mask, annotated_image = self._fallback_contour_detector(image_bgr)

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": class_names,
            "obstacle_mask": obstacle_mask,
            "annotated_image": annotated_image,
            "total_count": len(boxes)
        }

    def _fallback_contour_detector(self, image_bgr: np.ndarray) -> Tuple:
        """Heuristic obstacle detector using edge & contour saliency."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        scores = []
        class_names = []
        obstacle_mask = np.zeros((h, w), dtype=bool)
        annotated = image_bgr.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter ground terrain noise, keep solid objects
            if 600 < area < (h * w * 0.4):
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                # Ignore items in high sky
                if y + h_box > int(h * 0.4):
                    boxes.append([x, y, x + w_box, y + h_box])
                    scores.append(0.85)
                    class_names.append("obstacle")
                    obstacle_mask[y:y + h_box, x:x + w_box] = True
                    cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                    cv2.putText(annotated, "obstacle", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return boxes, scores, class_names, obstacle_mask, annotated
