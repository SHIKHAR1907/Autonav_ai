"""
Direction Planner Module for AutoNav AI.
Evaluates directional sector traversability and suggests safe navigation headings
with Heads-Up Display (HUD) trajectory overlays for Unmanned Ground Vehicles.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class DirectionPlanner:
    """Plans steering direction and renders robotic HUD trajectory overlays."""

    def __init__(self, forward_clearance_threshold: float = 0.55):
        self.forward_threshold = forward_clearance_threshold

    def plan(self, costmap: np.ndarray, base_rgb_image: np.ndarray) -> Dict:
        """
        Plans safe navigation direction based on ground traversability.
        
        Args:
            costmap: 2D float array (0.0 to 1.0) of traversability.
            base_rgb_image: Original RGB image to overlay HUD telemetry.
            
        Returns:
            Dictionary containing:
                - 'direction': str ('FORWARD', 'STEER_LEFT', 'STEER_RIGHT', 'STOP')
                - 'action_label': Human-readable recommendation string
                - 'sector_scores': Dict of { 'left': float, 'center': float, 'right': float }
                - 'hud_overlay': RGB image with trajectory arrows and telemetry HUD
        """
        h, w = costmap.shape[:2]

        # Consider ground travel zone: from bottom (h) up to horizon (0.45 * h)
        ground_top = int(h * 0.45)
        ground_roi = costmap[ground_top:, :]

        col_w = w // 3
        left_roi = ground_roi[:, :col_w]
        center_roi = ground_roi[:, col_w: 2 * col_w]
        right_roi = ground_roi[:, 2 * col_w:]

        # Compute sector safety scores (0 to 100%)
        left_score = round(float(np.mean(left_roi)) * 100, 1) if left_roi.size > 0 else 0.0
        center_score = round(float(np.mean(center_roi)) * 100, 1) if center_roi.size > 0 else 0.0
        right_score = round(float(np.mean(right_roi)) * 100, 1) if right_roi.size > 0 else 0.0

        # Check immediate near-field hazard directly in front of vehicle (bottom 25% of center)
        near_field_center = ground_roi[int(ground_roi.shape[0] * 0.75):, col_w: 2 * col_w]
        near_hazard = np.mean(near_field_center) < 0.35 if near_field_center.size > 0 else False

        # Direction Decision Logic
        if not near_hazard and center_score >= (self.forward_threshold * 100):
            direction = "FORWARD"
            action_label = "🟢 PROCEED FORWARD (Path Clear)"
            status_color = (0, 220, 50)  # Green
        elif left_score > right_score and left_score >= 35.0:
            direction = "STEER_LEFT"
            action_label = "🟡 STEER LEFT (Avoid Obstacle Ahead)"
            status_color = (255, 180, 0)  # Yellow
        elif right_score >= 35.0:
            direction = "STEER_RIGHT"
            action_label = "🟡 STEER RIGHT (Avoid Obstacle Ahead)"
            status_color = (255, 180, 0)  # Yellow
        else:
            direction = "STOP"
            action_label = "🔴 STOP / ALL SECTORS IMPASSABLE"
            status_color = (230, 30, 30)  # Red

        # Render HUD Overlay
        hud_overlay = self._draw_hud(
            base_rgb_image.copy(),
            direction,
            action_label,
            status_color,
            left_score,
            center_score,
            right_score
        )

        return {
            "direction": direction,
            "action_label": action_label,
            "sector_scores": {
                "left": left_score,
                "center": center_score,
                "right": right_score
            },
            "hud_overlay": hud_overlay
        }

    def _draw_hud(
        self,
        img: np.ndarray,
        direction: str,
        action_label: str,
        status_color: Tuple[int, int, int],
        l_score: float,
        c_score: float,
        r_score: float
    ) -> np.ndarray:
        """Draws robotic telemetry HUD, sector lines, and navigation trajectory arrows."""
        h, w = img.shape[:2]
        ground_top = int(h * 0.45)
        col_w = w // 3

        # Draw transparent sector overlay on ground
        overlay = img.copy()

        # Vertical sector dividers
        cv2.line(overlay, (col_w, ground_top), (col_w, h), (255, 255, 255), 2)
        cv2.line(overlay, (2 * col_w, ground_top), (2 * col_w, h), (255, 255, 255), 2)

        # Highlight best sector with soft tint
        if direction == "FORWARD":
            cv2.rectangle(overlay, (col_w, ground_top), (2 * col_w, h), (0, 220, 50), -1)
        elif direction == "STEER_LEFT":
            cv2.rectangle(overlay, (0, ground_top), (col_w, h), (255, 180, 0), -1)
        elif direction == "STEER_RIGHT":
            cv2.rectangle(overlay, (2 * col_w, ground_top), (w, h), (255, 180, 0), -1)
        elif direction == "STOP":
            cv2.rectangle(overlay, (0, ground_top), (w, h), (220, 30, 30), -1)

        # Blend overlay (alpha = 0.22)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

        # Draw trajectory navigation arrow from vehicle bottom center
        start_pt = (w // 2, h - 30)

        if direction == "FORWARD":
            end_pt = (w // 2, ground_top + 50)
            cv2.arrowedLine(img, start_pt, end_pt, (0, 255, 0), 6, tipLength=0.2)
        elif direction == "STEER_LEFT":
            control_pt = (col_w // 2, ground_top + 60)
            cv2.arrowedLine(img, start_pt, control_pt, (255, 200, 0), 6, tipLength=0.2)
        elif direction == "STEER_RIGHT":
            control_pt = (w - col_w // 2, ground_top + 60)
            cv2.arrowedLine(img, start_pt, control_pt, (255, 200, 0), 6, tipLength=0.2)
        else:
            # STOP indicator
            cv2.line(img, (w // 2 - 40, h - 120), (w // 2 + 40, h - 60), (255, 0, 0), 6)
            cv2.line(img, (w // 2 + 40, h - 120), (w // 2 - 40, h - 60), (255, 0, 0), 6)

        # Header Navigation Banner
        banner_h = 55
        cv2.rectangle(img, (0, 0), (w, banner_h), (25, 28, 36), -1)
        cv2.putText(
            img,
            f"AUTONAV AI | {action_label}",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # Sector telemetry labels at bottom
        cv2.putText(img, f"LEFT: {l_score:.0f}%", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"CTR: {c_score:.0f}%", (col_w + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"RIGHT: {r_score:.0f}%", (2 * col_w + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img
