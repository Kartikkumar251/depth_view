"""
layout_detector.py - Real-Time Geometric Object Layout Detection
Extracts real-time object boundaries, footprints, and contours from depth feeds.
No machine learning or heavy neural networks: fast, deterministic depth geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np
import cv2


@dataclass
class DetectedLayout:
    """Holds geometric attributes of a real-time detected object layout."""
    contour_2d: np.ndarray             # (N, 1, 2) pixel contour points
    polygon_corners_2d: np.ndarray     # (K, 2) simplified polygon vertex pixels
    oriented_box_2d: np.ndarray        # (4, 2) minimum area bounding box corners
    center_pixel: Tuple[int, int]      # (u, v) centroid
    box_width_px: float                # Bounding box width in pixels
    box_height_px: float               # Bounding box height in pixels
    orientation_deg: float             # In-plane rotation angle (degrees)
    object_depth_m: float              # Median metric depth of the object surface
    base_depth_m: float                # Median metric depth of background/table base plane
    elevation_height_m: float          # Height of object above base plane
    pixel_area: float                  # Contour area in pixels
    is_valid: bool                     # Whether a stable object layout is detected


class RealTimeLayoutDetector:
    """
    Detects rigid object layouts in real-time from depth maps:
    1. Identifies the reference ground/table surface plane.
    2. Segments elevated foreground objects (delta-Z > min_elevation).
    3. Finds the dominant object contour and fits polygon vertices.
    4. Calculates in-plane rotation and bounding box.
    """

    def __init__(
        self,
        min_elevation_m: float = 0.015,   # Minimum 1.5 cm height above surface
        max_distance_m: float = 1.50,     # Max sensor operating distance
        min_contour_area_px: int = 600,   # Minimum pixel size to filter noise
    ):
        self.min_elevation_m = min_elevation_m
        self.max_distance_m = max_distance_m
        self.min_contour_area_px = min_contour_area_px
        self.last_stable_layout: Optional[DetectedLayout] = None
        # Pre-allocated morphology kernels
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect_layout(
        self,
        depth_m: np.ndarray,
        search_roi: Optional[Tuple[int, int, int, int]] = None,
        color_bgr: Optional[np.ndarray] = None,
    ) -> Optional[DetectedLayout]:
        """
        Analyzes a single metric depth map (and optional color frame for FPV mode)
        and returns the detected object layout.
        Optimized with strided percentiles and localized ROI masking for maximum FPS.
        """
        h, w = depth_m.shape
        if search_roi is not None:
            ymin, ymax, xmin, xmax = search_roi
            ymin = max(0, min(ymin, h - 1))
            ymax = max(ymin + 1, min(ymax, h))
            xmin = max(0, min(xmin, w - 1))
            xmax = max(xmin + 1, min(xmax, w))
            roi_mask = np.zeros((h, w), dtype=bool)
            roi_mask[ymin:ymax, xmin:xmax] = True
        else:
            roi_mask = np.ones((h, w), dtype=bool)

        valid_depth = (depth_m > 0.15) & (depth_m <= self.max_distance_m) & roi_mask
        n_valid = np.count_nonzero(valid_depth)
        if n_valid < 1000:
            return None

        # 1. Fast estimated ground plane depth via strided percentile
        valid_depth_vals = depth_m[valid_depth]
        stride = max(1, len(valid_depth_vals) // 8000)
        sampled_vals = valid_depth_vals[::stride]
        base_depth_m = float(np.percentile(sampled_vals, 85))

        # 2. Segment elevated foreground object
        object_depth_threshold = base_depth_m - self.min_elevation_m
        foreground_mask = valid_depth & (depth_m < object_depth_threshold)

        if np.count_nonzero(foreground_mask) < self.min_contour_area_px:
            closest_z = float(np.percentile(sampled_vals, 5))
            foreground_mask = valid_depth & (depth_m <= closest_z + 0.10)
            base_depth_m = closest_z + 0.15

        # 3. Morphological cleanup with pre-allocated kernel
        mask_uint8 = foreground_mask.astype(np.uint8)
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, self.morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.morph_kernel)

        # 4. Contour detection
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest continuous contour
        largest_cnt = None
        max_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_contour_area_px and area > max_area:
                max_area = area
                largest_cnt = c

        if largest_cnt is None and color_bgr is not None:
            # Fallback to high-contrast visual contour segmentation for FPV feeds
            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
            color_contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in color_contours:
                area = cv2.contourArea(c)
                if self.min_contour_area_px <= area < (w * h * 0.90) and area > max_area:
                    max_area = area
                    largest_cnt = c

        if largest_cnt is None:
            return None

        area_px = float(max_area)

        # 5. Polygon Simplification
        peri = cv2.arcLength(largest_cnt, True)
        approx_poly = cv2.approxPolyDP(largest_cnt, 0.025 * peri, True)
        polygon_corners = approx_poly.reshape(-1, 2)

        # 6. Minimum Area Oriented Bounding Box
        rect = cv2.minAreaRect(largest_cnt)
        (center, (dim1, dim2), angle) = rect
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        box_w = max(dim1, dim2)
        box_h = min(dim1, dim2)

        if dim1 < dim2:
            angle = angle + 90.0

        # 7. Localized ROI object surface depth extraction (avoids full-frame allocation)
        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
        sub_depth = depth_m[by:by+bh, bx:bx+bw]
        sub_mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted_cnt = largest_cnt - np.array([bx, by])
        cv2.drawContours(sub_mask, [shifted_cnt], -1, 1, thickness=-1)

        obj_depths = sub_depth[(sub_mask == 1) & (sub_depth > 0)]
        if len(obj_depths) > 0:
            if len(obj_depths) > 4000:
                obj_depths = obj_depths[::(len(obj_depths) // 2000)]
            object_depth_m = float(np.median(obj_depths))
        else:
            object_depth_m = float(np.percentile(sampled_vals, 10))

        elevation_m = max(0.001, base_depth_m - object_depth_m)

        layout = DetectedLayout(
            contour_2d=largest_cnt,
            polygon_corners_2d=polygon_corners,
            oriented_box_2d=box_pts,
            center_pixel=(int(center[0]), int(center[1])),
            box_width_px=float(box_w),
            box_height_px=float(box_h),
            orientation_deg=round(float(angle), 1),
            object_depth_m=round(object_depth_m, 4),
            base_depth_m=round(base_depth_m, 4),
            elevation_height_m=round(elevation_m, 4),
            pixel_area=area_px,
            is_valid=True,
        )

        self.last_stable_layout = layout
        return layout
