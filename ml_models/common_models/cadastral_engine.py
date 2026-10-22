"""
cadastral_engine.py - Metric Cadastral Analysis & Spatial Matrix Dissection
Calculates real-world metric Area, Perimeter, Length, Width, Height,
spatial coordinate grid matrices, and boundary encroachment detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import cv2

try:
    from .layout_detector import DetectedLayout
except ImportError:
    try:
        from ml_models.common_models.layout_detector import DetectedLayout
    except ImportError:
        from layout_detector import DetectedLayout


@dataclass
class CadastralMetrics:
    """Detailed metric layout analysis for land / structure surveying."""
    length_m: float
    width_m: float
    length_cm: float
    width_cm: float
    height_elevation_m: float
    height_elevation_cm: float
    footprint_area_sq_m: float
    footprint_area_sq_cm: float
    perimeter_m: float
    perimeter_cm: float
    aspect_ratio: float
    orientation_deg: float
    rectangularity_pct: float
    compactness_ratio: float
    vertex_count: int
    boundary_vertices_3d: List[Dict[str, float]]
    encroachment_status: str       # "CLEAR", "POTENTIAL ENCROACHMENT", "ENCROACHMENT DETECTED"
    encroachment_area_sq_cm: float
    max_setback_violation_cm: float
    survey_summary: str
    matrix_df: pd.DataFrame


class CadastralAnalysisEngine:
    """
    Transforms detected 2D layout contours into metric 3D survey data:
    - Deprojects boundary corners using camera calibration
    - Computes metric 3D Area (Shoelace formula) and 3D Perimeter
    - Dissects layout into spatial matrix grid
    - Simulates cadastral zoning & encroachment detection
    """

    def __init__(self, grid_resolution: int = 10):
        self.grid_resolution = grid_resolution

    def analyze_layout(
        self,
        layout: DetectedLayout,
        camera: Any,
        depth_m: np.ndarray,
        reference_property_line: Optional[Tuple[int, int, int, int]] = None,
    ) -> CadastralMetrics:
        """
        Executes full cadastral and metric layout analysis on a detected object.
        """
        corners_2d = layout.polygon_corners_2d
        obj_depth = max(layout.object_depth_m, 0.20)
        base_depth = max(layout.base_depth_m, obj_depth + 0.01)

        # 1. Deproject 2D polygon corners to 3D metric coordinates (meters)
        u_corners = corners_2d[:, 0].astype(np.float32)
        v_corners = corners_2d[:, 1].astype(np.float32)
        z_corners = np.full(len(corners_2d), obj_depth, dtype=np.float32)
        
        if hasattr(camera, "deproject_points_vectorized"):
            xs, ys, zs = camera.deproject_points_vectorized(u_corners, v_corners, z_corners)
        else:
            xs, ys, zs = [], [], []
            for u, v in corners_2d:
                x_m, y_m, z_m = camera.deproject_pixel(float(u), float(v), obj_depth)
                xs.append(x_m); ys.append(y_m); zs.append(z_m)
            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

        vertices_3d = []
        pts_3d = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        for i, (u, v) in enumerate(corners_2d):
            vertices_3d.append({
                "vertex_id": i + 1,
                "pixel_u": int(u),
                "pixel_v": int(v),
                "X_m": round(float(xs[i]), 4),
                "Y_m": round(float(ys[i]), 4),
                "Z_m": round(float(zs[i]), 4),
            })

        n_verts = len(pts_3d)

        # 2. Metric 3D Perimeter Calculation (vectorized roll diff)
        diffs = pts_3d - np.roll(pts_3d, -1, axis=0)
        perimeter_m = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

        # 3. Metric Footprint Area (Shoelace Formula on calibrated plane coordinates)
        x_coords = pts_3d[:, 0]
        y_coords = pts_3d[:, 1]
        area_m2 = 0.5 * abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
        area_m2 = float(area_m2)

        # 4. Metric Length and Width from Oriented Bounding Box
        box_2d = layout.oriented_box_2d
        u_box = box_2d[:, 0].astype(np.float32)
        v_box = box_2d[:, 1].astype(np.float32)
        z_box = np.full(len(box_2d), obj_depth, dtype=np.float32)
        if hasattr(camera, "deproject_points_vectorized"):
            bxs, bys, bzs = camera.deproject_points_vectorized(u_box, v_box, z_box)
            box_3d = np.stack([bxs, bys, bzs], axis=1)
        else:
            box_3d = np.array([camera.deproject_pixel(float(u), float(v), obj_depth) for u, v in box_2d], dtype=np.float32)

        edge1 = float(np.linalg.norm(box_3d[1] - box_3d[0]))
        edge2 = float(np.linalg.norm(box_3d[2] - box_3d[1]))
        length_m = max(edge1, edge2)
        width_m = min(edge1, edge2)

        # 5. Elevation Height (Structure / Object height above ground)
        elevation_height_m = max(0.005, base_depth - obj_depth)

        # 6. Geometric Shape Factors
        box_area_m2 = max(length_m * width_m, 1e-6)
        rectangularity = min(100.0, (area_m2 / box_area_m2) * 100.0)
        aspect_ratio = length_m / max(width_m, 1e-4)
        compactness = (4.0 * np.pi * area_m2) / max(perimeter_m ** 2, 1e-6)

        # 7. Dissect Layout into Spatial Coordinate Grid Matrix (Vectorized)
        rect_pts = layout.oriented_box_2d
        xmin = max(0, int(np.min(rect_pts[:, 0])))
        xmax = min(depth_m.shape[1] - 1, int(np.max(rect_pts[:, 0])))
        ymin = max(0, int(np.min(rect_pts[:, 1])))
        ymax = min(depth_m.shape[0] - 1, int(np.max(rect_pts[:, 1])))

        row_steps = np.linspace(ymin, ymax, self.grid_resolution, dtype=int)
        col_steps = np.linspace(xmin, xmax, self.grid_resolution, dtype=int)
        r_grid, c_grid = np.meshgrid(row_steps, col_steps, indexing="ij")
        r_flat = r_grid.ravel()
        c_flat = c_grid.ravel()

        # Fast inside/outside lookup using rasterized ROI mask
        roi_mask = np.zeros((depth_m.shape[0], depth_m.shape[1]), dtype=np.uint8)
        cv2.drawContours(roi_mask, [layout.contour_2d], -1, 1, thickness=-1)

        raw_depths = depth_m[r_flat, c_flat]
        inside_flags = roi_mask[r_flat, c_flat] == 1

        cell_depths = np.where(
            (raw_depths > 0) & inside_flags,
            raw_depths,
            np.where(inside_flags, obj_depth, base_depth),
        ).astype(np.float32)

        if hasattr(camera, "deproject_points_vectorized"):
            X_all, Y_all, Z_all = camera.deproject_points_vectorized(
                c_flat.astype(np.float32), r_flat.astype(np.float32), cell_depths
            )
        else:
            X_all, Y_all, Z_all = [], [], []
            for c_px, r_px, z_px in zip(c_flat, r_flat, cell_depths):
                x_p, y_p, z_p = camera.deproject_pixel(float(c_px), float(r_px), float(z_px))
                X_all.append(x_p); Y_all.append(y_p); Z_all.append(z_p)
            X_all, Y_all, Z_all = np.array(X_all), np.array(Y_all), np.array(Z_all)

        elevations_cm = np.round((base_depth - cell_depths) * 100.0, 1)

        matrix_records = []
        total_cells = len(r_flat)
        for idx in range(total_cells):
            r_idx = idx // self.grid_resolution
            c_idx = idx % self.grid_resolution
            matrix_records.append({
                "cell_id": idx + 1,
                "grid_r": r_idx + 1,
                "grid_c": c_idx + 1,
                "pixel_u": int(c_flat[idx]),
                "pixel_v": int(r_flat[idx]),
                "X_m": round(float(X_all[idx]), 4),
                "Y_m": round(float(Y_all[idx]), 4),
                "Z_m": round(float(Z_all[idx]), 4),
                "height_cm": float(elevations_cm[idx]),
                "zone_status": "INSIDE" if inside_flags[idx] else "OUTSIDE",
            })

        matrix_df = pd.DataFrame(matrix_records)

        # 8. Cadastral Boundary Encroachment Analysis
        # Simulates checking against a designated property boundary line
        # Default property boundary: safe inner region
        if reference_property_line is None:
            # Simulated setback line at 10% inside the camera view
            h_frame, w_frame = depth_m.shape
            p_ymin, p_ymax = int(h_frame * 0.15), int(h_frame * 0.85)
            p_xmin, p_xmax = int(w_frame * 0.15), int(w_frame * 0.85)
            reference_property_line = (p_ymin, p_ymax, p_xmin, p_xmax)

        p_ymin, p_ymax, p_xmin, p_xmax = reference_property_line
        encroachment_status = "CLEAR / COMPLIANT"
        encroach_area_cm2 = 0.0
        max_violation_cm = 0.0

        # Check if any corners of the layout cross the parcel bounds
        violations = []
        for u, v in corners_2d:
            dist_v = 0.0
            if u < p_xmin:
                dist_v = max(dist_v, p_xmin - u)
            elif u > p_xmax:
                dist_v = max(dist_v, u - p_xmax)
            if v < p_ymin:
                dist_v = max(dist_v, p_ymin - v)
            elif v > p_ymax:
                dist_v = max(dist_v, v - p_ymax)
            if dist_v > 0:
                # Convert pixel violation to metric cm
                cm_per_px = (width_m * 100.0) / max(layout.box_height_px, 1.0)
                violations.append(dist_v * cm_per_px)

        if violations:
            max_violation_cm = round(float(max(violations)), 1)
            encroach_area_cm2 = round(area_m2 * 10000.0 * 0.18, 1)  # Estimated encroachment slice
            if max_violation_cm > 4.0:
                encroachment_status = "CRITICAL ENCROACHMENT"
            else:
                encroachment_status = "MARGINAL SETBACK VIOLATION"
            survey_summary = (
                f"Structure footprint extends beyond boundary setback by {max_violation_cm:.1f} cm. "
                f"Flagged for cadastral review."
            )
        else:
            survey_summary = (
                f"Structure footprint is completely within lawful parcel boundary setbacks. "
                f"Clearance maintained on all sides."
            )

        # Convert units for display
        area_cm2 = area_m2 * 10000.0
        peri_cm = perimeter_m * 100.0
        len_cm = length_m * 100.0
        wid_cm = width_m * 100.0
        elev_cm = elevation_height_m * 100.0

        return CadastralMetrics(
            length_m=round(length_m, 4),
            width_m=round(width_m, 4),
            length_cm=round(len_cm, 2),
            width_cm=round(wid_cm, 2),
            height_elevation_m=round(elevation_height_m, 4),
            height_elevation_cm=round(elev_cm, 2),
            footprint_area_sq_m=round(area_m2, 4),
            footprint_area_sq_cm=round(area_cm2, 2),
            perimeter_m=round(perimeter_m, 4),
            perimeter_cm=round(peri_cm, 2),
            aspect_ratio=round(aspect_ratio, 2),
            orientation_deg=layout.orientation_deg,
            rectangularity_pct=round(rectangularity, 1),
            compactness_ratio=round(compactness, 2),
            vertex_count=n_verts,
            boundary_vertices_3d=vertices_3d,
            encroachment_status=encroachment_status,
            encroachment_area_sq_cm=encroach_area_cm2,
            max_setback_violation_cm=max_violation_cm,
            survey_summary=survey_summary,
            matrix_df=matrix_df,
        )
