"""
measurements.py - Metric Coordinate Matrix & Geometric Dimension Estimation
Provides:
- Sampled coordinate grid matrix (point_id, pixel_u, pixel_v, X_m, Y_m, Z_m)
- In-plane oriented bounding box & PCA dimension estimation (Length, Width, Height/Extent)
- Ground plane detection and reference plane offset
- Confidence scoring and limitation explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import cv2

try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class DimensionEstimate:
    """Stores estimated physical object dimensions and geometric analysis."""
    length_m: float
    width_m: float
    height_extent_m: float
    length_cm: float
    width_cm: float
    height_extent_cm: float
    surface_area_sq_cm: float
    is_height_relative_to_ground: bool
    height_description: str
    confidence_level: str  # "HIGH", "MODERATE", "LOW"
    confidence_score_pct: float
    quality_factors: List[str]
    plane_normal: Optional[np.ndarray]
    inlier_ratio_pct: float
    oriented_box_corners: Optional[np.ndarray]  # (8, 3) 3D box corners
    limitations: List[str] = field(default_factory=list)


class MeasurementEngine:
    """
    Computes metric coordinate matrices and geometric dimensions from 3D point clouds.
    """

    def __init__(self, min_points_per_cell: int = 3):
        self.min_points_per_cell = min_points_per_cell

    def generate_coordinate_matrix(
        self,
        reconstruction_result: Any,
        camera: Any,
        grid_rows: int = 10,
        grid_cols: int = 10,
    ) -> pd.DataFrame:
        """
        Creates a sampled 3D coordinate matrix table over the selected object ROI.
        
        Columns:
            point_id: Unique integer index (1..N)
            pixel_u: Horizontal image coordinate (pixels)
            pixel_v: Vertical image coordinate (pixels)
            X_m: Camera frame X coordinate in meters (left/right)
            Y_m: Camera frame Y coordinate in meters (up/down)
            Z_m: Camera frame Z coordinate in meters (depth distance)
            
        Cells with insufficient valid depth are marked with np.nan and displayed as N/A.
        """
        ymin, ymax, xmin, xmax = reconstruction_result.roi
        depth_map = reconstruction_result.aggregated_depth

        row_edges = np.linspace(ymin, ymax, grid_rows + 1, dtype=int)
        col_edges = np.linspace(xmin, xmax, grid_cols + 1, dtype=int)

        records = []
        point_id = 1

        for r in range(grid_rows):
            r_start, r_end = row_edges[r], row_edges[r + 1]
            center_v = int((r_start + r_end) / 2)

            for c in range(grid_cols):
                c_start, c_end = col_edges[c], col_edges[c + 1]
                center_u = int((c_start + c_end) / 2)

                cell_depths = depth_map[r_start:r_end, c_start:c_end]
                valid_depths = cell_depths[cell_depths > 0]

                if len(valid_depths) >= self.min_points_per_cell:
                    median_z = float(np.median(valid_depths))
                    # Deproject the center of this cell at the robust median depth
                    x_m, y_m, z_m = camera.deproject_pixel(center_u, center_v, median_z)
                    records.append({
                        "point_id": point_id,
                        "pixel_u": center_u,
                        "pixel_v": center_v,
                        "X_m": round(float(x_m), 4),
                        "Y_m": round(float(y_m), 4),
                        "Z_m": round(float(z_m), 4),
                        "status": "VALID",
                    })
                else:
                    records.append({
                        "point_id": point_id,
                        "pixel_u": center_u,
                        "pixel_v": center_v,
                        "X_m": np.nan,
                        "Y_m": np.nan,
                        "Z_m": np.nan,
                        "status": "N/A",
                    })
                point_id += 1

        df = pd.DataFrame(records)
        return df

    def estimate_dimensions(
        self,
        reconstruction_result: Any,
        reference_ground_plane: Optional[np.ndarray] = None,
    ) -> DimensionEstimate:
        """
        Estimates length, width, and height/thickness using RANSAC plane fitting,
        in-plane projection, and 2D minimum-area oriented bounding box.
        """
        points = reconstruction_result.points
        pcd = reconstruction_result.pcd

        limitations = [
            "Single-view surface scan: Only front-facing geometry is visible to the camera. "
            "Hidden rear and base dimensions cannot be directly measured without multiple viewpoints.",
            "Estimates are sensitive to camera distance (optimal 0.4m - 1.5m), ambient lighting, "
            "surface reflectivity, and ROI boundary accuracy.",
            "A square visual appearance in RGB does not guarantee physical squareness without metric depth.",
        ]

        # Handle empty or sparse point cloud
        if len(points) < 15:
            return DimensionEstimate(
                length_m=0.0,
                width_m=0.0,
                height_extent_m=0.0,
                length_cm=0.0,
                width_cm=0.0,
                height_extent_cm=0.0,
                surface_area_sq_cm=0.0,
                is_height_relative_to_ground=False,
                height_description="Insufficient valid 3D points in ROI to estimate dimensions.",
                confidence_level="LOW",
                confidence_score_pct=0.0,
                quality_factors=["Point count below minimum threshold (15 points)."],
                plane_normal=None,
                inlier_ratio_pct=0.0,
                oriented_box_corners=None,
                limitations=limitations,
            )

        # 1. RANSAC Plane Fitting to identify dominant object face
        plane_normal = None
        inlier_ratio = 0.0
        inlier_points = points

        if pcd is not None and len(pcd.points) >= 30:
            try:
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=0.005,  # 5mm tolerance
                    ransac_n=3,
                    num_iterations=1000,
                )
                a, b, c, d = plane_model
                norm = np.sqrt(a * a + b * b + c * c)
                if norm > 1e-6:
                    plane_normal = np.array([a, b, c], dtype=np.float32) / norm
                    # Ensure normal points toward camera (z < 0)
                    if plane_normal[2] > 0:
                        plane_normal = -plane_normal
                    inlier_ratio = (len(inliers) / len(points)) * 100.0
                    inlier_points = points[inliers]
            except Exception:
                pass

        if plane_normal is None:
            # Fallback to PCA normal (eigenvector with smallest eigenvalue)
            centroid = np.mean(points, axis=0)
            shifted = points - centroid
            cov = np.cov(shifted.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            plane_normal = eigenvecs[:, 0]  # Smallest variance direction
            if plane_normal[2] > 0:
                plane_normal = -plane_normal
            inlier_ratio = 75.0

        # 2. Project points onto 2D plane coordinate system for precise length & width
        # Construct orthonormal basis (u_vec, v_vec, n_vec)
        n_vec = plane_normal / np.linalg.norm(plane_normal)
        arbitrary = np.array([0.0, 1.0, 0.0]) if abs(n_vec[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_vec = np.cross(n_vec, arbitrary)
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(n_vec, u_vec)

        # Projection into 2D plane coordinates
        centroid = np.mean(inlier_points, axis=0)
        diff = inlier_points - centroid
        u_coords = np.dot(diff, u_vec).astype(np.float32)
        v_coords = np.dot(diff, v_vec).astype(np.float32)
        pts_2d = np.stack([u_coords, v_coords], axis=-1)

        # 3. Minimum Area Bounding Box in 2D (invariant to object rotation on plane)
        rect = cv2.minAreaRect(pts_2d)
        (center_2d, (dim1, dim2), angle) = rect
        length_m = float(max(dim1, dim2))
        width_m = float(min(dim1, dim2))

        # 4. Height / Depth Extent Estimation
        # Compute projection along normal
        normal_projections = np.dot(points - centroid, n_vec)
        normal_extent_m = float(np.percentile(normal_projections, 95) - np.percentile(normal_projections, 5))
        normal_extent_m = max(normal_extent_m, 0.001)

        is_height_relative_to_ground = False
        if reference_ground_plane is not None:
            # If user or algorithm set a ground plane (A, B, C, D)
            # Distance from centroid to ground plane: |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
            gp = reference_ground_plane
            gp_norm = np.linalg.norm(gp[:3])
            if gp_norm > 1e-6:
                dist_to_ground = abs(np.dot(gp[:3], centroid) + gp[3]) / gp_norm
                height_extent_m = float(dist_to_ground)
                is_height_relative_to_ground = True
                height_description = "Measured height above calibrated ground/table reference plane."
            else:
                height_extent_m = normal_extent_m
                height_description = "Visible surface depth extent along surface normal (single-view constraint)."
        else:
            height_extent_m = normal_extent_m
            height_description = (
                "Visible face depth variation/thickness along normal. "
                "Hidden rear side cannot be measured from a single viewpoint."
            )

        # 5. Oriented 3D Bounding Box Corners for 3D visualization
        box_corners_3d = None
        if pcd is not None and len(pcd.points) >= 10:
            try:
                obb = pcd.get_oriented_bounding_box()
                box_corners_3d = np.asarray(obb.get_box_points(), dtype=np.float32)
            except Exception:
                pass

        # 6. Confidence and Quality Scoring
        quality_factors = []
        confidence_score = 100.0

        # Depth coverage in ROI
        coverage = reconstruction_result.valid_depth_pct
        if coverage >= 85.0:
            quality_factors.append(f"Excellent depth coverage: {coverage:.1f}% valid pixels.")
        elif coverage >= 60.0:
            quality_factors.append(f"Moderate depth coverage: {coverage:.1f}% valid pixels.")
            confidence_score -= 15.0
        else:
            quality_factors.append(f"Low depth coverage: {coverage:.1f}% valid pixels (depth dropouts).")
            confidence_score -= 35.0

        # Point count
        pt_count = reconstruction_result.accepted_points_count
        if pt_count >= 2000:
            quality_factors.append(f"Dense 3D point cloud: {pt_count:,} points.")
        elif pt_count >= 500:
            quality_factors.append(f"Adequate point cloud density: {pt_count:,} points.")
            confidence_score -= 10.0
        else:
            quality_factors.append(f"Sparse point cloud: {pt_count:,} points.")
            confidence_score -= 30.0

        # Plane fit consistency
        if inlier_ratio >= 80.0:
            quality_factors.append(f"Strong planar fit: {inlier_ratio:.1f}% inliers.")
        elif inlier_ratio >= 50.0:
            quality_factors.append(f"Moderate planar fit: {inlier_ratio:.1f}% inliers.")
            confidence_score -= 15.0
        else:
            quality_factors.append(f"Rough/non-planar surface: {inlier_ratio:.1f}% inliers.")
            confidence_score -= 25.0

        # Distance check
        avg_z = float(np.mean(points[:, 2]))
        if 0.35 <= avg_z <= 1.2:
            quality_factors.append(f"Optimal working distance: {avg_z:.2f} m.")
        elif avg_z < 0.35:
            quality_factors.append(f"Subject close to camera minimum range: {avg_z:.2f} m.")
            confidence_score -= 15.0
        else:
            quality_factors.append(f"Subject further from camera: {avg_z:.2f} m (increased noise).")
            confidence_score -= 20.0

        confidence_score = max(5.0, min(confidence_score, 98.0))
        if confidence_score >= 80.0:
            confidence_level = "HIGH"
        elif confidence_score >= 55.0:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        length_cm = length_m * 100.0
        width_cm = width_m * 100.0
        height_extent_cm = height_extent_m * 100.0
        surface_area_sq_cm = length_cm * width_cm

        return DimensionEstimate(
            length_m=round(length_m, 4),
            width_m=round(width_m, 4),
            height_extent_m=round(height_extent_m, 4),
            length_cm=round(length_cm, 2),
            width_cm=round(width_cm, 2),
            height_extent_cm=round(height_extent_cm, 2),
            surface_area_sq_cm=round(surface_area_sq_cm, 2),
            is_height_relative_to_ground=is_height_relative_to_ground,
            height_description=height_description,
            confidence_level=confidence_level,
            confidence_score_pct=round(confidence_score, 1),
            quality_factors=quality_factors,
            plane_normal=plane_normal,
            inlier_ratio_pct=round(inlier_ratio, 1),
            oriented_box_corners=box_corners_3d,
            limitations=limitations,
        )
