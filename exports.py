"""
exports.py - Data Export Utilities for Bhoomi Drishti
Exports:
- Sampled 3D coordinate matrix as CSV
- Coloured 3D point cloud as standard PLY (compatible with MeshLab, CloudCompare, Blender)
- Scan summary and dimensional report as JSON
"""

from __future__ import annotations

import os
import json
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    import open3d as o3d
except ImportError:
    o3d = None


class DataExporter:
    """Handles persistent filesystem exports and in-memory downloads."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_timestamp_slug() -> str:
        """Returns a formatted timestamp slug for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_csv(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """
        Exports the sampled coordinate matrix to a CSV file.
        
        Returns:
            Tuple of (output_path, csv_string_data)
        """
        if filename is None:
            filename = f"coordinate_matrix_{self.get_timestamp_slug()}.csv"

        filepath = self.output_dir / filename

        # Ensure header comments documenting metadata
        header_comments = (
            "# Bhoomi Drishti - Sampled 3D Coordinate Matrix\n"
            "# Camera coordinate frame in METRES (X: Right, Y: Down, Z: Forward Depth)\n"
            "# Note: Sampled representative grid coordinates, not a literal per-pixel measurement.\n"
        )

        csv_content = df.to_csv(index=False)
        full_content = header_comments + csv_content

        filepath.write_text(full_content, encoding="utf-8")
        return filepath, full_content

    def export_ply(
        self,
        reconstruction_result: Any,
        filename: Optional[str] = None,
    ) -> Tuple[Path, bytes]:
        """
        Exports the 3D point cloud with RGB vertex colors to a standard PLY file.
        
        Returns:
            Tuple of (output_path, ply_bytes)
        """
        if filename is None:
            filename = f"point_cloud_{self.get_timestamp_slug()}.ply"

        filepath = self.output_dir / filename
        pcd = reconstruction_result.pcd
        points = reconstruction_result.points
        colors = reconstruction_result.colors

        # If Open3D is available and pcd is populated, use write_point_cloud
        if o3d is not None and pcd is not None and len(pcd.points) > 0:
            o3d.io.write_point_cloud(str(filepath), pcd, write_ascii=False)
            ply_bytes = filepath.read_bytes()
            return filepath, ply_bytes

        # Fallback binary PLY generation directly from NumPy arrays
        ply_bytes = self.create_binary_ply_bytes(points, colors, reconstruction_result.normals)
        filepath.write_bytes(ply_bytes)
        return filepath, ply_bytes

    @staticmethod
    def create_binary_ply_bytes(
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bytes:
        """Constructs a binary little-endian PLY file in memory."""
        n_points = len(points)
        has_normals = normals is not None and len(normals) == n_points

        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"comment Bhoomi Drishti RealSense 3D Scanner Export - {n_points} vertices",
            f"element vertex {n_points}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_normals:
            header.extend([
                "property float nx",
                "property float ny",
                "property float nz",
            ])

        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ])

        header_bytes = "\n".join(header).encode("ascii")

        # Color values normalized to [0, 255]
        if np.max(colors) <= 1.0:
            rgb_uint8 = (colors * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_uint8 = colors.clip(0, 255).astype(np.uint8)

        pts_float32 = points.astype("<f4")

        # Interleave vertex data
        data_buffer = io.BytesIO()
        data_buffer.write(header_bytes)

        if has_normals:
            norms_float32 = normals.astype("<f4")
            for i in range(n_points):
                data_buffer.write(pts_float32[i].tobytes())
                data_buffer.write(norms_float32[i].tobytes())
                data_buffer.write(rgb_uint8[i].tobytes())
        else:
            for i in range(n_points):
                data_buffer.write(pts_float32[i].tobytes())
                data_buffer.write(rgb_uint8[i].tobytes())

        return data_buffer.getvalue()

    def export_summary_json(
        self,
        dimensions: Any,
        reconstruction_result: Any,
        camera_info: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Exports full scan metadata and dimension report as JSON."""
        if filename is None:
            filename = f"scan_report_{self.get_timestamp_slug()}.json"

        filepath = self.output_dir / filename

        report = {
            "application": "Bhoomi Drishti - RealSense 3D Scanner",
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "name": camera_info.get("name", "Intel RealSense D435i"),
                "serial": camera_info.get("serial", "N/A"),
                "firmware": camera_info.get("firmware", "N/A"),
                "usb_type": camera_info.get("usb_type", "N/A"),
                "depth_scale_meters": camera_info.get("depth_scale_meters", 0.001),
                "resolution": camera_info.get("stream_resolution", "640x480"),
            },
            "scan_quality": {
                "duration_seconds": reconstruction_result.duration_seconds,
                "frames_captured": reconstruction_result.frame_count,
                "valid_depth_coverage_pct": round(reconstruction_result.valid_depth_pct, 2),
                "raw_point_count": reconstruction_result.raw_points_count,
                "accepted_point_count": reconstruction_result.accepted_points_count,
                "depth_range_meters": {
                    "min_z": round(reconstruction_result.depth_range_m[0], 4),
                    "max_z": round(reconstruction_result.depth_range_m[1], 4),
                },
                "confidence_level": dimensions.confidence_level,
                "confidence_score_pct": dimensions.confidence_score_pct,
                "quality_factors": dimensions.quality_factors,
            },
            "dimensions": {
                "length_cm": dimensions.length_cm,
                "width_cm": dimensions.width_cm,
                "height_extent_cm": dimensions.height_extent_cm,
                "length_m": dimensions.length_m,
                "width_m": dimensions.width_m,
                "height_extent_m": dimensions.height_extent_m,
                "surface_area_sq_cm": dimensions.surface_area_sq_cm,
                "is_height_relative_to_ground": dimensions.is_height_relative_to_ground,
                "height_description": dimensions.height_description,
                "inlier_ratio_pct": dimensions.inlier_ratio_pct,
            },
            "limitations": dimensions.limitations,
        }

        json_str = json.dumps(report, indent=2)
        filepath.write_text(json_str, encoding="utf-8")
        return filepath, json_str
