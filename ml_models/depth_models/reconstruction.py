"""
reconstruction.py - 3D Surface Reconstruction & Temporal Depth Filtering
Handles multi-frame depth aggregation, noise rejection, Open3D point cloud generation,
voxel downsampling, and statistical outlier removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class ScanReconstructionResult:
    """Holds the 3D point cloud, aggregated maps, and metadata from a scan."""
    pcd: Optional[Any]  # Open3D PointCloud
    points: np.ndarray  # (N, 3) float32 in meters
    colors: np.ndarray  # (N, 3) float32 in range [0, 1]
    normals: Optional[np.ndarray]  # (N, 3) float32
    pixel_coords: np.ndarray  # (N, 2) int32 [u, v]
    aggregated_depth: np.ndarray  # (H, W) float32 in meters
    representative_color: np.ndarray  # (H, W, 3) uint8
    roi: Tuple[int, int, int, int]  # (ymin, ymax, xmin, xmax)
    valid_depth_pct: float  # Percentage of valid depth pixels in ROI
    total_roi_pixels: int
    raw_points_count: int
    accepted_points_count: int
    depth_range_m: Tuple[float, float]
    frame_count: int
    duration_seconds: float
    metadata: Dict[str, Any]


class PointCloudReconstructor:
    """
    Robust 3D Point Cloud Generator for Intel RealSense D435i.
    - Multi-frame temporal median filtering (rejects depth noise & flicker)
    - Open3D voxel downsampling and statistical outlier removal
    - Normal estimation
    """

    def __init__(
        self,
        voxel_size_m: float = 0.004,  # 4 mm resolution
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ):
        self.voxel_size_m = voxel_size_m
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def aggregate_frames(
        self,
        frames: List[Dict[str, Any]],
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes temporal median depth across captured frames to reject noise and flicker.
        
        Args:
            frames: List of frame dicts with 'depth_m' and 'color'.
            roi: Optional (ymin, ymax, xmin, xmax) bounding box.
            
        Returns:
            median_depth (np.ndarray): HxW float32 metric depth
            rep_color (np.ndarray): HxWx3 uint8 RGB image
            valid_pct (float): valid depth percentage within the ROI
        """
        if not frames:
            raise ValueError("No frames provided for temporal aggregation.")

        h, w = frames[0]["depth_m"].shape

        # Stack depth maps into 3D array (num_frames, H, W)
        depth_stack = np.array([f["depth_m"] for f in frames], dtype=np.float32)

        # Replace non-positive depth with NaN for nanmedian calculation
        depth_masked = np.where(depth_stack > 0, depth_stack, np.nan)

        # Suppress all-NaN slice warnings from NumPy when pixels have no valid depth in any frame
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_depth = np.nanmedian(depth_masked, axis=0)

        # Replace remaining NaNs with 0.0
        median_depth = np.nan_to_num(median_depth, nan=0.0).astype(np.float32)

        # Representative color: pick the middle frame
        mid_idx = len(frames) // 2
        rep_color = frames[mid_idx]["color"].copy()

        # Compute valid depth coverage percentage in ROI
        if roi is not None:
            ymin, ymax, xmin, xmax = roi
            roi_depth = median_depth[ymin:ymax, xmin:xmax]
            total_pixels = roi_depth.size
            valid_pixels = np.count_nonzero(roi_depth > 0)
        else:
            total_pixels = median_depth.size
            valid_pixels = np.count_nonzero(median_depth > 0)

        valid_pct = (valid_pixels / max(total_pixels, 1)) * 100.0

        return median_depth, rep_color, valid_pct

    def reconstruct_from_frames(
        self,
        frames: List[Dict[str, Any]],
        camera: Any,
        roi: Optional[Tuple[int, int, int, int]] = None,
        duration_seconds: float = 10.0,
    ) -> ScanReconstructionResult:
        """
        Executes end-to-end 3D reconstruction from a multi-frame sequence.
        
        Args:
            frames: Captured sequence of aligned RGB-D frames.
            camera: RealSenseCamera instance (for intrinsics & deprojection).
            roi: (ymin, ymax, xmin, xmax) pixel bounds of target object.
            duration_seconds: Duration of capture.
            
        Returns:
            ScanReconstructionResult object.
        """
        if not frames:
            raise ValueError("Cannot reconstruct from empty frame sequence.")

        h, w = frames[0]["depth_m"].shape
        if roi is None:
            roi = (0, h, 0, w)
        ymin, ymax, xmin, xmax = roi

        # 1. Temporal aggregation
        agg_depth, rep_color, valid_depth_pct = self.aggregate_frames(frames, roi=roi)

        # 2. Vectorized deprojection for selected ROI
        raw_points, raw_colors, pixel_coords = camera.deproject_depth_map(
            depth_meters=agg_depth,
            roi=roi,
            color_rgb=rep_color,
        )

        total_roi_pixels = (ymax - ymin) * (xmax - xmin)
        raw_count = len(raw_points)

        if raw_count == 0:
            return ScanReconstructionResult(
                pcd=None,
                points=np.empty((0, 3), dtype=np.float32),
                colors=np.empty((0, 3), dtype=np.float32),
                normals=None,
                pixel_coords=np.empty((0, 2), dtype=np.int32),
                aggregated_depth=agg_depth,
                representative_color=rep_color,
                roi=roi,
                valid_depth_pct=0.0,
                total_roi_pixels=total_roi_pixels,
                raw_points_count=0,
                accepted_points_count=0,
                depth_range_m=(0.0, 0.0),
                frame_count=len(frames),
                duration_seconds=duration_seconds,
                metadata={"status": "No valid depth data in selected ROI"},
            )

        # 3. Open3D Point Cloud Processing
        pcd = None
        normals = None
        accepted_points = raw_points
        accepted_colors = raw_colors

        if o3d is not None and raw_count >= 10:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(raw_colors.astype(np.float64))

            # Voxel downsampling (removes redundant points and regularizes density)
            if self.voxel_size_m > 0:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size_m)

            # Statistical outlier removal
            if len(pcd.points) > self.nb_neighbors:
                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
                )
                pcd = cl

            # Surface normal estimation
            if len(pcd.points) >= 10:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=max(self.voxel_size_m * 4, 0.02), max_nn=30
                    )
                )
                pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
                normals = np.asarray(pcd.normals, dtype=np.float32)

            accepted_points = np.asarray(pcd.points, dtype=np.float32)
            accepted_colors = np.asarray(pcd.colors, dtype=np.float32)

        z_vals = accepted_points[:, 2] if len(accepted_points) > 0 else np.array([0.0])
        depth_range_m = (float(np.min(z_vals)), float(np.max(z_vals)))

        metadata = {
            "voxel_size_m": self.voxel_size_m,
            "nb_neighbors": self.nb_neighbors,
            "std_ratio": self.std_ratio,
            "camera_resolution": f"{w}x{h}",
            "roi_box": {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax},
        }

        return ScanReconstructionResult(
            pcd=pcd,
            points=accepted_points,
            colors=accepted_colors,
            normals=normals,
            pixel_coords=pixel_coords,
            aggregated_depth=agg_depth,
            representative_color=rep_color,
            roi=roi,
            valid_depth_pct=valid_depth_pct,
            total_roi_pixels=total_roi_pixels,
            raw_points_count=raw_count,
            accepted_points_count=len(accepted_points),
            depth_range_m=depth_range_m,
            frame_count=len(frames),
            duration_seconds=duration_seconds,
            metadata=metadata,
        )
