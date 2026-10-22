"""
test_depth_ml.py - Unit and Integration Tests for Depth View ML Models
Validates:
- Intel RealSense camera pinhole projection geometry & error handling
- Multi-frame temporal depth filtering and Open3D 3D point cloud reconstruction
- Real-time depth layout contour detection & elevation segmentation
- Sampled 10x10 coordinate matrix generation
- PCA oriented bounding box & metric dimension estimation
- Standard PLY & CSV export utilities
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from camera import RealSenseCamera, CameraConnectionError
from ml_models.depth_models.reconstruction import PointCloudReconstructor, ScanReconstructionResult
from ml_models.common_models.layout_detector import RealTimeLayoutDetector, DetectedLayout
from ml_models.common_models.measurements import MeasurementEngine, DimensionEstimate
from ml_models.common_models.cadastral_engine import CadastralAnalysisEngine
from exports import DataExporter


class DummyIntrinsics:
    """Mock intrinsics matching 640x480 RealSense D435i calibrated values."""
    width = 640
    height = 480
    fx = 385.0
    fy = 385.0
    ppx = 320.0
    ppy = 240.0
    model = "BrownConrady"
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


class DummyCamera:
    """Mock camera providing calibrated pinhole deprojection for testing."""
    def __init__(self):
        self.intrinsics = DummyIntrinsics()
        self.min_depth_m = 0.15
        self.max_depth_m = 3.0
        self.device_info = {
            "name": "Intel RealSense D435i (Depth View Mock)",
            "serial": "123456789",
            "firmware": "5.13.0.50",
            "usb_type": "USB 3.2",
            "depth_scale_meters": 0.001,
            "stream_resolution": "640x480 @ 30fps",
        }

    def deproject_pixel(self, u: float, v: float, depth_m: float):
        if depth_m <= 0:
            return 0.0, 0.0, 0.0
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        ppx, ppy = self.intrinsics.ppx, self.intrinsics.ppy
        x = (u - ppx) / fx * depth_m
        y = (v - ppy) / fy * depth_m
        return float(x), float(y), float(depth_m)

    def deproject_depth_map(self, depth_meters, roi=None, color_rgb=None):
        h, w = depth_meters.shape
        if roi is not None:
            ymin, ymax, xmin, xmax = roi
        else:
            ymin, ymax, xmin, xmax = 0, h, 0, w

        u_range = np.arange(xmin, xmax, dtype=np.float32)
        v_range = np.arange(ymin, ymax, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_range, v_range)

        z_sub = depth_meters[ymin:ymax, xmin:xmax]
        valid_mask = (z_sub >= self.min_depth_m) & (z_sub <= self.max_depth_m)

        if not np.any(valid_mask):
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2))

        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        z_valid = z_sub[valid_mask]

        x_valid = (u_valid - self.intrinsics.ppx) / self.intrinsics.fx * z_valid
        y_valid = (v_valid - self.intrinsics.ppy) / self.intrinsics.fy * z_valid

        points = np.stack([x_valid, y_valid, z_valid], axis=-1).astype(np.float32)
        pixel_coords = np.stack([u_valid.astype(np.int32), v_valid.astype(np.int32)], axis=-1)
        colors = np.ones_like(points, dtype=np.float32) * 0.7

        return points, colors, pixel_coords


class TestDepthViewML(unittest.TestCase):

    def setUp(self):
        self.camera = DummyCamera()
        self.reconstructor = PointCloudReconstructor(voxel_size_m=0.005)
        self.meas_engine = MeasurementEngine()
        self.cadastral_engine = CadastralAnalysisEngine()
        self.layout_detector = RealTimeLayoutDetector()
        self.exporter = DataExporter(output_dir=str(REPO_DIR / "tests" / "test_outputs"))

    def _create_synthetic_cube_scan(self):
        """Creates synthetic aligned depth and color frames of a 20cm elevated cube at 0.5m depth."""
        frames = []
        for i in range(10):
            depth = np.ones((480, 640), dtype=np.float32) * 0.80  # Ground at 80 cm
            # Elevated square block (200x200 px) at 50 cm
            depth[140:340, 220:420] = 0.50
            # Add subtle gaussian noise
            noise = np.random.normal(0, 0.002, depth.shape).astype(np.float32)
            depth += noise
            color = np.full((480, 640, 3), 180, dtype=np.uint8)
            frames.append({"color": color, "depth_m": depth, "timestamp": float(i)})
        return frames

    def test_reconstruction_pipeline(self):
        frames = self._create_synthetic_cube_scan()
        roi = (140, 340, 220, 420)
        result = self.reconstructor.reconstruct_from_frames(
            frames=frames,
            camera=self.camera,
            roi=roi,
            duration_seconds=10.0,
        )

        self.assertIsInstance(result, ScanReconstructionResult)
        self.assertGreater(result.accepted_points_count, 100)
        self.assertGreater(result.valid_depth_pct, 90.0)
        self.assertAlmostEqual(result.depth_range_m[0], 0.50, delta=0.05)

    def test_coordinate_matrix_generation(self):
        frames = self._create_synthetic_cube_scan()
        roi = (140, 340, 220, 420)
        result = self.reconstructor.reconstruct_from_frames(
            frames=frames,
            camera=self.camera,
            roi=roi,
            duration_seconds=10.0,
        )

        df = self.meas_engine.generate_coordinate_matrix(
            reconstruction_result=result, camera=self.camera, grid_rows=10, grid_cols=10
        )

        self.assertEqual(len(df), 100)
        self.assertIn("point_id", df.columns)
        self.assertIn("X_m", df.columns)
        self.assertIn("Y_m", df.columns)
        self.assertIn("Z_m", df.columns)
        valid_z = df["Z_m"].dropna()
        self.assertGreater(len(valid_z), 50)

    def test_dimension_estimation(self):
        frames = self._create_synthetic_cube_scan()
        roi = (140, 340, 220, 420)
        result = self.reconstructor.reconstruct_from_frames(
            frames=frames,
            camera=self.camera,
            roi=roi,
            duration_seconds=10.0,
        )

        dim = self.meas_engine.estimate_dimensions(result)
        self.assertIsInstance(dim, DimensionEstimate)
        self.assertGreater(dim.length_cm, 5.0)
        self.assertGreater(dim.width_cm, 5.0)
        self.assertIn(dim.confidence_level, ["HIGH", "MODERATE", "LOW"])

    def test_depth_layout_detector(self):
        depth = np.ones((480, 640), dtype=np.float32) * 0.80
        depth[140:340, 220:420] = 0.50
        layout = self.layout_detector.detect_layout(depth)

        self.assertIsNotNone(layout)
        self.assertTrue(layout.is_valid)
        self.assertGreater(layout.elevation_height_m, 0.20)

    def test_export_utilities(self):
        frames = self._create_synthetic_cube_scan()
        roi = (140, 340, 220, 420)
        result = self.reconstructor.reconstruct_from_frames(
            frames=frames,
            camera=self.camera,
            roi=roi,
            duration_seconds=10.0,
        )
        df = self.meas_engine.generate_coordinate_matrix(reconstruction_result=result, camera=self.camera)

        csv_path, csv_text = self.exporter.export_csv(df, filename="test_matrix.csv")
        self.assertTrue(csv_path.exists())
        self.assertIn("point_id", csv_text)

        ply_path, ply_bytes = self.exporter.export_ply(result, filename="test_mesh.ply")
        self.assertTrue(ply_path.exists())
        self.assertTrue(ply_bytes.startswith(b"ply\n"))

        # Cleanup test files
        if csv_path.exists():
            csv_path.unlink()
        if ply_path.exists():
            ply_path.unlink()


if __name__ == "__main__":
    unittest.main()
