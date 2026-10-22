"""
run_depth_scan.py - CLI Launcher for Depth View 3D ML Pipeline
Executes calibrated 3D depth acquisition, temporal filtering,
layout boundary detection, 10x10 coordinate matrix generation,
and PLY/CSV export.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure local imports
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from camera import RealSenseCamera, CameraConnectionError
from ml_models.depth_models.reconstruction import PointCloudReconstructor
from ml_models.common_models.layout_detector import RealTimeLayoutDetector
from ml_models.common_models.measurements import MeasurementEngine
from ml_models.common_models.cadastral_engine import CadastralAnalysisEngine
from exports import DataExporter


def run_pipeline():
    print("=" * 60)
    print("Depth View — 3D Machine Learning & Reconstruction Pipeline")
    print("=" * 60)

    # 1. Initialize Camera
    cam = None
    use_synthetic = False
    try:
        print("[1/5] Initializing Intel RealSense D435i...")
        cam = RealSenseCamera(width=640, height=480, fps=30)
        info = cam.start()
        print(f"      Connected: {info.get('name')} (Serial: {info.get('serial')})")
    except Exception as e:
        print(f"      Physical RealSense not detected ({e}). Using synthetic calibrated benchmark feed.")
        use_synthetic = True

    # 2. Acquire Sequence
    print("\n[2/5] Capturing 3D depth frame sequence...")
    frames = []
    if not use_synthetic and cam is not None:
        try:
            frames = cam.capture_sequence(
                duration_seconds=3.0,
                target_fps=15,
                progress_callback=lambda p, el, cnt: print(f"      Progress: {p*100:.0f}% ({cnt} frames captured)", end="\r"),
            )
            print()
        finally:
            cam.stop()
    else:
        # Generate synthetic benchmark cube at 0.5m depth
        for i in range(15):
            depth = np.ones((480, 640), dtype=np.float32) * 0.85
            depth[140:340, 220:420] = 0.50 + np.random.normal(0, 0.002, (200, 200)).astype(np.float32)
            color = np.full((480, 640, 3), 190, dtype=np.uint8)
            frames.append({"color": color, "depth_m": depth, "timestamp": float(i)})
        print(f"      Generated {len(frames)} calibrated benchmark frames.")

    # 3. Detect Layout & Reconstruct Point Cloud
    print("\n[3/5] Executing Depth Machine Learning Models...")
    detector = RealTimeLayoutDetector()
    sample_depth = frames[-1]["depth_m"]
    layout = detector.detect_layout(sample_depth)

    roi = (140, 340, 220, 420)
    if layout and layout.is_valid:
        box = layout.oriented_box_2d
        ymin = max(0, int(np.min(box[:, 1])))
        ymax = min(480, int(np.max(box[:, 1])))
        xmin = max(0, int(np.min(box[:, 0])))
        xmax = min(640, int(np.max(box[:, 0])))
        roi = (ymin, ymax, xmin, xmax)
        print(f"      Detected Object Layout: {layout.box_width_px:.1f}x{layout.box_height_px:.1f} px, elevation: {layout.elevation_height_m*100:.1f} cm")

    reconstructor = PointCloudReconstructor(voxel_size_m=0.004)
    # Using dummy camera projection for synthetic or real camera
    from tests.test_depth_ml import DummyCamera
    proj_camera = cam if (cam and cam.intrinsics) else DummyCamera()

    recon_res = reconstructor.reconstruct_from_frames(
        frames=frames,
        camera=proj_camera,
        roi=roi,
        duration_seconds=3.0,
    )
    print(f"      Reconstruction: {recon_res.accepted_points_count:,} points accepted, valid depth: {recon_res.valid_depth_pct:.1f}%")

    # 4. Measure Dimensions & Matrix
    print("\n[4/5] Computing 3D Metrics & Spatial Matrix Grid...")
    meas_engine = MeasurementEngine()
    dim = meas_engine.estimate_dimensions(recon_res)
    print(f"      Dimensions: {dim.length_cm:.1f} x {dim.width_cm:.1f} cm (Confidence: {dim.confidence_level})")

    matrix_df = meas_engine.generate_coordinate_matrix(recon_res, proj_camera, grid_rows=10, grid_cols=10)
    valid_points = matrix_df["Z_m"].dropna().count()
    print(f"      Generated 10x10 Matrix: {valid_points}/100 valid spatial cells")

    # 5. Export
    print("\n[5/5] Exporting Data...")
    exporter = DataExporter(output_dir="outputs")
    csv_path, _ = exporter.export_csv(matrix_df)
    ply_path, _ = exporter.export_ply(recon_res)
    print(f"      PLY Point Cloud saved: {ply_path}")
    print(f"      CSV Coordinate Matrix saved: {csv_path}")
    print("\n[Done] Pipeline executed successfully.")


if __name__ == "__main__":
    run_pipeline()
