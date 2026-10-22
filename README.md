# Depth View — 3D Computer Vision & Machine Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-3D%20Point%20Cloud-green.svg)](http://www.open3d.org/)
[![Intel RealSense](https://img.shields.io/badge/Intel%20RealSense-D435i%20Calibrated-orange.svg)](https://www.intelrealsense.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Depth View** is a specialized 3D Computer Vision and Machine Learning repository designed for high-precision stereoscopic and active IR depth scanning, point cloud surface reconstruction, real-time geometric layout detection, and metric spatial coordinate analysis.

This repository focuses exclusively on **Depth View ML and 3D geometric pipelines** using the Intel RealSense D435i (and compatible calibrated depth sensors).

---

## Architecture & Subsystems

```
depth_view/
├── ml_models/
│   ├── __init__.py                    # Unified export of all Depth ML models
│   │
│   ├── depth_models/                  # 3D Depth & Point Cloud Generation Models
│   │   ├── __init__.py
│   │   └── reconstruction.py          # Temporal median depth filtering, Open3D voxel
│   │                                  # downsampler, statistical outlier removal (SOR),
│   │                                  # and 3D surface normal estimation
│   │
│   └── common_models/                 # Geometric Machine Learning & Cadastral Analysis
│       ├── __init__.py
│       ├── layout_detector.py         # Real-time elevation gradient segmentation,
│       │                              # foreground contour extraction, oriented 2D bounding box
│       ├── measurements.py            # Sampled 10x10 coordinate matrix, PCA 3D principal
│       │                              # axes dimension estimator, RANSAC ground plane fitting
│       └── cadastral_engine.py        # 3D polygon metric deprojection, Shoelace area,
│                                      # boundary encroachment classification
│
├── camera.py                          # Calibrated Intel RealSense D435i hardware driver,
│                                      # USB 3.0/2.1 adaptive streaming, pinhole deprojection
├── exports.py                         # PLY point cloud and CSV coordinate matrix exporter
├── run_depth_scan.py                  # Standalone CLI scan & ML pipeline launcher
├── tests/
│   ├── __init__.py
│   └── test_depth_ml.py               # Unit & integration test suite
├── requirements.txt                   # Dependency specifications
└── README.md
```

---

## Core Machine Learning & Vision Models

### 1. `PointCloudReconstructor` (`ml_models/depth_models/reconstruction.py`)
- **Temporal Median Depth Filtering**: Multi-frame depth stack aggregation rejecting transient flicker, specular dropouts, and flying-pixel noise.
- **Voxel Grid Downsampling**: Uniform spatial decimation producing clean, metrically uniform point clouds.
- **Statistical Outlier Removal (SOR)**: Removes floating sensor noise based on mean distance to $k$-nearest neighbors.
- **Surface Normal Estimation**: Computes 3D surface normal vectors for geometric feature extraction.

### 2. `RealTimeLayoutDetector` (`ml_models/common_models/layout_detector.py`)
- **Elevation Gradient Segmentation**: Fast, deterministic thresholding of foreground object surfaces relative to the ground/table base plane ($\Delta Z > \text{min\_elevation}$).
- **Rotated Minimum-Area Bounding Box**: Calculates in-plane rotation angle, metric centroid, and corner vertices.
- **Polygon Simplification**: Uses `approxPolyDP` to fit tight geometric polygons around detected objects.

### 3. `MeasurementEngine` (`ml_models/common_models/measurements.py`)
- **Sampled 10×10 Coordinate Matrix**: Generates a standardized spatial matrix (`point_id, pixel_u, pixel_v, X_m, Y_m, Z_m`) using robust median cell statistics.
- **PCA Principal Axes Fitting**: Estimates true metric length, width, and height by projecting 3D point clusters onto principal eigenvectors.
- **RANSAC Ground Plane Detection**: Detects surface reference planes to ensure elevation measurements reflect true physical height.

### 4. `CadastralAnalysisEngine` (`ml_models/common_models/cadastral_engine.py`)
- **3D Metric Polygon Deprojection**: Maps 2D image contours to real-world metric 3D space.
- **Shoelace 3D Area & Perimeter**: Computes true ground area and perimeter with boundary encroachment checks.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Kartikkumar251/depth_view.git
cd depth_view

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### Run the Depth Scan Pipeline
Execute the complete end-to-end depth pipeline (automatically connects to Intel RealSense D435i if present, or uses calibrated benchmark frames):
```bash
python run_depth_scan.py
```

### Run the ML Test Suite
Validate all depth models, reconstruction pipelines, and exports:
```bash
python -m unittest tests/test_depth_ml.py
```

---

## Programmatic Usage

```python
from camera import RealSenseCamera
from ml_models.depth_models import PointCloudReconstructor
from ml_models.common_models import MeasurementEngine, RealTimeLayoutDetector
from exports import DataExporter

# 1. Start RealSense D435i
with RealSenseCamera(width=640, height=480, fps=30) as cam:
    frames = cam.capture_sequence(duration_seconds=3.0)

# 2. Reconstruct filtered 3D point cloud
reconstructor = PointCloudReconstructor(voxel_size_m=0.004)
recon_result = reconstructor.reconstruct_from_frames(frames, camera=cam)

# 3. Estimate physical dimensions and 10x10 coordinate matrix
engine = MeasurementEngine()
dimensions = engine.estimate_dimensions(recon_result)
matrix_df = engine.generate_coordinate_matrix(recon_result, camera=cam)

print(f"Object: {dimensions.length_cm:.1f} cm x {dimensions.width_cm:.1f} cm")

# 4. Export
exporter = DataExporter(output_dir="outputs")
exporter.export_ply(recon_result, "point_cloud.ply")
exporter.export_csv(matrix_df, "coordinate_matrix.csv")
```

---

## License
MIT License.
