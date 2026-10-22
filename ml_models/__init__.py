"""
Depth View — 3D Machine Learning & Computer Vision Models Package
Specialized pipeline for 3D stereoscopic and active IR depth scanning:
1. depth_models: 3D point cloud generation, temporal depth filtering, Open3D voxel downsampling, statistical outlier removal, normal estimation.
2. common_models: Real-time depth layout detection, cadastral matrix dissection, and metric dimension estimation.
"""

from .common_models import (
    RealTimeLayoutDetector,
    DetectedLayout,
    CadastralAnalysisEngine,
    CadastralMetrics,
    MeasurementEngine,
    DimensionEstimate,
)
from .depth_models import (
    PointCloudReconstructor,
    ScanReconstructionResult,
)

__all__ = [
    # Subpackages
    "common_models",
    "depth_models",
    # Depth ML Models
    "PointCloudReconstructor",
    "ScanReconstructionResult",
    # Common Geometry & Layout ML Models
    "RealTimeLayoutDetector",
    "DetectedLayout",
    "CadastralAnalysisEngine",
    "CadastralMetrics",
    "MeasurementEngine",
    "DimensionEstimate",
]
