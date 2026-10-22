"""
Common Models (Shared Between FPV and Depth Pipelines)
Algorithms and geometric models shared across both FPV video feeds and RealSense 3D depth feeds:
- RealTimeLayoutDetector / DetectedLayout: Dual-mode footprint boundary segmentation
  (Uses depth elevation maps when available, and color saliency / Otsu / edge filters for monocular FPV)
- CadastralAnalysisEngine / CadastralMetrics: Metric 3D area, perimeter, oriented bounding box,
  10x10 spatial matrix grid dissection, and property line setback encroachment detection
- MeasurementEngine / DimensionEstimate: Spatial PCA dimensions, coordinate matrix, ground plane RANSAC
"""

from .layout_detector import (
    RealTimeLayoutDetector,
    DetectedLayout,
)
from .cadastral_engine import (
    CadastralAnalysisEngine,
    CadastralMetrics,
)
from .measurements import (
    MeasurementEngine,
    DimensionEstimate,
)

__all__ = [
    "RealTimeLayoutDetector",
    "DetectedLayout",
    "CadastralAnalysisEngine",
    "CadastralMetrics",
    "MeasurementEngine",
    "DimensionEstimate",
]
