"""
Depth Models & 3D Point Cloud Algorithms
Specialized 3D depth models for Intel RealSense D435i stereoscopic and active IR depth feeds:
- PointCloudReconstructor: Temporal median depth filtering, voxel downsampling, statistical outlier removal
- ScanReconstructionResult: 3D point cloud, normal vectors, and metric survey container
"""

from .reconstruction import (
    PointCloudReconstructor,
    ScanReconstructionResult,
)

__all__ = [
    "PointCloudReconstructor",
    "ScanReconstructionResult",
]
