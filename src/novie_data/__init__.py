"""Module contains every novie data class definition."""

from .arm_coverage_data import SpiralArmCoverageData
from .arm_data import SpiralClusterResidualsData
from .cluster_data import SpiralClusterData
from .corrugation_data import CorrugationData
from .corrugation_residuals_data import CorrugationResidualsData
from .file_types import NovieFileType, get_novie_file_type
from .neighbourhood_data import SphericalNeighbourhoodData
from .perturber_data import PerturberData
from .ridge_data import RidgeData
from .snail_data import SnailData
from .snapshot_data import SnapshotData
from .solar_circle_data import SolarCircleData
from .surface_density_data import SurfaceDensityData
from .wrinkle_data import WrinkleData
from .wrinkle_residuals_data import WrinkleResidualsData

__all__ = [
    "SurfaceDensityData",
    "CorrugationData",
    "SnailData",
    "CorrugationResidualsData",
    "SpiralClusterData",
    "SnapshotData",
    "SolarCircleData",
    "get_novie_file_type",
    "NovieFileType",
    "SpiralClusterResidualsData",
    "SpiralArmCoverageData",
    "WrinkleData",
    "WrinkleResidualsData",
    "SphericalNeighbourhoodData",
    "PerturberData",
    "RidgeData",
]

__version__: str = "1.0.1"
