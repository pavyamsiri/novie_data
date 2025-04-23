"""Module contains every novie data class definition."""

from .arm_coverage_data import SpiralArmCoverageData
from .corrugation_data import CorrugationData
from .corrugation_residuals_data import CorrugationResidualsData
from .interface import NovieData
from .neighbourhood_data import SphericalNeighbourhoodData
from .perturber_data import PerturberData
from .ridge_data import RidgeData
from .snail_data import SnailData
from .snapshot_data import SnapshotData
from .solar_circle_data import SolarCircleData
from .surface_density_data import GridData
from .wrinkle_data import WrinkleData
from .wrinkle_residuals_data import WrinkleResidualsData

__all__ = [
    "CorrugationData",
    "CorrugationResidualsData",
    "GridData",
    "NovieData",
    "PerturberData",
    "RidgeData",
    "SnailData",
    "SnapshotData",
    "SolarCircleData",
    "SphericalNeighbourhoodData",
    "SpiralArmCoverageData",
    "WrinkleData",
    "WrinkleResidualsData",
]

__version__: str = "4.0.0"
