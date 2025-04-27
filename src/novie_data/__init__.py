"""Module contains every novie data class definition."""

from .arm_coverage_data import SpiralArmCoverageData
from .corrugation_data import CorrugationData
from .corrugation_residuals_data import CorrugationResidualsData
from .grid_data import GridData
from .interface import NovieData
from .perturber_data import PerturberData
from .ridge_data import RidgeData
from .snail_data import SnailData
from .snapshot_data import SnapshotData
from .solar_circle_data import SolarCircleData
from .velocity_grid import VelocityGridData
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
    "SpiralArmCoverageData",
    "VelocityGridData",
    "WrinkleData",
    "WrinkleResidualsData",
]

__version__: str = "4.0.0"
