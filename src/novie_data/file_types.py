"""Module containing the types of novie files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from h5py import File as Hdf5File

from novie_data.snapshot_data import SnapshotData
from novie_data.solar_circle_data import SolarCircleData

from .arm_coverage_data import SpiralArmCoverageData
from .corrugation_data import CorrugationData
from .corrugation_residuals_data import CorrugationResidualsData
from .perturber_data import PerturberData
from .ridge_data import RidgeData
from .snail_data import SnailData
from .surface_density_data import SurfaceDensityData
from .wrinkle_data import WrinkleData
from .wrinkle_residuals_data import WrinkleResidualsData

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .interface import NovieData


def get_novie_type(input_path: Path) -> type[NovieData]:
    """Determine the novie type.

    This assumes the file is a valid novie file.

    Parameters
    ----------
    input_path : Path
        The path to the file.

    Returns
    -------
    file_type : type[NovieData]
        The novie file type.

    """
    file_types: Sequence[type[NovieData]] = [
        CorrugationData,
        CorrugationResidualsData,
        SnailData,
        SurfaceDensityData,
        SpiralArmCoverageData,
        WrinkleData,
        WrinkleResidualsData,
        PerturberData,
        RidgeData,
        SnapshotData,
        SolarCircleData,
    ]

    with Hdf5File(input_path, "r") as file:
        file_type_str: str = str(file.attrs["type"])
        for file_type in file_types:
            if file_type_str == file_type.DATA_FILE_TYPE:
                return file_type
    msg = f"Unsupported file type {file_type_str}"
    raise ValueError(msg)
