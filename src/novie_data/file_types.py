"""Module containing the types of novie files."""

from enum import IntEnum, auto
from pathlib import Path

from h5py import File as Hdf5File

from novie_data.snapshot_data import SnapshotData
from novie_data.solar_circle_data import SolarCircleData

from .arm_coverage_data import SpiralArmCoverageData
from .arm_data import SpiralClusterResidualsData
from .cluster_data import SpiralClusterData
from .corrugation_data import CorrugationData
from .corrugation_residuals_data import CorrugationResidualsData
from .perturber_data import PerturberData
from .ridge_data import RidgeData
from .snail_data import SnailData
from .surface_density_data import SurfaceDensityData
from .wrinkle_data import WrinkleData
from .wrinkle_residuals_data import WrinkleResidualsData


class NovieFileType(IntEnum):
    """The novie file types.

    Variants
    --------
    ARM_RESIDUALS
        The spiral arm residuals.
    ARM_COVERAGE
        The arm coverage.
    CORRUGATION
        The corrugation data.
    CORRUGATION_RESIDUALS
        The corrugation residuals.
    SPIRAL_CLUSTER
        The spiral clusters.
    SNAIL
        The phase spiral data.
    SURFACE_DENSITY
        The surface density data.
    WRINKLE
        The wrinkle data.
    WRINKLE_RESIDUALS
        The wrinkle residuals.
    PERTURBER
        The perturber location.
    RIDGE
        The ridge/R-Vphi phase space.

    """

    ARM_RESIDUALS = auto()
    ARM_COVERAGE = auto()
    CORRUGATION = auto()
    CORRUGATION_RESIDUALS = auto()
    SPIRAL_CLUSTER = auto()
    SNAIL = auto()
    SURFACE_DENSITY = auto()
    WRINKLE = auto()
    WRINKLE_RESIDUALS = auto()
    PERTURBER = auto()
    RIDGE = auto()
    SNAPSHOT = auto()
    SOLAR_CIRCLE = auto()


def get_novie_file_type(input_path: Path) -> NovieFileType:
    """Determine the novie file type of the file at the given path.

    This assumes the file is a valid novie file.

    Parameters
    ----------
    input_path : Path
        The path to the file.

    Returns
    -------
    file_type : NovieFileType
        The novie file type of the file.

    """
    file_type_to_enum: dict[str, NovieFileType] = {
        CorrugationData.DATA_FILE_TYPE: NovieFileType.CORRUGATION,
        CorrugationResidualsData.DATA_FILE_TYPE: NovieFileType.CORRUGATION_RESIDUALS,
        SpiralClusterData.DATA_FILE_TYPE: NovieFileType.SPIRAL_CLUSTER,
        SnailData.DATA_FILE_TYPE: NovieFileType.SNAIL,
        SpiralClusterResidualsData.DATA_FILE_TYPE: NovieFileType.ARM_RESIDUALS,
        SurfaceDensityData.DATA_FILE_TYPE: NovieFileType.SURFACE_DENSITY,
        SpiralArmCoverageData.DATA_FILE_TYPE: NovieFileType.ARM_COVERAGE,
        WrinkleData.DATA_FILE_TYPE: NovieFileType.WRINKLE,
        WrinkleResidualsData.DATA_FILE_TYPE: NovieFileType.WRINKLE_RESIDUALS,
        PerturberData.DATA_FILE_TYPE: NovieFileType.PERTURBER,
        RidgeData.DATA_FILE_TYPE: NovieFileType.RIDGE,
        SnapshotData.DATA_FILE_TYPE: NovieFileType.SNAPSHOT,
        SolarCircleData.DATA_FILE_TYPE: NovieFileType.SOLAR_CIRCLE,
    }

    with Hdf5File(input_path, "r") as file:
        file_type_str: str = str(file.attrs["type"])
        file_type: NovieFileType | None = file_type_to_enum.get(file_type_str)
        if file_type is None:
            msg = f"Unsupported file type {file_type_str}"
            raise ValueError(msg)
        return file_type
