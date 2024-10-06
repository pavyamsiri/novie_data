"""Data representing the coverage of observed arms with spiral arm masks from simulations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from .serde.accessors import get_string_sequence_from_hdf5, read_dataset_from_hdf5
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData
from .solar_circle_data import SolarCircleData

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy import uint32
    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SpiralArmCoverageData:
    """Data class to store spiral arm coverage.

    Attributes
    ----------
    num_covered_arm_pixels : NDArray[uint32]
        The number of covered arm pixels.
    num_total_arm_pixels : NDArray[uint32]
        The total number of arm pixels.
    solar_circle_data : SolarCircleData
        The solar circle data.
    snapshot_data : SnapshotData
        The data describing each snapshot.

    """

    num_covered_arm_pixels: NDArray[uint32]
    num_total_arm_pixels: NDArray[uint32]
    covered_arm_normalised_densities: NDArray[float32]
    solar_circle_data: SolarCircleData
    snapshot_data: SnapshotData
    arm_names: Sequence[str]

    DATA_FILE_TYPE: ClassVar[str] = "SpiralArmCoverage"
    VERSION: ClassVar[Version] = Version("1.1.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the arrays the correct size
        num_neighbourhoods: int = self.num_covered_arm_pixels.shape[0]
        num_arms: int = self.num_covered_arm_pixels.shape[1]
        num_frames: int = self.num_covered_arm_pixels.shape[2]

        common_shape: tuple[int, int, int] = (num_neighbourhoods, num_arms, num_frames)
        if self.num_covered_arm_pixels.shape != common_shape:
            msg = f"Expected covered arm pixels' array shape to be {common_shape} but got {self.num_covered_arm_pixels.shape}"
            raise ValueError(msg)
        if self.num_total_arm_pixels.shape != common_shape:
            msg = f"Expected total arm pixels' array shape to be {common_shape} but got {self.num_total_arm_pixels.shape}"
            raise ValueError(msg)
        if self.covered_arm_normalised_densities.shape != common_shape:
            current_shape = self.covered_arm_normalised_densities.shape
            msg = f"Expected normalised densities' array shape to be {common_shape} but got {current_shape}"
            raise ValueError(msg)
        if num_frames != self.snapshot_data.num_frames:
            msg = "Expected the snapshot data to have the same number of frames as the other arrays."
            raise ValueError(msg)
        if len(self.arm_names) != num_arms:
            msg = f"Expected the number of arms to be {num_arms} but got {len(self.arm_names)}"
            raise ValueError(msg)

        self.arm_coverage: NDArray[float32] = np.copy(self.num_covered_arm_pixels).astype(float32)
        self.arm_coverage[self.num_total_arm_pixels > 0] /= self.num_total_arm_pixels[self.num_total_arm_pixels > 0]

        num_total_arm_pixels_sum = np.sum(self.num_total_arm_pixels, axis=1)
        self.average_total_arm_coverage: NDArray[float32] = np.sum(self.num_covered_arm_pixels, axis=1).astype(float32)
        self.average_total_arm_coverage[num_total_arm_pixels_sum > 0] /= num_total_arm_pixels_sum[num_total_arm_pixels_sum > 0]
        self.average_arm_coverage: NDArray[float32] = np.mean(self.arm_coverage, axis=1)

        self.average_covered_arm_normalised_densities: NDArray[float32] = np.mean(self.covered_arm_normalised_densities, axis=1)

        self.num_arms: int = num_arms
        self.num_neighbourhoods: int = num_neighbourhoods

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize phase spiral data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        SpiralClusterData
            The deserialized data.

        """
        with Hdf5File(path, "r") as in_file:
            verify_file_type_from_hdf5(in_file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(in_file, cls.VERSION)

            num_covered_arm_pixels: NDArray[uint32]
            num_covered_arm_pixels = read_dataset_from_hdf5(in_file, "num_covered_arm_pixels")
            num_total_arm_pixels: NDArray[uint32]
            num_total_arm_pixels = read_dataset_from_hdf5(in_file, "num_total_arm_pixels")
            covered_arm_normalised_densities: NDArray[float32]
            covered_arm_normalised_densities = read_dataset_from_hdf5(in_file, "covered_arm_normalised_densities")
            arm_names: Sequence[str] = get_string_sequence_from_hdf5(in_file, "arm_names")
            snapshot_data = SnapshotData.load_from(in_file)
            solar_circle_data = SolarCircleData.load_from(in_file)
        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]",
            cls.__name__,
            path.absolute(),
        )
        return cls(
            num_covered_arm_pixels=num_covered_arm_pixels,
            num_total_arm_pixels=num_total_arm_pixels,
            solar_circle_data=solar_circle_data,
            snapshot_data=snapshot_data,
            covered_arm_normalised_densities=covered_arm_normalised_densities,
            arm_names=arm_names,
        )

    def dump(self, path: Path) -> None:
        """Serialize phase spiral data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        cls = type(self)
        with Hdf5File(path, "w") as out_file:
            # General
            out_file.attrs["type"] = cls.DATA_FILE_TYPE
            out_file.attrs["version"] = str(cls.VERSION)

            out_file.create_dataset("num_covered_arm_pixels", data=self.num_covered_arm_pixels)
            out_file.create_dataset("num_total_arm_pixels", data=self.num_total_arm_pixels)
            out_file.create_dataset("covered_arm_normalised_densities", data=self.covered_arm_normalised_densities)
            out_file.create_dataset("arm_names", data=self.arm_names)
            self.snapshot_data.dump_into(out_file)
            self.solar_circle_data.dump_into(out_file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    def get_starting_angles_deg(self) -> NDArray[float32]:
        """Return the angle of the start locations in degrees.

        Returns
        -------
        starting_angles : NDArray[float32]
            The angle of the start locations in degrees.

        """
        return np.linspace(0, 360, self.num_neighbourhoods, endpoint=False)
