"""Data representing the coverage of observed arms with spiral arm masks from simulations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32, uint32
from packaging.version import Version

from .serde.accessors import get_str_attr_from_hdf5, get_string_sequence_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class SpiralArmCoverageData:
    """Data class to store spiral arm coverage.

    Attributes
    ----------
    num_covered_arm_pixels : NDArray[uint32]
        The number of covered arm pixels.
    num_total_arm_pixels : NDArray[uint32]
        The total number of arm pixels.
    name : str
        The name of the dataset.

    """

    DATA_FILE_TYPE: ClassVar[str] = "SpiralArmCoverage"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __init__(
        self,
        *,
        name: str,
        num_covered_arm_pixels: NDArray[uint32],
        num_total_arm_pixels: NDArray[uint32],
        covered_arm_normalised_densities: NDArray[float32],
        arm_names: Sequence[str],
    ) -> None:
        """Perform post-initialisation verification.

        Parameters
        ----------
        name : str
            The name of the dataset.
        num_covered_arm_pixels : NDArray[uint32]
            The number of covered arm pixels.
        num_total_arm_pixels : NDArray[uint32]
            The total number of arm pixels.
        covered_arm_normalised_densities : NDArray[float32]
            The mean normalised density within the covered region.
        arm_names : Sequence[str]
            The name of each arm.

        """
        self.name: str = name
        self.num_covered_arm_pixels: NDArray[uint32] = num_covered_arm_pixels
        self.num_total_arm_pixels: NDArray[uint32] = num_total_arm_pixels
        self.covered_arm_normalised_densities: NDArray[float32] = covered_arm_normalised_densities
        self.arm_names: Sequence[str] = arm_names

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

    def __eq__(self, other: object, /) -> bool:
        """Compare for equality.

        Parameters
        ----------
        other : object
            The object to compare to.

        Returns
        -------
        bool
            `True` if the other object is equal to this object, `False` otherwise.

        Notes
        -----
        Equality means all fields are equal.

        """
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.name == other.name
        equality &= len(self.arm_names) == len(other.arm_names)
        equality &= all(this_name == other_name for this_name, other_name in zip(self.arm_names, other.arm_names, strict=False))
        equality &= np.all(self.num_covered_arm_pixels == other.num_covered_arm_pixels)
        equality &= np.all(self.num_total_arm_pixels == other.num_total_arm_pixels)
        return bool(equality)

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

            name: str = get_str_attr_from_hdf5(in_file, "name")
            num_covered_arm_pixels = read_dataset_from_hdf5_with_dtype(in_file, "num_covered_arm_pixels", dtype=uint32)
            num_total_arm_pixels = read_dataset_from_hdf5_with_dtype(in_file, "num_total_arm_pixels", dtype=uint32)
            covered_arm_normalised_densities = read_dataset_from_hdf5_with_dtype(
                in_file, "covered_arm_normalised_densities", dtype=float32
            )
            arm_names: Sequence[str] = get_string_sequence_from_hdf5(in_file, "arm_names")
        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]",
            cls.__name__,
            path.absolute(),
        )
        return cls(
            num_covered_arm_pixels=num_covered_arm_pixels,
            num_total_arm_pixels=num_total_arm_pixels,
            covered_arm_normalised_densities=covered_arm_normalised_densities,
            arm_names=arm_names,
            name=name,
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
            out_file.attrs["name"] = self.name

            out_file.create_dataset("num_covered_arm_pixels", data=self.num_covered_arm_pixels)
            out_file.create_dataset("num_total_arm_pixels", data=self.num_total_arm_pixels)
            out_file.create_dataset("covered_arm_normalised_densities", data=self.covered_arm_normalised_densities)
            out_file.create_dataset("arm_names", data=self.arm_names)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    def get_starting_angles_deg(self) -> NDArray[float32]:
        """Return the angle of the start locations in degrees.

        Returns
        -------
        starting_angles : NDArray[float32]
            The angle of the start locations in degrees.

        """
        return np.linspace(0, 360, self.num_neighbourhoods, endpoint=False)
