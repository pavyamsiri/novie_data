"""The animation data class for radial ridges in Vr and Vphi."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from numpy.typing import NDArray
from packaging.version import Version

from .serde.accessors import get_float_attr_from_hdf5, get_int_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class RidgeData:
    """The surface densities of a snapshot."""

    # Data products
    mass_density: NDArray[float32]
    number_density: NDArray[float32]

    # Metadata
    num_radial_bins: int
    num_velocity_bins: int
    min_radius: float
    max_radius: float
    min_velocity: float
    max_velocity: float

    DATA_FILE_TYPE: ClassVar[str] = "Ridge"
    VERSION: ClassVar[Version] = Version("1.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Validate shapes
        same_shape = self.mass_density.shape == self.number_density.shape and self.mass_density.shape == self.number_density.shape
        if not same_shape:
            msg = "The colorings differ in shape!"
            raise ValueError(msg)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize phase spiral data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        RidgeData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            num_radial_bins: int = get_int_attr_from_hdf5(file, "num_radial_bins")
            num_velocity_bins: int = get_int_attr_from_hdf5(file, "num_velocity_bins")
            min_radius: float = get_float_attr_from_hdf5(file, "min_radius")
            max_radius: float = get_float_attr_from_hdf5(file, "max_radius")
            min_velocity: float = get_float_attr_from_hdf5(file, "min_velocity")
            max_velocity: float = get_float_attr_from_hdf5(file, "max_velocity")

            # Arrays
            mass_density = read_dataset_from_hdf5_with_dtype(file, "mass_density", dtype=float32)
            number_density = read_dataset_from_hdf5_with_dtype(file, "number_density", dtype=float32)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            mass_density=mass_density,
            number_density=number_density,
            num_radial_bins=num_radial_bins,
            num_velocity_bins=num_velocity_bins,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            min_radius=min_radius,
            max_radius=max_radius,
        )

    def dump(self, path: Path) -> None:
        """Serialize phase spiral data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        cls = type(self)
        with Hdf5File(path, "w") as file:
            # General
            file.attrs["type"] = cls.DATA_FILE_TYPE
            file.attrs["version"] = str(cls.VERSION)
            file.attrs["num_radial_bins"] = self.num_radial_bins
            file.attrs["num_velocity_bins"] = self.num_velocity_bins
            file.attrs["min_radius"] = self.min_radius
            file.attrs["max_radius"] = self.max_radius
            file.attrs["min_velocity"] = self.min_velocity
            file.attrs["max_velocity"] = self.max_velocity

            file.create_dataset("mass_density", data=self.mass_density)
            file.create_dataset("number_density", data=self.number_density)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.mass_density.shape[2]

    def get_radial_limits(self) -> tuple[float, float]:
        """Return the radial limits.

        Returns
        -------
        limits : tuple[float, float]
            The radial limits.

        """
        return (self.min_radius, self.max_radius)

    def get_velocity_limits(self) -> tuple[float, float]:
        """Return the vertical velocity limits.

        Returns
        -------
        limits : tuple[float, float]
            The vertical velocity limits.

        """
        return (self.min_velocity, self.max_velocity)

    def get_2d_limits(self) -> tuple[float, float, float, float]:
        """Return the 2D limits of the surface density grid.

        Returns
        -------
        limits : tuple[float, float, float, float]
            The 2D limits.

        """
        return (self.min_radius, self.max_radius, self.min_velocity, self.max_velocity)

    def get_dummy_data(self) -> NDArray[float32]:
        """Return an array of ones with the same shape as the grid.

        Returns
        -------
        NDArray[float32]
            The array of ones with the same shape as the grid.

        """
        # NOTE: Transpose to return as row-major with the velocity being on the vertical axis.
        return np.zeros((self.num_velocity_bins, self.num_radial_bins), dtype=float32)
