"""The animation data class for Lz vs mean Vr."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from .neighbourhood_data import SphericalNeighbourhoodData
from .serde.accessors import get_float_attr_from_hdf5, get_int_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .solar_circle_data import SolarCircleData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class WrinkleData:
    """The surface densities of a snapshot in all three cardinal projections.

    Attributes
    ----------
    angular_momentum : NDArray[float]
        The central angular momementum values of each bin in kpc km/s.
    mean_radial_velocity : NDArray[float]
        The mean radial velocity of each bin in km/s.
    mean_radial_velocity_error : NDArray[float]
        The Poisson error of the mean radial velocity of each bin in km/s.
        This is given by error = std / sqrt(N).
    min_lz : float
        The minimum angular momentum in kpc km/s.
    max_lz : float
        The maximum angular momentum in kpc km/s.
    num_bins : int
        The number of bins in angular momentum Lz.
    solar_circle_data : SolarCircleData
        The solar circle used.
    neighbourhood_data : SphericalNeighbourhoodData
        The neighbourhood configuration.

    """

    angular_momentum: NDArray[float32]
    mean_radial_velocity: NDArray[float32]
    mean_radial_velocity_error: NDArray[float32]
    min_lz: float
    max_lz: float
    num_bins: int
    solar_circle_data: SolarCircleData
    neighbourhood_data: SphericalNeighbourhoodData

    DATA_FILE_TYPE: ClassVar[str] = "Wrinkle"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the arrays are the same size
        same_shape = self.mean_radial_velocity.shape == self.mean_radial_velocity_error.shape
        if not same_shape:
            msg = "The projections differ in shape!"
            msg += f"Vr = {self.mean_radial_velocity.shape}, err(Vr) = {self.mean_radial_velocity_error.shape}"
            raise ValueError(msg)
        # Verify that the array has the expected dimension
        shape = self.mean_radial_velocity.shape
        if len(shape) != 3:
            msg = "Expected array to be a 2D array of `(num_bins, num_frames)`"
            raise ValueError(msg)
        # Verify that the array has expected shape
        if shape[0] != self.num_bins:
            msg = f"Expected the number of bins to be {self.num_bins} but got {shape[0]}"
            raise ValueError(msg)
        # Verify that the array has the expected number of neighbourhoods
        if shape[2] != self.neighbourhood_data.num_spheres:
            msg = f"Expected the number of spheres to be {self.neighbourhood_data.num_spheres} but got {shape[2]}"
            raise ValueError(msg)

        # Verify angular momentum array
        if len(self.angular_momentum.shape) != 1:
            msg = f"Expected array to be a 1D array of `(num_bins)` but got {self.angular_momentum.shape}"
            raise ValueError(msg)
        if self.angular_momentum.shape[0] != self.num_bins:
            msg = f"Expected the number of bins to be {self.num_bins} but got {self.angular_momentum.shape[0]}"
            raise ValueError(msg)

        # Useful values
        self.num_spheres: int = self.neighbourhood_data.num_spheres
        self.num_frames: int = shape[1]

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        SurfaceDensityData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            num_bins: int = get_int_attr_from_hdf5(file, "num_bins")
            min_lz: float = get_float_attr_from_hdf5(file, "min_lz")
            max_lz: float = get_float_attr_from_hdf5(file, "max_lz")

            # Projections
            angular_momentum = read_dataset_from_hdf5_with_dtype(file, "angular_momentum", dtype=float32)
            mean_radial_velocity = read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity", dtype=float32)
            mean_radial_velocity_error = read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity_error", dtype=float32)

            solar_circle_data = SolarCircleData.load_from(file)
            neighbourhood_data = SphericalNeighbourhoodData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            angular_momentum=angular_momentum,
            mean_radial_velocity=mean_radial_velocity,
            mean_radial_velocity_error=mean_radial_velocity_error,
            min_lz=min_lz,
            max_lz=max_lz,
            num_bins=num_bins,
            solar_circle_data=solar_circle_data,
            neighbourhood_data=neighbourhood_data,
        )

    def dump(self, path: Path) -> None:
        """Serialize data to disk.

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
            file.attrs["num_bins"] = self.num_bins
            file.attrs["min_lz"] = self.min_lz
            file.attrs["max_lz"] = self.max_lz

            file.create_dataset("angular_momentum", data=self.angular_momentum)
            file.create_dataset("mean_radial_velocity", data=self.mean_radial_velocity)
            file.create_dataset("mean_radial_velocity_error", data=self.mean_radial_velocity_error)
            self.solar_circle_data.dump_into(file)
            self.neighbourhood_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions
    def get_limits(self) -> tuple[float, float]:
        """Return the 1D limits of the wrinkle.

        Returns
        -------
        limits : tuple[float, float]
            The 1D limits.

        """
        return (self.min_lz, self.max_lz)
