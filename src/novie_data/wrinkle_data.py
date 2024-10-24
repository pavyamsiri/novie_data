"""The animation data class for Lz vs mean Vr."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import Array1D, Array3D, verify_array_is_1d, verify_array_is_3d
from novie_data.errors import (
    verify_arrays_have_correct_length,
    verify_arrays_have_same_shape,
    verify_value_is_nonnegative,
    verify_value_is_positive,
)

from .neighbourhood_data import SphericalNeighbourhoodData
from .serde.accessors import (
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path


_Array1D_f32: TypeAlias = Array1D[np.float32]
_Array3D_f32: TypeAlias = Array3D[np.float32]


log: logging.Logger = logging.getLogger(__name__)


class WrinkleData:
    """The surface densities of a snapshot in all three cardinal projections.

    Attributes
    ----------
    name : str
        The name of the dataset.
    angular_momentum : Array1D[f32]
        The central angular momementum values of each bin in kpc km/s.
    mean_radial_velocity : Array3D[f32]
        The mean radial velocity of each bin in km/s.
    mean_radial_velocity_error : Array3D[f32]
        The Poisson error of the mean radial velocity of each bin in km/s.
        This is given by error = std / sqrt(N).
    min_lz : float
        The minimum angular momentum in kpc km/s.
    max_lz : float
        The maximum angular momentum in kpc km/s.
    num_bins : int
        The number of bins in angular momentum Lz.
    neighbourhood_data : SphericalNeighbourhoodData
        The neighbourhood configuration.
    distance_error : float
        The error on the LOS distance (1 sigma) as a percentage of the distance.

    """

    DATA_FILE_TYPE: ClassVar[str] = "Wrinkle"
    VERSION: ClassVar[Version] = Version("3.0.0")

    def __init__(
        self,
        *,
        name: str,
        angular_momentum: _Array1D_f32,
        mean_radial_velocity: _Array3D_f32,
        mean_radial_velocity_error: _Array3D_f32,
        min_lz: float,
        max_lz: float,
        num_bins: int,
        neighbourhood_data: SphericalNeighbourhoodData,
        distance_error: float,
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the dataset.
        angular_momentum : Array1D[f32]
            The central angular momementum values of each bin in kpc km/s.
        mean_radial_velocity : Array3D[f32]
            The mean radial velocity of each bin in km/s.
        mean_radial_velocity_error : Array3D[f32]
            The Poisson error of the mean radial velocity of each bin in km/s.
            This is given by error = std / sqrt(N).
        min_lz : float
            The minimum angular momentum in kpc km/s.
        max_lz : float
            The maximum angular momentum in kpc km/s.
        num_bins : int
            The number of bins in angular momentum Lz.
        neighbourhood_data : SphericalNeighbourhoodData
            The neighbourhood configuration.
        distance_error : float
            The error on the LOS distance (1 sigma) as a percentage of the distance.

        """
        self.name: str = name
        self.angular_momentum: _Array1D_f32 = angular_momentum
        self.mean_radial_velocity: _Array3D_f32 = mean_radial_velocity
        self.mean_radial_velocity_error: _Array3D_f32 = mean_radial_velocity_error
        self.min_lz: float = min_lz
        self.max_lz: float = max_lz
        self.num_bins: int = num_bins
        self.neighbourhood_data: SphericalNeighbourhoodData = neighbourhood_data
        self.distance_error: float = distance_error

        verify_value_is_positive(self.num_bins, msg="Expected the number of bins to be positive!")
        verify_value_is_nonnegative(self.min_lz, msg="Expected the minimum angular momentum to be non-negative!")
        if self.min_lz > self.max_lz:
            msg = "Expected the maximum angular momentum to be strictly greater than the minimum angular momentum!"
            raise ValueError(msg)
        verify_value_is_nonnegative(self.distance_error, msg="Expected the distance error to be non-negative!")

        # Verify that the arrays are the same size
        verify_arrays_have_same_shape(
            [self.mean_radial_velocity, self.mean_radial_velocity_error],
            msg="Expected the mean radial velocity arrays to have the same shape!",
        )
        verify_arrays_have_correct_length(
            [(self.mean_radial_velocity, 0), (self.angular_momentum, 0)],
            num_bins,
            msg=f"Expected the mean radial velocity arrays to have {num_bins} rows!",
        )
        verify_arrays_have_correct_length(
            [(self.mean_radial_velocity, 2)],
            neighbourhood_data.num_spheres,
            msg=f"Expected the mean radial velocity arrays's 3rd axis to have {neighbourhood_data.num_spheres} cells!",
        )

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
        equality &= self.min_lz == other.min_lz
        equality &= self.max_lz == other.max_lz
        equality &= self.num_bins == other.num_bins
        equality &= self.neighbourhood_data == other.neighbourhood_data
        equality &= self.distance_error == other.distance_error
        equality &= np.all(self.angular_momentum == other.angular_momentum)
        equality &= np.all(self.mean_radial_velocity == other.mean_radial_velocity)
        equality &= np.all(self.mean_radial_velocity_error == other.mean_radial_velocity_error)
        return bool(equality)

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
            distance_error: float = get_float_attr_from_hdf5(file, "distance_error")
            name: str = get_str_attr_from_hdf5(file, "name")

            # Projections
            angular_momentum = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "angular_momentum", dtype=np.float32))
            mean_radial_velocity = verify_array_is_3d(
                read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity", dtype=np.float32)
            )
            mean_radial_velocity_error = verify_array_is_3d(
                read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity_error", dtype=np.float32)
            )

            neighbourhood_data = SphericalNeighbourhoodData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            angular_momentum=angular_momentum,
            mean_radial_velocity=mean_radial_velocity,
            mean_radial_velocity_error=mean_radial_velocity_error,
            min_lz=min_lz,
            max_lz=max_lz,
            num_bins=num_bins,
            neighbourhood_data=neighbourhood_data,
            distance_error=distance_error,
            name=name,
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
            file.attrs["distance_error"] = self.distance_error
            file.attrs["name"] = self.name

            file.create_dataset("angular_momentum", data=self.angular_momentum)
            file.create_dataset("mean_radial_velocity", data=self.mean_radial_velocity)
            file.create_dataset("mean_radial_velocity_error", data=self.mean_radial_velocity_error)
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
