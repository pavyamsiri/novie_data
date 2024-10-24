"""The animation data class for radial ridges in Vr and Vphi."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from numpy.typing import NDArray
from packaging.version import Version

from novie_data.errors import (
    verify_arrays_have_correct_length,
    verify_arrays_have_same_shape,
    verify_value_is_nonnegative,
    verify_value_is_positive,
)

from .serde.accessors import (
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class RidgeData:
    """The surface densities of a snapshot."""

    DATA_FILE_TYPE: ClassVar[str] = "Ridge"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __init__(
        self,
        *,
        name: str,
        mass_density: NDArray[float32],
        number_density: NDArray[float32],
        num_radial_bins: int,
        num_velocity_bins: int,
        min_radius: float,
        max_radius: float,
        min_velocity: float,
        max_velocity: float,
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the dataset.
        mass_density : Array3D[f32]
            The mass density projection in r-Vr phase space.
        number_density : Array3D[f32]
            The number density projection in r-Vr phase space.
        num_radial_bins : int
            The number of radial bins.
        num_velocity_bins : int
            The number of velocity bins.
        min_radius : float
            The minimum radius in units of kpc.
        max_radius : float
            The maximum radius in units of kpc.
        min_velocity : float
            The minimum velocity in units of km/s.
        max_velocity : float
            The maximum velocity in units of km/s.

        """
        self.name: str = name
        self.mass_density: NDArray[float32] = mass_density
        self.number_density: NDArray[float32] = number_density
        self.num_radial_bins: int = num_radial_bins
        self.num_velocity_bins: int = num_velocity_bins
        self.min_radius: float = min_radius
        self.max_radius: float = max_radius
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity

        verify_value_is_positive(self.num_radial_bins, msg="Expected the number of radial bins to be positive!")
        verify_value_is_positive(self.num_velocity_bins, msg="Expected the number of velocity bins to be positive!")
        verify_value_is_nonnegative(self.min_radius, msg="Expected the minimum radius to be non-negative!")
        if self.min_radius >= self.max_radius:
            msg = "Expected the maximum radius to be strictly greater than the minimum radius!"
            raise ValueError(msg)
        if self.min_velocity >= self.max_velocity:
            msg = "Expected the maximum velocity to be strictly greater than the minimum velocity!"
            raise ValueError(msg)

        verify_arrays_have_same_shape(
            [self.mass_density, self.number_density],
            msg="Expected the mass density and number density array to have the same shape!",
        )
        verify_arrays_have_correct_length(
            [(self.mass_density, 0)],
            self.num_velocity_bins,
            msg=f"Expected the mass/number density array to have {self.num_velocity_bins} rows.",
        )
        verify_arrays_have_correct_length(
            [(self.mass_density, 1)],
            self.num_radial_bins,
            msg=f"Expected the mass/number density array to have {self.num_radial_bins} columns.",
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
        equality &= self.num_radial_bins == other.num_radial_bins
        equality &= self.num_velocity_bins == other.num_velocity_bins
        equality &= self.min_radius == other.min_radius
        equality &= self.max_radius == other.max_radius
        equality &= self.min_velocity == other.min_velocity
        equality &= self.max_velocity == other.max_velocity
        equality &= np.all(self.mass_density == other.mass_density)
        equality &= np.all(self.number_density == other.number_density)
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
            name: str = get_str_attr_from_hdf5(file, "name")

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
            file.attrs["name"] = self.name

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
