"""The animation data class for phase spirals or "snail shells"."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Self, assert_never

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from numpy.typing import NDArray
from packaging.version import Version
from scipy.ndimage import gaussian_filter

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

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class SnailPlotColoring(Enum):
    """The phase spiral coloring.

    Variants
    --------
    DENSITY
        The number density.
    OVERDENSITY
        The overdensity.
    AZIMUTHAL_VELOCITY
        The mean azimuthal velocity.
    RADIAL_VELOCITY
        The mean radial velocity.

    """

    DENSITY = "density"
    OVERDENSITY = "overdensity"
    AZIMUTHAL_VELOCITY = "vphi"
    RADIAL_VELOCITY = "vr"

    def __str__(self) -> str:
        """Return the string representation.

        Returns
        -------
        str
            The string representation.

        """
        return self.value

    def get_label(self) -> str:
        """Return the label for the coloring in matplotlib Mathtext.

        Returns
        -------
        str
            The label.

        """
        match self:
            # Surface density
            case SnailPlotColoring.DENSITY:
                return r"$\rho$"
            # Overdensity
            case SnailPlotColoring.OVERDENSITY:
                return r"$\delta \rho$"
            # Azimuthal velocity
            case SnailPlotColoring.AZIMUTHAL_VELOCITY:
                return r"$ \langle v_{\phi} \rangle \;(\mathrm{km} \, \mathrm{s}^{-1})$"
            # Radial velocity
            case SnailPlotColoring.RADIAL_VELOCITY:
                return r"$ \langle v_{R} \rangle \;(\mathrm{km} \, \mathrm{s}^{-1})$"
            case _:
                assert_never(self)

    @staticmethod
    def from_str(value: str) -> SnailPlotColoring | None:
        """Parse a string into a `SnailPlotColoring`.

        Parameters
        ----------
        value : str
            The string to parse.

        Returns
        -------
        SnailPlotColoring | None
            The parsed coloring or `None` if the string was not valid.

        """
        match value:
            case "density":
                return SnailPlotColoring.DENSITY
            case "vphi":
                return SnailPlotColoring.AZIMUTHAL_VELOCITY
            case "vr":
                return SnailPlotColoring.RADIAL_VELOCITY
            case _:
                return None


@dataclass
class SnailData:
    """The snail data of a snapshot."""

    # Data products
    surface_density: NDArray[float32]
    azimuthal_velocity: NDArray[float32]
    radial_velocity: NDArray[float32]
    neighbourhood_data: SphericalNeighbourhoodData

    # Metadata
    num_height_bins: int
    num_velocity_bins: int
    max_height: float
    max_velocity: float

    name: str

    DATA_FILE_TYPE: ClassVar[str] = "Snail"
    VERSION: ClassVar[Version] = Version("4.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Validate shapes
        same_shape = (
            self.surface_density.shape == self.azimuthal_velocity.shape
            and self.surface_density.shape == self.radial_velocity.shape
        )
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
        SnailData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            num_height_bins: int = get_int_attr_from_hdf5(file, "num_height_bins")
            num_velocity_bins: int = get_int_attr_from_hdf5(file, "num_velocity_bins")
            max_height: float = get_float_attr_from_hdf5(file, "max_height")
            max_velocity: float = get_float_attr_from_hdf5(file, "max_velocity")
            name: str = get_str_attr_from_hdf5(file, "name")

            # Arrays
            surface_density = read_dataset_from_hdf5_with_dtype(file, "surface_density", dtype=float32)
            azimuthal_velocity = read_dataset_from_hdf5_with_dtype(file, "azimuthal_velocity", dtype=float32)
            radial_velocity = read_dataset_from_hdf5_with_dtype(file, "radial_velocity", dtype=float32)

            neighbourhood_data = SphericalNeighbourhoodData.load_from(file)
        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            surface_density=surface_density,
            azimuthal_velocity=azimuthal_velocity,
            radial_velocity=radial_velocity,
            num_height_bins=num_height_bins,
            num_velocity_bins=num_velocity_bins,
            max_height=max_height,
            max_velocity=max_velocity,
            neighbourhood_data=neighbourhood_data,
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
            file.attrs["num_height_bins"] = self.num_height_bins
            file.attrs["num_velocity_bins"] = self.num_velocity_bins
            file.attrs["max_height"] = self.max_height
            file.attrs["max_velocity"] = self.max_velocity
            file.attrs["name"] = self.name

            file.create_dataset("surface_density", data=self.surface_density)
            file.create_dataset("azimuthal_velocity", data=self.azimuthal_velocity)
            file.create_dataset("radial_velocity", data=self.radial_velocity)
            self.neighbourhood_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.surface_density.shape[2]

    def get_height_limits(self) -> tuple[float, float]:
        """Return the height limits.

        Returns
        -------
        limits : tuple[float, float]
            The height limits.

        """
        return (-self.max_height, self.max_height)

    def get_velocity_limits(self) -> tuple[float, float]:
        """Return the vertical velocity limits.

        Returns
        -------
        limits : tuple[float, float]
            The vertical velocity limits.

        """
        return (-self.max_velocity, self.max_velocity)

    def get_2d_limits(self) -> tuple[float, float, float, float]:
        """Return the 2D limits of the surface density grid.

        Returns
        -------
        limits : tuple[float, float, float, float]
            The 2D limits.

        """
        return (-self.max_height, self.max_height, -self.max_velocity, self.max_velocity)

    def get_min_value(self, coloring: SnailPlotColoring) -> float:
        """Return the minimum (non-zero) density across all projections.

        Parameters
        ----------
        coloring : SnailPlotColoring
            The coloring.

        Returns
        -------
        float
            The minimum density across all projections.

        """
        match coloring:
            case SnailPlotColoring.DENSITY:
                array = self.surface_density
            case SnailPlotColoring.OVERDENSITY:
                array = self.overdensity
            case SnailPlotColoring.AZIMUTHAL_VELOCITY:
                array = self.azimuthal_velocity
            case SnailPlotColoring.RADIAL_VELOCITY:
                array = self.radial_velocity
            case _:
                assert_never(coloring)
        return float(np.nanmin(array[array != 0]))

    def get_max_value(self, coloring: SnailPlotColoring) -> float:
        """Return the maximum density across all projections.

        Parameters
        ----------
        coloring : SnailPlotColoring
            The coloring.

        Returns
        -------
        float
            The maximum density across all projections.

        """
        match coloring:
            case SnailPlotColoring.DENSITY:
                array = self.surface_density
            case SnailPlotColoring.OVERDENSITY:
                array = self.overdensity
            case SnailPlotColoring.AZIMUTHAL_VELOCITY:
                array = self.azimuthal_velocity
            case SnailPlotColoring.RADIAL_VELOCITY:
                array = self.radial_velocity
            case _:
                assert_never(coloring)
        return float(np.nanmax(array[array != 0]))

    @property
    def overdensity(self) -> NDArray[float32]:
        """NDArray[float32]: The overdensity map blurred using a width of 4 bins."""
        return self.get_overdensity(4)

    def get_overdensity(self, blur_width: float) -> NDArray[float32]:
        """Return the overdensity map given a blurring width in units of bins.

        Parameters
        ----------
        blur_width : float
            The blur width for both axes in number of bins.

        Returns
        -------
        overdensity : NDArray[float32]
            The overdensity.

        """
        mean_density: NDArray[float32] = gaussian_filter(
            self.surface_density, sigma=(blur_width, blur_width), axes=(0, 1)
        ).astype(float32)
        norm_density = np.copy(self.surface_density)
        norm_density[norm_density != 0] /= mean_density[norm_density != 0]
        return (norm_density - 1).astype(float32)

    def get_dummy_data(self) -> NDArray[float32]:
        """Return an array of ones with the same shape as the grid.

        Returns
        -------
        NDArray[float32]
            The array of ones with the same shape as the grid.

        """
        # NOTE: Transpose to return as row-major with the velocity being on the vertical axis.
        return np.zeros((self.num_height_bins, self.num_velocity_bins), dtype=float32).T
