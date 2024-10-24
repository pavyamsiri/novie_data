"""The animation data class for side on view projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from novie_data.errors import verify_arrays_have_correct_length, verify_value_is_nonnegative, verify_value_is_positive

from .serde.accessors import (
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class RadialBinningData:
    """The binning configuration along the radial axis.

    Attributes
    ----------
    num_bins : int
        The number of radial bins.
    min_radius : float
        The minimum radius in kpc.
    max_radius : float
        The maximum radius in kpc.

    """

    num_bins: int
    min_radius: float
    max_radius: float

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        verify_value_is_positive(self.num_bins, msg="Expected the number of bins to be positive!")
        verify_value_is_nonnegative(self.min_radius, msg="Expected the minimum radius to be non-negative!")
        if self.min_radius >= self.max_radius:
            msg = f"Expected the minimum radius {self.min_radius} kpc to be less than {self.max_radius} kpc."
            raise ValueError(msg)

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["num_radial_bins"] = self.num_bins
        out_file.attrs["min_radius"] = self.min_radius
        out_file.attrs["max_radius"] = self.max_radius
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        num_bins: int = get_int_attr_from_hdf5(in_file, "num_radial_bins")
        min_radius: float = get_float_attr_from_hdf5(in_file, "min_radius")
        max_radius: float = get_float_attr_from_hdf5(in_file, "max_radius")

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            num_bins=num_bins,
            min_radius=min_radius,
            max_radius=max_radius,
        )


@dataclass
class HeightBinningData:
    """The binning configuration along the vertical axis.

    Attributes
    ----------
    num_bins : int
        The number of height bins.
    max_height : float
        The maximum absolute height in kpc.
    cutoff_frequency : float
        The cutoff frequency of the OLPF in kpc**-1.

    """

    num_bins: int
    max_height: float
    cutoff_frequency: float

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        verify_value_is_positive(self.num_bins, msg="Expected the number of bins to be positive!")
        verify_value_is_positive(self.max_height, msg="Expected the maximum absolute height to be positive!")
        verify_value_is_nonnegative(self.cutoff_frequency, msg="Expected the cutoff frequency to be non-negative!")

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["num_height_bins"] = self.num_bins
        out_file.attrs["max_height"] = self.max_height
        out_file.attrs["cutoff_frequency"] = self.cutoff_frequency
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        num_bins: int = get_int_attr_from_hdf5(in_file, "num_height_bins")
        max_height: float = get_float_attr_from_hdf5(in_file, "max_height")
        cutoff_frequency: float = get_float_attr_from_hdf5(in_file, "cutoff_frequency")

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            num_bins=num_bins,
            max_height=max_height,
            cutoff_frequency=cutoff_frequency,
        )


@dataclass
class WedgeData:
    """The configuration of the wedges around the solar circle.

    Attributes
    ----------
    num_wedges : int
        The number of wedges.
    inner_radius : float
        The inner radius of the wedge in kpc.
    outer_radius : float
        The outer radius of the wedge in kpc.
    min_longitude_deg : float
        The minimum galactic longitude in degrees.
    max_longitude_deg : float
        The maximum galactic longitude in degrees.

    Notes
    -----
    All values are with respect to the solar frame.

    """

    num_wedges: int
    inner_radius: float
    outer_radius: float
    min_longitude_deg: float
    max_longitude_deg: float

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        verify_value_is_positive(self.num_wedges, msg="Expected the number of wedges to be positive!")
        verify_value_is_nonnegative(self.inner_radius, msg="Expected the inner radius to be non-negative!")
        verify_value_is_nonnegative(self.min_longitude_deg, msg="Expected the minimum longitude in degrees to be non-negative!")
        if self.inner_radius >= self.outer_radius:
            msg = (
                f"Expected the inner radius {self.inner_radius} kpc to be smaller than the outer radius {self.outer_radius} kpc."
            )
            raise ValueError(msg)
        if self.min_longitude_deg >= self.max_longitude_deg:
            msg = f"Expected longitudes to be monontonic increasing but {self.min_longitude_deg} < {self.max_longitude_deg}"
            raise ValueError(msg)
        self.width: float = self.outer_radius - self.inner_radius
        self.longitude_width_deg: float = self.max_longitude_deg - self.min_longitude_deg

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["num_filters"] = self.num_wedges
        out_file.attrs["inner_radius"] = self.inner_radius
        out_file.attrs["outer_radius"] = self.outer_radius
        out_file.attrs["min_longitude_deg"] = self.min_longitude_deg
        out_file.attrs["max_longitude_deg"] = self.max_longitude_deg
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        num_wedges: int = get_int_attr_from_hdf5(in_file, "num_filters")
        inner_radius: float = get_float_attr_from_hdf5(in_file, "inner_radius")
        outer_radius: float = get_float_attr_from_hdf5(in_file, "outer_radius")
        min_longitude_deg: float = get_float_attr_from_hdf5(in_file, "min_longitude_deg")
        max_longitude_deg: float = get_float_attr_from_hdf5(in_file, "max_longitude_deg")

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            num_wedges=num_wedges,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            min_longitude_deg=min_longitude_deg,
            max_longitude_deg=max_longitude_deg,
        )


class CorrugationData:
    """The side on view of a snapshot."""

    DATA_FILE_TYPE: ClassVar[str] = "Corrugation"
    VERSION: ClassVar[Version] = Version("3.0.0")

    def __init__(
        self,
        *,
        name: str,
        projection_rz: NDArray[float32],
        radii: NDArray[float32],
        mean_height: NDArray[float32],
        mean_height_error: NDArray[float32],
        radial_bins: RadialBinningData,
        height_bins: HeightBinningData,
        wedge_data: WedgeData,
        distance_error: float,
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the dataset.
        projection_rz : Array4D[f32]
            The GC radius and GC height phase space projection for each neighbourhood and frame.
        radii : Array1D[f32]
            The central radius value for each radial bin in units of kpc.
        mean_height : Array3D[f32]
            The mean height for each radial bin in units of kpc.
        mean_height_error : Array3D[f32]
            The error in the mean height for each radial bin in units of kpc.
        radial_bins : RadialBinningData
            The radial binning data.
        height_bins : HeightBinningData
            The height binning data.
        wedge_data : WedgeData
            The wedge data.
        distance_error : float
            The error in LOS distance as a percentage.

        """
        self.name: str = name
        self.projection_rz: NDArray[float32] = projection_rz
        self.radii: NDArray[float32] = radii
        self.mean_height: NDArray[float32] = mean_height
        self.mean_height_error: NDArray[float32] = mean_height_error
        self.radial_bins: RadialBinningData = radial_bins
        self.height_bins: HeightBinningData = height_bins
        self.wedge_data: WedgeData = wedge_data
        self.distance_error: float = distance_error

        # Validate projection
        verify_arrays_have_correct_length(
            [(self.projection_rz, 0)],
            height_bins.num_bins,
            msg=f"Expected the projection's 1st axis to have {height_bins.num_bins} cells.",
        )
        verify_arrays_have_correct_length(
            [(self.projection_rz, 1)],
            radial_bins.num_bins,
            msg=f"Expected the projection's 2nd axis to have {radial_bins.num_bins} cells.",
        )
        verify_arrays_have_correct_length(
            [(self.radii, 0), (self.mean_height, 0), (self.mean_height_error, 0)],
            radial_bins.num_bins,
            msg=f"Expected the mean height arrays and the radii array to have {radial_bins.num_bins} rows.",
        )
        verify_arrays_have_correct_length(
            [(self.projection_rz, 3)],
            wedge_data.num_wedges,
            msg=f"Expected the projection's 4th axis to have {wedge_data.num_wedges} cells.",
        )
        verify_arrays_have_correct_length(
            [(self.mean_height, 2), (self.mean_height_error, 2)],
            wedge_data.num_wedges,
            msg=f"Expected the mean height array's 3rd axis to have {wedge_data.num_wedges} cells.",
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
        equality &= self.radial_bins == other.radial_bins
        equality &= self.height_bins == other.height_bins
        equality &= self.wedge_data == other.wedge_data
        equality &= self.distance_error == other.distance_error
        equality &= np.all(self.projection_rz == other.projection_rz)
        equality &= np.all(self.radii == other.radii)
        equality &= np.all(self.mean_height == other.mean_height)
        equality &= np.all(self.mean_height_error == other.mean_height_error)
        return bool(equality)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize data from disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        CorrugationData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            name: str = get_str_attr_from_hdf5(file, "name")
            distance_error: float = get_float_attr_from_hdf5(file, "distance_error")

            # Projections
            projection_rz = read_dataset_from_hdf5_with_dtype(file, "projection_rz", dtype=float32)
            # Mean height
            radii = read_dataset_from_hdf5_with_dtype(file, "radii", dtype=float32)
            mean_height = read_dataset_from_hdf5_with_dtype(file, "mean_height", dtype=float32)
            mean_height_error = read_dataset_from_hdf5_with_dtype(file, "mean_height_error", dtype=float32)
            radial_bins = RadialBinningData.load_from(file)
            height_bins = HeightBinningData.load_from(file)
            wedge_data = WedgeData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            projection_rz=projection_rz,
            radii=radii,
            mean_height=mean_height,
            mean_height_error=mean_height_error,
            radial_bins=radial_bins,
            height_bins=height_bins,
            wedge_data=wedge_data,
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
            file.attrs["distance_error"] = self.distance_error
            file.attrs["name"] = self.name

            file.create_dataset("projection_rz", data=self.projection_rz)
            file.create_dataset("radii", data=self.radii)
            file.create_dataset("mean_height", data=self.mean_height)
            file.create_dataset("mean_height_error", data=self.mean_height_error)
            self.radial_bins.dump_into(file)
            self.height_bins.dump_into(file)
            self.wedge_data.dump_into(file)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.projection_rz.shape[2]

    def get_radial_limits(self) -> tuple[float, float]:
        """Return the radial limits.

        Returns
        -------
        limits : tuple[float, float]
            The radial limits.

        """
        return (self.radial_bins.min_radius, self.radial_bins.max_radius)

    def get_height_limits(self) -> tuple[float, float]:
        """Return the height limits.

        Returns
        -------
        limits : tuple[float, float]
            The height limits.

        """
        return (-self.height_bins.max_height, self.height_bins.max_height)

    def get_2d_limits(self) -> tuple[float, float, float, float]:
        """Return the 2D limits of the surface density grid.

        Returns
        -------
        limits : tuple[float, float, float, float]
            The 2D limits.

        """
        return (
            self.radial_bins.min_radius,
            self.radial_bins.max_radius,
            -self.height_bins.max_height,
            self.height_bins.max_height,
        )

    def get_min_density(self) -> float:
        """Return the minimum (non-zero) density across all projections.

        Returns
        -------
        float
            The minimum density across all projections.

        """
        return float(np.nanmin(self.projection_rz[self.projection_rz > 0]))

    def get_max_density(self) -> float:
        """Return the maximum density across all projections.

        Returns
        -------
        float
            The maximum density across all projections.

        """
        return float(np.nanmax(self.projection_rz[self.projection_rz > 0]))

    def get_dummy_data(self) -> NDArray[float32]:
        """Return an array of ones with the same shape as the grid.

        Returns
        -------
        NDArray[float32]
            The array of ones with the same shape as the grid.

        """
        # NOTE: Transpose to return as row-major, with the height being on the vertical axis.
        return np.ones((self.radial_bins.num_bins, self.height_bins.num_bins), dtype=float32).T
