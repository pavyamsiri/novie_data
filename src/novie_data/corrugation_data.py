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

from .serde.accessors import get_float_attr_from_hdf5, get_int_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData
from .solar_circle_data import SolarCircleData

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
        if self.num_bins < 1:
            msg = f"Expected the number of bins to be positive but got {self.num_bins}."
            raise ValueError(msg)
        if self.min_radius < 0 or self.max_radius <= 0:
            msg = f"Expected the limits to be non-negative but got [{self.min_radius}, {self.max_radius}] kpc."
            raise ValueError(msg)
        if self.min_radius >= self.max_radius:
            msg = f"Expected the minimum radius {self.min_radius} kpc to be less than {self.max_radius} kpc."
            raise ValueError(msg)

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

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
        """Serialize snapshot data from file.

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
        if self.num_bins < 1:
            msg = f"Expected the number of bins to be positive but got {self.num_bins}."
            raise ValueError(msg)
        if self.max_height <= 0:
            msg = f"Expected the limits to be non-negative but got max height {self.max_height} kpc."
            raise ValueError(msg)
        if self.cutoff_frequency < 0:
            msg = f"Expected the cutoff frequency to be non-negative but got max_height {self.cutoff_frequency} kpc^-1."
            raise ValueError(msg)

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

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
        """Serialize snapshot data from file.

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
        if self.num_wedges < 1:
            msg = f"Expected the number of wedges to be positive but got {self.num_wedges}."
            raise ValueError(msg)
        if self.inner_radius < 0 or self.outer_radius <= 0:
            msg = f"Expected the limits to be non-negative but got [{self.inner_radius}, {self.outer_radius}] kpc."
            raise ValueError(msg)
        if self.inner_radius >= self.outer_radius:
            msg = (
                f"Expected the inner radius {self.inner_radius} kpc to be smaller than the outer radius {self.outer_radius} kpc."
            )
            raise ValueError(msg)
        if self.min_longitude_deg >= self.max_longitude_deg:
            msg = f"Expected longitudes to be monontonic increasing but {self.min_longitude_deg} < {self.max_longitude_deg}"
            raise ValueError(msg)
        if self.min_longitude_deg < 0 or self.max_longitude_deg <= 0:
            msg = f"Expected longitudes to be non-negative but got [{self.min_longitude_deg}, {self.max_longitude_deg}]"
            raise ValueError(msg)
        self.width: float = self.outer_radius - self.inner_radius
        self.longitude_width_deg: float = self.max_longitude_deg - self.min_longitude_deg

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

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
        """Serialize snapshot data from file.

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


@dataclass
class CorrugationData:
    """The side on view of a snapshot."""

    projection_rz: NDArray[float32]
    radii: NDArray[float32]
    mean_height: NDArray[float32]
    mean_height_error: NDArray[float32]

    snapshot_data: SnapshotData
    radial_bins: RadialBinningData
    height_bins: HeightBinningData
    wedge_data: WedgeData
    solar_circle_data: SolarCircleData

    DATA_FILE_TYPE: ClassVar[str] = "Corrugation"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Validate projection
        projection_shape = self.projection_rz.shape
        if len(projection_shape) != 4:
            msg = f"Expected the projection to be 4D but it is instead {len(projection_shape)}D."
            raise ValueError(msg)
        expected_projection_shape = (
            self.height_bins.num_bins,
            self.radial_bins.num_bins,
            self.snapshot_data.num_frames,
            self.wedge_data.num_wedges,
        )
        expected_projection_shape_delete = (
            self.height_bins.num_bins,
            self.radial_bins.num_bins,
            self.snapshot_data.num_frames,
        )
        if projection_shape not in (expected_projection_shape, expected_projection_shape_delete):
            msg = f"Expected the projection to have the shape {expected_projection_shape} but got {projection_shape}."
            raise ValueError(msg)

        if len(self.radii.shape) != 1:
            msg = f"Expected the radial bin centres to be 1D but it is instead {len(self.radii.shape)}D."
            raise ValueError(msg)
        if self.radii.shape[0] != self.radial_bins.num_bins:
            msg = f"Expected the number of radial bin centres to be {self.radial_bins.num_bins} but got {self.radii.shape[0]}."
            raise ValueError(msg)

        if self.mean_height.shape != self.mean_height_error.shape:
            mean_height_shape = self.mean_height.shape
            error_shape = self.mean_height_error.shape
            msg = f"Expected the shape of mean height {mean_height_shape} to be same as its errors {error_shape}"
            raise ValueError(msg)

        mean_height_shape = self.mean_height.shape
        if len(mean_height_shape) != 3:
            msg = f"Expected the mean height and its error array to be 3D but it is instead {len(mean_height_shape)}D."
            raise ValueError(msg)

        expected_mean_height_shape = (self.radial_bins.num_bins, self.snapshot_data.num_frames, self.wedge_data.num_wedges)
        if mean_height_shape != expected_mean_height_shape:
            msg = f"Expected the mean height array to have the shape {expected_mean_height_shape} but got {mean_height_shape}."
            raise ValueError(msg)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize surface density data from file.

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

            # Projections
            projection_rz = read_dataset_from_hdf5_with_dtype(file, "projection_rz", dtype=float32)
            # Mean height
            radii = read_dataset_from_hdf5_with_dtype(file, "radii", dtype=float32)
            mean_height = read_dataset_from_hdf5_with_dtype(file, "mean_height", dtype=float32)
            mean_height_error = read_dataset_from_hdf5_with_dtype(file, "mean_height_error", dtype=float32)
            snapshot_data = SnapshotData.load_from(file)
            radial_bins = RadialBinningData.load_from(file)
            height_bins = HeightBinningData.load_from(file)
            wedge_data = WedgeData.load_from(file)
            solar_circle_data = SolarCircleData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            snapshot_data=snapshot_data,
            projection_rz=projection_rz,
            radii=radii,
            mean_height=mean_height,
            mean_height_error=mean_height_error,
            radial_bins=radial_bins,
            height_bins=height_bins,
            wedge_data=wedge_data,
            solar_circle_data=solar_circle_data,
        )

    def dump(self, path: Path) -> None:
        """Serialize surface density data to disk.

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

            file.create_dataset("projection_rz", data=self.projection_rz)
            file.create_dataset("radii", data=self.radii)
            file.create_dataset("mean_height", data=self.mean_height)
            file.create_dataset("mean_height_error", data=self.mean_height_error)
            self.snapshot_data.dump_into(file)
            self.radial_bins.dump_into(file)
            self.height_bins.dump_into(file)
            self.wedge_data.dump_into(file)
            self.solar_circle_data.dump_into(file)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.snapshot_data.num_frames

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
