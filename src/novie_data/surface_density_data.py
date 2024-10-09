"""The animation data class for surface density projections in Cartesian space."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from .serde.accessors import get_float_attr_from_hdf5, get_int_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class ExponentialDiscProfileData:
    """The exponential profile of a galactic disc.

    The profile is of the form Sigma = A*exp(-R/Rs).

    Attributes
    ----------
    scale_mass : float
        The scale mass of the disc in Msol.
    scale_length : float
        The scale length of the disc in kpc.

    """

    scale_mass: float
    scale_length: float

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize exponential disc parameters to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["disc_scale_mass"] = self.scale_mass
        out_file.attrs["disc_scale_length"] = self.scale_length
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize exponential disc parameters from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        scale_mass: float = get_float_attr_from_hdf5(in_file, "disc_scale_mass")
        scale_length: float = get_float_attr_from_hdf5(in_file, "disc_scale_length")

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            scale_mass=scale_mass,
            scale_length=scale_length,
        )


@dataclass
class SurfaceDensityData:
    """The surface densities of a snapshot in all three cardinal projections.

    Attributes
    ----------
    projection_xy : NDArray[float]
        The surface density in the xy projection.
    projection_xz : NDArray[float]
        The surface density in the xz projection.
    projection_yz : NDArray[float]
        The surface density in the yz projection.
    flat_projection_xy : NDArray[float]
        The flattened surface density in the xy projection.
    extent : float
        The half-width of all axes.
    num_bins : int
        The number of bins of all axes.
    disc_profile : ExponentialDiscProfileData
        The parameters determining exponential profile of the disc.
    snapshot_data : SnapshotData
        The number of frames and per snapshot data.

    Notes
    -----
    All projections are 3D arrays of floats with the shape `(num_bins, num_bins, num_frames)`.

    """

    projection_xy: NDArray[float32]
    projection_xz: NDArray[float32]
    projection_yz: NDArray[float32]
    flat_projection_xy: NDArray[float32]
    extent: float
    num_bins: int
    disc_profile: ExponentialDiscProfileData
    snapshot_data: SnapshotData

    DATA_FILE_TYPE: ClassVar[str] = "Grid"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the projections are the same size
        same_shape = (
            self.projection_xy.shape == self.projection_xz.shape
            and self.projection_xy.shape == self.projection_yz.shape
            and self.flat_projection_xy.shape == self.projection_xy.shape
        )
        if not same_shape:
            msg = "The projections differ in shape!"
            msg += f"xy = {self.projection_xy.shape}, xz = {self.projection_xz.shape}, yz = {self.projection_yz.shape}"
            raise ValueError(msg)
        # Verify that the projection has the expected dimension
        shape = self.projection_xz.shape
        if len(shape) != 3:
            msg = "Expected projections to be a 3D array of `(num_bins, num_bins, num_frames)`"
            raise ValueError(msg)
        # Verify that the projection has expected shape
        if shape[0] != self.num_bins or shape[1] != self.num_bins:
            msg = f"Expected each slice to be ({self.num_bins}, {self.num_bins}) but got ({shape[0]}, {shape[1]})"
            raise ValueError(msg)
        # Verify that the projection has the expected number of frames
        if shape[2] != self.snapshot_data.num_frames:
            msg = f"Expected the number of projection frames to be {self.snapshot_data.num_frames} but got {shape[2]}"
            raise ValueError(msg)

        # Useful properties
        self.pixel_to_distance: float = 2 * self.extent / self.num_bins
        self.overdensity: NDArray[float32] = np.divide(
            self.flat_projection_xy,
            self.flat_projection_xy[:, :, 0][:, :, None],
            where=self.flat_projection_xy[:, :, 0][:, :, None] != 0,
        )
        self.density_contrast: NDArray[float32] = self.overdensity - 1

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize surface density data from file.

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

            extent: float = get_float_attr_from_hdf5(file, "extent")
            num_bins: int = get_int_attr_from_hdf5(file, "num_bins")

            # Projections
            projection_xy = read_dataset_from_hdf5_with_dtype(file, "projection_xy", dtype=float32)
            projection_xz = read_dataset_from_hdf5_with_dtype(file, "projection_xz", dtype=float32)
            projection_yz = read_dataset_from_hdf5_with_dtype(file, "projection_yz", dtype=float32)

            flat_projection_xy = read_dataset_from_hdf5_with_dtype(file, "flat_projection_xy", dtype=float32)

            disc_profile = ExponentialDiscProfileData.load_from(file)
            snapshot_data = SnapshotData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            projection_xy=projection_xy,
            projection_xz=projection_xz,
            projection_yz=projection_yz,
            flat_projection_xy=flat_projection_xy,
            extent=extent,
            num_bins=num_bins,
            disc_profile=disc_profile,
            snapshot_data=snapshot_data,
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
            file.attrs["extent"] = self.extent
            file.attrs["num_bins"] = self.num_bins

            file.create_dataset("projection_xy", data=self.projection_xy)
            file.create_dataset("projection_xz", data=self.projection_xz)
            file.create_dataset("projection_yz", data=self.projection_yz)
            file.create_dataset("flat_projection_xy", data=self.flat_projection_xy)
            self.disc_profile.dump_into(file)
            self.snapshot_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions
    def get_limits(self) -> tuple[float, float]:
        """Return the 1D limits of the surface density grid.

        Returns
        -------
        limits : tuple[float, float]
            The 1D limits.

        """
        return (-self.extent, self.extent)

    def get_2d_limits(self) -> tuple[float, float, float, float]:
        """Return the 2D limits of the surface density grid.

        Returns
        -------
        limits : tuple[float, float, float, float]
            The 2D limits.

        """
        return (-self.extent, self.extent, -self.extent, self.extent)

    def get_min_density(self) -> float:
        """Return the minimum (non-zero) density across all projections.

        Returns
        -------
        float
            The minimum density across all projections.

        """
        min_xy = np.nanmin(self.projection_xy[self.projection_xy > 0])
        min_xz = np.nanmin(self.projection_xz[self.projection_xz > 0])
        min_yz = np.nanmin(self.projection_yz[self.projection_yz > 0])
        return float(np.min([min_xy, min_xz, min_yz]))

    def get_min_flat_density(self) -> float:
        """Return the minimum (non-zero) flattened density in the xy projection.

        Returns
        -------
        float
            The minimum flattened density in the xy projection.

        """
        return float(np.nanmin(self.flat_projection_xy[self.flat_projection_xy > 0]))

    def get_max_density(self) -> float:
        """Return the maximum density across all projections.

        Returns
        -------
        float
            The maximum density across all projections.

        """
        max_xy = np.nanmax(self.projection_xy)
        max_xz = np.nanmax(self.projection_xz)
        max_yz = np.nanmax(self.projection_yz)
        return float(np.max([max_xy, max_xz, max_yz]))

    def get_max_flat_density(self) -> float:
        """Return the maximum (non-zero) flattened density in the xy projection.

        Returns
        -------
        float
            The maximum flattened density in the xy projection.

        """
        return float(np.nanmax(self.flat_projection_xy))

    def get_dummy_data(self) -> NDArray[float32]:
        """Return an array of ones with the same shape as the grid.

        Returns
        -------
        NDArray[float32]
            The array of ones with the same shape as the grid.

        """
        return np.ones((self.num_bins, self.num_bins), dtype=float32)
