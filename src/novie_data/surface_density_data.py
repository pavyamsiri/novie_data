"""The animation data class for surface density projections in Cartesian space."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import Array2D, Array3D, verify_array_is_3d
from novie_data.errors import verify_arrays_have_correct_length, verify_arrays_have_same_shape

from .serde.accessors import (
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

_Array2D_f32: TypeAlias = Array2D[np.float32]
_Array3D_f32: TypeAlias = Array3D[np.float32]

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


class SurfaceDensityData:
    """The surface densities of a snapshot in all three cardinal projections.

    Attributes
    ----------
    name : str
        The name of the dataset.
    projection_xy : Array3D[f32]
        The surface density in the xy projection.
    projection_xz : Array3D[f32]
        The surface density in the xz projection.
    projection_yz : Array3D[f32]
        The surface density in the yz projection.
    flat_projection_xy : Array3D[f32]
        The flattened surface density in the xy projection.
    extent : float
        The half-width of all axes.
    num_bins : int
        The number of bins of all axes.
    disc_profile : ExponentialDiscProfileData
        The parameters determining exponential profile of the disc.

    Notes
    -----
    All projections are 3D arrays of floats with the shape `(num_bins, num_bins, num_frames)`.

    """

    DATA_FILE_TYPE: ClassVar[str] = "Grid"
    VERSION: ClassVar[Version] = Version("3.0.0")

    def __init__(
        self,
        *,
        name: str,
        projection_xy: _Array3D_f32,
        projection_xz: _Array3D_f32,
        projection_yz: _Array3D_f32,
        flat_projection_xy: _Array3D_f32,
        extent: float,
        num_bins: int,
        disc_profile: ExponentialDiscProfileData,
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the dataset.
        projection_xy : Array3D[f32]
            The surface density in the xy projection.
        projection_xz : Array3D[f32]
            The surface density in the xz projection.
        projection_yz : Array3D[f32]
            The surface density in the yz projection.
        flat_projection_xy : Array3D[f32]
            The flattened surface density in the xy projection.
        extent : float
            The half-width of all axes.
        num_bins : int
            The number of bins of all axes.
        disc_profile : ExponentialDiscProfileData
            The parameters determining exponential profile of the disc.

        Notes
        -----
        All projections are 3D arrays of floats with the shape `(num_bins, num_bins, num_frames)`.

        """
        self.name: str = name
        self.projection_xy: _Array3D_f32 = projection_xy
        self.projection_xz: _Array3D_f32 = projection_xz
        self.projection_yz: _Array3D_f32 = projection_yz
        self.flat_projection_xy: _Array3D_f32 = flat_projection_xy
        self.extent: float = extent
        self.num_bins: int = num_bins
        self.disc_profile: ExponentialDiscProfileData = disc_profile

        # Verify that the projections are the same size
        verify_arrays_have_same_shape(
            [self.projection_xy, self.projection_xz, self.projection_yz, self.flat_projection_xy],
            msg="Projections differ in shape!",
        )
        verify_arrays_have_correct_length(
            [(self.projection_xy, 0)], num_bins, msg=f"Expected the projection to have {num_bins} rows!"
        )
        verify_arrays_have_correct_length(
            [(self.projection_xy, 1)], num_bins, msg=f"Expected the projection to have {num_bins} columns!"
        )

        # Useful properties
        self.pixel_to_distance: float = 2 * self.extent / self.num_bins
        self.overdensity: _Array3D_f32 = np.divide(
            self.flat_projection_xy,
            self.flat_projection_xy[:, :, 0][:, :, None],
            where=self.flat_projection_xy[:, :, 0][:, :, None] != 0,
        )
        self.density_contrast: _Array3D_f32 = self.overdensity - 1

    def __eq__(self, other: object) -> bool:
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
        equality &= self.extent == other.extent
        equality &= self.num_bins == other.num_bins
        equality &= self.disc_profile == other.disc_profile
        equality &= np.all(self.projection_xy == other.projection_xy)
        equality &= np.all(self.projection_xz == other.projection_xz)
        equality &= np.all(self.projection_yz == other.projection_yz)
        equality &= np.all(self.flat_projection_xy == other.flat_projection_xy)
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

            extent: float = get_float_attr_from_hdf5(file, "extent")
            num_bins: int = get_int_attr_from_hdf5(file, "num_bins")
            name: str = get_str_attr_from_hdf5(file, "name")

            # Projections
            projection_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xy", dtype=np.float32))
            projection_xz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xz", dtype=np.float32))
            projection_yz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_yz", dtype=np.float32))
            flat_projection_xy = verify_array_is_3d(
                read_dataset_from_hdf5_with_dtype(file, "flat_projection_xy", dtype=np.float32)
            )

            disc_profile = ExponentialDiscProfileData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            projection_xy=projection_xy,
            projection_xz=projection_xz,
            projection_yz=projection_yz,
            flat_projection_xy=flat_projection_xy,
            extent=extent,
            num_bins=num_bins,
            disc_profile=disc_profile,
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
            file.attrs["extent"] = self.extent
            file.attrs["num_bins"] = self.num_bins
            file.attrs["name"] = self.name

            file.create_dataset("projection_xy", data=self.projection_xy)
            file.create_dataset("projection_xz", data=self.projection_xz)
            file.create_dataset("projection_yz", data=self.projection_yz)
            file.create_dataset("flat_projection_xy", data=self.flat_projection_xy)
            self.disc_profile.dump_into(file)
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

    def get_dummy_data(self) -> _Array2D_f32:
        """Return an array of ones with the same shape as the grid.

        Returns
        -------
        _Array2D_f32
            The array of ones with the same shape as the grid.

        """
        return np.ones((self.num_bins, self.num_bins), dtype=np.float32)
