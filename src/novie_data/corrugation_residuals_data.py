"""Data representing residuals and errors between observed and simulated corrugation data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from .serde.accessors import get_str_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

log: logging.Logger = logging.getLogger(__name__)


class CorrugationResidualsData:
    """Data class to store residuals and errors from corrugation data processing.

    Attributes
    ----------
    name : str
        The name of the data set.
    radii : Array1D[f32]
        The central radius value for each radial bin in units of kpc.
    residuals : Array3D[f32]
        The residuals between the test mean height and the expected mean height in units of kpc.
    relative_errors : Array3D[f32]
        The relative residuals between the test mean height and the expected mean height.
    sum_of_square_residuals: Array2D[f32]
        The sum of square residuals for each frame and filter in units of kpc**2.
    mean_absolute_relative_error : Array2D[f32]
        The mean absolute relative error.

    """

    DATA_FILE_TYPE: ClassVar[str] = "CorrugationResiduals"
    VERSION: ClassVar[Version] = Version("1.0.0")

    def __init__(
        self,
        *,
        name: str,
        radii: NDArray[float32],
        residuals: NDArray[float32],
        relative_errors: NDArray[float32],
        sum_of_square_residuals: NDArray[float32],
        mean_absolute_relative_error: NDArray[float32],
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the data set.
        radii : Array1D[f32]
            The central radius value for each radial bin in units of kpc.
        residuals : Array3D[f32]
            The residuals between the test mean height and the expected mean height in units of kpc.
        relative_errors : Array3D[f32]
            The relative residuals between the test mean height and the expected mean height.
        sum_of_square_residuals: Array2D[f32]
            The sum of square residuals for each frame and filter in units of kpc**2.
        mean_absolute_relative_error : Array2D[f32]
            The mean absolute relative error.

        """
        self.name: str = name
        self.radii: NDArray[float32] = radii
        self.residuals: NDArray[float32] = residuals
        self.relative_errors: NDArray[float32] = relative_errors
        self.sum_of_square_residuals: NDArray[float32] = sum_of_square_residuals
        self.mean_absolute_relative_error: NDArray[float32] = mean_absolute_relative_error

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
        equality &= np.all(self.radii == other.radii)
        equality &= np.all(self.residuals == other.residuals)
        equality &= np.all(self.relative_errors == other.relative_errors)
        equality &= np.all(self.sum_of_square_residuals == other.sum_of_square_residuals)
        equality &= np.all(self.mean_absolute_relative_error == other.mean_absolute_relative_error)
        return bool(equality)

    def get_argmin_over_absolute(self) -> tuple[int, int]:
        """Get the frame and filter indices that minimises the sum of square residuals.

        Returns
        -------
        min_frame : int
            The index of the frame that minimises the sum of square residuals.
        min_filter : int
            The index of the filter that minimises the sum of square residuals.

        """
        min_idx = np.nanargmin(self.sum_of_square_residuals)
        min_frame, min_filter = np.unravel_index(min_idx, self.sum_of_square_residuals.shape)
        return int(min_frame), int(min_filter)

    def get_argmin_over_relative(self) -> tuple[int, int]:
        """Get the frame and filter indices that minimises the mean absolute relative error.

        Returns
        -------
        min_frame : int
            The index of the frame that minimises the mean absolute relative error.
        min_filter : int
            The index of the filter that minimises the mean absolute relative error.

        """
        min_idx = np.nanargmin(self.mean_absolute_relative_error)
        min_frame, min_filter = np.unravel_index(min_idx, self.mean_absolute_relative_error.shape)
        return int(min_frame), int(min_filter)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize phase spiral data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        CorrugationResidualsData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            name: str = get_str_attr_from_hdf5(file, "name")

            # Arrays
            residuals = read_dataset_from_hdf5_with_dtype(file, "residuals", dtype=float32)
            sum_of_square_residuals = read_dataset_from_hdf5_with_dtype(file, "sum_of_square_residuals", dtype=float32)
            relative_errors = read_dataset_from_hdf5_with_dtype(file, "relative_errors", dtype=float32)
            mean_absolute_relative_error = read_dataset_from_hdf5_with_dtype(file, "mean_absolute_relative_error", dtype=float32)
            radii = read_dataset_from_hdf5_with_dtype(file, "radii", dtype=float32)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            residuals=residuals,
            sum_of_square_residuals=sum_of_square_residuals,
            relative_errors=relative_errors,
            mean_absolute_relative_error=mean_absolute_relative_error,
            radii=radii,
            name=name,
        )

    def dump(self, path: Path) -> None:
        """Serialize phase spiral data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        with Hdf5File(path, "w") as file:
            # General
            file.attrs["type"] = CorrugationResidualsData.DATA_FILE_TYPE
            file.attrs["version"] = str(CorrugationResidualsData.VERSION)
            file.attrs["name"] = self.name

            file.create_dataset("residuals", data=self.residuals)
            file.create_dataset("sum_of_square_residuals", data=self.sum_of_square_residuals)
            file.create_dataset("relative_errors", data=self.relative_errors)
            file.create_dataset("mean_absolute_relative_error", data=self.mean_absolute_relative_error)
            file.create_dataset("radii", data=self.radii)
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", CorrugationResidualsData.__name__, path.absolute()
        )

    # Convenience functions
    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.residuals.shape[1]

    @property
    def num_filters(self) -> int:
        """int: The number of filters."""
        return self.residuals.shape[2]

    def get_starting_angles_deg(self) -> NDArray[float32]:
        """Return the angle of the start locations in degrees.

        Returns
        -------
        starting_angles : NDArray[float32]
            The angle of the start locations in degrees.

        """
        return np.linspace(0, 360, self.num_filters, endpoint=False)
