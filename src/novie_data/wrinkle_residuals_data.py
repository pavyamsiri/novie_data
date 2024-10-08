"""Data representing residuals and errors between observed and simulated wrinkle data."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from numpy.typing import NDArray
from packaging.version import Version

from .serde.accessors import read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class WrinkleResidualsData:
    """Data class to store residuals and errors from wrinkle data processing.

    Attributes
    ----------
    residuals : NDArray[float32]
        The residuals between the test mean height and the expected mean height in units of km/s.
    errors : NDArray[float32]
        The sum of square residuals for each frame and filter in units of km/s.
    relative_residuals : NDArray[float32]
        The relative residuals between the test mean radial velocity and the expected mean radial velocity.
    relative_errors : NDArray[float32]
        The mean absolute relative error.
    angular_momentum : NDArray[float32]
        The central angular momentum value for each bin in units of kpc km/s.
    snapshot_data : SnapshotData
        The data generic over each snapshot.

    """

    residuals: NDArray[float32]
    sum_of_square_residuals: NDArray[float32]
    relative_errors: NDArray[float32]
    mean_absolute_relative_error: NDArray[float32]
    angular_momentum: NDArray[float32]
    snapshot_data: SnapshotData

    DATA_FILE_TYPE: ClassVar[str] = "WrinkleResidualsData"
    VERSION: ClassVar[Version] = Version("1.0.0")

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
        WrinkleResidualsData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            # Arrays
            residuals: NDArray[float32]
            residuals = read_dataset_from_hdf5_with_dtype(file, "residuals", dtype=float32)
            sum_of_square_residuals: NDArray[float32]
            sum_of_square_residuals = read_dataset_from_hdf5_with_dtype(file, "sum_of_square_residuals", dtype=float32)
            relative_errors: NDArray[float32]
            relative_errors = read_dataset_from_hdf5_with_dtype(file, "relative_errors", dtype=float32)
            mean_absolute_relative_error: NDArray[float32]
            mean_absolute_relative_error = read_dataset_from_hdf5_with_dtype(file, "mean_absolute_relative_error", dtype=float32)
            angular_momentum: NDArray[float32]
            angular_momentum = read_dataset_from_hdf5_with_dtype(file, "radii", dtype=float32)

            snapshot_data = SnapshotData.load_from(file)
        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            residuals=residuals,
            sum_of_square_residuals=sum_of_square_residuals,
            relative_errors=relative_errors,
            mean_absolute_relative_error=mean_absolute_relative_error,
            angular_momentum=angular_momentum,
            snapshot_data=snapshot_data,
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

            file.create_dataset("residuals", data=self.residuals)
            file.create_dataset("sum_of_square_residuals", data=self.sum_of_square_residuals)
            file.create_dataset("relative_errors", data=self.relative_errors)
            file.create_dataset("mean_absolute_relative_error", data=self.mean_absolute_relative_error)
            file.create_dataset("radii", data=self.angular_momentum)
            self.snapshot_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions
    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.snapshot_data.num_frames

    @property
    def num_spheres(self) -> int:
        """int: The number of spheres."""
        return self.residuals.shape[2]

    def get_starting_angles_deg(self) -> NDArray[float32]:
        """Return the angle of the start locations in degrees.

        Returns
        -------
        starting_angles : NDArray[float32]
            The angle of the start locations in degrees.

        """
        return np.linspace(0, 360, self.num_spheres, endpoint=False)
