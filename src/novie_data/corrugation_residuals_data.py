"""Data representing residuals and errors between observed and simulated corrugation data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import Array1D, Array2D, Array3D, verify_array_is_1d, verify_array_is_2d, verify_array_is_3d
from novie_data.errors import verify_arrays_are_consistent

from .serde.accessors import get_str_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

_Array1D_f32: TypeAlias = Array1D[np.float32]
_Array2D_f32: TypeAlias = Array2D[np.float32]
_Array3D_f32: TypeAlias = Array3D[np.float32]

log: logging.Logger = logging.getLogger(__name__)


class CorrugationResidualsData:
    """Data class to store residuals and errors from corrugation data processing.

    Attributes
    ----------
    name : str
        The name of the data set.
    metric_name : str
        The name of the metric used.
    metric : Array3D[f32]
        The value of the metric per point, frame and location.
    summary : Array2D[f32]
        The summary statistic of the metric per frame and location.
    bin_values : Array1D[f32]
        The central bin value for each bin.

    """

    DATA_FILE_TYPE: ClassVar[str] = "CorrugationResiduals"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __init__(
        self,
        *,
        name: str,
        metric_name: str,
        metric: _Array3D_f32,
        summary: _Array2D_f32,
        bin_values: _Array1D_f32,
    ) -> None:
        """Initialize the data class.

        Parameters
        ----------
        name : str
            The name of the data set.
        metric_name : str
            The name of the metric used.
        metric : Array3D[f32]
            The value of the metric per point, frame and location.
        summary : Array2D[f32]
            The summary statistic of the metric per frame and location.
        bin_values : Array1D[f32]
            The central bin value for each bin.

        """
        self.name: str = name
        self.metric_name: str = metric_name
        self.metric: _Array3D_f32 = metric
        self.summary: _Array2D_f32 = summary
        self.bin_values: _Array1D_f32 = bin_values

        verify_arrays_are_consistent(
            [(self.bin_values, 0), (self.metric, 0)],
            msg="Expected the bin values and metric arrays to have the same number of rows.",
        )
        verify_arrays_are_consistent(
            [
                (self.metric, 1),
                (self.summary, 0),
            ],
            msg="Expected the arrays to have the same number of frames!",
        )
        verify_arrays_are_consistent(
            [
                (self.metric, 2),
                (self.summary, 1),
            ],
            msg="Expected the arrays to have the same number of neighbourhoods!",
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
        equality &= self.metric_name == other.metric_name
        equality &= np.all(self.bin_values == other.bin_values)
        equality &= np.all(self.metric == other.metric)
        equality &= np.all(self.summary == other.summary)
        return bool(equality)

    def get_extremal_index_over_summary(self, extremum: Literal["min", "max"]) -> tuple[int, int]:
        """Get the frame and location indices that minimises the summary statistic.

        Returns
        -------
        frame_index : int
            The index of the frame that extremises the summary statistic.
        location_index : int
            The index of the location that extremises the summary statistic.

        """
        extremum_index = np.nanargmin(self.summary) if extremum == "min" else np.nanargmax(self.summary)
        frame_index, location_index = np.unravel_index(extremum_index, self.summary.shape)
        return int(frame_index), int(location_index)

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
            metric_name: str = get_str_attr_from_hdf5(file, "metric_name")

            # Arrays
            bin_values = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "bin_values", dtype=np.float32))
            metric = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "metric", dtype=np.float32))
            summary = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "summary", dtype=np.float32))

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            name=name,
            metric_name=metric_name,
            metric=metric,
            summary=summary,
            bin_values=bin_values,
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
            file.attrs["metric_name"] = self.metric_name
            file.create_dataset("metric", data=self.metric)
            file.create_dataset("summary", data=self.summary)
            file.create_dataset("radii", data=self.bin_values)
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", CorrugationResidualsData.__name__, path.absolute()
        )

    # Convenience functions
    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.metric.shape[1]

    @property
    def num_locations(self) -> int:
        """int: The number of filters."""
        return self.metric.shape[2]

    def get_starting_angles_deg(self) -> _Array1D_f32:
        """Return the angle of the start locations in degrees.

        Returns
        -------
        starting_angles : Array1D[f32]
            The angle of the start locations in degrees.

        """
        return np.linspace(0, 360, self.num_locations, endpoint=False)
