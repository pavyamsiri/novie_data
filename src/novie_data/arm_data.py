"""Data representing errors between observed spiral arms and found spiral clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from .serde.accessors import (
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    get_string_sequence_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SpiralArmErrorData:
    """Data that compares observed spiral arm data to the fitted clusters from SpArcFiRe.

    Attributes
    ----------
    names : Sequence[str]
        The spiral arm names.
    cluster_errors : NDArray[float32]
        The fit error of each spiral arm with respect to each cluster.
    fit_errors : NDArray[float32]
        The fit error of each spiral arm with respect to each cluster's fitted spiral.
    original_fit_errors : NDArray[float32]
        The fit error of each cluster's fitted log spiral.
    num_fit_points : int
        The number of fit points used to calculate fit error.

    """

    names: Sequence[str]
    cluster_errors: NDArray[float32]
    fit_errors: NDArray[float32]
    original_fit_errors: NDArray[float32]
    num_fit_points: int

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the arrays the correct size
        num_neighbourhoods: int = self.cluster_errors.shape[0]
        max_clusters: int = self.cluster_errors.shape[1]
        num_arms: int = self.cluster_errors.shape[2]
        num_frames: int = self.cluster_errors.shape[3]

        common_shape: tuple[int, int, int, int] = (num_neighbourhoods, max_clusters, num_arms, num_frames)
        if self.cluster_errors.shape != common_shape:
            msg = f"Expected cluster errors shape to be {common_shape} but got {self.cluster_errors.shape}"
            raise ValueError(msg)
        if self.fit_errors.shape != common_shape:
            msg = f"Expected fit errors shape to be {common_shape} but got {self.fit_errors.shape}"
            raise ValueError(msg)
        if self.original_fit_errors.shape != (max_clusters, num_frames):
            msg = (
                f"Expected original fit errors shape to be {(max_clusters, num_frames)} but got {self.original_fit_errors.shape}"
            )
            raise ValueError(msg)
        if len(self.names) != num_arms:
            msg = (
                f"The length of the arm names array {len(self.names)} is not the same as the expected number of arms {num_arms}!"
            )
        self.num_arms: int = num_arms
        self.num_frames: int = num_frames
        self.num_neighbourhoods: int = num_neighbourhoods

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        names: Sequence[str] = get_string_sequence_from_hdf5(in_file, "arm_names")
        cluster_errors = read_dataset_from_hdf5_with_dtype(in_file, "arm_cluster_errors", dtype=float32)
        fit_errors = read_dataset_from_hdf5_with_dtype(in_file, "arm_fit_errors", dtype=float32)
        original_fit_errors = read_dataset_from_hdf5_with_dtype(in_file, "arm_original_fit_errors", dtype=float32)
        num_fit_points: int = get_int_attr_from_hdf5(in_file, "num_fit_points")
        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            names=names,
            cluster_errors=cluster_errors,
            fit_errors=fit_errors,
            num_fit_points=num_fit_points,
            original_fit_errors=original_fit_errors,
        )

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["num_fit_points"] = int(self.num_fit_points)
        out_file.create_dataset("arm_names", data=self.names)
        out_file.create_dataset("arm_cluster_errors", data=self.cluster_errors)
        out_file.create_dataset("arm_fit_errors", data=self.fit_errors)
        out_file.create_dataset("arm_original_fit_errors", data=self.original_fit_errors)
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )


@dataclass
class SpiralClusterResidualsData:
    """Data class to store residuals and errors from corrugation data processing.

    Attributes
    ----------
    arm_error_data : SpiralArmErrorData
        The error between observed arms and found clusters.
    name : str
        The name of the dataset.

    """

    arm_error_data: SpiralArmErrorData
    name: str

    DATA_FILE_TYPE: ClassVar[str] = "SpiralClusterResiduals"
    VERSION: ClassVar[Version] = Version("2.0.0")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize phase spiral data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        SpiralClusterData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            name: str = get_str_attr_from_hdf5(file, "name")
            arm_error_data = SpiralArmErrorData.load_from(file)
        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]",
            cls.__name__,
            path.absolute(),
        )
        return cls(
            arm_error_data=arm_error_data,
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
            file.attrs["name"] = self.name

            self.arm_error_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.arm_error_data.num_frames

    @property
    def combined_errors(self) -> NDArray[float32]:
        """NDArray[float]: The sum of the cluster errors and the fit errors between arm and cluster pairs."""
        return self.arm_error_data.cluster_errors + self.arm_error_data.fit_errors
