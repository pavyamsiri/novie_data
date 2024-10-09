"""Data general to all snapshot-based data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

from h5py import File as Hdf5File
from numpy import float32, uint32
from packaging.version import Version

from .serde.accessors import read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class SnapshotData:
    """The data that is general to a collection of snapshots.

    Attributes
    ----------
    names : NDArray[uint32]
        The snapshot numbers.
    times : NDArray[float32]
        The time associated with each snapshot in Myr.

    """

    DATA_FILE_TYPE: ClassVar[str] = "Snapshot"
    VERSION: ClassVar[Version] = Version("0.1.0")

    def __init__(self, names: NDArray[uint32], times: NDArray[float32]) -> None:
        """Initialize the snapshot data.

        Parameters
        ----------
        names : NDArray[uint32]
            The snapshot numbers.
        times : NDArray[float32]
            The time associated with each snapshot in Myr.

        """
        times_dimension = len(times.shape)
        if times_dimension != 1:
            msg = f"Expected `times` to be a 1D array but it is {times_dimension}D instead!"
            raise ValueError(msg)
        num_times: int = len(self.times)
        num_names: int = len(self.names)

        if num_times != num_names:
            msg = f"The number of times {num_times} is not equal to the number of snapshot names {num_names}!"
            raise ValueError(msg)

        self.names: NDArray[uint32] = names
        self.times: NDArray[float32] = times
        self.num_frames: int = num_times

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

            file.create_dataset("names", data=self.names)
            file.create_dataset("times", data=self.times)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deerialize data from disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)
            names = read_dataset_from_hdf5_with_dtype(file, "names", dtype=uint32)
            times = read_dataset_from_hdf5_with_dtype(file, "times", dtype=float32)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            names=names,
            times=times,
        )
