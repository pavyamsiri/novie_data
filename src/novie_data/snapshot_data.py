"""Data general to all snapshot-based data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from .serde.accessors import get_and_read_dataset_from_hdf5, get_string_sequence_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Sequence

    from h5py import File as Hdf5File
    from numpy import float32
    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SnapshotData:
    """The data that is general to a collection of snapshots.

    Attributes
    ----------
    names : Sequence[str]
        The names of each snapshot.
    times : NDArray[float32]
        The time associated with each snapshot in Myr.

    """

    names: Sequence[str]
    times: NDArray[float32]

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        times_dimension = len(self.times.shape)
        if times_dimension != 1:
            msg = f"Expected `times` to be a 1D array but it is {times_dimension}D instead!"
            raise ValueError(msg)
        num_times: int = len(self.times)
        num_names: int = len(self.names)

        if num_times != num_names:
            msg = f"The number of times {num_times} is not equal to the number of snapshot names {num_names}!"
            raise ValueError(msg)

        self.num_frames: int = num_times

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.create_dataset("snapshot_names", data=self.names)
        out_file.create_dataset("times", data=self.times)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", type(self).__name__, out_file.filename)

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize snapshot data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        snapshot_names: Sequence[str] = get_string_sequence_from_hdf5(in_file, "snapshot_names")
        times, _ = get_and_read_dataset_from_hdf5(in_file, "times")

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, in_file.filename)
        return cls(
            names=snapshot_names,
            times=times,
        )

    @staticmethod
    def all_same(data_list: Sequence[SnapshotData]) -> bool:
        """Return whether the list of snapshot data are all the same.

        Parameters
        ----------
        data_list : Sequence[SnapshotData]
            The list of snapshot data.

        Returns
        -------
        all_same : bool
            This is `True` if all snapshot data are equal otherwise `False`.

        """
        if len(data_list) == 0:
            return True
        reference_data = data_list[0]
        for current_data in data_list[1:]:
            if current_data.num_frames != reference_data.num_frames:
                return False
            if not np.allclose(current_data.times, reference_data.times):
                return False
            if current_data.names != reference_data.names:
                return False
        return True
