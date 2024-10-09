"""Data general to all snapshot-based data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import numpy as np
from numpy import float32

from .serde.accessors import get_string_sequence_from_hdf5, read_dataset_from_hdf5_with_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from h5py import File as Hdf5File
    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class SnapshotData:
    """The data that is general to a collection of snapshots.

    Attributes
    ----------
    names : Sequence[str]
        The names of each snapshot.
    times : NDArray[float32]
        The time associated with each snapshot in Myr.

    """

    def __init__(self, names: Sequence[str], times: NDArray[float32]) -> None:
        """Initialize the snapshot data.

        Parameters
        ----------
        names : Sequence[str]
            The names of each snapshot.
        times : NDArray[float32]
            The time associated with each snapshot in Myr.

        """
        times_dimension = len(cast(tuple[int], times.shape))
        if times_dimension != 1:
            msg = f"Expected `times` to be a 1D array but it is {times_dimension}D instead!"
            raise ValueError(msg)
        num_times: int = len(self.times)
        num_names: int = len(self.names)

        if num_times != num_names:
            msg = f"The number of times {num_times} is not equal to the number of snapshot names {num_names}!"
            raise ValueError(msg)

        self.names: Sequence[str] = names
        self.times: NDArray[float32] = times
        self.num_frames: int = num_times

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        _ = out_file.create_dataset("snapshot_names", data=self.names)
        _ = out_file.create_dataset("times", data=self.times)
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
        snapshot_names: Sequence[str] = get_string_sequence_from_hdf5(in_file, "snapshot_names")
        times = read_dataset_from_hdf5_with_dtype(in_file, "times", dtype=float32)

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
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
