"""Test snapshot data."""

from pathlib import Path

import numpy as np
import pytest

from novie_data.errors import InconsistentArrayLengthError
from novie_data.interface import NovieData
from novie_data.snapshot_data import SnapshotData


def test_snapshot_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SnapshotData, NovieData)


def test_snapshot_data_init() -> None:
    """Test the constructor."""
    s = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint32), times=np.linspace(0, 1000, 5))
    count = len(tuple(s.snapshot_names()))
    assert count == 5


def test_snapshot_data_init_inconsistent_lengths() -> None:
    """Test that the constructor errors when the array lengths are inconsistent."""
    with pytest.raises(InconsistentArrayLengthError):
        _ = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint32), times=np.linspace(0, 1000, 2))


def test_snapshot_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "snapshot.hdf5"
    s = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint32), times=np.linspace(0, 1000, 5))
    s.dump(output_path)
    t = SnapshotData.load(output_path)
    assert s == t
