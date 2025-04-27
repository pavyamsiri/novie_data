"""Test snapshot data."""

from pathlib import Path

import numpy as np
import pytest
from novie_helpers.verifier import InconsistentArrayLengthError

from novie_data.interface import NovieData
from novie_data.snapshot_data import SnapshotData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SnapshotData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    s = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint16), times=np.linspace(0, 1000, 5, dtype=np.float64).reshape(5))
    assert s.num_frames == 5


def test_init_inconsistent_lengths() -> None:
    """Test that the constructor errors when the array lengths are inconsistent."""
    with pytest.raises(InconsistentArrayLengthError):
        _ = SnapshotData(
            name="test", codes=np.arange(3, dtype=np.uint16), times=np.linspace(0, 1000, 5, dtype=np.float64).reshape(5)
        )


def test_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    s = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint16), times=np.linspace(0, 1000, 5, dtype=np.float64).reshape(5))
    s.dump(output_path)
    t = SnapshotData.load(output_path)
    assert s == t


def test_deserialization_v0() -> None:
    """Test deserialization of v3."""
    num_frames: int = 5

    input_path = Path("test_data/snapshot_v0.hdf5")
    t = SnapshotData.load(input_path)

    assert t.num_frames == num_frames


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    s = SnapshotData.empty(n=10)
    s.dump(output_path)

    s = SnapshotData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    SnapshotData.save_init(output_path, name="test")
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    SnapshotData.save_frame(output_path, 0, codes=156, times=0.004)
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.codes[0] == 156
    assert s.times[0] == 0.004

    for i in range(1, 5):
        current_code = i
        current_time = 0.1 * i + 0.1
        SnapshotData.save_frame(
            output_path,
            i,
            codes=current_code,
            times=current_time,
        )
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_code = i
        current_time = 0.1 * i + 0.1
        assert s.codes[i] == current_code
        assert s.times[i] == current_time
        assert s.completeness[i]
