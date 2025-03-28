"""Test snapshot data."""

from pathlib import Path

import numpy as np
import pytest
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import verify_array_is_1d
from novie_data.errors import InconsistentArrayShapeError
from novie_data.interface import NovieData
from novie_data.serde.accessors import get_file_version, read_dataset_from_hdf5_with_dtype
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
    with pytest.raises(InconsistentArrayShapeError):
        _ = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint32), times=np.linspace(0, 1000, 2))


def test_snapshot_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    s = SnapshotData(name="test", codes=np.arange(5, dtype=np.uint32), times=np.linspace(0, 1000, 5))
    s.dump(output_path)
    t = SnapshotData.load(output_path)
    assert s == t


def test_snapshot_data_deserialization_v0() -> None:
    """Test deserialization of v0."""
    num_frames: int = 5

    input_path = Path("test_data/snapshot_v0.hdf5")
    t = SnapshotData.load(input_path)

    assert t.num_frames == num_frames
    assert t.is_complete()
    assert np.all(t.completeness)


def test_snapshot_data_deserialization_v1() -> None:
    """Test deserialization of v1."""
    num_frames: int = 5

    input_path = Path("test_data/snapshot_v1.hdf5")
    t = SnapshotData.load(input_path)

    assert t.num_frames == num_frames
    assert not t.is_complete()
    assert t.completeness[0]
    assert np.all(~t.completeness[1:])


def test_snapshot_data_incremental_per_frame(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    s = SnapshotData.empty(10)
    s.dump(output_path)

    s = SnapshotData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    SnapshotData.save_init("test", output_path)
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    SnapshotData.save_frame(0, 1, 0.1, output_path)
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.times[0] == 0.1
    assert s.codes[0] == 1
    assert s.completeness[0]
    assert np.all(~s.completeness[1:])

    for i in range(1, 5):
        current_code = i
        current_time = 0.1 * i + 0.1
        SnapshotData.save_frame(i, current_code, current_time, output_path)
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_code = i
        current_time = 0.1 * i + 0.1
        assert s.codes[i] == current_code
        assert s.times[i] == current_time
        assert s.completeness[i]

    SnapshotData.save_chunk(np.s_[7:9], np.array([3, 5], dtype=np.uint32), np.array([0.344, 0.32], dtype=np.float32), output_path)
    s = SnapshotData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.codes[7] == 3
    assert s.codes[8] == 5
    assert s.times[7] == 0.344
    assert s.times[8] == 0.32


def test_snapshot_data_convert_v0_to_v1(tmp_path: Path) -> None:
    """Test incremental serialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    v0_path = tmp_path / "test_v0.hdf5"

    with Hdf5File(v0_path, "w") as v0_file:
        _create_snapshot_data_v0(v0_file, 5)
        old_version = get_file_version(v0_file)
        assert old_version.major == 0
        SnapshotData.migrate_version(v0_file, Version("1.0.0"), is_complete=False)

        new_version = get_file_version(v0_file)
        assert new_version.major == 1
        complete = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(v0_file, "completeness", dtype=np.bool_))
        assert np.all(~complete)

    with Hdf5File(v0_path, "w") as v0_file:
        _create_snapshot_data_v0(v0_file, 5)
        old_version = get_file_version(v0_file)
        assert old_version.major == 0
        SnapshotData.migrate_version(v0_file, Version("1.0.0"), is_complete=True)

        new_version = get_file_version(v0_file)
        assert new_version.major == 1
        complete = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(v0_file, "completeness", dtype=np.bool_))
        assert np.all(complete)


def _create_snapshot_data_v0(file: Hdf5File, num_frames: int) -> None:
    codes = np.arange(1, num_frames + 1, dtype=np.uint32)
    times = np.linspace(0, 1000, num_frames, dtype=np.float32)

    # General
    file.attrs["type"] = SnapshotData.DATA_FILE_TYPE
    file.attrs["version"] = str(Version("0.0.0"))
    file.attrs["name"] = "test_v0"

    file.create_dataset("codes", data=codes)
    file.create_dataset("times", data=times)
