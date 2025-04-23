"""Test perturber data."""

from pathlib import Path

import numpy as np
import pytest

from novie_data.interface import NovieData
from novie_data.perturber_data import PerturberData
from novie_data.novie_data_gen.builtins._verifier import InconsistentArrayLengthError, UnexpectedArrayLengthError


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(PerturberData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float64)
    vel = np.zeros((3, num_frames), dtype=np.float64)
    mass = np.zeros(num_frames, dtype=np.float64)
    s = PerturberData(name="test", position=pos, velocity=vel, mass=mass)
    assert np.all(pos == s.position)
    assert np.all(vel == s.velocity)
    assert np.all(mass == s.mass)


def test_init_wrong_position_dim() -> None:
    """Test the constructor with an invalid position array."""
    num_frames = 20
    pos = np.zeros((2, num_frames), dtype=np.float64)
    vel = np.zeros((2, num_frames), dtype=np.float64)
    mass = np.zeros(num_frames, dtype=np.float64)
    with pytest.raises(UnexpectedArrayLengthError):
        _ = PerturberData(name="test", position=pos, velocity=vel, mass=mass)


def test_init_wrong_velocity_dim() -> None:
    """Test the constructor with an invalid velocity array."""
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float64)
    vel = np.zeros((2, num_frames), dtype=np.float64)
    mass = np.zeros(num_frames, dtype=np.float64)
    with pytest.raises(InconsistentArrayLengthError):
        _ = PerturberData(name="test", position=pos, velocity=vel, mass=mass)


def test_init_inconsistent_dims() -> None:
    """Test the constructor with an invalid velocity array."""
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float64)
    vel = np.zeros((3, num_frames + 1), dtype=np.float64)
    mass = np.zeros(num_frames - 2, dtype=np.float64)
    with pytest.raises(InconsistentArrayLengthError):
        _ = PerturberData(name="test", position=pos, velocity=vel, mass=mass)


def test_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float64)
    vel = np.zeros((3, num_frames), dtype=np.float64)
    mass = np.zeros(num_frames, dtype=np.float64)
    s = PerturberData(name="test", position=pos, velocity=vel, mass=mass)
    s.dump(output_path)
    t = PerturberData.load(output_path)
    assert s == t


def test_deserialization_v2() -> None:
    """Test deserialization of v2."""
    num_frames: int = 20
    input_path = Path("test_data/perturber_v2.hdf5")
    t = PerturberData.load(input_path)

    assert t.num_frames == num_frames


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    s = PerturberData.empty(n=10)
    s.dump(output_path)

    s = PerturberData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    PerturberData.save_init(output_path, name="test")
    s = PerturberData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    initial_position = np.array([0.01, 0.84, -12.43], dtype=np.float64).reshape(3)
    initial_velocity = np.array([9.4, -0.04, 41], dtype=np.float64).reshape(3)
    acceleration = np.array([0.01, -0.5, -0.0007], dtype=np.float64).reshape(3)
    PerturberData.save_frame(
        output_path,
        0,
        mass=421.46574,
        position=initial_position,
        velocity=initial_velocity,
    )
    s = PerturberData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.mass[0] == 421.46574
    assert np.array_equal(s.position[:, 0], initial_position)
    assert np.array_equal(s.velocity[:, 0], initial_velocity)

    for i in range(1, 5):
        current_time = 0.74 * i + 598
        current_mass = current_time
        current_velocity = acceleration * current_time + initial_velocity
        current_position = 0.5 * acceleration * current_time**2 + initial_velocity * current_time + initial_position
        PerturberData.save_frame(
            output_path,
            i,
            mass=current_mass,
            position=current_position.reshape(3),
            velocity=current_velocity.reshape(3),
        )
    s = PerturberData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 0.74 * i + 598
        current_mass = current_time
        current_velocity = acceleration * current_time + initial_velocity
        current_position = 0.5 * acceleration * current_time**2 + initial_velocity * current_time + initial_position
        assert s.mass[i] == current_mass
        assert np.array_equal(s.position[:, i], current_position)
        assert np.array_equal(s.velocity[:, i], current_velocity)
        assert s.completeness[i]
