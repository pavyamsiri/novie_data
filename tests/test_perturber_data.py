"""Test snapshot data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.perturber_data import PerturberData


def test_perturber_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(PerturberData, NovieData)


def test_perturber_data_init() -> None:
    """Test the constructor."""
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float32)
    vel = np.zeros((3, num_frames), dtype=np.float32)
    mass = np.zeros(num_frames, dtype=np.float32)
    s = PerturberData(name="test", position=pos, velocity=vel, mass=mass)
    assert np.all(pos == s.position)
    assert np.all(vel == s.velocity)
    assert np.all(mass == s.mass)


def test_perturber_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "snapshot.hdf5"
    num_frames = 20
    pos = np.zeros((3, num_frames), dtype=np.float32)
    vel = np.zeros((3, num_frames), dtype=np.float32)
    mass = np.zeros(num_frames, dtype=np.float32)
    s = PerturberData(name="test", position=pos, velocity=vel, mass=mass)
    s.dump(output_path)
    t = PerturberData.load(output_path)
    assert s == t
