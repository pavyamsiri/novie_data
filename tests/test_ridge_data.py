"""Test ridge data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.ridge_data import RidgeData


def test_ridge_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(RidgeData, NovieData)


def test_ridge_data_init() -> None:
    """Test the constructor."""
    num_radial_bins: int = 13
    num_velocity_bins: int = 123
    num_frames: int = 3
    min_radius: float = 0.0
    max_radius: float = 123.0
    min_velocity: float = -12
    max_velocity: float = -10
    density = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float32)

    s = RidgeData(
        name="test",
        mass_density=density,
        number_density=density,
        num_radial_bins=num_radial_bins,
        num_velocity_bins=num_velocity_bins,
        min_radius=min_radius,
        max_radius=max_radius,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    assert s.min_radius == min_radius


def test_ridge_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_radial_bins: int = 13
    num_velocity_bins: int = 123
    num_frames: int = 3
    min_radius: float = 0.0
    max_radius: float = 123.0
    min_velocity: float = -12
    max_velocity: float = -10
    density = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float32)

    s = RidgeData(
        name="test",
        mass_density=density,
        number_density=density,
        num_radial_bins=num_radial_bins,
        num_velocity_bins=num_velocity_bins,
        min_radius=min_radius,
        max_radius=max_radius,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    s.dump(output_path)
    t = RidgeData.load(output_path)
    assert s == t
