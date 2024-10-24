"""Test wrinkle data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.neighbourhood_data import SphericalNeighbourhoodData
from novie_data.wrinkle_data import WrinkleData


def test_wrinkle_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(WrinkleData, NovieData)


def test_wrinkle_data_init() -> None:
    """Test the constructor."""
    num_bins: int = 12
    num_frames: int = 9
    num_neighbourhoods: int = 32
    min_lz: float = 0
    max_lz: float = 1e5
    neighbourhood = SphericalNeighbourhoodData(
        num_spheres=num_neighbourhoods,
        radius=1.0,
    )
    angular_momentum = np.zeros(num_bins, dtype=np.float32)
    mean_velocity = np.zeros((num_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    s = WrinkleData(
        name="test",
        angular_momentum=angular_momentum,
        mean_radial_velocity=mean_velocity,
        mean_radial_velocity_error=mean_velocity,
        min_lz=min_lz,
        max_lz=max_lz,
        num_bins=num_bins,
        neighbourhood_data=neighbourhood,
        distance_error=1.01,
    )
    assert s.distance_error == 1.01


def test_wrinkle_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_bins: int = 12
    num_frames: int = 9
    num_neighbourhoods: int = 32
    min_lz: float = 0
    max_lz: float = 1e5
    neighbourhood = SphericalNeighbourhoodData(
        num_spheres=num_neighbourhoods,
        radius=1.0,
    )
    angular_momentum = np.zeros(num_bins, dtype=np.float32)
    mean_velocity = np.zeros((num_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    s = WrinkleData(
        name="test",
        angular_momentum=angular_momentum,
        mean_radial_velocity=mean_velocity,
        mean_radial_velocity_error=mean_velocity,
        min_lz=min_lz,
        max_lz=max_lz,
        num_bins=num_bins,
        neighbourhood_data=neighbourhood,
        distance_error=1.01,
    )
    s.dump(output_path)
    t = WrinkleData.load(output_path)
    assert s == t
