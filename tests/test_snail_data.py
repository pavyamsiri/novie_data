"""Test snail data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.neighbourhood_data import SphericalNeighbourhoodData
from novie_data.snail_data import SnailData


def test_snail_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SnailData, NovieData)


def test_snail_data_init() -> None:
    """Test the constructor."""
    num_height_bins: int = 36
    num_velocity_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4
    proj = np.zeros((num_velocity_bins, num_height_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    neighbourhood = SphericalNeighbourhoodData(
        num_spheres=num_neighbourhoods,
        radius=1.0,
    )
    s = SnailData(
        name="test",
        surface_density=proj,
        azimuthal_velocity=proj,
        radial_velocity=proj,
        neighbourhood_data=neighbourhood,
        num_height_bins=num_height_bins,
        num_velocity_bins=num_velocity_bins,
        max_height=1,
        max_velocity=60,
    )
    assert s.max_height == 1


def test_snail_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_height_bins: int = 36
    num_velocity_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4
    proj = np.zeros((num_velocity_bins, num_height_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    neighbourhood = SphericalNeighbourhoodData(
        num_spheres=num_neighbourhoods,
        radius=1.0,
    )
    s = SnailData(
        name="test",
        surface_density=proj,
        azimuthal_velocity=proj,
        radial_velocity=proj,
        neighbourhood_data=neighbourhood,
        num_height_bins=num_height_bins,
        num_velocity_bins=num_velocity_bins,
        max_height=1,
        max_velocity=60,
    )
    s.dump(output_path)
    t = SnailData.load(output_path)
    assert s == t


def test_snail_data_deserialization_v4() -> None:
    """Test deserialization of v4."""
    num_height_bins: int = 36
    num_velocity_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4

    input_path = Path("test_data/snail_v4.hdf5")
    t = SnailData.load(input_path)

    assert t.num_height_bins == num_height_bins
    assert t.num_velocity_bins == num_velocity_bins
    assert t.num_frames == num_frames
    assert t.surface_density.shape == (num_velocity_bins, num_height_bins, num_frames, num_neighbourhoods)
