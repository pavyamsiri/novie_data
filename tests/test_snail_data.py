"""Test snail data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.snail_data import SnailData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SnailData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_height_bins: int = 36
    num_velocity_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4
    proj = np.zeros((num_velocity_bins, num_height_bins, num_frames, num_neighbourhoods), dtype=np.float64)
    s = SnailData(
        name="test",
        surface_density=proj,
        azimuthal_velocity=proj,
        radial_velocity=proj,
        sphere_radius=1.0,
        max_height=1,
        max_velocity=60,
    )
    assert s.max_height == 1


def test_serde(tmp_path: Path) -> None:
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
    proj = np.zeros((num_velocity_bins, num_height_bins, num_frames, num_neighbourhoods), dtype=np.float64)
    s = SnailData(
        name="test",
        surface_density=proj,
        azimuthal_velocity=proj,
        radial_velocity=proj,
        sphere_radius=1.0,
        max_height=1,
        max_velocity=60,
    )
    s.dump(output_path)
    t = SnailData.load(output_path)
    assert s == t


def test_deserialization_v4() -> None:
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


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    num_frames = 10
    num_height_bins = 20
    num_locations = 6
    num_velocity_bins = 11

    s = SnailData.empty(
        num_frames=num_frames,
        num_height_bins=num_height_bins,
        num_locations=num_locations,
        num_velocity_bins=num_velocity_bins,
    )
    s.dump(output_path)

    s = SnailData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    SnailData.save_init(
        output_path,
        max_height=2,
        omega=0.11,
        max_velocity=4,
        name="test",
        sphere_radius=2,
    )
    s = SnailData.load(output_path)
    assert s.omega == 0.11
    assert s.max_height == 2
    assert s.max_velocity == 4
    assert s.sphere_radius == 2
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    surface_density = np.full((num_velocity_bins, num_height_bins, num_locations), 0.743, dtype=np.float64)
    azimuthal_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), 4.05, dtype=np.float64)
    radial_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), -120.4, dtype=np.float64)
    SnailData.save_frame(
        output_path,
        0,
        azimuthal_velocity=azimuthal_velocity,
        radial_velocity=radial_velocity,
        surface_density=surface_density,
    )
    s = SnailData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.array_equal(s.surface_density[:, :, 0, :], surface_density)
    assert np.array_equal(s.azimuthal_velocity[:, :, 0, :], azimuthal_velocity)
    assert np.array_equal(s.radial_velocity[:, :, 0, :], radial_velocity)

    for i in range(1, 5):
        current_time = 0.74 * i + 598
        surface_density = np.full((num_velocity_bins, num_height_bins, num_locations), 3 * current_time, dtype=np.float64)
        azimuthal_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), current_time + 123, dtype=np.float64)
        radial_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), current_time + 3, dtype=np.float64)
        SnailData.save_frame(
            output_path,
            i,
            azimuthal_velocity=azimuthal_velocity,
            radial_velocity=radial_velocity,
            surface_density=surface_density,
        )
    s = SnailData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 0.74 * i + 598
        surface_density = np.full((num_velocity_bins, num_height_bins, num_locations), 3 * current_time, dtype=np.float64)
        azimuthal_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), current_time + 123, dtype=np.float64)
        radial_velocity = np.full((num_velocity_bins, num_height_bins, num_locations), current_time + 3, dtype=np.float64)

        assert np.array_equal(s.surface_density[:, :, i, :], surface_density)
        assert np.array_equal(s.azimuthal_velocity[:, :, i, :], azimuthal_velocity)
        assert np.array_equal(s.radial_velocity[:, :, i, :], radial_velocity)
        assert s.completeness[i]
