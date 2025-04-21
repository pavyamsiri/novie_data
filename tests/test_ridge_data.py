"""Test ridge data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.ridge_data import RidgeData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(RidgeData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_radial_bins: int = 13
    num_velocity_bins: int = 123
    num_frames: int = 3
    min_radius: float = 0.0
    max_radius: float = 123.0
    min_velocity: float = -12
    max_velocity: float = -10
    density = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float64)

    s = RidgeData(
        name="test",
        mass_density=density,
        number_density=density,
        min_radius=min_radius,
        max_radius=max_radius,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    assert s.min_radius == min_radius


def test_serde(tmp_path: Path) -> None:
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
    density = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float64)

    s = RidgeData(
        name="test",
        mass_density=density,
        number_density=density,
        min_radius=min_radius,
        max_radius=max_radius,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    s.dump(output_path)
    t = RidgeData.load(output_path)
    assert s == t


def test_deserialization_v2() -> None:
    """Test deserialization of v2."""
    num_radial_bins: int = 13
    num_velocity_bins: int = 123
    num_frames: int = 3
    min_radius: float = 0.0
    max_radius: float = 123.0
    min_velocity: float = -12
    max_velocity: float = -10

    input_path = Path("test_data/ridge_v2.hdf5")
    t = RidgeData.load(input_path)

    assert t.num_radial_bins == num_radial_bins
    assert t.num_velocity_bins == num_velocity_bins
    assert t.min_radius == min_radius
    assert t.max_radius == max_radius
    assert t.min_velocity == min_velocity
    assert t.max_velocity == max_velocity
    assert t.num_frames == num_frames
    assert t.mass_density.shape == (num_velocity_bins, num_radial_bins, num_frames)


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    num_velocity_bins = 36
    num_radial_bins = 20
    num_frames = 10
    s = RidgeData.empty(
        num_frames=num_frames,
        num_velocity_bins=num_velocity_bins,
        num_radial_bins=num_radial_bins,
    )
    s.dump(output_path)

    s = RidgeData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    RidgeData.save_init(
        output_path,
        max_radius=12,
        max_velocity=41,
        min_radius=3,
        min_velocity=-123,
        name="test",
    )
    s = RidgeData.load(output_path)
    assert s.name == "test"
    assert s.max_radius == 12
    assert s.max_velocity == 41
    assert s.min_radius == 3
    assert s.min_velocity == -123
    assert np.all(~s.completeness)

    mass_density = np.full((num_velocity_bins, num_radial_bins), 1743.023, dtype=np.float64)
    number_density = np.full((num_velocity_bins, num_radial_bins), 312.854, dtype=np.float64)
    RidgeData.save_frame(
        output_path,
        0,
        mass_density=mass_density,
        number_density=number_density,
    )
    s = RidgeData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.array_equal(s.mass_density[:, :, 0], mass_density)
    assert np.array_equal(s.number_density[:, :, 0], number_density)

    for i in range(1, 5):
        current_time = 0.41 * i - 10
        mass_density = np.full((num_velocity_bins, num_radial_bins), current_time + 9743.1222, dtype=np.float64)
        number_density = np.full((num_velocity_bins, num_radial_bins), 12 * current_time + 0.57, dtype=np.float64)
        RidgeData.save_frame(
            output_path,
            i,
            mass_density=mass_density,
            number_density=number_density,
        )
    s = RidgeData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 0.41 * i - 10
        mass_density = np.full((num_velocity_bins, num_radial_bins), current_time + 9743.1222, dtype=np.float64)
        number_density = np.full((num_velocity_bins, num_radial_bins), 12 * current_time + 0.57, dtype=np.float64)

        assert np.array_equal(s.mass_density[:, :, i], mass_density)
        assert np.array_equal(s.number_density[:, :, i], number_density)
        assert s.completeness[i]
