"""Test wrinkle data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.wrinkle_data import WrinkleData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(WrinkleData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_bins: int = 12
    num_frames: int = 9
    num_neighbourhoods: int = 32
    min_lz: float = 0
    max_lz: float = 1e5
    angular_momentum = np.zeros(num_bins, dtype=np.float64)
    mean_velocity = np.zeros((num_bins, num_frames, num_neighbourhoods), dtype=np.float64)
    s = WrinkleData(
        name="test",
        angular_momentum=angular_momentum,
        mean_radial_velocity=mean_velocity,
        mean_radial_velocity_error=mean_velocity,
        min_lz=min_lz,
        max_lz=max_lz,
        sphere_radius=1.0,
        distance_error=1.01,
    )
    assert s.distance_error == 1.01


def test_serde(tmp_path: Path) -> None:
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
    angular_momentum = np.zeros(num_bins, dtype=np.float64)
    mean_velocity = np.zeros((num_bins, num_frames, num_neighbourhoods), dtype=np.float64)
    s = WrinkleData(
        name="test",
        angular_momentum=angular_momentum,
        mean_radial_velocity=mean_velocity,
        mean_radial_velocity_error=mean_velocity,
        min_lz=min_lz,
        max_lz=max_lz,
        sphere_radius=1.0,
        distance_error=1.01,
    )
    s.dump(output_path)
    t = WrinkleData.load(output_path)
    assert s == t


def test_deserialization_v3() -> None:
    """Test deserialization of v3."""
    num_bins: int = 12
    num_frames: int = 9
    num_neighbourhoods: int = 32
    min_lz: float = 0
    max_lz: float = 1e5
    radius: float = 1.0
    distance_error: float = 1.01

    input_path = Path("test_data/wrinkle_v3.hdf5")
    t = WrinkleData.load(input_path)

    assert t.distance_error == distance_error
    assert t.min_lz == min_lz
    assert t.max_lz == max_lz
    assert t.sphere_radius == radius
    assert t.mean_radial_velocity.shape == (num_bins, num_frames, num_neighbourhoods)


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    num_momentum_bins = 20
    num_frames = 10
    num_locations = 7
    s = WrinkleData.empty(
        num_frames=num_frames,
        num_momentum_bins=num_momentum_bins,
        num_locations=num_locations,
    )
    s.dump(output_path)

    s = WrinkleData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    max_radius = 11.11
    min_radius = 0
    angular_momentum = np.linspace(min_radius, max_radius, num_momentum_bins, dtype=np.float64).reshape(num_momentum_bins)
    WrinkleData.save_init(
        output_path,
        angular_momentum=angular_momentum,
        distance_error=1.3,
        max_lz=11.11,
        omega=-0.44,
        min_lz=0,
        name="test",
        sphere_radius=1.0,
    )
    s = WrinkleData.load(output_path)

    assert s.distance_error == 1.3
    assert s.omega == -0.44
    assert s.max_lz == 11.11
    assert s.min_lz == 0
    assert s.name == "test"
    assert s.sphere_radius == 1.0
    assert np.array_equal(s.angular_momentum, angular_momentum)
    assert np.all(~s.completeness)

    mean_radial_velocity = np.full((num_momentum_bins, num_locations), 89.02, dtype=np.float64)
    mean_radial_velocity_error = np.full((num_momentum_bins, num_locations), 32.129, dtype=np.float64)
    WrinkleData.save_frame(
        output_path,
        0,
        mean_radial_velocity=mean_radial_velocity,
        mean_radial_velocity_error=mean_radial_velocity_error,
    )
    s = WrinkleData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.array_equal(s.mean_radial_velocity[:, 0, :], mean_radial_velocity)
    assert np.array_equal(s.mean_radial_velocity_error[:, 0, :], mean_radial_velocity_error)

    for i in range(1, 5):
        current_time = 52.0 * i + 123.0
        current_mean_radial_velocity_value = 100.4 * current_time + 0.041
        current_mean_radial_velocity_error_value = -0.0152 * current_time
        mean_radial_velocity = np.full((num_momentum_bins, num_locations), current_mean_radial_velocity_value, dtype=np.float64)
        mean_radial_velocity_error = np.full(
            (num_momentum_bins, num_locations), current_mean_radial_velocity_error_value, dtype=np.float64
        )
        WrinkleData.save_frame(
            output_path,
            i,
            mean_radial_velocity=mean_radial_velocity,
            mean_radial_velocity_error=mean_radial_velocity_error,
        )
    s = WrinkleData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 52.0 * i + 123.0
        current_mean_radial_velocity_value = 100.4 * current_time + 0.041
        current_mean_radial_velocity_error_value = -0.0152 * current_time
        mean_radial_velocity = np.full((num_momentum_bins, num_locations), current_mean_radial_velocity_value, dtype=np.float64)
        mean_radial_velocity_error = np.full(
            (num_momentum_bins, num_locations), current_mean_radial_velocity_error_value, dtype=np.float64
        )

        assert np.array_equal(s.mean_radial_velocity[:, i, :], mean_radial_velocity)
        assert np.array_equal(s.mean_radial_velocity_error[:, i, :], mean_radial_velocity_error)
        assert s.completeness[i]
