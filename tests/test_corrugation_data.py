"""Test corrugation data."""

from pathlib import Path

import numpy as np

from novie_data.corrugation_data import CorrugationData
from novie_data.interface import NovieData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(CorrugationData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_height_bins: int = 36
    num_radial_bins: int = 20
    num_locations: int = 7
    num_frames: int = 4
    proj = np.zeros((num_height_bins, num_radial_bins, num_frames, num_locations), dtype=np.float64)
    radii = np.zeros(num_radial_bins, dtype=np.float64)
    mean_height = np.zeros((num_radial_bins, num_frames, num_locations), dtype=np.float64)
    min_radius = 0
    max_radius = 12
    max_height = 2
    cutoff_frequency = 0.5
    inner_radius = 0
    outer_radius = 2
    min_longitude_deg = 220
    max_longitude_deg = 240
    s = CorrugationData(
        name="test",
        projection_rz=proj,
        radii=radii,
        mean_height=mean_height,
        mean_height_error=mean_height,
        distance_error=1.12,
        min_radius=min_radius,
        max_radius=max_radius,
        max_height=max_height,
        cutoff_frequency=cutoff_frequency,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
    )
    assert s.distance_error == 1.12


def test_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_height_bins: int = 36
    num_radial_bins: int = 20
    num_locations: int = 7
    num_frames: int = 4
    proj = np.zeros((num_height_bins, num_radial_bins, num_frames, num_locations), dtype=np.float64)
    radii = np.zeros(num_radial_bins, dtype=np.float64)
    mean_height = np.zeros((num_radial_bins, num_frames, num_locations), dtype=np.float64)
    min_radius = 0
    max_radius = 12
    max_height = 2
    cutoff_frequency = 0.5
    inner_radius = 0
    outer_radius = 2
    min_longitude_deg = 220
    max_longitude_deg = 240
    s = CorrugationData(
        name="test",
        projection_rz=proj,
        radii=radii,
        mean_height=mean_height,
        mean_height_error=mean_height,
        distance_error=1.12,
        min_radius=min_radius,
        max_radius=max_radius,
        max_height=max_height,
        cutoff_frequency=cutoff_frequency,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
    )
    s.dump(output_path)
    t = CorrugationData.load(output_path)
    assert s == t


def test_deserialization_v3() -> None:
    """Test deserialization of v3."""
    num_height_bins: int = 36
    num_radial_bins: int = 20
    num_locations: int = 7
    num_frames: int = 4

    input_path = Path("test_data/corrugation_v3.hdf5")
    t = CorrugationData.load(input_path)

    assert t.num_radial_bins == num_radial_bins
    assert t.num_height_bins == num_height_bins
    assert t.num_locations == num_locations
    assert t.num_frames == num_frames
    assert t.projection_rz.shape == (num_height_bins, num_radial_bins, num_frames, num_locations)


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    num_height_bins = 36
    num_radial_bins = 20
    num_frames = 10
    num_locations = 7
    s = CorrugationData.empty(
        num_frames=num_frames,
        num_height_bins=num_height_bins,
        num_radial_bins=num_radial_bins,
        num_locations=num_locations,
    )
    s.dump(output_path)

    s = CorrugationData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    max_radius = 12
    min_radius = 8
    radii = np.linspace(min_radius, max_radius, num_radial_bins, dtype=np.float64).reshape(num_radial_bins)
    CorrugationData.save_init(
        output_path,
        cutoff_frequency=12.0,
        distance_error=1.033,
        inner_radius=0,
        outer_radius=7,
        max_height=2,
        min_longitude_deg=140,
        max_longitude_deg=200,
        max_radius=max_radius,
        min_radius=min_radius,
        name="test",
        radii=radii,
    )
    s = CorrugationData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.cutoff_frequency == 12.0
    assert s.distance_error == 1.033
    assert s.inner_radius == 0
    assert s.outer_radius == 7
    assert s.max_height == 2
    assert s.min_longitude_deg == 140
    assert s.max_longitude_deg == 200
    assert s.max_radius == 12
    assert s.min_radius == 8
    assert np.array_equal(s.radii, radii)
    assert np.all(~s.completeness)

    proj = np.full((num_height_bins, num_radial_bins, num_locations), 1.34, dtype=np.float64)
    mean_height = np.full((num_radial_bins, num_locations), 0.12, dtype=np.float64)
    mean_height_error = np.full((num_radial_bins, num_locations), 0.0001, dtype=np.float64)
    CorrugationData.save_frame(
        output_path,
        0,
        projection_rz=proj,
        mean_height=mean_height,
        mean_height_error=mean_height_error,
    )
    s = CorrugationData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.array_equal(s.projection_rz[:, :, 0, :], proj)
    assert np.array_equal(s.mean_height[:, 0, :], mean_height)
    assert np.array_equal(s.mean_height_error[:, 0, :], mean_height_error)

    for i in range(1, 5):
        current_time = 0.41 * i - 10
        current_proj_value = np.square(current_time) + 0.001
        current_mean_height_value = -0.71 * current_time + 410.57403
        current_mean_height_error_value = 13.0 * current_time
        proj = np.full((num_height_bins, num_radial_bins, num_locations), current_proj_value, dtype=np.float64)
        mean_height = np.full((num_radial_bins, num_locations), current_mean_height_value, dtype=np.float64)
        mean_height_error = np.full((num_radial_bins, num_locations), current_mean_height_error_value, dtype=np.float64)
        CorrugationData.save_frame(
            output_path,
            i,
            projection_rz=proj,
            mean_height=mean_height,
            mean_height_error=mean_height_error,
        )
    s = CorrugationData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 0.41 * i - 10
        current_proj_value = np.square(current_time) + 0.001
        current_mean_height_value = -0.71 * current_time + 410.57403
        current_mean_height_error_value = 13.0 * current_time
        proj = np.full((num_height_bins, num_radial_bins, num_locations), current_proj_value, dtype=np.float64)
        mean_height = np.full((num_radial_bins, num_locations), current_mean_height_value, dtype=np.float64)
        mean_height_error = np.full((num_radial_bins, num_locations), current_mean_height_error_value, dtype=np.float64)

        assert np.array_equal(s.projection_rz[:, :, i, :], proj)
        assert np.array_equal(s.mean_height[:, i, :], mean_height)
        assert np.array_equal(s.mean_height_error[:, i, :], mean_height_error)
        assert s.completeness[i]
