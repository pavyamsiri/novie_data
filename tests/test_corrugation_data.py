"""Test corrugation data."""

from pathlib import Path

import numpy as np

from novie_data.corrugation_data import CorrugationData, HeightBinningData, RadialBinningData, WedgeData
from novie_data.interface import NovieData


def test_corrugation_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(CorrugationData, NovieData)


def test_corrugation_data_init() -> None:
    """Test the constructor."""
    num_height_bins: int = 36
    num_radial_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4
    proj = np.zeros((num_height_bins, num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    radii = np.zeros(num_radial_bins, dtype=np.float32)
    mean_height = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    radial_bins = RadialBinningData(
        num_bins=num_radial_bins,
        min_radius=0,
        max_radius=12,
    )
    height_bins = HeightBinningData(
        num_bins=num_height_bins,
        max_height=2,
        cutoff_frequency=0.5,
    )
    wedge = WedgeData(
        num_wedges=num_neighbourhoods,
        inner_radius=0,
        outer_radius=2,
        min_longitude_deg=220,
        max_longitude_deg=240,
    )
    s = CorrugationData(
        name="test",
        projection_rz=proj,
        radii=radii,
        mean_height=mean_height,
        mean_height_error=mean_height,
        radial_bins=radial_bins,
        height_bins=height_bins,
        wedge_data=wedge,
        distance_error=1.12,
    )
    assert s.distance_error == 1.12


def test_corrugation_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_height_bins: int = 36
    num_radial_bins: int = 20
    num_neighbourhoods: int = 7
    num_frames: int = 4
    proj = np.zeros((num_height_bins, num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    radii = np.zeros(num_radial_bins, dtype=np.float32)
    mean_height = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    radial_bins = RadialBinningData(
        num_bins=num_radial_bins,
        min_radius=0,
        max_radius=12,
    )
    height_bins = HeightBinningData(
        num_bins=num_height_bins,
        max_height=2,
        cutoff_frequency=0.5,
    )
    wedge = WedgeData(
        num_wedges=num_neighbourhoods,
        inner_radius=0,
        outer_radius=2,
        min_longitude_deg=220,
        max_longitude_deg=240,
    )
    s = CorrugationData(
        name="test",
        projection_rz=proj,
        radii=radii,
        mean_height=mean_height,
        mean_height_error=mean_height,
        radial_bins=radial_bins,
        height_bins=height_bins,
        wedge_data=wedge,
        distance_error=1.12,
    )
    s.dump(output_path)
    t = CorrugationData.load(output_path)
    assert s == t
