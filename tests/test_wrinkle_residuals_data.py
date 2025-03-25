"""Test wrinkle data."""

from pathlib import Path

import numpy as np

from novie_data.interface import NovieData
from novie_data.wrinkle_residuals_data import WrinkleResidualsData


def test_wrinkle_residuals_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(WrinkleResidualsData, NovieData)


def test_wrinkle_residuals_data_init() -> None:
    """Test the constructor."""
    num_radial_bins: int = 123
    num_frames: int = 7
    num_neighbourhoods: int = 87
    angular_momentum = np.zeros(num_radial_bins, dtype=np.float32)
    residuals = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    sse = np.zeros((num_frames, num_neighbourhoods), dtype=np.float32)
    s = WrinkleResidualsData(
        name="test",
        metric_name="sse",
        metric=residuals,
        summary=sse,
        bin_values=angular_momentum,
    )
    assert s.name == "test"


def test_wrinkle_residuals_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_radial_bins: int = 123
    num_frames: int = 7
    num_neighbourhoods: int = 87
    angular_momentum = np.zeros(num_radial_bins, dtype=np.float32)
    residuals = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    sse = np.zeros((num_frames, num_neighbourhoods), dtype=np.float32)
    s = WrinkleResidualsData(
        name="test",
        metric_name="sse",
        metric=residuals,
        summary=sse,
        bin_values=angular_momentum,
    )
    s.dump(output_path)
    t = WrinkleResidualsData.load(output_path)
    assert s == t


def test_wrinkle_residuals_data_deserialization_v3() -> None:
    """Test deserialization of v3."""
    num_radial_bins: int = 123
    num_frames: int = 7
    num_neighbourhoods: int = 87

    input_path = Path("test_data/wrinkle_residuals_v3.hdf5")
    t = WrinkleResidualsData.load(input_path)

    assert t.num_frames == num_frames
    assert t.num_locations == num_neighbourhoods
    assert t.metric.shape == (num_radial_bins, num_frames, num_neighbourhoods)
