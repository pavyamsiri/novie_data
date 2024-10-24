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
        angular_momentum=angular_momentum,
        residuals=residuals,
        relative_errors=residuals,
        sum_of_square_residuals=sse,
        mean_absolute_relative_error=sse,
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
        angular_momentum=angular_momentum,
        residuals=residuals,
        relative_errors=residuals,
        sum_of_square_residuals=sse,
        mean_absolute_relative_error=sse,
    )
    s.dump(output_path)
    t = WrinkleResidualsData.load(output_path)
    assert s == t
