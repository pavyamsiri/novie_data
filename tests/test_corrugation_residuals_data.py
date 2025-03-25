"""Test corrugation data."""

from pathlib import Path

import numpy as np

from novie_data.corrugation_residuals_data import CorrugationResidualsData
from novie_data.interface import NovieData


def test_corrugation_residuals_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(CorrugationResidualsData, NovieData)


def test_corrugation_residuals_data_init() -> None:
    """Test the constructor."""
    num_radial_bins: int = 123
    num_frames: int = 7
    num_neighbourhoods: int = 87
    radii = np.zeros(num_radial_bins, dtype=np.float32)
    residuals = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    sse = np.zeros((num_frames, num_neighbourhoods), dtype=np.float32)
    s = CorrugationResidualsData(
        name="test",
        metric_name="sse",
        metric=residuals,
        summary=sse,
        bin_values=radii,
    )
    assert s.name == "test"


def test_corrugation_residuals_data_serde(tmp_path: Path) -> None:
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
    radii = np.zeros(num_radial_bins, dtype=np.float32)
    residuals = np.zeros((num_radial_bins, num_frames, num_neighbourhoods), dtype=np.float32)
    sse = np.zeros((num_frames, num_neighbourhoods), dtype=np.float32)
    s = CorrugationResidualsData(
        name="test",
        metric_name="sse",
        metric=residuals,
        summary=sse,
        bin_values=radii,
    )
    s.dump(output_path)
    t = CorrugationResidualsData.load(output_path)
    assert s == t


def test_corrugation_residuals_data_deserialization_v2() -> None:
    """Test deserialization of v2."""
    num_radial_bins: int = 123
    num_frames: int = 7
    num_neighbourhoods: int = 87

    input_path = Path("test_data/corrugation_residuals_v2.hdf5")
    t = CorrugationResidualsData.load(input_path)

    assert t.metric.shape == (num_radial_bins, num_frames, num_neighbourhoods)
