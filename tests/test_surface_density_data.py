"""Test surface density data."""

from pathlib import Path

import numpy as np
import pytest

from novie_data.interface import NovieData
from novie_data.surface_density_data import GridData
from novie_data_gen.builtins._verifier import InconsistentArrayLengthError


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(GridData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float64)
    s = GridData(
        name="test",
        projection_xy=proj,
        projection_xz=proj,
        projection_yz=proj,
        flat_projection_xy=proj,
        extent=20.0,
        disc_scale_mass=1e5,
        disc_scale_length=3,
    )
    assert s.extent == 20.0


def test_init_inconsistent_shapes() -> None:
    """Test that the constructor errors when the arrays are inconsistent."""
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float64)
    wrong_proj = np.zeros((num_bins, num_bins, 3), dtype=np.float64)
    with pytest.raises(InconsistentArrayLengthError):
        _ = GridData(
            name="test",
            projection_xy=proj,
            projection_xz=proj,
            projection_yz=proj,
            flat_projection_xy=wrong_proj,
            extent=20.0,
            disc_scale_mass=1e5,
            disc_scale_length=3,
        )


def test_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float64)
    s = GridData(
        name="test",
        projection_xy=proj,
        projection_xz=proj,
        projection_yz=proj,
        flat_projection_xy=proj,
        extent=20.0,
        disc_scale_mass=1e5,
        disc_scale_length=3,
    )
    s.dump(output_path)
    t = GridData.load(output_path)
    assert s == t


def test_deserialization_v3() -> None:
    """Test deserialization of v3."""
    scale_mass: float = 1e5
    scale_length: float = 3
    extent: float = 20.0
    num_bins: int = 2
    num_frames: int = 7

    input_path = Path("test_data/surface_density_v3.hdf5")
    t = GridData.load(input_path)

    assert t.num_bins == num_bins
    assert t.projection_xy.shape == (num_bins, num_bins, num_frames)
    assert t.extent == extent
    assert t.disc_scale_mass == scale_mass
    assert t.disc_scale_length == scale_length
