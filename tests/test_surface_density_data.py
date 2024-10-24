"""Test surface density data."""

from pathlib import Path

import numpy as np
import pytest

from novie_data.errors import InconsistentArrayShapeError, WrongArrayLengthError
from novie_data.interface import NovieData
from novie_data.surface_density_data import ExponentialDiscProfileData, SurfaceDensityData


def test_surface_density_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SurfaceDensityData, NovieData)


def test_surface_density_data_init() -> None:
    """Test the constructor."""
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float32)
    disc_profile = ExponentialDiscProfileData(
        scale_mass=1e5,
        scale_length=3,
    )
    s = SurfaceDensityData(
        name="test",
        projection_xy=proj,
        projection_xz=proj,
        projection_yz=proj,
        flat_projection_xy=proj,
        extent=20.0,
        num_bins=num_bins,
        disc_profile=disc_profile,
    )
    assert s.extent == 20.0


def test_surface_density_data_init_inconsistent_shapes() -> None:
    """Test that the constructor errors when the arrays are inconsistent."""
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float32)
    wrong_proj = np.zeros((num_bins, num_bins, 3), dtype=np.float32)
    disc_profile = ExponentialDiscProfileData(
        scale_mass=1e5,
        scale_length=3,
    )
    with pytest.raises(InconsistentArrayShapeError):
        _ = SurfaceDensityData(
            name="test",
            projection_xy=proj,
            projection_xz=proj,
            projection_yz=proj,
            flat_projection_xy=wrong_proj,
            extent=20.0,
            num_bins=num_bins,
            disc_profile=disc_profile,
        )


def test_surface_density_data_init_wrong_length() -> None:
    """Test that the constructor errors when the arrays are inconsistent."""
    num_bins: int = 2
    proj = np.zeros((num_bins // 2, num_bins + 1, 7), dtype=np.float32)
    disc_profile = ExponentialDiscProfileData(
        scale_mass=1e5,
        scale_length=3,
    )
    with pytest.raises(WrongArrayLengthError):
        _ = SurfaceDensityData(
            name="test",
            projection_xy=proj,
            projection_xz=proj,
            projection_yz=proj,
            flat_projection_xy=proj,
            extent=20.0,
            num_bins=num_bins,
            disc_profile=disc_profile,
        )


def test_surface_density_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_bins: int = 2
    proj = np.zeros((num_bins, num_bins, 7), dtype=np.float32)
    disc_profile = ExponentialDiscProfileData(
        scale_mass=1e5,
        scale_length=3,
    )
    s = SurfaceDensityData(
        name="test",
        projection_xy=proj,
        projection_xz=proj,
        projection_yz=proj,
        flat_projection_xy=proj,
        extent=20.0,
        num_bins=num_bins,
        disc_profile=disc_profile,
    )
    s.dump(output_path)
    t = SurfaceDensityData.load(output_path)
    assert s == t
