"""Test surface density data."""

from pathlib import Path

import numpy as np

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


def test_surface_density_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "surface_density.hdf5"
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
